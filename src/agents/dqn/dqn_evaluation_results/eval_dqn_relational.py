"""
eval_dqn_relational.py
======================
Evaluate the DQN + UAVAttentionExtractor model (models/dqn_relational/) across
the same 5 scalability conditions and 50 seeds used for every other agent.

Writes:
  baseline_results/sweep/dqn_relational_results.csv  (raw per-seed rows)
  baseline_results/dqn_relational_summary.csv         (per-condition aggregates)

Also prints an NDR / Jain's table so you can copy numbers directly into
Table tab:arch_algo_decomp.

Usage (RunPod — run from project root):
    PYTHONIOENCODING=utf-8 uv run python \\
        src/agents/dqn/dqn_evaluation_results/eval_dqn_relational.py

    # Fewer seeds for a quick sanity check:
    PYTHONIOENCODING=utf-8 uv run python \\
        src/agents/dqn/dqn_evaluation_results/eval_dqn_relational.py --seeds 10
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

# ── Path bootstrap ────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent          # dqn_evaluation_results/
_DQN  = _HERE.parent                             # dqn/
_SRC  = _HERE.parents[2]                         # src/
for _p in (str(_SRC), str(_SRC / "environment"), str(_DQN)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import gymnasium
import torch
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

from environment.uav_env import UAVEnvironment

# ── Paths ─────────────────────────────────────────────────────────────────────
MODEL_PATH  = _DQN / "models" / "dqn_relational" / "dqn_final.zip"
CONFIG_PATH = _DQN / "models" / "dqn_relational" / "training_config.json"
OUT_DIR     = _HERE / "baseline_results"
SWEEP_DIR   = OUT_DIR / "sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SWEEP_DIR.mkdir(parents=True, exist_ok=True)
RAW_CSV     = SWEEP_DIR / "dqn_relational_results.csv"
SUMMARY_CSV = OUT_DIR / "dqn_relational_summary.csv"

# ── Evaluation conditions (must match algorithm_architecture_comparison.py) ───
CONDITIONS = [
    {"label": "100x100_N10",  "grid_size": (100, 100), "n_sensors": 10},
    {"label": "200x200_N20",  "grid_size": (200, 200), "n_sensors": 20},
    {"label": "300x300_N30",  "grid_size": (300, 300), "n_sensors": 30},
    {"label": "400x400_N40",  "grid_size": (400, 400), "n_sensors": 40},
    {"label": "500x500_N50",  "grid_size": (500, 500), "n_sensors": 50},
]

N_SEEDS     = 50
MAX_BATTERY = 274.0
MAX_SENSORS = 50

ENV_BASE = {
    "max_steps":          2100,
    "path_loss_exponent": 3.8,
    "rssi_threshold":     -85.0,
    "sensor_duty_cycle":  10.0,
    "max_battery":        274.0,
    "render_mode":        None,
}

RAW_FIELDS = [
    "seed", "label", "grid_w", "grid_h", "n_sensors",
    "ndr_pct", "jfi", "bytes_per_wh", "battery_pct",
    "total_data_bytes", "steps",
]


# ── Padded env wrapper ────────────────────────────────────────────────────────

class _PaddedEnv(UAVEnvironment):
    """Zero-pads observations to MAX_SENSORS slots; snapshots episode state."""

    def __init__(self, max_sensors_limit: int = MAX_SENSORS, **kw):
        self.max_sensors_limit = max_sensors_limit
        self._snap: dict | None = None
        super().__init__(**kw)

        raw = self.observation_space.shape[0]
        self._fps = 0
        for uav_f in range(raw + 1):
            rem = raw - uav_f
            if rem > 0 and rem % self.num_sensors == 0:
                self._fps = rem // self.num_sensors
                break
        if self._fps == 0:
            raise ValueError(f"Cannot infer fps: raw={raw}, n={self.num_sensors}")

        padded = raw + (self.max_sensors_limit - self.num_sensors) * self._fps
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(padded,), dtype=np.float32
        )

    def _pad(self, obs: np.ndarray) -> np.ndarray:
        pad = np.zeros(
            (self.max_sensors_limit - self.num_sensors) * self._fps, dtype=np.float32
        )
        return np.concatenate([obs, pad]).astype(np.float32)

    def reset(self, **kw):
        obs, info = super().reset(**kw)
        return self._pad(obs), info

    def step(self, action):
        obs, r, te, tr, info = super().step(action)
        if te or tr:
            self._snap = {
                "sensors":         self.sensors,
                "sensors_visited": self.sensors_visited,
                "battery":         float(self.uav.battery),
                "total_collected": float(self.total_data_collected),
                "n_sensors":       self.num_sensors,
            }
        return self._pad(obs), r, te, tr, info


# ── Metric extraction ─────────────────────────────────────────────────────────

def _metrics(snap: dict) -> dict:
    sensors  = snap["sensors"]
    n        = snap["n_sensors"]
    ndr      = len(snap["sensors_visited"]) / n * 100.0

    rates = [float(s.total_data_transmitted) for s in sensors]
    s_sum = sum(rates)
    sq_sum = sum(r ** 2 for r in rates)
    jfi   = (s_sum ** 2) / (n * sq_sum) if sq_sum > 0 else 0.0

    battery  = snap["battery"]
    energy   = max(MAX_BATTERY - battery, 1e-6)
    bpwh     = snap["total_collected"] / energy

    return {
        "ndr_pct":          ndr,
        "jfi":              jfi,
        "bytes_per_wh":     bpwh,
        "battery_pct":      battery / MAX_BATTERY * 100.0,
        "total_data_bytes": snap["total_collected"],
    }


# ── Single episode runner ─────────────────────────────────────────────────────

def _run_episode(model, cfg: dict, cond: dict, seed: int) -> dict:
    n_stack = cfg.get("n_stack", 4)

    env_kw = {
        **ENV_BASE,
        "grid_size":          cond["grid_size"],
        "num_sensors":        cond["n_sensors"],
        "max_sensors_limit":  MAX_SENSORS,
    }
    vec = DummyVecEnv([lambda: _PaddedEnv(**env_kw)])
    vec = VecFrameStack(vec, n_stack=n_stack)

    inner = vec
    while hasattr(inner, "venv"):
        inner = inner.venv
    base: _PaddedEnv = inner.envs[0]

    import random
    random.seed(seed)
    np.random.seed(seed)
    obs   = vec.reset()
    done  = np.array([False])
    steps = 0
    while not done[0]:
        with torch.no_grad():
            action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = vec.step(action)
        done   = dones
        steps += 1
    vec.close()

    if base._snap is None:
        return {}

    m = _metrics(base._snap)
    m["steps"] = steps
    return m


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=N_SEEDS,
                        help="number of seeds per condition (default 50)")
    args = parser.parse_args()
    seeds = list(range(args.seeds))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Device: {}".format(device))
    print("Loading model from: {}".format(MODEL_PATH))

    if not MODEL_PATH.exists():
        print("ERROR: model not found — train first with train_dqn_relational.py")
        sys.exit(1)

    model = DQN.load(str(MODEL_PATH), device=device)
    model.policy.set_training_mode(False)

    cfg = {"use_frame_stacking": True, "n_stack": 4, "max_sensors_limit": MAX_SENSORS}
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            cfg.update(json.load(f))
    print("Config: n_stack={}, max_sensors={}".format(
        cfg.get("n_stack", 4), cfg.get("max_sensors_limit", MAX_SENSORS)))
    print()

    # Resume support: skip already-done (label, seed) pairs
    done_keys: set[tuple] = set()
    if RAW_CSV.exists():
        with open(RAW_CSV, newline="") as f:
            for row in csv.DictReader(f):
                done_keys.add((row["label"], int(row["seed"])))
        print("Resuming — {} rows already done.".format(len(done_keys)))

    # Collect raw results
    raw_rows: list[dict] = []
    total = len(CONDITIONS) * len(seeds)
    done_count = 0

    for cond in CONDITIONS:
        label = cond["label"]
        cond_rows = []
        for seed in seeds:
            if (label, seed) in done_keys:
                done_count += 1
                continue
            m = _run_episode(model, cfg, cond, seed)
            if not m:
                continue
            row = {
                "seed":             seed,
                "label":            label,
                "grid_w":           cond["grid_size"][0],
                "grid_h":           cond["grid_size"][1],
                "n_sensors":        cond["n_sensors"],
                "ndr_pct":          m["ndr_pct"],
                "jfi":              m["jfi"],
                "bytes_per_wh":     m["bytes_per_wh"],
                "battery_pct":      m["battery_pct"],
                "total_data_bytes": m["total_data_bytes"],
                "steps":            m["steps"],
            }
            cond_rows.append(row)
            raw_rows.append(row)
            done_count += 1

            # Append immediately (crash-safe)
            write_header = not RAW_CSV.exists()
            with open(RAW_CSV, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=RAW_FIELDS)
                if write_header:
                    w.writeheader()
                w.writerow(row)

        ndrs  = [r["ndr_pct"] for r in cond_rows]
        jains = [r["jfi"]     for r in cond_rows]
        if ndrs:
            print("{:<16}  NDR={:5.1f}% +/-{:.1f}  Jain={:.3f} +/-{:.3f}  (n={})".format(
                label,
                np.mean(ndrs), 1.96 * np.std(ndrs, ddof=1) / np.sqrt(len(ndrs)),
                np.mean(jains), 1.96 * np.std(jains, ddof=1) / np.sqrt(len(jains)),
                len(ndrs),
            ))

    print("\nDone. {}/{} episodes evaluated.".format(done_count, total))

    # ── Aggregate summary CSV ─────────────────────────────────────────────────
    # Load ALL rows (including pre-existing ones)
    all_rows: list[dict] = []
    if RAW_CSV.exists():
        with open(RAW_CSV, newline="") as f:
            all_rows = list(csv.DictReader(f))

    summary_rows = []
    for cond in CONDITIONS:
        label = cond["label"]
        subset = [r for r in all_rows if r["label"] == label]
        for metric in ("ndr_pct", "jfi", "bytes_per_wh"):
            vals = np.array([float(r[metric]) for r in subset])
            n    = len(vals)
            if n == 0:
                continue
            mean = float(vals.mean())
            std  = float(vals.std(ddof=1)) if n > 1 else 0.0
            ci   = 1.96 * std / np.sqrt(n)
            summary_rows.append({
                "agent":     "DQN-Relational",
                "label":     label,
                "grid":      "{}x{}".format(cond["grid_size"][0], cond["grid_size"][1]),
                "n_sensors": cond["n_sensors"],
                "metric":    metric,
                "mean":      f"{mean:.4f}",
                "std":       f"{std:.4f}",
                "ci_half":   f"{ci:.4f}",
                "ci_lo":     f"{mean - ci:.4f}",
                "ci_hi":     f"{mean + ci:.4f}",
                "n_samples": n,
            })

    if summary_rows:
        with open(SUMMARY_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            w.writerows(summary_rows)
        print("Summary written: {}".format(SUMMARY_CSV))

    # ── Dissertation table (copy-pasteable) ───────────────────────────────────
    print("\n" + "=" * 72)
    print("  Table numbers for tab:arch_algo_decomp  (DQN + Relational column)")
    print("=" * 72)
    print("  {:20s}  {:>10s}  {:>10s}".format("Condition", "NDR (%)", "Jain's"))
    print("  " + "-" * 48)
    for cond in CONDITIONS:
        label = cond["label"]
        subset = [r for r in all_rows if r["label"] == label]
        if not subset:
            print("  {:20s}  {:>10s}  {:>10s}".format(label, "n/a", "n/a"))
            continue
        ndrs  = np.array([float(r["ndr_pct"]) for r in subset])
        jains = np.array([float(r["jfi"])     for r in subset])
        ndr_ci   = 1.96 * ndrs.std(ddof=1)  / np.sqrt(len(ndrs))
        jain_ci  = 1.96 * jains.std(ddof=1) / np.sqrt(len(jains))
        print("  {:20s}  {:5.1f} +/-{:.1f}  {:.3f} +/-{:.3f}".format(
            label, ndrs.mean(), ndr_ci, jains.mean(), jain_ci))
    print("=" * 72)


if __name__ == "__main__":
    main()
