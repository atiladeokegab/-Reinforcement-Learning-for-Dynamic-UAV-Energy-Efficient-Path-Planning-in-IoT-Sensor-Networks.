"""
Component ablation study: DQN Full Model vs A1 / A2 / A3 / A4.

  A1 — No Capture Effect  : same-SF co-transmissions all fail (no winner)
  A2 — Instant ADR        : adr_lambda = 1.0 (no EMA smoothing)
  A3 — No AoI Observation : urgency features zeroed in observation at inference
  A4 — No Domain Rand     : dqn_no_dr/dqn_final.zip (fixed-env trained model)

Evaluation: four (grid, N) conditions matching the scalability sweep,
20 seeds per condition, Welch's t-test vs Full Model.
NDR = sensors_visited / total_sensors × 100  (coverage %, consistent with dqn.py).

Usage:
    PYTHONIOENCODING=utf-8 uv run python src/agents/dqn/dqn_evaluation_results/ablation_study.py

Note on A4: trained on fixed 500×500, N=20 without domain randomisation.
At conditions with N≠20 the extra sensor slots carry real data the model
never saw during training; the performance drop therefore reflects both the
absence of DR and the out-of-distribution sensor count.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

_HERE = Path(__file__).resolve().parent
_SRC  = _HERE.parent.parent.parent
for _p in (str(_SRC / "environment"), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from environment.uav_env import UAVEnvironment

# ── Paths ─────────────────────────────────────────────────────────────────────
_MODELS   = _HERE.parent / "models"
FULL_PATH = _MODELS / "dqn_v3_retrain" / "dqn_final.zip"
A4_PATH   = _MODELS / "dqn_no_dr"      / "dqn_final.zip"
OUT_DIR   = _HERE / "baseline_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Eval config ───────────────────────────────────────────────────────────────
N_SEEDS         = 50
MAX_BATTERY     = 274.0
MAX_STEPS       = 2100
N_STACK         = 4
MAX_SENSORS_LIM = 50
FEATURES_PER    = 3   # buffer, urgency, link_quality per sensor

# Five conditions matching the scalability sweep in compare_agents.py exactly
CONDITIONS = [
    {"grid": (100, 100), "n_sensors": 10},
    {"grid": (200, 200), "n_sensors": 20},
    {"grid": (300, 300), "n_sensors": 30},
    {"grid": (400, 400), "n_sensors": 40},
    {"grid": (500, 500), "n_sensors": 50},
]


# ── Environment wrappers ──────────────────────────────────────────────────────

class _PaddedEnv(UAVEnvironment):
    """Zero-pads observations to MAX_SENSORS_LIM so the trained model fits."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        raw = self.observation_space.shape[0]
        padded = raw + (MAX_SENSORS_LIM - self.num_sensors) * FEATURES_PER
        import gymnasium
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(padded,), dtype=np.float32
        )
        self._last_terminal_stats: dict | None = None

    def _pad(self, obs: np.ndarray) -> np.ndarray:
        extra = (MAX_SENSORS_LIM - self.num_sensors) * FEATURES_PER
        return np.concatenate([obs, np.zeros(extra, dtype=np.float32)])

    def _capture_terminal_stats(self) -> dict:
        cr_list = [
            s.total_data_transmitted / max(s.total_data_generated, 1e-9)
            for s in self.sensors
        ]
        ndr = (len(self.sensors_visited) / self.num_sensors) * 100.0
        energy_used = self.uav.max_battery - self.uav.battery
        efficiency = self.total_data_collected / energy_used if energy_used > 0 else 0.0
        return {
            "cr_list":         cr_list,
            "ndr":             ndr,
            "efficiency":      efficiency,
            "sensors_visited": len(self.sensors_visited),
        }

    def reset(self, **kw):
        obs, info = super().reset(**kw)
        return self._pad(obs), info

    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        if term or trunc:
            info["terminal_stats"] = self._capture_terminal_stats()
        return self._pad(obs), r, term, trunc, info


class _A1NoCaptureEnv(_PaddedEnv):
    """A1: disables capture effect — all same-SF co-transmissions fail."""

    def _execute_collect_action(self, step_data_loss: float) -> float:
        urgencies_before = self._get_sensor_urgencies()
        self.uav.hover(duration=self.collection_duration)
        battery_used = self.uav.battery_drain_hover * self.collection_duration

        transmission_attempts: dict = {}
        for sensor in self.sensors:
            if sensor.data_buffer <= 0:
                continue
            sensor.update_spreading_factor(
                tuple(self.uav.position), current_step=self.current_step
            )
            p_link  = sensor.get_success_probability(
                tuple(self.uav.position), use_advanced_model=True
            )
            p_cycle = sensor.duty_cycle_probability
            if p_link * p_cycle > random.random():
                sf = sensor.spreading_factor
                transmission_attempts.setdefault(sf, []).append(sensor)

        # No capture effect: only singletons win; collisions always destroy
        successful_sf_slots: dict = {}
        collision_count = 0
        for sf, attempting in transmission_attempts.items():
            if len(attempting) == 1:
                successful_sf_slots[sf] = attempting[0]
            else:
                collision_count += len(attempting)

        total_bytes = 0.0
        new_sensors: list = []
        self.last_successful_collections = []
        for sf, winner in successful_sf_slots.items():
            b, ok = winner.collect_data(
                uav_position=tuple(self.uav.position),
                collection_duration=self.collection_duration,
            )
            if ok and b > 0:
                total_bytes += b
                self.total_data_collected += b
                if winner.sensor_id not in self.sensors_visited:
                    new_sensors.append(winner.sensor_id)
                    self.sensors_visited.add(winner.sensor_id)
                self.last_successful_collections.append((winner, sf))

        attempted_empty = any(s.data_buffer <= 0 for s in self.sensors)
        urgency_reduced = float(
            np.sum(np.maximum(0, urgencies_before - self._get_sensor_urgencies()))
        )
        all_collected = all(s.data_buffer <= 0 for s in self.sensors)
        self.last_step_bytes_collected = total_bytes
        mean_urgency = (
            float(np.mean([self._calculate_urgency(s) for s in successful_sf_slots.values()]))
            if successful_sf_slots else 0.0
        )
        return self.reward_fn.calculate_collection_reward(
            bytes_collected=total_bytes,
            was_new_sensor=len(new_sensors) > 0,
            was_empty=attempted_empty,
            all_sensors_collected=all_collected,
            battery_used=battery_used,
            collision_count=collision_count,
            data_loss=step_data_loss,
            urgency_reduced=urgency_reduced,
            sensor_buffers=[float(s.data_buffer) for s in self.sensors],
            sensor_urgency=mean_urgency,
        )


class _A2InstantADREnv(_PaddedEnv):
    """A2: sensors use adr_lambda = 1.0 (instant SF switching, no EMA lag)."""

    def reset(self, **kw):
        obs, info = super().reset(**kw)
        for s in self.sensors:
            s.adr_lambda = 1.0
        return obs, info


class _A3NoAoIEnv(_PaddedEnv):
    """A3: urgency features zeroed in observation at inference time."""

    def _zero_urgency(self, obs: np.ndarray) -> np.ndarray:
        out = obs.copy()
        for i in range(MAX_SENSORS_LIM):
            urgency_idx = 3 + FEATURES_PER * i + 1
            if urgency_idx < len(out):
                out[urgency_idx] = 0.0
        return out

    def reset(self, **kw):
        obs, info = super().reset(**kw)
        return self._zero_urgency(obs), info

    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        return self._zero_urgency(obs), r, term, trunc, info


# ── Env factory ───────────────────────────────────────────────────────────────

def _make_vec(env_cls, seed: int, grid_size: tuple, num_sensors: int) -> VecFrameStack:
    np.random.seed(seed)
    random.seed(seed)

    def _factory():
        return env_cls(
            grid_size=grid_size,
            num_sensors=num_sensors,
            max_battery=MAX_BATTERY,
            max_steps=MAX_STEPS,
        )

    vec = DummyVecEnv([_factory])
    vec = VecFrameStack(vec, n_stack=N_STACK)
    return vec


# ── Episode runner ────────────────────────────────────────────────────────────

def _run_episode(model: DQN, env_cls, seed: int,
                 grid_size: tuple, num_sensors: int) -> dict:
    vec = _make_vec(env_cls, seed, grid_size, num_sensors)

    base = vec.venv.envs[0]
    try:
        import gymnasium
        base.np_random, _ = gymnasium.utils.seeding.np_random(seed)
    except Exception:
        base.np_random = np.random.RandomState(seed)
    np.random.seed(seed)
    random.seed(seed)

    obs = vec.reset()
    cumulative_reward = 0.0
    terminal_stats: dict | None = None

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = vec.step(action)
        cumulative_reward += float(rewards[0])
        if dones[0]:
            terminal_stats = infos[0].get("terminal_stats")
            break

    vec.close()

    if terminal_stats is None:
        return {
            "cumulative_reward": cumulative_reward,
            "ndr": 0.0, "jains": 0.0, "gini": 0.0, "efficiency": 0.0,
        }

    cr_list = terminal_stats["cr_list"]
    arr = np.array(cr_list, dtype=float)
    jains = float(arr.sum() ** 2 / (len(arr) * (arr ** 2).sum())) if arr.sum() > 0 else 0.0
    sorted_arr = np.sort(arr)
    n = len(sorted_arr)
    gini = float(
        (2 * (np.arange(1, n + 1) * sorted_arr).sum()) / (n * sorted_arr.sum()) - (n + 1) / n
    ) if sorted_arr.sum() > 0 else 0.0

    return {
        "cumulative_reward": cumulative_reward,
        "ndr":               terminal_stats["ndr"],
        "jains":             jains,
        "gini":              gini,
        "efficiency":        terminal_stats["efficiency"],
    }


# ── Evaluate a condition ──────────────────────────────────────────────────────

def evaluate(label: str, model_path: Path, env_cls, seeds: list[int],
             grid_size: tuple, num_sensors: int) -> pd.DataFrame:
    tag = f"{grid_size[0]}×{grid_size[1]} N={num_sensors}"
    print(f"\n{'='*60}\nEvaluating: {label} | {tag}\nModel: {model_path}\n{'='*60}")
    vec_tmp = _make_vec(env_cls, 0, grid_size, num_sensors)
    # buffer_size=2: inference only — avoids allocating the full replay buffer
    model = DQN.load(str(model_path), env=vec_tmp,
                     custom_objects={"buffer_size": 2, "optimize_memory_usage": False})
    vec_tmp.close()

    rows = []
    for seed in seeds:
        r = _run_episode(model, env_cls, seed, grid_size, num_sensors)
        r["seed"] = seed
        rows.append(r)
        print(
            f"  seed {seed:>3}: reward={r['cumulative_reward']:>10.1f}  "
            f"NDR={r['ndr']:.1f}%  Jain={r['jains']:.3f}"
        )
    df = pd.DataFrame(rows)
    df["condition"] = label
    return df


# ── Statistics ────────────────────────────────────────────────────────────────

def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    pooled = np.sqrt((a.std(ddof=1) ** 2 + b.std(ddof=1) ** 2) / 2)
    return float((a.mean() - b.mean()) / pooled) if pooled > 0 else 0.0


def _effect_label(d: float) -> str:
    ad = abs(d)
    if ad < 0.2:  return "negligible"
    if ad < 0.5:  return "small"
    if ad < 0.8:  return "medium"
    return "large"


def welch_row(metric: str, full: pd.DataFrame, abl: pd.DataFrame) -> dict:
    a, b = full[metric].values, abl[metric].values
    t, p = stats.ttest_ind(a, b, equal_var=False)
    d = _cohens_d(a, b)
    return {
        "metric":        metric,
        "full_mean":     a.mean(),  "full_std":     a.std(ddof=1),
        "ablation_mean": b.mean(),  "ablation_std": b.std(ddof=1),
        "t": t, "p": p,
        "cohens_d": d,  "effect": _effect_label(d),
    }


# ── Per-condition bar chart ───────────────────────────────────────────────────

def plot_ablation(full_df: pd.DataFrame, ablation_dfs: dict[str, pd.DataFrame],
                  grid_size: tuple, num_sensors: int) -> None:
    metrics = [
        ("ndr",               "NDR (%)"),
        ("jains",             "Jain's Fairness"),
        ("cumulative_reward", "Cumulative Reward"),
        ("efficiency",        "Energy Efficiency (B/Wh)"),
    ]
    labels  = ["Full Model"] + list(ablation_dfs.keys())
    palette = ["#1f77b4", "#d62728", "#ff7f0e", "#2ca02c", "#9467bd"]

    fig, axes = plt.subplots(1, len(metrics), figsize=(16, 4))
    for ax, (col, title) in zip(axes, metrics):
        for i, label in enumerate(labels):
            df   = full_df if label == "Full Model" else ablation_dfs[label]
            vals = df[col].values
            ax.bar(i, vals.mean(), yerr=vals.std(ddof=1),
                   color=palette[i], capsize=5, alpha=0.85, width=0.6)
        ax.set_title(title, fontsize=9)
        ax.set_xticks([])
        ax.tick_params(axis="y", labelsize=7)

    patches = [plt.Rectangle((0, 0), 1, 1, color=palette[i]) for i in range(len(labels))]
    fig.legend(patches, labels, loc="lower center", ncol=len(labels),
               fontsize=8, bbox_to_anchor=(0.5, -0.06))
    fig.suptitle(
        f"Component Ablation: Full DQN vs A1/A2/A3/A4\n"
        f"({grid_size[0]}×{grid_size[1]}, N={num_sensors}, {N_SEEDS} seeds)",
        fontsize=10,
    )
    plt.tight_layout()
    tag = f"{grid_size[0]}x{grid_size[1]}_N{num_sensors}"
    out = OUT_DIR / f"ablation_comparison_{tag}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Plot saved → {out}")
    plt.close()


# ── Combined NDR summary across all conditions ────────────────────────────────

def plot_combined_ndr(all_records: list[dict]) -> None:
    """Line plot of NDR vs (grid, N) condition for Full Model and each ablation."""
    df = pd.DataFrame(all_records)
    ndr_df = df[df["metric"] == "ndr"].copy()

    condition_labels = [
        f"{c['grid'][0]}×{c['grid'][1]}\nN={c['n_sensors']}" for c in CONDITIONS
    ]
    ablation_names = ["Full Model", "A1: No Capture", "A2: Instant ADR",
                      "A3: No AoI Obs", "A4: No Domain Rand"]
    palette = ["#1f77b4", "#d62728", "#ff7f0e", "#2ca02c", "#9467bd"]
    markers = ["o", "s", "^", "D", "v"]

    fig, ax = plt.subplots(figsize=(8, 5))

    for i, name in enumerate(ablation_names):
        y_vals, y_err = [], []
        for cond in CONDITIONS:
            tag = f"{cond['grid'][0]}x{cond['grid'][1]}_N{cond['n_sensors']}"
            subset = ndr_df[
                (ndr_df["ablation"] == name) &
                (ndr_df["condition_tag"] == tag)
            ]
            if subset.empty:
                y_vals.append(np.nan)
                y_err.append(0)
            else:
                row = subset.iloc[0]
                mean_col = "full_mean" if name == "Full Model" else "ablation_mean"
                std_col  = "full_std"  if name == "Full Model" else "ablation_std"
                y_vals.append(row[mean_col])
                y_err.append(row[std_col])

        xs = list(range(len(CONDITIONS)))
        ax.errorbar(xs, y_vals, yerr=y_err,
                    label=name, color=palette[i], marker=markers[i],
                    linewidth=1.8, markersize=6, capsize=4)

    ax.set_xticks(range(len(CONDITIONS)))
    ax.set_xticklabels(condition_labels, fontsize=8)
    ax.set_ylabel("NDR (%)", fontsize=10)
    ax.set_ylim(0, 105)
    ax.set_xlabel("Condition (grid size, sensor count)", fontsize=10)
    ax.set_title(
        f"Ablation Study: NDR across Scales ({N_SEEDS} seeds/condition, 5 grid sizes)",
        fontsize=11,
    )
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = OUT_DIR / "ablation_ndr_combined.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Combined NDR plot saved → {out}")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    seeds   = list(range(N_SEEDS))
    metrics = ["cumulative_reward", "ndr", "jains", "gini", "efficiency"]
    all_stat_rows: list[dict] = []

    for cond in CONDITIONS:
        grid_size   = cond["grid"]
        num_sensors = cond["n_sensors"]
        tag         = f"{grid_size[0]}x{grid_size[1]}_N{num_sensors}"

        print(f"\n{'#'*70}")
        print(f"# CONDITION: {grid_size[0]}×{grid_size[1]}, N={num_sensors}")
        print(f"{'#'*70}")

        full_df = evaluate("Full Model",         FULL_PATH, _PaddedEnv,       seeds, grid_size, num_sensors)
        a1_df   = evaluate("A1: No Capture",     FULL_PATH, _A1NoCaptureEnv,  seeds, grid_size, num_sensors)
        a2_df   = evaluate("A2: Instant ADR",    FULL_PATH, _A2InstantADREnv, seeds, grid_size, num_sensors)
        a3_df   = evaluate("A3: No AoI Obs",     FULL_PATH, _A3NoAoIEnv,      seeds, grid_size, num_sensors)
        a4_df   = evaluate("A4: No Domain Rand", A4_PATH,   _PaddedEnv,       seeds, grid_size, num_sensors)

        ablation_dfs = {
            "A1: No Capture":     a1_df,
            "A2: Instant ADR":    a2_df,
            "A3: No AoI Obs":     a3_df,
            "A4: No Domain Rand": a4_df,
        }

        # Save per-condition raw CSVs
        for lbl, df in [("full", full_df), ("a1", a1_df), ("a2", a2_df),
                         ("a3", a3_df), ("a4", a4_df)]:
            df.to_csv(OUT_DIR / f"ablation_{lbl}_{tag}.csv", index=False)

        # Per-condition Welch stats
        print(f"\n{'='*70}\nWelch's t-test — {grid_size[0]}×{grid_size[1]}, N={num_sensors}\n{'='*70}")
        for abl_label, abl_df in ablation_dfs.items():
            print(f"\n--- {abl_label} ---")
            for col in metrics:
                r = welch_row(col, full_df, abl_df)
                r["ablation"]      = abl_label
                r["grid"]          = f"{grid_size[0]}×{grid_size[1]}"
                r["n_sensors"]     = num_sensors
                r["condition_tag"] = tag
                # Add Full Model row for the combined NDR plot
                if col == "ndr":
                    fm_row = {
                        "metric": col, "ablation": "Full Model",
                        "grid": f"{grid_size[0]}×{grid_size[1]}",
                        "n_sensors": num_sensors, "condition_tag": tag,
                        "full_mean": r["full_mean"], "full_std": r["full_std"],
                        "ablation_mean": r["full_mean"], "ablation_std": r["full_std"],
                        "t": 0.0, "p": 1.0, "cohens_d": 0.0, "effect": "negligible",
                    }
                    if not any(
                        x["ablation"] == "Full Model" and
                        x["condition_tag"] == tag and
                        x["metric"] == col
                        for x in all_stat_rows
                    ):
                        all_stat_rows.append(fm_row)
                all_stat_rows.append(r)
                sig = ("***" if r["p"] < 0.001 else "**"  if r["p"] < 0.01
                       else "*"   if r["p"] < 0.05  else "ns")
                print(
                    f"  {col:22s}: Full={r['full_mean']:.3f}±{r['full_std']:.3f}  "
                    f"Abl={r['ablation_mean']:.3f}±{r['ablation_std']:.3f}  "
                    f"t={r['t']:+.3f}  p={r['p']:.4f}{sig}  "
                    f"d={r['cohens_d']:+.3f} ({r['effect']})"
                )

        plot_ablation(full_df, ablation_dfs, grid_size, num_sensors)

    # Save combined stats CSV (used for dissertation tables and combined plot)
    combined_df = pd.DataFrame(all_stat_rows)
    csv_path = OUT_DIR / "ablation_study_results.csv"
    combined_df.to_csv(csv_path, index=False)
    print(f"\nCombined results saved → {csv_path}")

    plot_combined_ndr(all_stat_rows)

    # ── Print dissertation-format summary table (NDR, per condition) ──────────
    print(f"\n{'='*70}")
    print("NDR summary — Full Model vs each ablation, all conditions")
    print(f"{'='*70}")
    ndr_rows = [r for r in all_stat_rows if r["metric"] == "ndr"
                and r["ablation"] != "Full Model"]
    for cond in CONDITIONS:
        tag = f"{cond['grid'][0]}x{cond['grid'][1]}_N{cond['n_sensors']}"
        label = f"{cond['grid'][0]}×{cond['grid'][1]}, N={cond['n_sensors']}"
        print(f"\n{label}")
        for r in ndr_rows:
            if r["condition_tag"] == tag:
                sig = ("***" if r["p"] < 0.001 else "**" if r["p"] < 0.01
                       else "*" if r["p"] < 0.05 else "ns")
                print(
                    f"  {r['ablation']:22s}: "
                    f"Full={r['full_mean']:.1f}±{r['full_std']:.1f}  "
                    f"Abl={r['ablation_mean']:.1f}±{r['ablation_std']:.1f}  "
                    f"t={r['t']:+.2f}  p={r['p']:.4f}{sig}  "
                    f"d={r['cohens_d']:+.2f} ({r['effect']})"
                )


if __name__ == "__main__":
    main()
