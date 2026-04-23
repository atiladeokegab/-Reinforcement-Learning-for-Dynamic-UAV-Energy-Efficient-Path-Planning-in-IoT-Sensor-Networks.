"""
generalization_eval.py
======================
Multi-seed, multi-grid generalization evaluation.

Compares DQN, Relational RL (PPO), and GTrXL Transformer across the full
curriculum grid range (100×100 to 500×500, 10–50 sensors) over N_SEEDS
independent seeds to confirm that results are not flukes.

Models are loaded once; missing checkpoints are skipped with a warning so the
script runs in partial form on machines that only have some models available.

Outputs (all written to baseline_results/generalization/):
  * generalization_results.csv    — raw per-episode rows
  * generalization_summary.csv    — mean ± std per (model, config)
  * generalization_ndr.png        — NDR (%) vs grid config, ±1σ error bars
  * generalization_jfi.png        — Jain's Fairness Index vs grid config
  * generalization_efficiency.png — Bytes / Wh vs grid config

Usage (local Windows or remote Linux pod):
    PYTHONIOENCODING=utf-8 uv run python \\
        src/agents/dqn/dqn_evaluation_results/generalization_eval.py
"""

from __future__ import annotations

import json
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import gymnasium
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# ---------------------------------------------------------------------------
# Path setup — works from any CWD on Windows or Linux
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve().parent          # …/dqn_evaluation_results/
_DQN_DIR = _HERE.parent                          # …/dqn/
_SRC = _HERE.parents[3]                          # …/src/
_ROOT = _HERE.parents[4]                         # project root

for _p in (str(_SRC), str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Relational policy source — RLlib needs RelationalUAVModule on sys.path
_REL_SRC = _SRC / "experiments" / "relational_policy"
if str(_REL_SRC) not in sys.path:
    sys.path.insert(0, str(_REL_SRC))

# GTrXL source
_TFM_SRC = _SRC / "experiments" / "transformer_policy"
if str(_TFM_SRC) not in sys.path:
    sys.path.insert(0, str(_TFM_SRC))

# gnn_extractor (required by DQN policy custom feature extractor)
for _gnn in list(_DQN_DIR.rglob("gnn_extractor.py")):
    _gd = str(_gnn.parent)
    if _gd not in sys.path:
        sys.path.insert(0, _gd)

from environment.uav_env import UAVEnvironment  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

try:
    import ieee_style
    ieee_style.apply()
except ImportError:
    log.warning("ieee_style not found — using matplotlib defaults")

# ---------------------------------------------------------------------------
# ── CONFIGURATION ────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

EVAL_CONFIGS: list[dict] = [
    {"grid_size": (100, 100), "num_sensors": 10, "label": "100×100\nN=10"},
    {"grid_size": (200, 200), "num_sensors": 20, "label": "200×200\nN=20"},
    {"grid_size": (300, 300), "num_sensors": 30, "label": "300×300\nN=30"},
    {"grid_size": (400, 400), "num_sensors": 40, "label": "400×400\nN=40"},
    {"grid_size": (500, 500), "num_sensors": 50, "label": "500×500\nN=50"},
]

N_SEEDS: int = 10
SEEDS: list[int] = list(range(N_SEEDS))

# Shared physics / episode params forwarded to UAVEnvironment
ENV_BASE: dict[str, Any] = {
    "max_steps":          2100,
    "path_loss_exponent": 3.8,
    "rssi_threshold":     -85.0,
    "sensor_duty_cycle":  10.0,
}

MAX_BATTERY: float = 274.0  # Wh — max battery capacity

# ── Model checkpoints (relative to project root; missing → skipped) ─────────

# DQN (Stable-Baselines3)
DQN_MODEL_PATH = _DQN_DIR / "models" / "dqn_v3_retrain" / "dqn_final.zip"
DQN_CONFIG_PATH = _DQN_DIR / "models" / "dqn_v3_retrain" / "training_config.json"
DQN_MAX_SENSORS = 50           # max the DQN policy was trained with

# Relational RL (RLlib PPO, new API stack / RLModule)
REL_CKPT_DIR = (
    _DQN_DIR / "models" / "relational_rl" /
    "results" / "checkpoints" / "stage_4" / "final"
)

# GTrXL Transformer (RLlib PPO, old API stack) — prefers stage-1 checkpoint
GTRXL_CKPT_DIR = _ROOT / "models" / "transformer_gtrxl_stage1" / "stage_1_progress"
GTRXL_FALLBACK  = _ROOT / "models" / "transformer_gtrxl" / "stage_0_progress"

# ── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR = _HERE / "baseline_results" / "generalization"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# ── DQN: zero-padded env wrapper (matches training conditions) ─────────────
# ---------------------------------------------------------------------------

class _AnalysisUAVEnv(UAVEnvironment):
    """Zero-pads observations to match DQN's fixed-size input (max_sensors=50)."""

    def __init__(self, max_sensors_limit: int = DQN_MAX_SENSORS, **kwargs):
        self.max_sensors_limit = max_sensors_limit
        super().__init__(**kwargs)

        raw = self.observation_space.shape[0]
        self._fps: int = 0
        for uav_f in range(raw + 1):
            rem = raw - uav_f
            if rem > 0 and rem % self.num_sensors == 0:
                self._fps = rem // self.num_sensors
                break
        if self._fps == 0:
            raise ValueError(
                f"Cannot infer features_per_sensor: raw={raw}, num_sensors={self.num_sensors}"
            )
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
        obs, r, terminated, truncated, info = super().step(action)
        return self._pad(obs), r, terminated, truncated, info


# ---------------------------------------------------------------------------
# ── Shared metric helpers ──────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def _jfi(sensors) -> float:
    """Jain's Fairness Index on per-sensor collection rates."""
    rates = []
    for s in sensors:
        gen = float(s.total_data_generated)
        tx  = float(s.total_data_transmitted)
        rates.append((tx / gen * 100.0) if gen > 0 else 0.0)
    n  = len(rates)
    s2 = sum(r ** 2 for r in rates)
    return float(sum(rates) ** 2 / (n * s2)) if n > 0 and s2 > 0 else 0.0


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _episode_metrics(base_env: Any, total_reward: float, steps: int) -> dict:
    """Extract standard metrics from a completed episode."""
    energy = MAX_BATTERY - float(base_env.uav.battery)
    ndr    = (len(base_env.sensors_visited) / base_env.num_sensors) * 100.0
    jfi    = _jfi(base_env.sensors)
    bpwh   = (float(base_env.total_data_collected) / energy) if energy > 0 else 0.0
    return {
        "ndr_pct":      ndr,
        "jfi":          jfi,
        "bytes_per_wh": bpwh,
        "total_reward": total_reward,
        "steps":        steps,
    }


# ---------------------------------------------------------------------------
# ── DQN runner ─────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def _load_dqn():
    if not DQN_MODEL_PATH.exists():
        log.warning("DQN checkpoint not found at %s — skipping", DQN_MODEL_PATH)
        return None, None
    defaults = {"use_frame_stacking": True, "n_stack": 4, "max_sensors_limit": DQN_MAX_SENSORS}
    try:
        with open(DQN_CONFIG_PATH) as f:
            cfg = {**defaults, **json.load(f)}
    except FileNotFoundError:
        log.warning("DQN config not found — using defaults")
        cfg = defaults
    model = DQN.load(str(DQN_MODEL_PATH))
    log.info("DQN loaded from %s", DQN_MODEL_PATH)
    return model, cfg


def _run_dqn_episode(model, dqn_cfg: dict, grid_size, num_sensors, seed: int) -> dict:
    _seed_all(seed)
    env_kwargs = {
        **ENV_BASE,
        "grid_size":          grid_size,
        "num_sensors":        num_sensors,
        "max_sensors_limit":  dqn_cfg["max_sensors_limit"],
        "include_sensor_positions": False,
    }
    vec_env = DummyVecEnv([lambda: _AnalysisUAVEnv(**env_kwargs)])
    if dqn_cfg.get("use_frame_stacking", True):
        vec_env = VecFrameStack(vec_env, n_stack=dqn_cfg.get("n_stack", 4))

    # Unwrap to access base env for metrics
    inner = vec_env
    while hasattr(inner, "venv"):
        inner = inner.venv
    base_env = inner.envs[0]

    obs = vec_env.reset()
    total_reward = 0.0
    steps = 0
    done = np.array([False])
    while not done[0]:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, _ = vec_env.step(action)
        total_reward += float(rewards[0])
        steps += 1
        done = dones

    vec_env.close()
    return _episode_metrics(base_env, total_reward, steps)


# ---------------------------------------------------------------------------
# ── Relational RL runner ───────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def _load_relational():
    if not REL_CKPT_DIR.exists():
        log.warning("Relational RL checkpoint not found at %s — skipping", REL_CKPT_DIR)
        return None
    from relational_rl_runner import load_relational_rl_module
    return load_relational_rl_module(REL_CKPT_DIR)


def _run_relational_episode(rl_module, grid_size, num_sensors, seed: int) -> dict:
    import torch
    from ray.rllib.core.columns import Columns
    from relational_rl_runner import InferenceRelationalUAVEnv

    _seed_all(seed)
    env = InferenceRelationalUAVEnv(
        n_max=50,
        grid_size=grid_size,
        num_sensors=num_sensors,
        **ENV_BASE,
    )
    obs, _ = env.reset(seed=seed)
    total_reward = 0.0
    steps = 0

    while True:
        batch = {Columns.OBS: {
            k: torch.as_tensor(np.asarray(v)).unsqueeze(0)
            for k, v in obs.items()
        }}
        with torch.no_grad():
            out = rl_module._forward_inference(batch)
        action = int(torch.argmax(out[Columns.ACTION_DIST_INPUTS], dim=-1).item())
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        steps += 1
        if terminated or truncated:
            break

    return _episode_metrics(env, total_reward, steps)


# ---------------------------------------------------------------------------
# ── GTrXL Transformer runner ───────────────────────────────────────────────
# ---------------------------------------------------------------------------

_GTRXL_ALGO = None   # loaded once, reused across all episodes


def _resolve_gtrxl_ckpt() -> Path | None:
    if GTRXL_CKPT_DIR.exists():
        return GTRXL_CKPT_DIR
    if GTRXL_FALLBACK.exists():
        log.info("GTrXL stage-1 ckpt not found; using fallback %s", GTRXL_FALLBACK)
        return GTRXL_FALLBACK
    return None


def _load_gtrxl():
    global _GTRXL_ALGO
    ckpt = _resolve_gtrxl_ckpt()
    if ckpt is None:
        log.warning(
            "GTrXL checkpoint not found at %s or %s — skipping",
            GTRXL_CKPT_DIR, GTRXL_FALLBACK,
        )
        return None

    import ray
    from ray import tune
    from ray.rllib.algorithms.ppo import PPOConfig
    from transformer_model import register_model
    from env_wrapper import TransformerObsWrapper

    try:
        import torch
        n_gpus = min(torch.cuda.device_count(), 2)
    except ImportError:
        n_gpus = 0

    ray.init(ignore_reinit_error=True, num_cpus=4, num_gpus=n_gpus)
    tune.register_env("UAVTransformerEnv", lambda cfg: TransformerObsWrapper(cfg))
    model_cfg = register_model()

    # Build with a dummy stage-0 config — obs dim is always 253 regardless of grid
    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(
            env="UAVTransformerEnv",
            env_config={"grid_size": (100, 100), "num_sensors": 10},
        )
        .training(model=model_cfg)
        .env_runners(num_env_runners=0)
        .resources(num_gpus=n_gpus)
        .debugging(log_level="ERROR")
    )
    algo = config.build_algo()
    algo.restore(str(ckpt))
    log.info("GTrXL loaded from %s", ckpt)
    _GTRXL_ALGO = algo
    return algo


def _run_gtrxl_episode(algo, grid_size, num_sensors, seed: int) -> dict:
    from env_wrapper import TransformerObsWrapper

    _seed_all(seed)
    env = TransformerObsWrapper({
        "grid_size":  grid_size,
        "num_sensors": num_sensors,
        **ENV_BASE,
    })
    obs, _ = env.reset(seed=seed)
    policy = algo.get_policy()
    total_reward = 0.0
    steps = 0
    state = policy.get_initial_state()

    while True:
        action, state, _ = policy.compute_single_action(
            obs, state=state, explore=False
        )
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += float(reward)
        steps += 1
        if terminated or truncated:
            break

    return _episode_metrics(env.env, total_reward, steps)


# ---------------------------------------------------------------------------
# ── Main evaluation loop ───────────────────────────────────────────────────
# ---------------------------------------------------------------------------

def run_all() -> pd.DataFrame:
    # ── Load models (skip missing) ─────────────────────────────────────────
    runners: dict[str, Any] = {}

    dqn_model, dqn_cfg = _load_dqn()
    if dqn_model is not None:
        runners["DQN"] = lambda g, n, s: _run_dqn_episode(dqn_model, dqn_cfg, g, n, s)

    rel_module = _load_relational()
    if rel_module is not None:
        runners["Relational RL"] = lambda g, n, s: _run_relational_episode(rel_module, g, n, s)

    gtrxl_algo = _load_gtrxl()
    if gtrxl_algo is not None:
        runners["GTrXL"] = lambda g, n, s: _run_gtrxl_episode(gtrxl_algo, g, n, s)

    if not runners:
        log.error("No models loaded — nothing to evaluate.")
        return pd.DataFrame()

    log.info("Active models: %s", list(runners.keys()))
    log.info("Configs: %d  |  Seeds: %d  |  Total episodes: %d",
             len(EVAL_CONFIGS), N_SEEDS,
             len(EVAL_CONFIGS) * N_SEEDS * len(runners))

    rows: list[dict] = []
    total = len(EVAL_CONFIGS) * N_SEEDS * len(runners)
    done  = 0

    for cfg in EVAL_CONFIGS:
        g = cfg["grid_size"]
        n = cfg["num_sensors"]
        label = f"{g[0]}x{g[1]}"
        for seed in SEEDS:
            for model_name, runner in runners.items():
                done += 1
                log.info("[%d/%d] %s | %s | seed=%d", done, total, model_name, label, seed)
                try:
                    metrics = runner(g, n, seed)
                    rows.append({
                        "model":       model_name,
                        "grid":        label,
                        "grid_size":   f"{g[0]}x{g[1]}",
                        "n_sensors":   n,
                        "seed":        seed,
                        **metrics,
                    })
                except Exception as exc:
                    log.warning("Episode failed (%s | %s | seed=%d): %s",
                                model_name, label, seed, exc)

    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "generalization_results.csv"
    df.to_csv(csv_path, index=False)
    log.info("Raw results → %s  (%d rows)", csv_path, len(df))
    return df


# ---------------------------------------------------------------------------
# ── Aggregation & summary table ───────────────────────────────────────────
# ---------------------------------------------------------------------------

def summarize(df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        df.groupby(["model", "grid", "n_sensors"])[["ndr_pct", "jfi", "bytes_per_wh"]]
        .agg(["mean", "std"])
        .round(3)
    )
    summary.columns = ["_".join(c) for c in summary.columns]
    summary = summary.reset_index()
    csv_path = OUTPUT_DIR / "generalization_summary.csv"
    summary.to_csv(csv_path, index=False)
    log.info("Summary → %s", csv_path)
    print("\n" + summary.to_string(index=False))
    return summary


# ---------------------------------------------------------------------------
# ── Plots ──────────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

_COLORS = {
    "DQN":          "#1f78b4",
    "Relational RL": "#33a02c",
    "GTrXL":        "#e31a1c",
}
_MARKERS = {
    "DQN":          "o",
    "Relational RL": "s",
    "GTrXL":        "^",
}


def _plot_metric(df: pd.DataFrame, metric: str, ylabel: str, filename: str) -> None:
    configs_ordered = [c["grid_size"] for c in EVAL_CONFIGS]
    x_labels = [f"{g[0]}×{g[1]}\nN={n}" for g, n in
                [(c["grid_size"], c["num_sensors"]) for c in EVAL_CONFIGS]]
    x = np.arange(len(EVAL_CONFIGS))

    fig, ax = plt.subplots(figsize=(7, 4))
    models = sorted(df["model"].unique())

    for model in models:
        means, stds = [], []
        for cfg in EVAL_CONFIGS:
            label = f"{cfg['grid_size'][0]}x{cfg['grid_size'][1]}"
            sub = df[(df["model"] == model) & (df["grid"] == label)][metric]
            means.append(sub.mean() if len(sub) > 0 else np.nan)
            stds.append(sub.std()  if len(sub) > 1 else 0.0)

        ax.errorbar(
            x, means, yerr=stds,
            label=model,
            color=_COLORS.get(model, None),
            marker=_MARKERS.get(model, "o"),
            linewidth=1.8,
            markersize=6,
            capsize=4,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, fontsize=8)
    ax.set_xlabel("Environment Configuration")
    ax.set_ylabel(ylabel)
    ax.legend(framealpha=0.9)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    out = OUTPUT_DIR / filename
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Plot saved → %s", out)


def plot_all(df: pd.DataFrame) -> None:
    _plot_metric(df, "ndr_pct",      "NDR (%)",        "generalization_ndr.png")
    _plot_metric(df, "jfi",          "Jain's FI",      "generalization_jfi.png")
    _plot_metric(df, "bytes_per_wh", "Efficiency (B/Wh)", "generalization_efficiency.png")


# ---------------------------------------------------------------------------
# ── Entry point ────────────────────────────────────────────────────────────
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    df = run_all()
    if df.empty:
        sys.exit(1)
    summarize(df)
    plot_all(df)
    log.info("Done. Results in %s", OUTPUT_DIR)
