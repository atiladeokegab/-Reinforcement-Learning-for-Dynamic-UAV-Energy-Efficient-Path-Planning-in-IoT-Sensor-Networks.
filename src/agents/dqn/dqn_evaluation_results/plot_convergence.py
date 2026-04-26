"""
plot_convergence.py
===================
Generates an IEEE-standard DQN convergence figure for the dissertation.

Evaluates each periodic checkpoint (every 200k steps) on a fixed test
environment (500×500, N=20, 5 seeds) to produce NDR and Jain's Fairness
Index learning curves, with vertical lines at each curriculum graduation.

Output:
  baseline_results/convergence_curve.png / .pdf

Usage:
    PYTHONIOENCODING=utf-8 uv run python \\
        src/agents/dqn/dqn_evaluation_results/plot_convergence.py
"""

from __future__ import annotations

import sys
import json
import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

_HERE = Path(__file__).resolve().parent
_SRC  = _HERE.parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    import ieee_style
    ieee_style.apply()
except ImportError:
    pass

import gymnasium
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv
from environment.uav_env import UAVEnvironment

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_DIR  = _HERE.parent / "models" / "dqn_v3_retrain"
GRAD_LOG   = MODEL_DIR / "graduation_log.json"
OUTPUT_DIR = _HERE / "baseline_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EVAL_GRID        = (500, 500)
EVAL_N_SENSORS   = 20
MAX_SENSORS_LIMIT = 50       # must match training (features_per_sensor=3, N_max=50)
FEATURES_PER_SENSOR = 3      # from training_config.json
N_SEEDS          = 5
N_STACK          = 4
MAX_STEPS        = 2100


# ---------------------------------------------------------------------------
# Zero-padded env wrapper (mirrors AnalysisUAVEnv in compare_agents.py)
# ---------------------------------------------------------------------------

class PaddedUAVEnv(UAVEnvironment):
    """Pad observation to max_sensors_limit so DQN network input matches training."""

    def __init__(self, max_sensors_limit: int = MAX_SENSORS_LIMIT, **kwargs):
        self.max_sensors_limit = max_sensors_limit
        super().__init__(**kwargs)
        raw_size = self.observation_space.shape[0]
        pad = (max_sensors_limit - self.num_sensors) * FEATURES_PER_SENSOR
        padded_size = raw_size + pad
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(padded_size,), dtype=np.float32
        )

    def _pad(self, obs: np.ndarray) -> np.ndarray:
        pad = (self.max_sensors_limit - self.num_sensors) * FEATURES_PER_SENSOR
        return np.concatenate([obs, np.zeros(pad, dtype=np.float32)])

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        return self._pad(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return self._pad(obs), reward, terminated, truncated, info


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_env(seed: int):
    def _init():
        np.random.seed(seed)
        env = PaddedUAVEnv(
            grid_size=EVAL_GRID,
            num_sensors=EVAL_N_SENSORS,
            max_steps=MAX_STEPS,
            include_sensor_positions=False,  # training used False → 3 features/sensor
        )
        return env
    return _init


class TerminalCapturePaddedEnv(PaddedUAVEnv):
    """PaddedUAVEnv that saves terminal NDR/Jain's before SB3 auto-resets it."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.terminal_ndr  = 0.0
        self.terminal_jfi  = 0.0

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        if terminated or truncated:
            n   = self.num_sensors
            vis = len(self.sensors_visited)
            self.terminal_ndr = (vis / n) * 100.0
            crs = np.array([
                s.total_data_transmitted / max(s.total_data_generated, 1e-9)
                for s in self.sensors
            ])
            self.terminal_jfi = float((crs.sum() ** 2) / (n * (crs ** 2).sum() + 1e-12))
        return obs, reward, terminated, truncated, info


def make_capture_env(seed: int):
    def _init():
        np.random.seed(seed)
        return TerminalCapturePaddedEnv(
            grid_size=EVAL_GRID,
            num_sensors=EVAL_N_SENSORS,
            max_steps=MAX_STEPS,
            include_sensor_positions=False,
        )
    return _init


def evaluate_checkpoint(zip_path: Path, seeds: list[int]) -> tuple[float, float, float, float]:
    """Return (mean_ndr, std_ndr, mean_jains, std_jains) over seeds.

    Uses SB3's VecFrameStack for correct frame ordering, and captures terminal
    metrics inside step() before the VecEnv auto-reset fires.
    """
    model = DQN.load(str(zip_path))

    ndrs, jains_list = [], []
    for seed in seeds:
        venv = VecFrameStack(DummyVecEnv([make_capture_env(seed)]), n_stack=N_STACK)
        obs  = venv.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, _, dones, _ = venv.step(action)
            done = bool(dones[0])

        # Terminal metrics were saved inside step() before auto-reset
        base_env = venv.envs[0]
        while hasattr(base_env, "env"):
            base_env = base_env.env

        ndrs.append(base_env.terminal_ndr)
        jains_list.append(base_env.terminal_jfi)
        venv.close()

    return float(np.mean(ndrs)), float(np.std(ndrs)), float(np.mean(jains_list)), float(np.std(jains_list))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    # ── Load graduation log ────────────────────────────────────────────────
    with open(GRAD_LOG) as f:
        grad_events = json.load(f)

    # The graduation log records rolling NDR/Jain's at the moment each stage
    # gate was passed — this IS the training convergence signal (curriculum-
    # aligned, evaluated on the conditions the agent was actually training on).
    grad_ts     = [g["ts"] / 1e6  for g in grad_events]     # x-axis (M steps)
    grad_ndr    = [g["rolling_ndr"]   for g in grad_events]
    grad_jains  = [g["rolling_jains"] for g in grad_events]
    gate_ndr    = [g["threshold_ndr"]   for g in grad_events]
    gate_jains  = [g["threshold_jains"] for g in grad_events]

    stage_labels = [f"S{g['from_stage']}→{g['to_stage']}" for g in grad_events]

    # Stage boundary x-positions (add t=0 and t=5M as start/end)
    stage_starts = [0.0] + grad_ts
    stage_ends   = grad_ts + [5.0]
    stage_names  = [
        "Stage 0\n100×100", "Stage 1\n200×200",
        "Stage 2\n300×300", "Stage 3\n400×400", "Stage 4\n500×500",
    ]

    # ── Colours ───────────────────────────────────────────────────────────
    teal       = "#1b9e77"
    orange     = "#d95f02"
    shade_cols = ["#e8f4f8", "#fef9e7", "#e8f8f0", "#fef0e7", "#ede8f8"]
    grad_color = "#555555"

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5.5, 4.5), sharex=True,
                                    gridspec_kw={"hspace": 0.08})

    # ── Shade curriculum stages ────────────────────────────────────────────
    for ax in (ax1, ax2):
        for xs, xe, col in zip(stage_starts, stage_ends, shade_cols):
            ax.axvspan(xs, xe, color=col, alpha=0.45, linewidth=0)

    # ── NDR panel ─────────────────────────────────────────────────────────
    ax1.plot(grad_ts, grad_ndr, color=teal, linewidth=1.6,
             marker="o", markersize=5, zorder=5, label="Rolling NDR at gate")
    # Gate threshold markers
    ax1.scatter(grad_ts, gate_ndr, color=teal, marker="x", s=40,
                linewidths=1.2, zorder=6, label="Gate threshold")
    for x, y, g in zip(grad_ts, grad_ndr, stage_labels):
        ax1.annotate(g, (x, y), textcoords="offset points", xytext=(5, 3),
                     fontsize=7, color=grad_color)
    ax1.set_ylabel("NDR (%)")
    ax1.set_ylim(0, 105)
    ax1.yaxis.set_major_locator(mticker.MultipleLocator(20))
    ax1.legend(loc="lower left", fontsize=7.5, ncol=2)

    # ── Jain's panel ──────────────────────────────────────────────────────
    ax2.plot(grad_ts, grad_jains, color=orange, linewidth=1.6,
             marker="s", markersize=5, zorder=5, label="Rolling Jain's FI at gate")
    ax2.scatter(grad_ts, gate_jains, color=orange, marker="x", s=40,
                linewidths=1.2, zorder=6, label="Gate threshold")
    ax2.set_ylabel("Jain's Fairness Index")
    ax2.set_xlabel("Training Timesteps (×10⁶)")
    ax2.set_ylim(0, 1.05)
    ax2.yaxis.set_major_locator(mticker.MultipleLocator(0.2))
    ax2.legend(loc="upper right", fontsize=7.5, ncol=2)

    # ── Stage name labels on top of shaded bands ──────────────────────────
    for xs, xe, name in zip(stage_starts, stage_ends, stage_names):
        mid = (xs + xe) / 2
        ax1.text(mid, 102, name, ha="center", va="top",
                 fontsize=6.5, color="#444444", style="italic")

    # ── Vertical graduation lines ──────────────────────────────────────────
    for ax in (ax1, ax2):
        for ts in grad_ts:
            ax.axvline(ts, color=grad_color, linewidth=0.7,
                       linestyle="--", alpha=0.6, zorder=3)

    ax1.set_xlim(0, 5.1)
    ax2.set_xlim(0, 5.1)

    ieee_style.save(fig, str(OUTPUT_DIR / "convergence_curve"), formats=("png", "pdf"))
    plt.close(fig)
    print(f"\nConvergence plot → {OUTPUT_DIR / 'convergence_curve'}.png/.pdf")


if __name__ == "__main__":
    main()
