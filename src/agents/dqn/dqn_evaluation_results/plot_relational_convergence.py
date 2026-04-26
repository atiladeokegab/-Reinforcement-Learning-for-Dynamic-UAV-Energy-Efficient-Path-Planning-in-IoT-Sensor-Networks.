"""
plot_relational_convergence.py
==============================
Evaluates each Relational RL curriculum stage checkpoint on its own training
config and plots NDR + Jain's FI as a grouped bar chart — convergence evidence
to accompany the DQN TensorBoard curve in the dissertation.

Curriculum (from train_relational.py):
  Stage 0: 100×100, N=10
  Stage 1: 200×200, N=15
  Stage 2: 300×300, N=20
  Stage 3: 400×400, N=20
  Stage 4: 500×500, N=20

Each stage checkpoint is evaluated on 10 seeds of its own config.

Output:
  baseline_results/relational_convergence.png / .pdf

Usage:
    PYTHONIOENCODING=utf-8 uv run python \\
        src/agents/dqn/dqn_evaluation_results/plot_relational_convergence.py
"""

from __future__ import annotations

import sys
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
_ROOT = _SRC.parent
for _p in (str(_SRC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import ieee_style
    ieee_style.apply()
except ImportError:
    pass

from environment.uav_env import UAVEnvironment
from agents.dqn.dqn_evaluation_results.relational_rl_runner import (
    load_relational_rl_module,
    InferenceRelationalUAVEnv,
)

import torch
from ray.rllib.core.columns import Columns

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CKPT_BASE = _ROOT / "src" / "agents" / "dqn" / "models" / "relational_rl" / \
            "results" / "checkpoints"
OUTPUT_DIR = _HERE / "baseline_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_SEEDS = 10
N_MAX   = 50   # must match RelationalUAVModule training

STAGES = [
    {"idx": 0, "grid": (100, 100), "n": 10,  "label": "Stage 0\n100×100\nN=10"},
    {"idx": 1, "grid": (200, 200), "n": 15,  "label": "Stage 1\n200×200\nN=15"},
    {"idx": 2, "grid": (300, 300), "n": 20,  "label": "Stage 2\n300×300\nN=20"},
    {"idx": 3, "grid": (400, 400), "n": 20,  "label": "Stage 3\n400×400\nN=20"},
    {"idx": 4, "grid": (500, 500), "n": 20,  "label": "Stage 4\n500×500\nN=20"},
]

# Light→dark green (Dark2[4] hue family)
STAGE_COLORS = ["#b3e2cd", "#7ecba8", "#4dac84", "#2a8a64", "#1a6645"]


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def run_one_episode(rl_module, env) -> tuple[float, float]:
    """Run one deterministic episode. Return (ndr_pct, jfi)."""
    obs, _ = env.reset()
    done = False
    while not done:
        batch = {Columns.OBS: {
            k: torch.as_tensor(np.asarray(v)).unsqueeze(0)
            for k, v in obs.items()
        }}
        with torch.no_grad():
            out = rl_module._forward_inference(batch)
        action = int(torch.argmax(out[Columns.ACTION_DIST_INPUTS], dim=-1).item())
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    n       = env.num_sensors
    visited = len(env.sensors_visited)
    ndr     = (visited / n) * 100.0

    crs = np.array([
        s.total_data_transmitted / max(s.total_data_generated, 1e-9)
        for s in env.sensors
    ])
    jfi = float((crs.sum() ** 2) / (n * (crs ** 2).sum() + 1e-12))
    return ndr, jfi


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ndr_means, ndr_stds = [], []
    jfi_means, jfi_stds = [], []

    for stage in STAGES:
        idx   = stage["idx"]
        grid  = stage["grid"]
        n_sen = stage["n"]
        ckpt  = CKPT_BASE / f"stage_{idx}" / "final"

        print(f"\nStage {idx}: {grid[0]}×{grid[1]}, N={n_sen}", flush=True)
        rl_module = load_relational_rl_module(ckpt)

        ndrs, jfis = [], []
        for seed in range(N_SEEDS):
            env = InferenceRelationalUAVEnv(
                n_max=N_MAX,
                grid_size=grid,
                num_sensors=n_sen,
                max_steps=2100,
                include_sensor_positions=True,
            )
            np.random.seed(seed)
            ndr, jfi = run_one_episode(rl_module, env)
            ndrs.append(ndr)
            jfis.append(jfi)
            print(f"  seed {seed}: NDR={ndr:.0f}%  Jain's={jfi:.3f}")

        ndr_means.append(np.mean(ndrs))
        ndr_stds .append(np.std(ndrs))
        jfi_means.append(np.mean(jfis))
        jfi_stds .append(np.std(jfis))
        print(f"  → mean NDR={ndr_means[-1]:.1f}%  Jain's={jfi_means[-1]:.3f}")

    # ── Plot ──────────────────────────────────────────────────────────────
    x      = np.arange(len(STAGES))
    labels = [s["label"] for s in STAGES]
    colors = STAGE_COLORS
    width  = 0.38

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 3.2),
                                    gridspec_kw={"wspace": 0.38})

    def bar_panel(ax, means, stds, ylabel, ylim, ytick):
        bars = ax.bar(x, means, width=0.6, color=colors, alpha=0.88,
                      edgecolor="white", linewidth=0.5, zorder=3)
        ax.errorbar(x, means, yerr=stds, fmt="none",
                    color="#333333", capsize=4, linewidth=1.2, zorder=4)
        # Individual seed dots (strip plot)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7.5)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_locator(mticker.MultipleLocator(ytick))
        ax.set_xlabel("Curriculum Stage", fontsize=9)
        return bars

    bar_panel(ax1, ndr_means, ndr_stds,
              "NDR (%)", (-5, 108), 20)
    ax1.set_title("(a) Node Discovery Rate", fontsize=9, pad=4)

    bar_panel(ax2, jfi_means, jfi_stds,
              "Jain's Fairness Index", (-0.05, 1.08), 0.2)
    ax2.set_title("(b) Jain's Fairness Index", fontsize=9, pad=4)

    # Annotate bar tops
    for ax, means, fmt in [(ax1, ndr_means, "{:.0f}%"), (ax2, jfi_means, "{:.2f}")]:
        for xi, m in zip(x, means):
            ax.text(xi, m + (2 if ax is ax1 else 0.02), fmt.format(m),
                    ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    fig.suptitle("Relational RL — Curriculum Stage Progression (10 seeds per stage)",
                 fontsize=9, y=1.01)

    ieee_style.save(fig, str(OUTPUT_DIR / "relational_convergence"),
                    formats=("png", "pdf"))
    plt.close(fig)
    print(f"\nSaved → {OUTPUT_DIR / 'relational_convergence'}.png/.pdf")


if __name__ == "__main__":
    main()
