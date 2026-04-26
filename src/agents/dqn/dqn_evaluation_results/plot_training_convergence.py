"""
plot_training_convergence.py
============================
Reproduces a figure-7-style training convergence plot from TensorBoard logs.

Left panel  — Mean Episode Reward per curriculum stage (light→dark blue).
Right panel — TD Loss per curriculum stage (light→dark red).
Raw values plotted transparently; EMA (w=0.9) overlaid in bold.
Curriculum stage boundaries from graduation_log.json.

Output:
  baseline_results/training_convergence.png / .pdf

Usage:
    PYTHONIOENCODING=utf-8 uv run python \\
        src/agents/dqn/dqn_evaluation_results/plot_training_convergence.py
"""

from __future__ import annotations

import sys
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D

_HERE = Path(__file__).resolve().parent
_SRC  = _HERE.parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

try:
    import ieee_style
    ieee_style.apply()
except ImportError:
    pass

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LOG_DIR    = _HERE.parents[3] / "logs" / "dqn_v3_retrain"
GRAD_LOG   = _HERE.parent / "models" / "dqn_v3_retrain" / "graduation_log.json"
OUTPUT_DIR = _HERE / "baseline_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Use DQN_3 — the complete 0–5M step training run
TB_RUN = LOG_DIR / "DQN_3"

EMA_ALPHA = 0.9   # EMA smoothing weight (heavier = smoother)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_scalar(run_dir: Path, tag: str) -> tuple[np.ndarray, np.ndarray]:
    """Return (steps, values) arrays for a scalar tag."""
    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": 0})
    ea.Reload()
    events = ea.Scalars(tag)
    steps  = np.array([e.step  for e in events], dtype=np.float64)
    values = np.array([e.value for e in events], dtype=np.float64)
    return steps, values


def ema(values: np.ndarray, alpha: float = EMA_ALPHA) -> np.ndarray:
    """Exponential moving average, causal."""
    out = np.empty_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * out[i - 1] + (1 - alpha) * values[i]
    return out


def stage_colors(n: int, cmap_name: str) -> list:
    """n colours from light→dark in cmap_name."""
    cmap = plt.get_cmap(cmap_name)
    return [cmap(0.25 + 0.65 * i / max(n - 1, 1)) for i in range(n)]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    # ── Load graduation log ───────────────────────────────────────────────
    with open(GRAD_LOG) as f:
        grad = json.load(f)

    # Stage boundaries in training steps
    # Stage 0 starts at 0, each stage ends when the next begins
    stage_boundaries = [0] + [g["ts"] for g in grad] + [5_000_000]
    stage_names = [
        "Stage 0 (100×100, N=10)",
        "Stage 1 (200×200, N=20)",
        "Stage 2 (300×300, N=30)",
        "Stage 3 (400×400, N=40)",
        "Stage 4 (500×500, N=50)",
    ]
    n_stages = len(stage_names)

    blue_cols = stage_colors(n_stages, "Blues")
    red_cols  = stage_colors(n_stages, "Reds")

    # ── Load TensorBoard scalars ──────────────────────────────────────────
    print(f"Loading TensorBoard data from {TB_RUN.name}...")
    rew_steps, rew_vals = load_scalar(TB_RUN, "rollout/ep_rew_mean")
    loss_steps, loss_vals = load_scalar(TB_RUN, "train/loss")
    print(f"  Reward: {len(rew_steps)} points  "
          f"({rew_steps[0]/1e6:.2f}M – {rew_steps[-1]/1e6:.2f}M steps)")
    print(f"  Loss:   {len(loss_steps)} points  "
          f"({loss_steps[0]/1e6:.2f}M – {loss_steps[-1]/1e6:.2f}M steps)")

    # ── Plot ──────────────────────────────────────────────────────────────
    fig, (ax_r, ax_l) = plt.subplots(1, 2, figsize=(7.5, 3.4),
                                      gridspec_kw={"wspace": 0.35})

    def plot_panel(ax, steps, vals, colors, ylabel, title, show_legend):
        smooth = ema(vals)
        x = steps / 1e6   # x-axis in millions of steps

        for s_idx in range(n_stages):
            lo = stage_boundaries[s_idx]
            hi = stage_boundaries[s_idx + 1]
            mask = (steps >= lo) & (steps < hi)
            if not mask.any():
                continue
            col = colors[s_idx]
            # Raw (transparent)
            ax.plot(x[mask], vals[mask],
                    color=col, alpha=0.25, linewidth=0.6)
            # EMA (bold)
            ax.plot(x[mask], smooth[mask],
                    color=col, linewidth=1.6,
                    label=f"Stage {s_idx}")

        ax.set_xlabel("Global Training Steps (×10⁶)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=9, pad=4)
        ax.set_xlim(x[0], x[-1])

        if show_legend:
            handles = [
                Line2D([0], [0], color=colors[i], linewidth=1.6,
                       label=stage_names[i])
                for i in range(n_stages)
                if ((steps >= stage_boundaries[i]) &
                    (steps < stage_boundaries[i+1])).any()
            ]
            ax.legend(handles=handles, fontsize=6.5,
                      loc="upper left", framealpha=0.9,
                      title="Curriculum stage", title_fontsize=6.5)

    plot_panel(ax_r, rew_steps, rew_vals, blue_cols,
               "Mean Episode Reward", "(a) Episode Reward Convergence",
               show_legend=True)

    plot_panel(ax_l, loss_steps, loss_vals, red_cols,
               "TD Loss", "(b) TD Loss Convergence",
               show_legend=False)

    # Shared stage-boundary vertical lines
    for ax in (ax_r, ax_l):
        for ts in [g["ts"] / 1e6 for g in grad]:
            ax.axvline(ts, color="#888888", linewidth=0.7,
                       linestyle="--", alpha=0.6, zorder=0)

    # Colorbar-style stage legend on right panel (matching reference style)
    sm_r = plt.cm.ScalarMappable(
        cmap="Reds",
        norm=plt.Normalize(vmin=0, vmax=n_stages - 1)
    )
    sm_r.set_array([])
    cbar = fig.colorbar(sm_r, ax=ax_l, fraction=0.046, pad=0.04,
                        ticks=range(n_stages))
    cbar.ax.set_yticklabels([f"Stage {i}" for i in range(n_stages)],
                             fontsize=6.5)
    cbar.set_label("Curriculum stage", fontsize=7)

    fig.tight_layout()

    ieee_style.save(fig, str(OUTPUT_DIR / "training_convergence"),
                    formats=("png", "pdf"))
    plt.close(fig)
    print(f"\nSaved → {OUTPUT_DIR / 'training_convergence'}.png/.pdf")


if __name__ == "__main__":
    main()
