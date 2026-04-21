"""
Training convergence figure — single continuous timeline with gradient colour.

- Steps are concatenated across all curriculum stages (no reset to 0).
- Vertical dashed lines mark stage boundaries.
- Raw data shown in light colour; EMA (w=0.9) as bold gradient line.
- Gradient runs from light→dark blue (reward) / light→dark red (loss)
  representing progression from Stage 1 → Stage 16.
"""

import os
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.collections import LineCollection
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ieee_style
ieee_style.apply()

# ── Paths ──────────────────────────────────────────────────────────────────
LOG_ROOT = os.path.join(os.path.dirname(__file__),
                        "..", "logs", "dqn_domain_rand")
OUT_DIR  = os.path.join(os.path.dirname(__file__), "training_results")
os.makedirs(OUT_DIR, exist_ok=True)

TAG_REWARD = "rollout/ep_rew_mean"
TAG_LOSS   = "train/loss"
MIN_POINTS = 50          # skip aborted runs with fewer logged episodes
EMA_WEIGHT = 0.9


# ── Helpers ────────────────────────────────────────────────────────────────

def load_scalar(run_dir, tag):
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    if tag not in ea.Tags().get("scalars", []):
        return np.array([]), np.array([])
    events = ea.Scalars(tag)
    return (np.array([e.step  for e in events]),
            np.array([e.value for e in events]))


def collect_runs(log_root, tag, min_points=MIN_POINTS):
    """Return list of (name, steps, values) sorted by run index."""
    runs = []
    for name in sorted(os.listdir(log_root),
                       key=lambda x: int(x.split("_")[1])
                                     if x.startswith("DQN_") else 999):
        if not name.startswith("DQN_"):
            continue
        path = os.path.join(log_root, name)
        if not os.path.isdir(path):
            continue
        s, v = load_scalar(path, tag)
        if len(s) >= min_points:
            runs.append((name, s, v))
    return runs


def stitch(runs):
    """
    Offset each run so steps are globally continuous.
    Returns (global_steps, values, transition_steps, run_labels).
    transition_steps[i] = global step at which run i+1 begins.
    """
    g_steps, g_vals, transitions, labels = [], [], [], []
    offset = 0
    for i, (name, steps, vals) in enumerate(runs):
        # drop last point (environment-reset artefact at stage boundary)
        if len(steps) > 1:
            steps, vals = steps[:-1], vals[:-1]
        g_steps.append(steps + offset)
        g_vals.append(vals)
        labels.append(name.replace("DQN_", "Stage "))
        offset += int(steps[-1]) + 1
        if i < len(runs) - 1:
            transitions.append(offset)
    return (np.concatenate(g_steps),
            np.concatenate(g_vals),
            transitions,
            labels)


def ema(values, w=EMA_WEIGHT):
    out, last = [], float(values[0])
    for v in values:
        last = w * last + (1 - w) * v
        out.append(last)
    return np.array(out)


def gradient_line(ax, x, y, cmap_name, lw=2.2, alpha=1.0):
    """
    Draw a single line coloured by a gradient along cmap_name.
    Uses LineCollection so the colour changes smoothly.
    """
    points  = np.array([x, y]).T.reshape(-1, 1, 2)
    segs    = np.concatenate([points[:-1], points[1:]], axis=1)
    t       = np.linspace(0, 1, len(segs))
    cmap    = plt.get_cmap(cmap_name)
    lc      = LineCollection(segs, array=t, cmap=cmap,
                             linewidth=lw, alpha=alpha)
    ax.add_collection(lc)
    return lc


def add_transitions(ax, transitions, labels, ymin, ymax):
    """Vertical dashed lines + stage labels at curriculum boundaries."""
    for i, t in enumerate(transitions):
        x = t / 1e6
        ax.axvline(x, color="dimgray", lw=0.9, ls="--", alpha=0.55)
        ax.text(x + 0.02, ymax - (ymax - ymin) * 0.04,
                labels[i + 1],
                fontsize=7, color="dimgray", va="top", rotation=90)


# ── Main plot ──────────────────────────────────────────────────────────────

def plot(reward_runs, loss_runs, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    # ── Panel 1: Episode reward ──────────────────────────────────────────
    ax = axes[0]
    r_x, r_y, r_trans, r_labels = stitch(reward_runs)
    x_m = r_x / 1e6                              # convert to millions

    # Ghost raw data — very light so EMA trend dominates
    ax.plot(x_m, r_y, color="#9ecae1", lw=0.4, alpha=0.20, zorder=1)

    # Thick EMA gradient line — the main feature
    r_smooth = ema(r_y)
    lc = gradient_line(ax, x_m, r_smooth, "Blues", lw=3.0)
    ax.autoscale()

    # y-axis limits: 2nd–98th percentile to suppress extreme early stages
    ylo = np.percentile(r_y, 2)
    yhi = np.percentile(r_y, 98)
    pad = (yhi - ylo) * 0.08
    ax.set_ylim(ylo - pad, yhi + pad)

    add_transitions(ax, r_trans, r_labels, ylo - pad, yhi + pad)

    sm = plt.cm.ScalarMappable(cmap="Blues",
                               norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.01)
    cb.set_ticks([0, 1])
    cb.set_ticklabels([r_labels[0], r_labels[-1]])
    cb.set_label("Curriculum stage")

    ax.set_xlabel(r"Global Training Steps ($\times 10^6$)")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("(a) Episode Reward Convergence", fontweight="bold")
    ax.yaxis.set_major_formatter(
        ticker.FuncFormatter(
            lambda v, _: f"{v/1e6:.1f}M" if abs(v) >= 1e5 else f"{v:.0f}"
        )
    )
    ieee_style.clean_axes(ax)

    # ── Panel 2: TD Loss ─────────────────────────────────────────────────
    ax = axes[1]
    l_x, l_y, l_trans, l_labels = stitch(loss_runs)
    x_m = l_x / 1e6

    # Clip spikes (99th percentile cap)
    p99   = np.percentile(l_y, 99)
    l_y_c = np.clip(l_y, None, p99)

    # Ghost raw data
    ax.plot(x_m, l_y_c, color="#fc9272", lw=0.4, alpha=0.20, zorder=1)

    # Thick EMA trend
    l_smooth = ema(l_y_c)
    gradient_line(ax, x_m, l_smooth, "Reds", lw=3.0)
    ax.autoscale()

    ylo = np.percentile(l_y_c, 2)
    yhi = p99
    pad = (yhi - ylo) * 0.08
    ax.set_ylim(ylo - pad, yhi + pad)

    add_transitions(ax, l_trans, l_labels, ylo - pad, yhi + pad)

    sm2 = plt.cm.ScalarMappable(cmap="Reds",
                                norm=plt.Normalize(vmin=0, vmax=1))
    sm2.set_array([])
    cb2 = fig.colorbar(sm2, ax=ax, fraction=0.03, pad=0.01)
    cb2.set_ticks([0, 1])
    cb2.set_ticklabels([l_labels[0], l_labels[-1]])
    cb2.set_label("Curriculum stage")

    ax.set_xlabel(r"Global Training Steps ($\times 10^6$)")
    ax.set_ylabel("TD Loss")
    ax.set_title("(b) TD Loss Convergence", fontweight="bold")
    ieee_style.clean_axes(ax)

    plt.tight_layout()
    # Save as PDF and EPS (strip .png suffix if present)
    stem = str(out_path).removesuffix(".png")
    ieee_style.save(fig, stem)
    plt.close()


def main():
    print(f"Scanning: {LOG_ROOT}")
    reward_runs = collect_runs(LOG_ROOT, TAG_REWARD)
    loss_runs   = collect_runs(LOG_ROOT, TAG_LOSS)
    print(f"  Reward stages : {[r[0] for r in reward_runs]}")
    print(f"  Loss   stages : {[r[0] for r in loss_runs]}")

    out_path = os.path.join(OUT_DIR, "fig_training_convergence.png")
    plot(reward_runs, loss_runs, out_path)


if __name__ == "__main__":
    main()
