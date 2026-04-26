"""
IEEE-compliant matplotlib style for UAV DQN dissertation figures.
=================================================================
Import and call ``apply()`` at the top of any plotting script to get:
  - Times New Roman serif font, 10 pt base size
  - White background, no top/right spines
  - Light gray dashed y-axis gridlines only
  - High-contrast Set1 colour palette
  - High-resolution PNG output helpers (600 DPI)

Usage::

    import ieee_style
    ieee_style.apply()
    fig, ax = plt.subplots(...)
    ieee_style.save(fig, "output_dir/fig_name")  # writes .png at 600 DPI
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
# ColorBrewer Dark2 qualitative palette (8 colours, print/screen safe)
DARK2 = [
    "#1b9e77",   # teal
    "#d95f02",   # orange
    "#7570b3",   # purple
    "#e7298a",   # pink
    "#66a61e",   # green
    "#e6ab02",   # gold
    "#a6761d",   # brown
    "#666666",   # grey
]

# Legacy Set1 alias kept for backwards compatibility
SET1 = DARK2

# Three-agent canonical mapping (Dark2 subset, colourblind + print safe)
# Paired with distinct linestyles/markers so figures read in greyscale
AGENT_COLORS = {
    "DQN":                    "#1b9e77",   # teal   (Dark2[0])
    "DQN Agent":              "#1b9e77",
    "DQN Agent (Proposed)":   "#1b9e77",
    "SF-Aware Greedy":        "#d95f02",   # orange (Dark2[1])
    "SF-Aware Greedy V2":     "#d95f02",
    "Smart Greedy V2":        "#d95f02",
    "Nearest Greedy":         "#7570b3",   # purple (Dark2[2])
    "Nearest Sensor Greedy":  "#7570b3",
    "Relational RL":          "#66a61e",   # green  (Dark2[4])
    "Relational RL (PPO)":    "#66a61e",
}

AGENT_LINESTYLES = {
    "DQN":                    "-",
    "DQN Agent":              "-",
    "DQN Agent (Proposed)":   "-",
    "SF-Aware Greedy":        "--",
    "SF-Aware Greedy V2":     "--",
    "Smart Greedy V2":        "--",
    "Nearest Greedy":         ":",
    "Nearest Sensor Greedy":  ":",
    "Relational RL":          (0, (3, 1, 1, 1)),
    "Relational RL (PPO)":    (0, (3, 1, 1, 1)),
}

AGENT_MARKERS = {
    "DQN":                    "o",
    "DQN Agent":              "o",
    "DQN Agent (Proposed)":   "o",
    "SF-Aware Greedy":        "s",
    "SF-Aware Greedy V2":     "s",
    "Smart Greedy V2":        "s",
    "Nearest Greedy":         "^",
    "Nearest Sensor Greedy":  "^",
    "Relational RL":          "v",
    "Relational RL (PPO)":    "v",
}

# Spreading-factor colour map (SF7→green, SF12→purple)
SF_COLORS = {
    7:  "#4DAF4A",
    8:  "#A6D854",
    9:  "#FDB462",
    10: "#FF7F00",
    11: "#E41A1C",
    12: "#984EA3",
}

# Ablation condition colours (5 conditions)
ABLATION_COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]

# ---------------------------------------------------------------------------
# Core rcParams
# ---------------------------------------------------------------------------
RC = {
    # Font
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "DejaVu Serif", "serif"],
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "legend.title_fontsize": 9,
    # Spines / axes
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.8,
    # Grid: y-only, light gray dashed
    "axes.grid":          True,
    "axes.grid.axis":     "y",
    "grid.color":         "#CCCCCC",
    "grid.linestyle":     "--",
    "grid.linewidth":     0.5,
    "grid.alpha":         0.7,
    # Background
    "axes.facecolor":     "white",
    "figure.facecolor":   "white",
    # Lines
    "lines.linewidth":    1.5,
    # Ticks
    "xtick.direction":    "out",
    "ytick.direction":    "out",
    "xtick.major.size":   3.5,
    "ytick.major.size":   3.5,
    # Legend
    "legend.frameon":     True,
    "legend.framealpha":  0.9,
    "legend.edgecolor":   "#CCCCCC",
    # Saving
    "savefig.dpi":        600,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    # PDF metadata
    "pdf.fonttype":       42,   # embed fonts (TrueType) for Acrobat compatibility
    "ps.fonttype":        42,
}


def apply():
    """Apply IEEE rcParams to the current matplotlib session."""
    mpl.rcParams.update(RC)


# ---------------------------------------------------------------------------
# Spine / grid cleanup (for axes created before apply() was called)
# ---------------------------------------------------------------------------
def clean_axes(ax):
    """Remove top/right spines; enable y-only light-gray dashed grid."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="#CCCCCC", linestyle="--", linewidth=0.5, alpha=0.7)
    ax.grid(False, axis="x")


def clean_figure(fig):
    """Apply clean_axes to every axes in *fig*."""
    for ax in fig.axes:
        clean_axes(ax)


# ---------------------------------------------------------------------------
# Saving helpers
# ---------------------------------------------------------------------------
def save(fig, stem: str, formats=("png",)):
    """
    Save *fig* to ``<stem>.png`` at 600 DPI (default).

    Parameters
    ----------
    fig    : matplotlib Figure
    stem   : path without extension, e.g. ``"results/fig_trajectory"``
    formats: tuple of extensions to write (default: png)
    """
    from pathlib import Path
    stem = str(stem)
    for ext in formats:
        out = f"{stem}.{ext}"
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=600, bbox_inches="tight", format=ext)
        print(f"  Saved: {out}")


# ---------------------------------------------------------------------------
# Strip-plot overlay (for bar charts with per-seed variance)
# ---------------------------------------------------------------------------
def add_stripplot(ax, data_groups, x_positions, colors,
                  jitter=0.08, s=30, alpha=0.7, zorder=6):
    """
    Overlay individual data points on a bar chart.

    Parameters
    ----------
    ax           : matplotlib Axes
    data_groups  : list of 1-D arrays, one per bar
    x_positions  : list of x-coordinates matching data_groups
    colors       : list of colours matching data_groups
    jitter       : horizontal jitter width
    s            : marker size
    alpha        : marker alpha
    zorder       : drawing order (should be above bars)
    """
    rng = np.random.default_rng(0)
    for x, vals, col in zip(x_positions, data_groups, colors):
        jx = rng.uniform(-jitter, jitter, size=len(vals))
        ax.scatter(x + jx, vals, color=col, s=s, alpha=alpha,
                   edgecolors="white", linewidths=0.4, zorder=zorder)
