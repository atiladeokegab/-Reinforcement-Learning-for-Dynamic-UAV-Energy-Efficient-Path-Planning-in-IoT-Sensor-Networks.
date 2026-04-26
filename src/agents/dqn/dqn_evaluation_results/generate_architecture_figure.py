"""Generate the Relational-RL architecture diagram (Fig 6 in the figure list).

Encoder -- sum-pool -- head schematic. Writes to
baseline_results/relational_architecture.png.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parent / "baseline_results" / "relational_architecture.png"

fig, ax = plt.subplots(figsize=(11, 5))
ax.set_xlim(0, 12); ax.set_ylim(0, 5.5); ax.axis("off")


def box(xy, w, h, text, fc="#e3f2fd", ec="#1565c0", fs=10, bold=False):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                                 facecolor=fc, edgecolor=ec, lw=1.5))
    ax.text(x + w / 2, y + h / 2, text,
            ha="center", va="center",
            fontsize=fs,
            fontweight="bold" if bold else "normal")


def arrow(xy1, xy2, lw=1.4, color="#333"):
    ax.add_patch(FancyArrowPatch(xy1, xy2,
                                 arrowstyle="-|>",
                                 mutation_scale=14,
                                 lw=lw, color=color))


# Column 1: sensor inputs (set of N)
for i, y in enumerate([4.0, 3.1, 2.2, 1.3]):
    box((0.2, y), 1.6, 0.65,
        f"$s_{{{i+1}}}$:  $[\\Delta x, \\Delta y, b, u, v]$",
        fc="#fff3e0", ec="#e65100", fs=9)
ax.text(1.0, 0.85, "$\\vdots$", ha="center", fontsize=14)
ax.text(1.0, 4.9, "Per-sensor features\n(masked, $n_{\\max}$ slots)",
        ha="center", fontsize=9, fontweight="bold")

# Column 2: shared encoder phi
box((2.4, 2.2), 1.6, 1.4,
    "$\\phi$ (shared MLP)\n2 $\\times$ 128, ReLU",
    fc="#e1f5fe", ec="#01579b", fs=10, bold=True)
for y in [4.3, 3.4, 2.5, 1.6]:
    arrow((1.8, y + 0.32), (2.4, 2.9))

# Column 3: masked sum
box((4.6, 2.4), 1.6, 1.0,
    "$\\sum_i m_i \\cdot \\phi(s_i)$\n(masked sum-pool)",
    fc="#f3e5f5", ec="#4a148c", fs=10, bold=True)
arrow((4.0, 2.9), (4.6, 2.9))

# Column 4: rho
box((6.8, 2.4), 1.5, 1.0,
    "$\\rho$ (post-pool MLP)\n2 $\\times$ 128, ReLU",
    fc="#e1f5fe", ec="#01579b", fs=10, bold=True)
arrow((6.2, 2.9), (6.8, 2.9))

# UAV branch
box((6.8, 0.8), 1.5, 0.8,
    "UAV state\n$[x, y, E]$",
    fc="#fff3e0", ec="#e65100", fs=9)

# Concat
box((8.8, 1.9), 1.3, 1.5,
    "concat\n$[\\text{uav}; z]$",
    fc="#ede7f6", ec="#311b92", fs=10)
arrow((8.3, 2.9), (8.8, 2.7))
arrow((8.3, 1.2), (8.8, 2.1))

# Head
box((10.3, 2.4), 1.5, 1.0,
    "head (linear)\nlogits $\\in \\mathbb{R}^5$, $V$",
    fc="#c8e6c9", ec="#1b5e20", fs=10, bold=True)
arrow((10.1, 2.65), (10.3, 2.9))

# Titles / annotations
ax.text(6.0, 5.2, "Relational-RL policy network (Deep-Sets style)",
        ha="center", fontsize=13, fontweight="bold")
ax.text(3.2, 0.25,
        "Permutation invariance: reordering $\\{s_i\\}$ leaves the masked sum unchanged.",
        ha="left", fontsize=9, style="italic", color="#4a148c")

fig.tight_layout()
fig.savefig(OUT, dpi=160, bbox_inches="tight")
print(f"wrote {OUT}")
