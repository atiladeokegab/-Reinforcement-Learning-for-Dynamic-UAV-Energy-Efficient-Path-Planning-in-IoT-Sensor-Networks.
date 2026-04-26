"""
Presentation-quality DRL system architecture figure.
Shows DQN (baseline) and Relational RL (proposed) side-by-side in the
style of the reference Fig. 2 DRL architecture diagram.

Output: baseline_results/drl_system_architecture.png
"""

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent / "baseline_results" / "drl_system_architecture.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

# ── colour palette ────────────────────────────────────────────────────────────
C_OBS   = "#FFF8E1"   # warm amber  – observation
C_OBS_E = "#F57F17"
C_DQN   = "#E3F2FD"   # light blue  – DQN
C_DQN_E = "#1565C0"
C_GNN   = "#F3E5F5"   # light purple – Relational RL
C_GNN_E = "#6A1B9A"
C_ACT   = "#E8F5E9"   # light green – action
C_ACT_E = "#2E7D32"
C_RWD   = "#FCE4EC"   # light red   – reward
C_RWD_E = "#B71C1C"
C_ENV   = "#E0F7FA"   # cyan        – environment
C_ENV_E = "#006064"
C_NODE_DQN = "#42A5F5"
C_NODE_GNN = "#AB47BC"
C_AGENT_BG = "#F5F5F5"

fig = plt.figure(figsize=(16, 9), facecolor="white")
ax  = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 16); ax.set_ylim(0, 9); ax.axis("off")

# ── helpers ───────────────────────────────────────────────────────────────────
def fbox(ax, xy, w, h, text, fc, ec, fs=9.5, bold=False, lw=1.6,
         va="center", ha="center", wrap=False, zorder=3, alpha=1.0,
         style="round,pad=0.10"):
    x, y = xy
    p = FancyBboxPatch((x, y), w, h, boxstyle=style,
                        facecolor=fc, edgecolor=ec, lw=lw,
                        zorder=zorder, alpha=alpha)
    ax.add_patch(p)
    ax.text(x + w/2, y + h/2, text,
            ha=ha, va=va, fontsize=fs,
            fontweight="bold" if bold else "normal",
            zorder=zorder+1, wrap=wrap,
            multialignment="center")
    return p

def arr(ax, xy1, xy2, color="#333333", lw=1.5, style="-|>",
        scale=14, ls="solid", zorder=4, conn="arc3,rad=0.0"):
    p = FancyArrowPatch(xy1, xy2,
                        arrowstyle=style,
                        connectionstyle=conn,
                        mutation_scale=scale,
                        lw=lw, color=color,
                        linestyle=ls, zorder=zorder)
    ax.add_patch(p)

def node(ax, cx, cy, r=0.13, fc=C_NODE_DQN, ec="white", lw=1.2, zorder=5):
    c = Circle((cx, cy), r, fc=fc, ec=ec, lw=lw, zorder=zorder)
    ax.add_patch(c)
    return cx, cy

def nn_layer(ax, cx, ys, r, fc, ec="white"):
    for y in ys:
        node(ax, cx, y, r=r, fc=fc, ec=ec)

def draw_dqn_network(ax, x0, y_center, fc=C_NODE_DQN):
    """Draw an MLP visualisation at (x0, y_center)."""
    r    = 0.11
    gap  = 0.35
    col0 = [y_center + d*gap for d in [-1, -0.5, 0, 0.5, 1]]      # input  (5)
    col1 = [y_center + d*gap for d in [-1, -0.5, 0, 0.5, 1]]      # 512
    col2 = [y_center + d*gap for d in [-0.75, -0.25, 0.25, 0.75]] # 512
    col3 = [y_center + d*gap for d in [-0.5, 0, 0.5]]             # 256
    col4 = [y_center + d*gap for d in [-0.5, -0.25, 0, 0.25, 0.5]]# Q(5)

    cols = [col0, col1, col2, col3, col4]
    xs   = [x0, x0+0.65, x0+1.30, x0+1.95, x0+2.60]

    # connections (light grey)
    for ci in range(len(cols)-1):
        for ya in cols[ci]:
            for yb in cols[ci+1]:
                ax.add_line(Line2D([xs[ci], xs[ci+1]], [ya, yb],
                                   color="#BDBDBD", lw=0.4, zorder=4))

    # nodes
    for ci, (cx, ys) in enumerate(zip(xs, cols)):
        c = fc if ci < len(cols)-1 else C_ACT_E
        for y in ys:
            node(ax, cx, y, r=r, fc=c, ec="white", lw=0.8)

    return xs[-1]  # rightmost x


def draw_gnn_network(ax, x0, y_center, fc=C_NODE_GNN):
    """Draw the GNN (attention→GRU→proj) block diagram inside the box."""
    bh = 0.55   # sub-block height
    bw = 1.20   # sub-block width
    gap = 0.20
    y0 = y_center - bh/2

    # Sub-block 1: Self-Attention
    xa = x0
    fbox(ax, (xa, y0), bw, bh,
         "Multi-Head\nSelf-Attention\n(4 heads, d=64)",
         fc="#EDE7F6", ec=C_GNN_E, fs=7.5, zorder=6)

    arr(ax, (xa+bw, y_center), (xa+bw+gap, y_center),
        color=C_GNN_E, lw=1.3, scale=10, zorder=7)

    # Sub-block 2: GRU
    xb = xa + bw + gap
    fbox(ax, (xb, y0), bw, bh,
         "GRU Temporal\nEncoder\n(k=10, h=128)",
         fc="#F8BBD9", ec="#880E4F", fs=7.5, zorder=6)

    arr(ax, (xb+bw, y_center), (xb+bw+gap, y_center),
        color=C_GNN_E, lw=1.3, scale=10, zorder=7)

    # Sub-block 3: Projection
    xc = xb + bw + gap
    fbox(ax, (xc, y0), bw, bh,
         "Linear\nProjection\n(d=256)",
         fc="#D1C4E9", ec=C_GNN_E, fs=7.5, zorder=6)

    return xc + bw   # rightmost x

# ══════════════════════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════════════════════
ax.text(8, 8.65, "Deep Reinforcement Learning Architecture: UAV IoT Data Collection",
        ha="center", va="center", fontsize=14, fontweight="bold", color="#1A237E")

# ══════════════════════════════════════════════════════════════════════════════
# DRL AGENT bounding box (dashed)
# ══════════════════════════════════════════════════════════════════════════════
agent_box = FancyBboxPatch((3.0, 3.8), 9.2, 4.1,
                            boxstyle="round,pad=0.15",
                            facecolor=C_AGENT_BG, edgecolor="#546E7A",
                            lw=2.0, linestyle="dashed", zorder=1)
ax.add_patch(agent_box)
ax.text(7.6, 8.05, "DRL Agent", ha="center", va="center",
        fontsize=11, fontweight="bold", color="#546E7A",
        bbox=dict(fc="white", ec="#546E7A", boxstyle="round,pad=0.2", lw=1.2))

# UAV icon (simple square drone symbol)
drone = FancyBboxPatch((7.25, 7.95), 0.70, 0.40,
                        boxstyle="round,pad=0.05",
                        fc="#CFD8DC", ec="#546E7A", lw=1.2, zorder=5)
ax.add_patch(drone)
ax.text(7.60, 8.15, "UAV", ha="center", va="center",
        fontsize=7.5, fontweight="bold", color="#37474F", zorder=6)

# ══════════════════════════════════════════════════════════════════════════════
# OBSERVATION BOX (left)
# ══════════════════════════════════════════════════════════════════════════════
fbox(ax, (0.3, 4.5), 2.4, 3.0,
     "UAV State\n$[x,\\ y,\\ E_{batt}]$\n"
     "──────────────\n"
     "Sensor Features\n"
     "$[b_i,\\ u_i,\\ v_i]_{i=1}^{N}$\n"
     "──────────────\n"
     "Frame stack:\n"
     "DQN $k=4$  |  Rel-RL $k=10$",
     fc=C_OBS, ec=C_OBS_E, fs=8.5, lw=2.0, zorder=3)

ax.text(1.5, 7.7, "Observation", ha="center", va="center",
        fontsize=9.5, fontweight="bold", color=C_OBS_E)

# ══════════════════════════════════════════════════════════════════════════════
# STATE box (inside agent, splits into two paths)
# ══════════════════════════════════════════════════════════════════════════════
STATE_Y  = 5.48   # bottom of State box
STATE_H  = 0.85
STATE_CY = STATE_Y + STATE_H / 2

fbox(ax, (3.2, STATE_Y), 1.35, STATE_H,
     "State\n(1530-dim)",
     fc="#ECEFF1", ec="#546E7A", fs=8.5, bold=True, lw=1.5)

# Obs → State arrow (horizontal from obs box right edge to state box left)
arr(ax, (2.70, 6.05), (3.20, STATE_CY), color=C_OBS_E, lw=2.0, scale=14)

# ══════════════════════════════════════════════════════════════════════════════
# DQN PATH (top half of agent, y_center ~ 6.85)
# ══════════════════════════════════════════════════════════════════════════════
DQN_Y = 6.85   # vertical centre of DQN row
GNN_Y = 4.60   # vertical centre of Relational RL row

# Label pill – floated left of the MLP
fbox(ax, (3.20, DQN_Y - 0.28), 1.90, 0.56,
     "DQN (Baseline)",
     fc=C_DQN, ec=C_DQN_E, fs=8.5, bold=True, lw=1.5)

# State → DQN label arrow (upward branch from state top)
arr(ax, (3.875, STATE_Y + STATE_H), (3.875, DQN_Y - 0.28),
    color=C_DQN_E, lw=1.6, scale=12)

# Draw MLP network
nn_x_end = draw_dqn_network(ax, x0=5.20, y_center=DQN_Y, fc=C_NODE_DQN)

# MLP caption below nodes
ax.text(5.20 + 1.30, DQN_Y - 0.65, "MLP  [512 → 512 → 256]",
        ha="center", va="center", fontsize=7.5,
        color=C_DQN_E, style="italic")

# Label pill → MLP entry connector
arr(ax, (5.10, DQN_Y), (5.20, DQN_Y), color=C_DQN_E, lw=1.4, scale=12)
ax.add_line(Line2D([5.10, 5.10], [DQN_Y - 0.00, DQN_Y],
                   color=C_DQN_E, lw=1.6, zorder=5))
ax.add_line(Line2D([4.55, 5.10], [DQN_Y, DQN_Y],
                   color=C_DQN_E, lw=1.6, zorder=5))
ax.add_line(Line2D([4.55, 4.55], [STATE_Y + STATE_H, DQN_Y],
                   color=C_DQN_E, lw=1.6, zorder=5))

# ══════════════════════════════════════════════════════════════════════════════
# RELATIONAL RL PATH (bottom half of agent, y_center ~ 4.60)
# ══════════════════════════════════════════════════════════════════════════════

# Label pill – floated left of the GNN blocks
fbox(ax, (3.20, GNN_Y - 0.28), 2.10, 0.56,
     "Relational RL (Proposed)",
     fc=C_GNN, ec=C_GNN_E, fs=8.5, bold=True, lw=1.5)

# State → Relational RL label arrow (downward branch from state bottom)
arr(ax, (3.875, STATE_Y), (3.875, GNN_Y + 0.28),
    color=C_GNN_E, lw=1.6, scale=12)

# Draw GNN sub-blocks
gnn_x_end = draw_gnn_network(ax, x0=5.20, y_center=GNN_Y, fc=C_NODE_GNN)

# Label pill → GNN entry connector
arr(ax, (5.10, GNN_Y), (5.20, GNN_Y), color=C_GNN_E, lw=1.4, scale=12)
ax.add_line(Line2D([4.55, 5.10], [GNN_Y, GNN_Y],
                   color=C_GNN_E, lw=1.6, zorder=5))
ax.add_line(Line2D([4.55, 4.55], [GNN_Y + 0.28, STATE_Y],
                   color=C_GNN_E, lw=1.6, zorder=5))

# Annotation: permutation invariance note
ax.text(6.9, GNN_Y - 0.70,
        "N sensors $\\rightarrow$ graph nodes  (permutation invariant)",
        ha="center", va="center", fontsize=7.5,
        color=C_GNN_E, style="italic",
        bbox=dict(fc="#F3E5F5", ec=C_GNN_E,
                  boxstyle="round,pad=0.15", lw=0.8))

# ══════════════════════════════════════════════════════════════════════════════
# Q-VALUES / ACTION SELECTION box (right, inside agent)
# ══════════════════════════════════════════════════════════════════════════════
Q_CX = 10.2   # left edge of Q-box
Q_Y  = 4.90   # bottom of Q-box
Q_H  = 2.60

fbox(ax, (Q_CX, Q_Y), 1.85, Q_H,
     "Q-values\n$Q(s, a)$\n──────\n$a_1$: North\n$a_2$: South\n$a_3$: East\n$a_4$: West\n$a_5$: Hover",
     fc="#E8EAF6", ec="#283593", fs=8.5, lw=1.6)

Q_MID = Q_Y + Q_H / 2   # vertical midpoint of Q-box

# DQN → Q-values
arr(ax, (nn_x_end, DQN_Y), (Q_CX, Q_MID + 0.4),
    color=C_DQN_E, lw=1.6, scale=13,
    conn="arc3,rad=-0.12")

# GNN → Q-values
arr(ax, (gnn_x_end, GNN_Y), (Q_CX, Q_MID - 0.4),
    color=C_GNN_E, lw=1.6, scale=13,
    conn="arc3,rad=0.12")

# ══════════════════════════════════════════════════════════════════════════════
# ACTION box (right, outside agent)
# ══════════════════════════════════════════════════════════════════════════════
ACT_Y = Q_Y
fbox(ax, (12.6, ACT_Y), 2.8, Q_H,
     "Action\n$a^* = \\arg\\max_a Q(s,a)$\n──────────────\n"
     "N / S / E / W / Hover\n(5 discrete actions)",
     fc=C_ACT, ec=C_ACT_E, fs=8.5, bold=False, lw=2.0)

ax.text(14.0, ACT_Y + Q_H + 0.18, "Action", ha="center", va="center",
        fontsize=9.5, fontweight="bold", color=C_ACT_E)

# Q-values → Action
arr(ax, (Q_CX + 1.85, Q_MID), (12.6, Q_MID),
    color="#283593", lw=2.0, scale=14)

# ══════════════════════════════════════════════════════════════════════════════
# REWARD box (bottom-centre, below agent)
# ══════════════════════════════════════════════════════════════════════════════
RWD_Y = 0.55
fbox(ax, (4.8, RWD_Y), 4.8, 1.55,
     "$R_t = +100 \\cdot B_t \\cdot u_i$  [throughput]\n"
     "$\\quad - 1000 \\cdot \\sigma^2_{norm}$  [fairness penalty]\n"
     "$\\quad + 5000 \\cdot \\mathbf{1}[new\\ sensor]$  [coverage bonus]",
     fc=C_RWD, ec=C_RWD_E, fs=9.5, lw=2.0)

ax.text(7.2, RWD_Y + 1.72, "Reward Function", ha="center", va="center",
        fontsize=9.5, fontweight="bold", color=C_RWD_E)

# ══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT box (bottom-right)
# ══════════════════════════════════════════════════════════════════════════════
ENV_Y = RWD_Y
fbox(ax, (12.6, ENV_Y), 2.8, 3.40,
     "",
     fc=C_ENV, ec=C_ENV_E, fs=8.5, lw=2.0)

ax.text(14.0, ENV_Y + 3.57, "Environment", ha="center", va="center",
        fontsize=9.5, fontweight="bold", color=C_ENV_E)

# Inset: sensor nodes + UAV dot + flight path
env_ax = ax.inset_axes([0.800, 0.065, 0.175, 0.340])
env_ax.set_xlim(0, 10); env_ax.set_ylim(0, 10)
env_ax.set_facecolor(C_ENV)
env_ax.set_xticks([]); env_ax.set_yticks([])
env_ax.spines[:].set_visible(False)

# LoRa sensor nodes (triangles)
rng = np.random.default_rng(42)
sx = rng.uniform(0.8, 9.2, 12)
sy = rng.uniform(0.8, 9.2, 12)
env_ax.scatter(sx, sy, c="#F57F17", s=55, marker="^",
               zorder=5, label="IoT sensor")

# UAV flight path
tx = [1.0, 2.5, 4.2, 5.8, 7.2, 8.6, 7.8, 6.1, 4.3, 2.8, 4.0, 5.5]
ty = [8.5, 7.2, 8.0, 6.3, 7.5, 5.0, 3.2, 4.1, 2.2, 3.5, 5.2, 7.0]
env_ax.plot(tx, ty, color="#1565C0", lw=1.4, alpha=0.65, zorder=3)

# Directional arrows along path
for i in range(0, len(tx)-1, 2):
    env_ax.annotate("", xy=(tx[i+1], ty[i+1]), xytext=(tx[i], ty[i]),
                    arrowprops=dict(arrowstyle="-|>", color="#1565C0",
                                   lw=1.1, mutation_scale=7))

# UAV represented as a filled circle (current position = last waypoint)
env_ax.plot(tx[-1], ty[-1], "o", color="#1565C0", ms=8,
            markeredgecolor="white", markeredgewidth=1.2,
            zorder=6, label="UAV")

# Mini legend inside inset
env_ax.scatter([], [], c="#F57F17", marker="^", s=40, label="Sensor")
env_ax.legend(loc="lower right", fontsize=4.5, framealpha=0.7,
              handlelength=1.0, borderpad=0.4)

env_ax.text(5, -1.5, "Grid: 100–1000 units  |  N = 10–40 sensors",
            ha="center", va="top", fontsize=5.2, color=C_ENV_E)

# ══════════════════════════════════════════════════════════════════════════════
# ARROWS: reward feedback loop (dashed)
# ══════════════════════════════════════════════════════════════════════════════
RWD_MID_Y = RWD_Y + 0.775

# Action → Environment (drop down right side, then left into env box)
ACT_MID_Y = ACT_Y + Q_H / 2
ax.add_line(Line2D([15.0, 15.0], [ACT_Y, ENV_Y + 3.40/2],
                   color=C_ACT_E, lw=1.8, linestyle="dashed", zorder=5))
arr(ax, (15.0, ENV_Y + 3.40/2), (15.40, ENV_Y + 3.40/2),
    color=C_ACT_E, lw=1.8, scale=13, ls="dashed")

# Environment → Reward (left along the bottom)
arr(ax, (12.6, RWD_MID_Y), (9.60, RWD_MID_Y),
    color=C_ENV_E, lw=1.8, scale=13, ls="dashed")

# Reward → State (up the left side back into agent)
ax.add_line(Line2D([4.80, 3.60], [RWD_MID_Y, RWD_MID_Y],
                   color=C_RWD_E, lw=1.8, linestyle="dashed", zorder=5))
ax.add_line(Line2D([3.60, 3.60], [RWD_MID_Y, STATE_CY],
                   color=C_RWD_E, lw=1.8, linestyle="dashed", zorder=5))
arr(ax, (3.60, STATE_CY), (3.20, STATE_CY),
    color=C_RWD_E, lw=1.8, scale=13, ls="dashed")

# ══════════════════════════════════════════════════════════════════════════════
# Obs → State horizontal arrow across boundary
# ══════════════════════════════════════════════════════════════════════════════
# (already drawn above at the State box)

# ══════════════════════════════════════════════════════════════════════════════
# LEGEND
# ══════════════════════════════════════════════════════════════════════════════
legend_elements = [
    mpatches.Patch(fc=C_DQN,  ec=C_DQN_E,  lw=1.5, label="DQN (Flat MLP baseline)"),
    mpatches.Patch(fc=C_GNN,  ec=C_GNN_E,  lw=1.5, label="Relational RL (GNN + GRU)"),
    Line2D([0], [0], color="#333", lw=1.5, ls="solid",  label="Forward pass"),
    Line2D([0], [0], color="#333", lw=1.5, ls="dashed", label="Feedback / reward"),
]
ax.legend(handles=legend_elements, loc="upper left",
          bbox_to_anchor=(0.01, 0.98), fontsize=8.5,
          framealpha=0.95, edgecolor="#546E7A", ncol=2)

# ══════════════════════════════════════════════════════════════════════════════
# CAPTION
# ══════════════════════════════════════════════════════════════════════════════
ax.text(8.0, 0.3,
        "Fig. DRL system architecture. Both agents share the same observation and action spaces. "
        "The Relational RL agent replaces the flat MLP extractor with\n"
        "sensor self-attention (spatial GNN) and a GRU temporal encoder, "
        "enabling permutation-invariant reasoning over a variable number of IoT sensors.",
        ha="center", va="center", fontsize=8.0, color="#37474F",
        style="italic", multialignment="center")

fig.savefig(OUT, dpi=180, bbox_inches="tight", facecolor="white")
print(f"Saved → {OUT}")
