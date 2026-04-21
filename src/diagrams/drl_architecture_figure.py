"""
DRL architecture figure — matches IEEE paper style (Fig. 3, arXiv 1810.07862).
White boxes, black borders, gray neurons, dashed reward arrow, serif font.
Output: figures/drl_architecture.pdf + .png  (600 DPI)
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent.parent.parent / \
      "MSc_and_BEng_Dissertation_Template_the_University_of_Manchester_EEE__2_" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family':         'serif',
    'font.serif':          ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size':           8,
    'text.usetex':         False,
    'pdf.fonttype':        42,
    'ps.fonttype':         42,
    'figure.facecolor':    'white',
    'axes.facecolor':      'white',
})

BLACK  = '#000000'
GRAY_N = '#AAAAAA'    # neuron fill (same as paper)
WHITE  = '#FFFFFF'
DASH_C = '#333333'    # dashed arrow colour

# ── Helpers ───────────────────────────────────────────────────────────────────

def box(ax, cx, cy, w, h, text, fontsize=7.5, lw=0.9, fc=WHITE, ec=BLACK):
    """Draw a plain rectangle with centred multi-line text."""
    rect = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                          boxstyle="square,pad=0",
                          facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3)
    ax.add_patch(rect)
    ax.text(cx, cy, text, ha='center', va='center',
            fontsize=fontsize, multialignment='center',
            linespacing=1.4, zorder=4)


def arrow(ax, x0, y0, x1, y1, label='', lfs=7, lside='above',
          color=BLACK, lw=0.9, dashed=False, rad=0.0):
    ls = 'dashed' if dashed else 'solid'
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                linestyle=ls, mutation_scale=9,
                                connectionstyle=f'arc3,rad={rad}'))
    if label:
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        dy = 0.038 if lside == 'above' else -0.038
        ax.text(mx, my + dy, label, ha='center', va='center',
                fontsize=lfs, style='italic', color=color, zorder=5)


def neuron(ax, cx, cy, r=0.028):
    c = plt.Circle((cx, cy), r, facecolor=GRAY_N, edgecolor=BLACK,
                   linewidth=0.7, zorder=4)
    ax.add_patch(c)
    return c


# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.6))
fig.patch.set_facecolor('white')
for ax in axes:
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis('off')

# ══════════════════════════════════════════════════════════════════════════════
# (a) Reinforcement Learning loop
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[0]
ax.set_title('(a) Reinforcement learning', fontsize=9, pad=7,
             fontfamily='serif')

# Boxes
box(ax, 0.22, 0.50, 0.31, 0.28,
    'Agent\nController\npolicy $\\pi$', fontsize=7.5)
box(ax, 0.78, 0.50, 0.28, 0.22,
    'Environment', fontsize=7.5)

# Action arrow (top, left → right)
arrow(ax, 0.375, 0.62, 0.64, 0.62,
      label='Action $a$', lfs=7, lside='above')

# Observed state arrow (bottom, right → left)
arrow(ax, 0.64, 0.39, 0.375, 0.39,
      label='Observed state $s$', lfs=7, lside='below')

# Immediate reward — dashed, curved over top
ax.annotate('', xy=(0.22, 0.68), xytext=(0.78, 0.68),
            arrowprops=dict(arrowstyle='->', color=DASH_C, lw=0.9,
                            linestyle='dashed', mutation_scale=9,
                            connectionstyle='arc3,rad=-0.32'))
ax.text(0.50, 0.90, 'Immediate reward $r$',
        ha='center', va='center', fontsize=7, style='italic',
        color=DASH_C, zorder=5)

# ══════════════════════════════════════════════════════════════════════════════
# (b) DQN Neural Network   Input→FC512→FC512→FC256→Output
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[1]
ax.set_title('(b) DQN neural network  [512, 512, 256]',
             fontsize=9, pad=7, fontfamily='serif')

layer_x = [0.10, 0.32, 0.52, 0.70, 0.90]
layer_n = [4,    5,    5,    4,    3   ]   # visible neurons

r = 0.030
node_ys = []
for n in layer_n:
    node_ys.append(np.linspace(0.25, 0.80, n))

# Weight lines (draw before neurons)
for i in range(len(layer_x) - 1):
    for y0 in node_ys[i]:
        for y1 in node_ys[i + 1]:
            ax.plot([layer_x[i] + r, layer_x[i+1] - r], [y0, y1],
                    color='#BBBBBB', lw=0.3, alpha=0.7, zorder=1)

# Neurons
for i, (lx, ys) in enumerate(zip(layer_x, node_ys)):
    for y in ys:
        neuron(ax, lx, y, r)
    if layer_n[i] == 5:
        ax.text(lx, 0.14, '$\\vdots$', ha='center', va='center',
                fontsize=9, color='#666666')

# Top group labels (spanning the layer columns)
ax.text(layer_x[0], 0.93, 'Input\nlayer',   ha='center', va='bottom', fontsize=7, fontfamily='serif')
ax.text(np.mean(layer_x[1:4]), 0.93, 'Hidden layers',  ha='center', va='bottom', fontsize=7, fontfamily='serif')
ax.text(layer_x[4], 0.93, 'Output\nlayer',  ha='center', va='bottom', fontsize=7, fontfamily='serif')

# Bottom per-column labels
bot = [('Inputs\n(1530-d)', layer_x[0]),
       ('Weights $\\theta$', np.mean(layer_x[1:4])),
       ('Outputs\n(5 actions)', layer_x[4])]
for lbl, lx in bot:
    ax.text(lx, 0.06, lbl, ha='center', va='top',
            fontsize=7, style='italic', fontfamily='serif',
            multialignment='center')

# Dimension annotations under each hidden column
for lx, lbl in zip(layer_x[1:4], ['FC 512', 'FC 512', 'FC 256']):
    ax.text(lx, 0.10, lbl, ha='center', va='top',
            fontsize=6, color='#555555', fontfamily='serif')

# ══════════════════════════════════════════════════════════════════════════════
# (c) Deep Q-Learning system
# ══════════════════════════════════════════════════════════════════════════════
ax = axes[2]
ax.set_title('(c) Deep Q-learning', fontsize=9, pad=7, fontfamily='serif')

# Outer Agent box (large)
box(ax, 0.24, 0.56, 0.36, 0.36,
    'Agent', fontsize=7.5)

# Mini neural network inside Agent box
mini_lx = [0.10, 0.17, 0.25, 0.33]
mini_n  = [2,    3,    3,    2   ]
mr = 0.014
mini_ys = [np.linspace(0.50, 0.64, n) for n in mini_n]

for i, (mx, mys) in enumerate(zip(mini_lx, mini_ys)):
    for my in mys:
        c = plt.Circle((mx, my), mr, facecolor=GRAY_N, edgecolor=BLACK,
                       linewidth=0.5, zorder=6)
        ax.add_patch(c)
    if i < len(mini_lx) - 1:
        for my in mys:
            for ny in mini_ys[i+1]:
                ax.plot([mx + mr, mini_lx[i+1] - mr], [my, ny],
                        color='#BBBBBB', lw=0.25, alpha=0.7, zorder=5)

ax.text(0.22, 0.71, 'Controller policy $\\pi$',
        ha='center', fontsize=6.5, style='italic', zorder=7)
ax.text(0.22, 0.46, 'features', ha='center', fontsize=6.5,
        style='italic', color='#444444', zorder=7)
ax.text(0.22, 0.41, 'Weights $\\theta$', ha='center', fontsize=6.5,
        style='italic', color='#444444', zorder=7)

# Environment box
box(ax, 0.80, 0.56, 0.28, 0.22, 'Environment', fontsize=7.5)

# Action arrow
arrow(ax, 0.42, 0.65, 0.66, 0.65,
      label='Action $a$', lfs=7, lside='above')

# State arrow
arrow(ax, 0.66, 0.48, 0.42, 0.48,
      label='Observed state $s$', lfs=7, lside='below')

# Reward — dashed curved
ax.annotate('', xy=(0.24, 0.76), xytext=(0.80, 0.76),
            arrowprops=dict(arrowstyle='->', color=DASH_C, lw=0.9,
                            linestyle='dashed', mutation_scale=9,
                            connectionstyle='arc3,rad=-0.28'))
ax.text(0.52, 0.90, 'Immediate reward $r$',
        ha='center', va='center', fontsize=7, style='italic',
        color=DASH_C, zorder=5)

# Replay buffer + target network (bottom, inside a plain box)
box(ax, 0.50, 0.16, 0.84, 0.20,
    'Experience replay buffer  $\\mathcal{D} = \\{(s,a,r,s^{\\prime})\\}$\n'
    'Online net $Q(s,a;\\theta)$ — gradient update\n'
    'Target net $\\hat{Q}(s,a;\\theta^{-})$ — frozen, copied every $\\tau$ steps',
    fontsize=6.2, lw=0.7)

# ── Save ─────────────────────────────────────────────────────────────────────
plt.tight_layout(pad=1.2, w_pad=1.5)
pdf_path = OUT / "drl_architecture.pdf"
png_path = OUT / "drl_architecture.png"
plt.savefig(pdf_path, dpi=600, bbox_inches='tight', facecolor='white')
plt.savefig(png_path, dpi=600, bbox_inches='tight', facecolor='white')
print("Saved:", pdf_path)
print("Saved:", png_path)
plt.close()
