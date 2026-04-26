"""Generate the scalability figure (Fig 9) from the actual sweep_summary.csv
data (50 seeds per condition). Writes:

  baseline_results/scalability_ndr.png      -- NDR% vs grid size/N
  baseline_results/scalability_jfi.png      -- Jain's fairness vs grid size/N
  baseline_results/scalability_combined.png -- two-panel main-body figure
"""

from pathlib import Path
import csv

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
OUT  = HERE / "baseline_results"
SWEEP = OUT / "sweep" / "sweep_summary.csv"

# Load
with open(SWEEP) as f:
    rows = list(csv.DictReader(f))

agents  = ["DQN", "Nearest Greedy", "Smart Greedy V2", "Relational RL", "TSP Oracle"]
configs = [("100x100", "10"), ("200x200", "20"), ("300x300", "30"),
           ("400x400", "40"), ("500x500", "50")]
config_labels = [f"{g}\nN={n}" for g, n in configs]

def grab(metric):
    m = np.zeros((len(agents), len(configs)))
    s = np.zeros_like(m)
    for i, a in enumerate(agents):
        for j, (g, n) in enumerate(configs):
            r = next(r for r in rows
                     if r["agent"] == a
                     and r["grid"] == g
                     and r["n_sensors"] == n
                     and r["metric"] == metric)
            m[i, j] = float(r["mean"])
            s[i, j] = float(r["ci_half"])
    return m, s

ndr_m, ndr_s = grab("ndr_pct")
jfi_m, jfi_s = grab("jfi")

# Colours (IEEE-ish)
colours = {
    "DQN":             "#d62728",
    "Nearest Greedy":  "#8c564b",
    "Smart Greedy V2": "#9467bd",
    "Relational RL":   "#1f77b4",
    "TSP Oracle":      "#2ca02c",
}
markers = {
    "DQN":             "o",
    "Nearest Greedy":  "^",
    "Smart Greedy V2": "s",
    "Relational RL":   "D",
    "TSP Oracle":      "*",
}

def plot_metric(ax, mean, ci, title, ylabel, ylim=None):
    x = np.arange(len(configs))
    for i, a in enumerate(agents):
        ax.errorbar(x, mean[i], yerr=ci[i],
                    label=a, color=colours[a], marker=markers[a],
                    lw=1.6, markersize=7, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(config_labels, fontsize=9)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11, fontweight="bold")
    if ylim: ax.set_ylim(*ylim)
    ax.grid(alpha=0.3)

# Two-panel
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
plot_metric(axes[0], ndr_m, ndr_s,
            "(a) Coverage (NDR %)", "NDR (%)", ylim=(0, 105))
plot_metric(axes[1], jfi_m, jfi_s,
            "(b) Jain's Fairness Index", "JFI", ylim=(0, 1.05))
axes[0].legend(loc="lower left", fontsize=8, frameon=True)
plt.tight_layout()
fig.savefig(OUT / "scalability_combined.png", dpi=160)
plt.close(fig)
print("wrote scalability_combined.png")

# Standalone NDR
fig, ax = plt.subplots(figsize=(7.5, 4.5))
plot_metric(ax, ndr_m, ndr_s,
            "Coverage (NDR %) across scales (50 seeds / condition)",
            "NDR (%)", ylim=(0, 105))
ax.legend(loc="lower left", fontsize=9)
plt.tight_layout()
fig.savefig(OUT / "scalability_ndr.png", dpi=160)
plt.close(fig)
print("wrote scalability_ndr.png")

# Standalone JFI
fig, ax = plt.subplots(figsize=(7.5, 4.5))
plot_metric(ax, jfi_m, jfi_s,
            "Jain's Fairness Index across scales",
            "JFI", ylim=(0, 1.05))
ax.legend(loc="lower left", fontsize=9)
plt.tight_layout()
fig.savefig(OUT / "scalability_jfi.png", dpi=160)
plt.close(fig)
print("wrote scalability_jfi.png")
