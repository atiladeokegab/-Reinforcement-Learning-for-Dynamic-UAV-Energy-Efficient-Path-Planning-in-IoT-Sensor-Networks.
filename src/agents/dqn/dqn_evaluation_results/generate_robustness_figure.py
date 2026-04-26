"""Generate the 4-panel sim-to-real robustness figure from
baseline_results/sim_to_real/robustness_raw.csv.

Writes baseline_results/sim_to_real/robustness_summary.png.
"""

from pathlib import Path
import csv
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RAW  = HERE / "baseline_results" / "sim_to_real" / "robustness_raw.csv"
OUT  = HERE / "baseline_results" / "sim_to_real" / "robustness_summary.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

if not RAW.exists():
    raise SystemExit(f"Missing raw data: {RAW}\nRun sim_to_real_sweep.py first.")

rows = list(csv.DictReader(RAW.open()))
for r in rows:
    r["value"] = float(r["value"])
    r["coverage_pct"] = float(r["coverage_pct"])
    r["reward"] = float(r["reward"])
    r["boundary_hits"] = int(float(r["boundary_hits"]))
    r["edge_step_pct"] = float(r["edge_step_pct"])

axes = ["gps_sigma", "path_loss", "shadow_sigma", "wind_drift"]
axis_labels = {
    "gps_sigma":    r"GPS noise $\sigma$ (grid units)",
    "path_loss":    r"Path-loss exponent $n$",
    "shadow_sigma": r"Shadowing std $\sigma_{\rm sh}$ (dB)",
    "wind_drift":   r"Wind drift (cells/step)",
}

fig, axarr = plt.subplots(2, 2, figsize=(11, 7.5))
for i, axis in enumerate(axes):
    ax = axarr[i // 2, i % 2]
    # pull rows for this axis only
    sub = [r for r in rows if r["axis"] == axis]
    # group by agent, value
    agg = defaultdict(list)
    for r in sub:
        agg[(r["agent"], r["value"])].append(r["coverage_pct"])
    values = sorted(set(r["value"] for r in sub))
    for agent, colour in [("DQN", "#d62728"),
                          ("Relational RL", "#1f77b4")]:
        means = [np.mean(agg[(agent, v)]) if agg[(agent, v)] else np.nan
                 for v in values]
        stds  = [np.std(agg[(agent, v)], ddof=1) / np.sqrt(max(len(agg[(agent, v)]), 1))
                 if agg[(agent, v)] else 0.0 for v in values]
        ci95  = [1.96 * s for s in stds]
        ax.errorbar(values, means, yerr=ci95, lw=1.8, marker="o",
                    capsize=3, label=agent, color=colour)
    ax.set_xlabel(axis_labels[axis])
    ax.set_ylabel("NDR (%)")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.set_title(axis_labels[axis], fontsize=10, fontweight="bold")

    # Flag the path-loss panel: its lines are flat by construction because the
    # Two-Ray implementation in iot_sensors.py uses hardcoded slope constants
    # (20 and 40 dB/decade). The `path_loss_exponent` attribute that the sweep
    # harness writes is therefore never consumed by the physics.
    if axis == "path_loss":
        ax.text(
            0.5, 0.45,
            "Perturbation no-op\n(hardcoded Two-Ray slopes)",
            transform=ax.transAxes,
            ha="center", va="center",
            fontsize=10, fontweight="bold",
            color="#8a1e1e",
            bbox=dict(boxstyle="round,pad=0.45",
                      facecolor="#fff1f0",
                      edgecolor="#8a1e1e",
                      linewidth=1.2,
                      alpha=0.92),
            zorder=5,
        )

axarr[0, 0].legend(loc="lower left", fontsize=9)
fig.suptitle("Sim-to-real robustness (10 seeds / cell, 200x200, N=20)",
             fontsize=12, fontweight="bold", y=1.00)
plt.tight_layout()
fig.savefig(OUT, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"wrote {OUT}")
