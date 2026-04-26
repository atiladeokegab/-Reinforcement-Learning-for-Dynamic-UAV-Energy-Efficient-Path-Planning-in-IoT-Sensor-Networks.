"""Generate the four data-driven figures for the dissertation:

  Fig 10/12  Side-by-side DQN vs Relational-RL position heatmaps
  Fig 13     Forest plot of Relational-RL - DQN deltas with 95% CIs
  Fig 1      System-model overview (simple schematic)

Outputs land in `baseline_results/` using the names main.tex expects:

  - dqn_boundary_heatmap.png          Fig 10
  - relational_boundary_heatmap.png   Fig 12
  - boundary_heatmap_compare.png      Combined 1x2 panel
  - forest_plot_dqn_vs_relational.png Fig 13
  - system_model.png                  Fig 1 (unblocks compile)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow, Rectangle

HERE   = Path(__file__).resolve().parent
OUTDIR = HERE / "baseline_results"
OUTDIR.mkdir(parents=True, exist_ok=True)

DQN_JSON = OUTDIR / "boundary_diagnostic.json"
REL_JSON = OUTDIR / "boundary_diagnostic_relational.json"

GRID_W = GRID_H = 200


def load_runs(path: Path):
    with open(path) as f:
        runs = json.load(f)
    pos = np.concatenate([np.array(r["positions"]) for r in runs], axis=0)
    return runs, pos


# ------------------------------------------------------------------
# Fig 10 + 12: boundary heatmaps
# ------------------------------------------------------------------

def make_heatmap(ax, positions, title, bins=50, cmap="magma"):
    H, xe, ye = np.histogram2d(
        positions[:, 0], positions[:, 1],
        bins=bins, range=[[0, GRID_W], [0, GRID_H]],
    )
    H = H.T
    H_norm = H / max(H.sum(), 1.0)
    im = ax.imshow(
        H_norm,
        origin="lower",
        extent=[0, GRID_W, 0, GRID_H],
        cmap=cmap,
        aspect="equal",
        vmin=0,
        vmax=np.percentile(H_norm[H_norm > 0], 99) if (H_norm > 0).any() else 1,
    )
    ax.set_xlim(0, GRID_W); ax.set_ylim(0, GRID_H)
    ax.set_xlabel("X coordinate")
    ax.set_ylabel("Y coordinate")
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.add_patch(Rectangle((0, 0), GRID_W, GRID_H, fill=False,
                           edgecolor="white", lw=0.8, ls="--", alpha=0.5))
    return im


print("[1/3] Boundary heatmaps")
dqn_runs, dqn_pos = load_runs(DQN_JSON)
rel_runs, rel_pos = load_runs(REL_JSON)

# Individual + combined
for pos, fname, title in [
    (dqn_pos, "dqn_boundary_heatmap.png",        "DQN (flat MLP): boundary-adjacent occupancy"),
    (rel_pos, "relational_boundary_heatmap.png", "Relational RL (PPO): interior exploration"),
]:
    fig, ax = plt.subplots(figsize=(6, 5.5))
    im = make_heatmap(ax, pos, title)
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Fraction of all-episode steps", fontsize=9)
    plt.tight_layout()
    fig.savefig(OUTDIR / fname, dpi=160)
    plt.close(fig)
    print(f"  wrote {fname}")

# Combined
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))
im0 = make_heatmap(axes[0], dqn_pos, "(a) DQN (flat MLP)")
im1 = make_heatmap(axes[1], rel_pos, "(b) Relational RL (PPO)")
for ax in axes:
    ax.set_aspect("equal")
plt.tight_layout()
fig.savefig(OUTDIR / "boundary_heatmap_compare.png", dpi=160)
plt.close(fig)
print("  wrote boundary_heatmap_compare.png")


# ------------------------------------------------------------------
# Fig 13: forest plot (RL - DQN deltas with bootstrap 95% CIs)
# ------------------------------------------------------------------

print("[2/3] Forest plot")


def bootstrap_ci(diffs, n_boot=10000, alpha=0.05):
    diffs = np.asarray(diffs, dtype=float)
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    boot_means = diffs[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return diffs.mean(), lo, hi


metrics = [
    ("edge_step_pct",  r"Edge-cell occupancy (\%)",  -1),  # lower is better
    ("boundary_hits",  "Wall collisions",             -1),  # lower is better
    ("coverage_pct",   r"Sensor coverage (\%)",       +1),  # higher is better
]

fig, ax = plt.subplots(figsize=(8.5, 3.5))
y_positions = np.arange(len(metrics))[::-1]

for y, (key, label, direction) in zip(y_positions, metrics):
    dqn_vals = np.array([r[key] for r in dqn_runs], dtype=float)
    rel_vals = np.array([r[key] for r in rel_runs], dtype=float)
    # seed-matched deltas: RL - DQN
    seed_map = {r["seed"]: r[key] for r in dqn_runs}
    pairs = [(rel_runs[i][key] - seed_map.get(rel_runs[i]["seed"], np.nan))
             for i in range(len(rel_runs))]
    pairs = np.array([p for p in pairs if not np.isnan(p)], dtype=float)
    mean, lo, hi = bootstrap_ci(pairs)
    colour = "#1b7837" if direction * mean > 0 else "#b2182b"
    ax.errorbar([mean], [y], xerr=[[mean - lo], [hi - mean]],
                fmt="o", color=colour, markersize=8, lw=1.8, capsize=4)
    # Annotate
    ax.text(mean, y + 0.18,
            f"{mean:+.2f}  [{lo:+.2f}, {hi:+.2f}]",
            ha="center", va="bottom", fontsize=9, color=colour)

ax.axvline(0, color="gray", lw=0.8, ls="--")
ax.set_yticks(y_positions)
ax.set_yticklabels([label for _, label, _ in metrics])
ax.set_xlabel(r"Per-seed delta (Relational RL $-$ DQN), 95\% bootstrap CI")
ax.set_title("Head-to-head: Relational RL vs DQN",
             fontsize=11, fontweight="bold")
ax.grid(axis="x", lw=0.4, alpha=0.4)
plt.tight_layout()
fig.savefig(OUTDIR / "forest_plot_dqn_vs_relational.png", dpi=160)
plt.close(fig)
print("  wrote forest_plot_dqn_vs_relational.png")


# ------------------------------------------------------------------
# Fig 1: system-model overview
# ------------------------------------------------------------------

print("[3/3] System-model overview (unblocks compile)")

fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, 10); ax.set_ylim(0, 6); ax.axis("off")

# Ground and sky
ax.add_patch(Rectangle((0, 0),   10, 1.0, facecolor="#c8e6c9",
                       edgecolor="none"))
ax.add_patch(Rectangle((0, 1.0), 10, 5.0, facecolor="#e3f2fd",
                       edgecolor="none"))

# UAV
ax.add_patch(Circle((5.0, 4.7), 0.22, facecolor="#ff8a1a",
                     edgecolor="#b22222", lw=1.8, zorder=5))
ax.text(5.0, 5.1, "UAV gateway", ha="center", fontsize=10,
        fontweight="bold")
ax.text(5.0, 4.35, "(100 m altitude, 274 Wh)", ha="center", fontsize=8)

# Sensors + links
rng = np.random.default_rng(1)
sensor_x = rng.uniform(0.7, 9.3, size=10)
sensor_y = rng.uniform(0.1, 0.9, size=10)
for sx, sy in zip(sensor_x, sensor_y):
    ax.plot([sx, 5.0], [sy, 4.7], color="#7c7c7c", lw=0.6, alpha=0.5,
            ls=":", zorder=1)
    ax.add_patch(Circle((sx, sy), 0.12, facecolor="#1d4ed8",
                         edgecolor="white", lw=0.8, zorder=3))

# Annotations
ax.annotate("LoRa uplinks\n(SF7--SF12, EMA-ADR,\nCapture Effect)",
            xy=(2.4, 2.3), xytext=(0.3, 2.7), fontsize=9,
            arrowprops=dict(arrowstyle="->", color="#444"))
ax.annotate("IoT sensors\n(10--40, 1000-byte buffer)",
            xy=(8.5, 0.6), xytext=(7.4, 1.8), fontsize=9,
            arrowprops=dict(arrowstyle="->", color="#444"))

ax.text(5.0, 5.7, "UAV-LoRaWAN data collection: system model",
        ha="center", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(OUTDIR / "system_model.png", dpi=160,
            bbox_inches="tight")
plt.close(fig)
print("  wrote system_model.png")

print("\nAll figures written to", OUTDIR)
