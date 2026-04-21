"""
System Model — Redesigned publication figure
IEEE top-down style: drawn UAV silhouette, LoRaWAN sensor icons,
semi-transparent SF coverage zones, DARK2 palette, 600 DPI PDF+PNG.
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines   as mlines
from matplotlib.patches import FancyBboxPatch, Circle, Polygon
from pathlib import Path

# ── Path setup ────────────────────────────────────────────────────────────────
SCRIPT = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT))
import ieee_style
ieee_style.apply()

OUT = SCRIPT / "baseline_results"
OUT.mkdir(parents=True, exist_ok=True)

# ── Reproducibility ───────────────────────────────────────────────────────────
np.random.seed(42)

# ── Deployment parameters ─────────────────────────────────────────────────────
GRID = 500
N    = 20
UNIT = 10          # metres per grid unit

sensor_pos = np.random.uniform(30, GRID - 30, size=(N, 2))
UAV_POS    = np.array([120.0, 80.0])
CHARGE_POS = np.array([22.0,  22.0])

# Collection radii (grid units) from link budget
RADII = {"SF7": 86, "SF9": 154, "SF11": 244, "SF12": 365}

# ── IEEE DARK2 palette ────────────────────────────────────────────────────────
SF_COL = {
    "SF7":  "#1b9e77",   # teal
    "SF9":  "#e6ab02",   # gold
    "SF11": "#d95f02",   # orange
    "SF12": "#7570b3",   # purple
}
SF_RING = {
    "SF12": dict(ls="--",  lw=1.0, fill_alpha=0.055),
    "SF11": dict(ls="-.",  lw=1.1, fill_alpha=0.075),
    "SF9":  dict(ls=":",   lw=1.2, fill_alpha=0.10 ),
    "SF7":  dict(ls="-",   lw=1.4, fill_alpha=0.14 ),
}
UAV_BODY  = "#e7298a"    # DARK2 pink
UAV_ARM   = "#333333"
CHARG_COL = "#e6ab02"    # DARK2 gold
TRAJ_COL  = "#444444"
RET_COL   = "#d95f02"    # DARK2 orange


def assign_sf(pos):
    d = np.linalg.norm(pos - UAV_POS)
    for sf in ("SF7", "SF9", "SF11", "SF12"):
        if d <= RADII[sf]:
            return sf
    return "SF12"

sensor_sfs = [assign_sf(p) for p in sensor_pos]

# Illustrative UAV trajectory waypoints
TRAJ = np.array([
    CHARGE_POS,
    [58,  40],
    [120, 80],
    [195, 160],
    [275, 108],
    [348,  72],
])


# ═════════════════════════════════════════════════════════════════════════════
# Icon drawing helpers
# ═════════════════════════════════════════════════════════════════════════════

def draw_uav(ax, cx, cy, scale=20):
    """
    Top-down quadrotor silhouette.
    scale ≈ half-span in grid units (arm length).
    """
    arm_len = scale * 0.82
    arm_w   = scale * 0.13
    body_r  = scale * 0.40
    rotor_r = scale * 0.34

    # Four arms at 45°, 135°, 225°, 315°
    for deg in (45, 135, 225, 315):
        theta = np.radians(deg)
        tip   = np.array([cx + arm_len * np.cos(theta),
                          cy + arm_len * np.sin(theta)])
        ctr   = np.array([cx, cy])
        perp  = np.array([-np.sin(theta), np.cos(theta)]) * arm_w * 0.5
        # Arm quad
        pts = [ctr + perp, ctr - perp, tip - perp, tip + perp]
        ax.add_patch(Polygon(pts, closed=True,
                             fc=UAV_ARM, ec="#111111",
                             lw=0.5, zorder=8))
        # Rotor disc
        ax.add_patch(Circle(tip, rotor_r,
                            fc=UAV_BODY, ec="#111111",
                            lw=0.8, alpha=0.88, zorder=9))
        # Rotor cross blades
        for blade_deg in (0, 90):
            ba = np.radians(deg + blade_deg)
            rv = rotor_r * 0.80
            ax.plot([tip[0] - rv*np.cos(ba), tip[0] + rv*np.cos(ba)],
                    [tip[1] - rv*np.sin(ba), tip[1] + rv*np.sin(ba)],
                    color="#111111", lw=0.7, alpha=0.75, zorder=10)

    # Central octagonal body
    n = 8
    bx = [cx + body_r * np.cos(2*np.pi*k/n + np.pi/8) for k in range(n)]
    by = [cy + body_r * np.sin(2*np.pi*k/n + np.pi/8) for k in range(n)]
    ax.add_patch(Polygon(list(zip(bx, by)), closed=True,
                         fc=UAV_BODY, ec="#111111",
                         lw=1.0, zorder=11))
    # Camera lens dot
    ax.add_patch(Circle((cx, cy), body_r * 0.30,
                         fc="#111111", ec="white",
                         lw=0.5, zorder=12))


def draw_sensor(ax, cx, cy, color, bw=7.5, bh=5.5):
    """LoRaWAN IoT node: rounded rectangle body + antenna + ball tip."""
    # Body
    ax.add_patch(FancyBboxPatch(
        (cx - bw/2, cy - bh/2), bw, bh,
        boxstyle="round,pad=0.6",
        fc=color, ec="#111111", lw=0.65, zorder=6, alpha=0.92))
    # Antenna stem
    ant_top = cy + bh/2 + bh * 1.25
    ax.plot([cx, cx], [cy + bh/2, ant_top],
            color="#111111", lw=0.85, zorder=7,
            solid_capstyle="round")
    # Antenna ball
    ax.add_patch(Circle((cx, ant_top), bh * 0.17,
                         fc="#111111", zorder=7))


def draw_charging_station(ax, cx, cy, sz=15):
    """Charging station: rounded box + lightning bolt."""
    hw, hh = sz * 0.56, sz * 0.44
    # Box
    ax.add_patch(FancyBboxPatch(
        (cx - hw, cy - hh), 2*hw, 2*hh,
        boxstyle="round,pad=0.6",
        fc=CHARG_COL, ec="#111111", lw=0.85, zorder=6, alpha=0.95))
    # Lightning bolt (simplified zig-zag polygon)
    s = hh * 0.68
    bolt_pts = [
        ( 0.22*s,  1.00*s),
        (-0.12*s,  0.12*s),
        ( 0.14*s,  0.12*s),
        (-0.22*s, -1.00*s),
        ( 0.12*s, -0.12*s),
        (-0.14*s, -0.12*s),
    ]
    ax.add_patch(Polygon(
        [(cx + p[0], cy + p[1]) for p in bolt_pts], closed=True,
        fc="white", ec="#555555", lw=0.5, zorder=7, alpha=0.92))


# ═════════════════════════════════════════════════════════════════════════════
# Figure canvas
# ═════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7.8, 7.2))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")
ax.set_xlim(-20, GRID + 20)
ax.set_ylim(-20, GRID + 20)
ax.grid(False)

# ── Terrain background ────────────────────────────────────────────────────────
ax.add_patch(FancyBboxPatch(
    (0, 0), GRID, GRID,
    boxstyle="square,pad=0",
    fc="#EDF4ED",            # very light green — open field feel
    ec="#2a2a2a", lw=1.4, zorder=0))

# White grid lines (subtle)
for v in range(0, GRID + 1, 100):
    ax.axvline(v, color="white", lw=0.7, alpha=0.85, zorder=1)
    ax.axhline(v, color="white", lw=0.7, alpha=0.85, zorder=1)

# ── SF coverage zones (largest → smallest so inner rings sit on top) ─────────
for sf in ("SF12", "SF11", "SF9", "SF7"):
    r   = RADII[sf]
    col = SF_COL[sf]
    sty = SF_RING[sf]
    # Filled disc (faint)
    ax.add_patch(Circle(UAV_POS, r, fc=col, ec="none",
                        alpha=sty["fill_alpha"], zorder=2))
    # Crisp ring edge
    ax.add_patch(Circle(UAV_POS, r, fc="none", ec=col,
                        ls=sty["ls"], lw=sty["lw"],
                        alpha=0.88, zorder=3))
    # Label at ~18° from horizontal, right side of ring
    ang = np.radians(18)
    lx  = UAV_POS[0] + r * np.cos(ang)
    ly  = UAV_POS[1] + r * np.sin(ang)
    if 2 < lx < GRID - 2 and 2 < ly < GRID - 2:
        ax.text(lx + 5, ly + 1,
                f"{sf}  ({r * UNIT / 1000:.2f} km)",
                fontsize=6.8, color=col,
                fontweight="bold", va="center", zorder=8)

# ── UAV trajectory ────────────────────────────────────────────────────────────
ax.plot(TRAJ[:, 0], TRAJ[:, 1],
        color=TRAJ_COL, lw=1.5, ls="--",
        dashes=(7, 3.5), alpha=0.88, zorder=4)
for i in range(len(TRAJ) - 1):
    mid = (TRAJ[i] + TRAJ[i + 1]) / 2
    d   = TRAJ[i + 1] - TRAJ[i]
    d   = d / np.linalg.norm(d)
    ax.annotate("",
                xy=mid + d * 5,
                xytext=mid - d * 5,
                arrowprops=dict(arrowstyle="-|>", color=TRAJ_COL,
                                lw=0.9, mutation_scale=9),
                zorder=5)

# ── Return-to-base path (dotted, orange) ─────────────────────────────────────
ax.annotate("",
            xy=CHARGE_POS + np.array([9, 9]),
            xytext=UAV_POS  - np.array([7, 4]),
            arrowprops=dict(arrowstyle="-|>", color=RET_COL, lw=1.1,
                            linestyle="dotted", mutation_scale=9,
                            connectionstyle="arc3,rad=-0.22"),
            zorder=5)

# ── Sensor icons ─────────────────────────────────────────────────────────────
for pos, sf in zip(sensor_pos, sensor_sfs):
    draw_sensor(ax, pos[0], pos[1], SF_COL[sf])

# ── Charging station ─────────────────────────────────────────────────────────
draw_charging_station(ax, CHARGE_POS[0], CHARGE_POS[1])

# ── UAV silhouette (on top of everything) ────────────────────────────────────
draw_uav(ax, UAV_POS[0], UAV_POS[1], scale=20)

# ── Annotations ──────────────────────────────────────────────────────────────
ax.annotate(
    "UAV (DJI TB60)\n$h_r = 100\\,\\mathrm{m}$",
    xy=UAV_POS + np.array([0, 20]),
    xytext=(UAV_POS[0] + 95, UAV_POS[1] + 80),
    fontsize=7.5, color="#111111",
    arrowprops=dict(arrowstyle="->", color="#444444", lw=0.8,
                    connectionstyle="arc3,rad=-0.15"),
    zorder=13)

ax.annotate(
    "Charging station\n(relay dispatch)",
    xy=CHARGE_POS + np.array([8, 14]),
    xytext=(CHARGE_POS[0] + 10, CHARGE_POS[1] + 135),
    fontsize=7.5, color="#111111",
    arrowprops=dict(arrowstyle="->", color="#444444", lw=0.8,
                    connectionstyle="arc3,rad=0.20"),
    zorder=13)

ax.text(195, 14,
        "Return: battery $\\leq 20\\%$",
        fontsize=6.8, color=RET_COL,
        style="italic", zorder=13)

# ── Axes formatting ───────────────────────────────────────────────────────────
ax.set_xlabel("$x$-position (grid units,  1 unit $= 10$ m)", fontsize=9)
ax.set_ylabel("$y$-position (grid units,  1 unit $= 10$ m)", fontsize=9)
ax.tick_params(labelsize=8)
for sp in ax.spines.values():
    sp.set_visible(True)
    sp.set_color("#2a2a2a")
    sp.set_linewidth(0.9)

# Secondary km axis
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim()[0] * UNIT / 1000,
             ax.get_xlim()[1] * UNIT / 1000)
ax2.set_xlabel("$x$-position (km)", fontsize=8)
ax2.tick_params(labelsize=7.5)

ax.set_title(
    "System model: UAV LoRaWAN data collection\n"
    r"$500 \times 500$ grid,  $N = 20$ sensors,  $h_r = 100$ m",
    fontsize=9.5, fontweight="bold", pad=6)

# ── Legend ────────────────────────────────────────────────────────────────────
leg_handles = [
    mpatches.Patch(fc=SF_COL["SF7"],  ec="#111", lw=0.6,
                   label=f"SF7 sensor   ({RADII['SF7']  * UNIT/1000:.2f} km)"),
    mpatches.Patch(fc=SF_COL["SF9"],  ec="#111", lw=0.6,
                   label=f"SF9 sensor   ({RADII['SF9']  * UNIT/1000:.2f} km)"),
    mpatches.Patch(fc=SF_COL["SF11"], ec="#111", lw=0.6,
                   label=f"SF11 sensor  ({RADII['SF11'] * UNIT/1000:.2f} km)"),
    mpatches.Patch(fc=SF_COL["SF12"], ec="#111", lw=0.6,
                   label=f"SF12 sensor  ({RADII['SF12'] * UNIT/1000:.2f} km)"),
    mpatches.Patch(fc=UAV_BODY,       ec="#111", lw=0.6,
                   label="UAV (DJI TB60)"),
    mpatches.Patch(fc=CHARG_COL,      ec="#111", lw=0.6,
                   label="Charging / relay station"),
    mlines.Line2D([0], [0], color=TRAJ_COL, lw=1.4,
                  ls="--", dashes=(6, 3),
                  label="UAV trajectory"),
    mlines.Line2D([0], [0], color=RET_COL, lw=1.1, ls=":",
                  label="Return-to-base path"),
]
ax.legend(handles=leg_handles, loc="upper right",
          fontsize=7, framealpha=1.0,
          edgecolor="#CCCCCC", frameon=True)

# ── Save ──────────────────────────────────────────────────────────────────────
fig.tight_layout(rect=[0, 0, 1, 0.97])
stem = OUT / "system_model"
for fmt in ("pdf", "png"):
    fig.savefig(f"{stem}.{fmt}", dpi=600, bbox_inches="tight", format=fmt)
    print(f"  Saved: {stem}.{fmt}")
plt.close()
