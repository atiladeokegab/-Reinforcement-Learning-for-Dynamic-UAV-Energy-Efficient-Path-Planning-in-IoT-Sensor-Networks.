"""
trajectory_grid.py — UAV trajectory grid across grid sizes × sensor counts
===========================================================================

Sweeps over 6 conditions: {300, 400, 500} × {10, 20 sensors}.
For each condition runs DQN + MaxThroughputGreedyV2 over SEEDS and produces:

  trajectory_grid.pdf   — 6-panel figure, one subplot per condition.
                           Each subplot overlays UAV paths across seeds (faded)
                           plus sensor positions colour-coded by SR.
  summary.csv           — per-condition mean NDR, Jain's, data efficiency,
                          B/Wh for DQN and greedy.

Usage
-----
  uv run python trajectory_grid.py --model <path-to-model-dir>

  <path-to-model-dir> should contain dqn_final.zip (and optionally
  training_config.json).  If omitted, defaults to models/smoke_test_400/.

Examples
--------
  uv run python trajectory_grid.py --model models/smoke_test_400
  uv run python trajectory_grid.py --model models/dqn_v3_retrain

Author: ATILADE GABRIEL OKE
"""

import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from matplotlib.lines import Line2D
import gymnasium
from pathlib import Path
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor

script_dir = Path(__file__).resolve().parent
src_dir    = script_dir.parent.parent.parent
sys.path.insert(0, str(script_dir))
sys.path.insert(0, str(script_dir.parent))
sys.path.insert(0, str(src_dir))

import ieee_style
ieee_style.apply()

_PDF_OUTPUT = False   # overridden by --pdf flag in main()

from environment.uav_env import UAVEnvironment
from greedy_agents import MaxThroughputGreedyV2, LawnmowerAgent

# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="UAV trajectory grid evaluation")
    p.add_argument(
        "--model", default=None,
        help="Path to model directory containing dqn_final.zip "
             "(default: models/smoke_test_400 relative to repo root)",
    )
    p.add_argument(
        "--seeds", default="42,123,256,789,1337",
        help="Comma-separated seed list (default: 5 seeds)",
    )
    p.add_argument(
        "--output", default=None,
        help="Output directory (default: trajectory_grid_results/<model_name>)",
    )
    return p.parse_args()

# ── Config ────────────────────────────────────────────────────────────────────

GRID_SIZES   = [(300, 300), (400, 400), (500, 500)]
SENSOR_COUNTS = [10, 20]
CONDITIONS   = [(g, n) for g in GRID_SIZES for n in SENSOR_COUNTS]  # 6 total

MAX_STEPS    = 2100
MAX_BATT     = 274.0
MAX_SENSORS_LIMIT = 50

BASE_ENV_KW = {
    "max_steps":          MAX_STEPS,
    "path_loss_exponent": 3.8,
    "rssi_threshold":     -85.0,
    "sensor_duty_cycle":  10.0,
    "max_battery":        MAX_BATT,
    "render_mode":        None,
}

# ── Zero-padding wrapper (same as compare_v3_multiseed) ───────────────────────

class PaddedEnv(UAVEnvironment):
    def __init__(self, max_sensors_limit: int = MAX_SENSORS_LIMIT, **kwargs):
        self._max_limit = max_sensors_limit
        super().__init__(**kwargs)
        raw = self.observation_space.shape[0]
        self._fps = 0
        for uav_f in range(raw + 1):
            rem = raw - uav_f
            if rem > 0 and rem % self.num_sensors == 0:
                self._fps = rem // self.num_sensors
                break
        if self._fps == 0:
            raise ValueError(
                "Cannot infer features_per_sensor: obs={}, N={}".format(
                    raw, self.num_sensors
                )
            )
        padded = raw + (self._max_limit - self.num_sensors) * self._fps
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(padded,), dtype=np.float32
        )

    def _pad(self, obs):
        extra = np.zeros(
            (self._max_limit - self.num_sensors) * self._fps, dtype=np.float32
        )
        return np.concatenate([obs, extra]).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        return self._pad(obs), info

    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        return self._pad(obs), r, term, trunc, info


# ── Model loader ─────────────────────────────────────────────────────────────

def load_model_and_config(model_dir: Path):
    defaults = {"use_frame_stacking": True, "n_stack": 4,
                "max_sensors_limit": MAX_SENSORS_LIMIT}
    # Search for config in model_dir then parent (handles best_metric_model/ subdir)
    config_path = next(
        (p for p in (model_dir / "training_config.json",
                     model_dir.parent / "training_config.json")
         if p.exists()),
        None,
    )
    if config_path:
        with open(config_path) as f:
            cfg = {**defaults, **json.load(f)}
        print("  Config: {}".format(config_path))
    else:
        print("  [warn] training_config.json not found — using defaults")
        cfg = defaults

    # Prefer dqn_final (end-of-training) for generalisation eval.
    # best_metric_model is saved early and may only know Stage-0 grids.
    for candidate in ("dqn_final.zip",
                      "best_metric_model/best_metric_model.zip",
                      "best_metric_model.zip"):
        p = model_dir / candidate
        if p.exists():
            model_path = p
            break
    else:
        raise FileNotFoundError(
            "No model zip found in {}".format(model_dir)
        )

    print("  Loading model: {}".format(model_path))

    # Build a dummy env of the right obs shape to load the model
    dummy = PaddedEnv(
        grid_size=(500, 500), num_sensors=20,
        max_sensors_limit=cfg["max_sensors_limit"],
        **BASE_ENV_KW,
    )
    vec = DummyVecEnv([lambda: Monitor(dummy)])
    if cfg.get("use_frame_stacking", True):
        vec = VecFrameStack(vec, n_stack=cfg.get("n_stack", 4))

    model = DQN.load(str(model_path), env=vec, device="auto")
    print("  Model loaded on device: {}".format(
        next(model.policy.parameters()).device
    ))
    return model, cfg


# ── Episode runners ───────────────────────────────────────────────────────────

def _jains(rates):
    n  = len(rates)
    s1 = sum(rates)
    s2 = sum(x**2 for x in rates)
    return (s1**2) / (n * s2) if n > 0 and s2 > 0 else 1.0


def run_dqn_episode(model, cfg, grid_size, n_sensors, seed, deterministic=True):
    rng = np.random.default_rng(seed + 77777)
    W, H = float(grid_size[0]), float(grid_size[1])
    uav_start = np.array(
        [float(rng.uniform(0.05 * W, 0.95 * W)),
         float(rng.uniform(0.05 * H, 0.95 * H))],
        dtype=np.float32,
    )

    env = PaddedEnv(
        grid_size=grid_size, num_sensors=n_sensors,
        max_sensors_limit=cfg["max_sensors_limit"],
        uav_start_position=uav_start,
        **BASE_ENV_KW,
    )
    vec = DummyVecEnv([lambda: env])
    if cfg.get("use_frame_stacking", True):
        vec = VecFrameStack(vec, n_stack=cfg.get("n_stack", 4))

    obs = vec.reset()
    trajectory, done = [], False
    while not done:
        trajectory.append(env.uav.position.copy())
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, _, dones, _ = vec.step(action)
        done = dones[0]
    trajectory.append(env.uav.position.copy())

    sensor_positions = [s.position.copy() for s in env.sensors]
    rates = [
        s.total_data_transmitted / s.total_data_generated
        for s in env.sensors if s.total_data_generated > 0
    ]
    ndr = len(env.sensors_visited) / n_sensors * 100
    jain = _jains([r * 100 for r in rates])
    de   = (sum(s.total_data_transmitted for s in env.sensors) /
            max(sum(s.total_data_generated  for s in env.sensors), 1)) * 100
    batt_used = max(MAX_BATT - env.uav.battery, 1e-6)
    bpwh = sum(s.total_data_transmitted for s in env.sensors) / batt_used

    vec.close()
    return {
        "trajectory": np.array(trajectory),
        "sensor_positions": sensor_positions,
        "sensor_rates": rates,
        "ndr": ndr, "jains": jain, "data_eff": de, "bpwh": bpwh,
    }


def run_greedy_episode(grid_size, n_sensors, seed):
    rng = np.random.default_rng(seed + 77777)
    W, H = float(grid_size[0]), float(grid_size[1])
    uav_start = np.array(
        [float(rng.uniform(0.05 * W, 0.95 * W)),
         float(rng.uniform(0.05 * H, 0.95 * H))],
        dtype=np.float32,
    )

    env = UAVEnvironment(
        grid_size=grid_size, num_sensors=n_sensors,
        uav_start_position=uav_start,
        **BASE_ENV_KW,
    )
    obs, _ = env.reset(seed=seed)
    agent   = MaxThroughputGreedyV2(env)

    trajectory, done = [], False
    while not done:
        trajectory.append(env.uav.position.copy())
        action = agent.select_action(obs)
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
    trajectory.append(env.uav.position.copy())

    sensor_positions = [s.position.copy() for s in env.sensors]
    rates = [
        s.total_data_transmitted / s.total_data_generated
        for s in env.sensors if s.total_data_generated > 0
    ]
    ndr  = len(env.sensors_visited) / n_sensors * 100
    jain = _jains([r * 100 for r in rates])
    de   = (sum(s.total_data_transmitted for s in env.sensors) /
            max(sum(s.total_data_generated  for s in env.sensors), 1)) * 100
    batt_used = max(MAX_BATT - env.uav.battery, 1e-6)
    bpwh = sum(s.total_data_transmitted for s in env.sensors) / batt_used

    env.close()
    return {
        "trajectory": np.array(trajectory),
        "sensor_positions": sensor_positions,
        "sensor_rates": rates,
        "ndr": ndr, "jains": jain, "data_eff": de, "bpwh": bpwh,
    }


def run_lawnmower_episode(grid_size, n_sensors, seed):
    rng = np.random.default_rng(seed + 77777)
    W, H = float(grid_size[0]), float(grid_size[1])
    uav_start = np.array(
        [float(rng.uniform(0.05 * W, 0.95 * W)),
         float(rng.uniform(0.05 * H, 0.95 * H))],
        dtype=np.float32,
    )
    env = UAVEnvironment(
        grid_size=grid_size, num_sensors=n_sensors,
        uav_start_position=uav_start,
        **BASE_ENV_KW,
    )
    obs, _ = env.reset(seed=seed)
    agent = LawnmowerAgent(env, strip_width=50)
    agent.reset()

    trajectory, done = [], False
    while not done:
        trajectory.append(env.uav.position.copy())
        action = agent.select_action(obs)
        obs, _, term, trunc, _ = env.step(action)
        done = term or trunc
    trajectory.append(env.uav.position.copy())

    sensor_positions = [s.position.copy() for s in env.sensors]
    rates = [
        s.total_data_transmitted / s.total_data_generated
        for s in env.sensors if s.total_data_generated > 0
    ]
    ndr  = len(env.sensors_visited) / n_sensors * 100
    jain = _jains([r * 100 for r in rates])
    de   = (sum(s.total_data_transmitted for s in env.sensors) /
            max(sum(s.total_data_generated  for s in env.sensors), 1)) * 100
    batt_used = max(MAX_BATT - env.uav.battery, 1e-6)
    bpwh = sum(s.total_data_transmitted for s in env.sensors) / batt_used

    env.close()
    return {
        "trajectory": np.array(trajectory),
        "sensor_positions": sensor_positions,
        "sensor_rates": rates,
        "ndr": ndr, "jains": jain, "data_eff": de, "bpwh": bpwh,
    }


# ── Plotting (IEEE-compliant) ─────────────────────────────────────────────────

# Canonical agent styles pulled from ieee_style
_DQN_COLOR    = ieee_style.AGENT_COLORS["DQN Agent"]        # teal  #1b9e77
_GREEDY_COLOR = ieee_style.AGENT_COLORS["SF-Aware Greedy"]  # orange #d95f02
_LAWN_COLOR   = "#e6ab02"                                    # gold
_CR_CMAP      = cm.RdYlGn
_CR_NORM      = mcolors.Normalize(vmin=0.0, vmax=1.0)

# IEEE double-column text width = 7.16 in; half-column = 3.5 in
# 3 rows × 2 cols → 7 in wide, 9.5 in tall
_FIG_W = 7.16
_FIG_H = 9.8


def _plot_condition(ax, grid_size, n_sensors, dqn_runs, greedy_runs, lawn_runs=None):
    """
    Render one IEEE-styled subplot.

    Layout
    ------
    • Sensor scatter  — coloured by collection ratio (RdYlGn), black edge
    • DQN paths       — teal solid lines, fading opacity across seeds
    • Greedy paths    — orange dashed lines, fading opacity across seeds
    • Start markers   — filled triangles at UAV start position per seed
    • Stats box       — mean NDR and Jain's for both agents, top-left
    • No x/y grid     — trajectory subplots are spatial; y-grid clutters
    """
    W, H   = grid_size
    n_runs = max(len(dqn_runs), 1)

    # ── Sensor scatter (first seed's layout as representative) ────────────────
    ref_pos   = np.array(dqn_runs[0]["sensor_positions"])
    ref_rates = np.array(dqn_runs[0]["sensor_rates"]) if dqn_runs[0]["sensor_rates"] else np.zeros(n_sensors)
    s_colors  = [_CR_CMAP(_CR_NORM(r)) for r in ref_rates]
    ax.scatter(
        ref_pos[:, 0], ref_pos[:, 1],
        c=s_colors, s=45, zorder=6,
        edgecolors="black", linewidths=0.5,
    )

    # ── DQN paths ─────────────────────────────────────────────────────────────
    for i, run in enumerate(dqn_runs):
        traj  = run["trajectory"]
        alpha = 0.30 + 0.55 * (i / n_runs)   # [0.30, 0.85], always < 1
        ax.plot(traj[:, 0], traj[:, 1],
                color=_DQN_COLOR, lw=0.8, alpha=alpha,
                linestyle="-", zorder=3)
        # Start marker — upward triangle
        ax.plot(traj[0, 0], traj[0, 1],
                marker="^", color=_DQN_COLOR, ms=4,
                alpha=min(alpha + 0.15, 1.0), zorder=5, linestyle="none")

    # ── Greedy paths ──────────────────────────────────────────────────────────
    for i, run in enumerate(greedy_runs):
        traj  = run["trajectory"]
        alpha = 0.25 + 0.50 * (i / n_runs)
        ax.plot(traj[:, 0], traj[:, 1],
                color=_GREEDY_COLOR, lw=0.8, alpha=alpha,
                linestyle="--", zorder=2)
        ax.plot(traj[0, 0], traj[0, 1],
                marker="^", color=_GREEDY_COLOR, ms=4,
                alpha=min(alpha + 0.15, 1.0), zorder=4, linestyle="none")

    # ── Lawnmower paths (optional) ────────────────────────────────────────────
    if lawn_runs:
        for i, run in enumerate(lawn_runs):
            traj  = run["trajectory"]
            alpha = 0.25 + 0.50 * (i / n_runs)
            ax.plot(traj[:, 0], traj[:, 1],
                    color=_LAWN_COLOR, lw=0.8, alpha=alpha,
                    linestyle=(0, (5, 2)), zorder=2)
            ax.plot(traj[0, 0], traj[0, 1],
                    marker="^", color=_LAWN_COLOR, ms=4,
                    alpha=min(alpha + 0.15, 1.0), zorder=4, linestyle="none")

    # ── Stats annotation ──────────────────────────────────────────────────────
    mean_ndr_dqn = np.mean([r["ndr"]   for r in dqn_runs])
    mean_j_dqn   = np.mean([r["jains"] for r in dqn_runs])
    mean_ndr_g   = np.mean([r["ndr"]   for r in greedy_runs])
    mean_j_g     = np.mean([r["jains"] for r in greedy_runs])
    lawn_line = ""
    if lawn_runs:
        mean_ndr_l = np.mean([r["ndr"]   for r in lawn_runs])
        mean_j_l   = np.mean([r["jains"] for r in lawn_runs])
        lawn_line  = "\n" + r"$\bf{Lawn}$" + "  NDR={:.0f}%  J={:.3f}".format(mean_ndr_l, mean_j_l)
    stats = (
        r"$\bf{DQN}$"    + "  NDR={:.0f}%  J={:.3f}\n".format(mean_ndr_dqn, mean_j_dqn) +
        r"$\bf{Greedy}$" + "  NDR={:.0f}%  J={:.3f}".format(mean_ndr_g, mean_j_g) +
        lawn_line
    )
    ax.text(
        0.03, 0.97, stats,
        transform=ax.transAxes,
        fontsize=7, verticalalignment="top", family="serif",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                  edgecolor="#CCCCCC", linewidth=0.6, alpha=0.90),
    )

    # ── Axes formatting ───────────────────────────────────────────────────────
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.set_aspect("equal", adjustable="box")

    # Keep all 4 spines — spatial domain has a meaningful boundary
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
    ax.grid(False)   # no grid on spatial plots

    ax.set_title(
        "{}$\\times${} grid,  $N={}$ sensors".format(W, H, n_sensors),
        fontsize=9, fontweight="bold", pad=4,
    )
    ax.set_xlabel("$x$ (grid units)", fontsize=9)
    ax.set_ylabel("$y$ (grid units)", fontsize=9)
    ax.tick_params(labelsize=8, length=3, width=0.6)


def build_figure(all_results, seeds, output_stem: str):
    """
    Build the 3×2 IEEE-styled trajectory grid and save as PNG + PDF.

    Parameters
    ----------
    all_results : dict  {(grid_size, n_sensors): {"dqn": [...], "greedy": [...]}}
    seeds       : list of ints
    output_stem : path without extension (ieee_style.save writes .png and .pdf)
    """
    n_cols = len(SENSOR_COUNTS)  # 2
    n_rows = len(GRID_SIZES)     # 3

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(_FIG_W, _FIG_H),
        constrained_layout=True,
    )

    col_labels = ["$N = {}$ sensors".format(n) for n in SENSOR_COUNTS]
    for col, label in enumerate(col_labels):
        axes[0, col].set_title(label, fontsize=10, fontweight="bold", pad=6)

    for row, grid_size in enumerate(GRID_SIZES):
        for col, n_sensors in enumerate(SENSOR_COUNTS):
            ax  = axes[row, col]
            key = (grid_size, n_sensors)
            _plot_condition(
                ax, grid_size, n_sensors,
                all_results[key]["dqn"],
                all_results[key]["greedy"],
                all_results[key].get("lawn"),
            )
            # Overwrite per-condition title with grid label only (N in col header)
            ax.set_title(
                "{}$\\times${}".format(*grid_size),
                fontsize=9, fontweight="bold", pad=3,
            )

    # ── Shared legend (figure-level) ─────────────────────────────────────────
    legend_handles = [
        Line2D([0], [0], color=_DQN_COLOR,    lw=1.5, ls="-",          label="DQN Agent"),
        Line2D([0], [0], color=_GREEDY_COLOR,  lw=1.5, ls="--",         label="SF-Aware Greedy"),
        Line2D([0], [0], color=_LAWN_COLOR,    lw=1.5, ls=(0, (5, 2)),  label="Lawnmower"),
        Line2D([0], [0], marker="^", color="grey", lw=0, ms=6,
               label="UAV start position", alpha=0.7),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center", ncol=4,
        fontsize=9, framealpha=0.95,
        edgecolor="#CCCCCC",
        bbox_to_anchor=(0.5, -0.01),
    )

    # ── Shared sensor CR colorbar ─────────────────────────────────────────────
    sm = cm.ScalarMappable(cmap=_CR_CMAP, norm=_CR_NORM)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, shrink=0.55, pad=0.03, aspect=25)
    cbar.set_label("Sensor collection ratio (CR)", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    formats = ["png"]
    if _PDF_OUTPUT:
        try:
            ieee_style.save(fig, output_stem, formats=("pdf",))
        except PermissionError as e:
            print("  [warn] Could not write PDF (file open or locked): {}".format(e))
    ieee_style.save(fig, output_stem, formats=("png",))
    plt.close(fig)


# ── Summary CSV ───────────────────────────────────────────────────────────────

def build_summary(all_results, output_path):
    rows = []
    for (grid_size, n_sensors), data in all_results.items():
        for agent_key, runs in (("DQN", data["dqn"]), ("SmartGreedy", data["greedy"]), ("Lawnmower", data.get("lawn", []))):
            rows.append({
                "grid":        "{}x{}".format(*grid_size),
                "n_sensors":   n_sensors,
                "agent":       agent_key,
                "mean_ndr":    round(np.mean([r["ndr"]      for r in runs]), 2),
                "std_ndr":     round(np.std ([r["ndr"]      for r in runs]), 2),
                "mean_jains":  round(np.mean([r["jains"]    for r in runs]), 4),
                "std_jains":   round(np.std ([r["jains"]    for r in runs]), 4),
                "mean_de":     round(np.mean([r["data_eff"] for r in runs]), 2),
                "mean_bpwh":   round(np.mean([r["bpwh"]     for r in runs]), 1),
                "n_seeds":     len(runs),
            })
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print("  Saved: {}".format(output_path))
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args  = parse_args()
    seeds = [int(s) for s in args.seeds.split(",")]

    # Resolve model directory — search in priority order:
    #   1. absolute path as given
    #   2. relative to src/agents/dqn/  (where models/ lives)
    #   3. relative to cwd
    if args.model:
        p = Path(args.model)
        candidates = [
            p,                                  # absolute or already correct
            script_dir.parent / p,              # src/agents/dqn/<model>
            Path.cwd() / p,                     # cwd/<model>
        ]
        model_dir = next((c for c in candidates if c.exists()), None)
        if model_dir is None:
            print("ERROR: model directory not found. Tried:")
            for c in candidates:
                print("  {}".format(c))
            sys.exit(1)
    else:
        model_dir = script_dir.parent / "models" / "smoke_test_400"

    if not model_dir.exists():
        print("ERROR: model directory not found: {}".format(model_dir))
        sys.exit(1)

    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = script_dir / "trajectory_grid_results" / model_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("TRAJECTORY GRID — {} conditions × {} seeds".format(
        len(CONDITIONS), len(seeds)))
    print("Model dir : {}".format(model_dir))
    print("Output dir: {}".format(output_dir))
    print("Conditions:", [(str(g[0])+"x"+str(g[1]), n) for g, n in CONDITIONS])
    print("Seeds     :", seeds)
    print("=" * 65)

    deterministic = not args.stochastic
    global _PDF_OUTPUT
    _PDF_OUTPUT = args.pdf
    model, cfg = load_model_and_config(model_dir)

    all_results = {}
    zero_ndr_count = 0
    for grid_size, n_sensors in CONDITIONS:
        key = (grid_size, n_sensors)
        print("\n[{}×{}, N={}]".format(grid_size[0], grid_size[1], n_sensors))
        dqn_runs, greedy_runs, lawn_runs = [], [], []

        for seed in seeds:
            print("  seed {}".format(seed), end="  ", flush=True)

            dqn_ep = run_dqn_episode(
                model, cfg, grid_size, n_sensors, seed, deterministic=deterministic
            )
            if dqn_ep["ndr"] == 0:
                zero_ndr_count += 1
            print("DQN  NDR={:.0f}%  J={:.3f}".format(
                dqn_ep["ndr"], dqn_ep["jains"]), end="   ", flush=True)

            g_ep = run_greedy_episode(grid_size, n_sensors, seed)
            print("Greedy  NDR={:.0f}%  J={:.3f}".format(
                g_ep["ndr"], g_ep["jains"]), end="   ", flush=True)

            l_ep = run_lawnmower_episode(grid_size, n_sensors, seed)
            print("Lawn  NDR={:.0f}%  J={:.3f}".format(
                l_ep["ndr"], l_ep["jains"]))

            dqn_runs.append(dqn_ep)
            greedy_runs.append(g_ep)
            lawn_runs.append(l_ep)

        all_results[key] = {"dqn": dqn_runs, "greedy": greedy_runs, "lawn": lawn_runs}

    total_runs = len(CONDITIONS) * len(seeds)
    if zero_ndr_count == total_runs:
        print(
            "\n[WARNING] DQN NDR=0% on every run. "
            "This indicates a degenerate deterministic policy — the model "
            "hovered in place rather than navigating. Likely cause: "
            "insufficient training (smoke test only). "
            "Re-run with --stochastic to verify env correctness, "
            "or evaluate the full 3M-step retrain model."
        )
    elif zero_ndr_count > total_runs // 2:
        print(
            "\n[WARNING] DQN NDR=0% on {}/{} runs. "
            "Model may not have generalised to these grid sizes.".format(
                zero_ndr_count, total_runs
            )
        )

    print("\nBuilding figure ...")
    build_figure(all_results, seeds, str(output_dir / "trajectory_grid"))

    print("Building summary CSV ...")
    df = build_summary(all_results, output_dir / "summary.csv")
    print("\n" + df.to_string(index=False))


def parse_args():
    p = argparse.ArgumentParser(description="UAV trajectory grid evaluation")
    p.add_argument(
        "--model", default=None,
        help="Model directory containing dqn_final.zip "
             "(default: models/smoke_test_400 relative to repo root)",
    )
    p.add_argument(
        "--seeds", default="42,123,256,789,1337",
        help="Comma-separated seed list",
    )
    p.add_argument(
        "--output", default=None,
        help="Output directory override",
    )
    p.add_argument(
        "--stochastic", action="store_true",
        help="Use stochastic policy (deterministic=False) — useful for "
             "diagnosing a degenerate deterministic policy on short-trained models",
    )
    p.add_argument(
        "--pdf", action="store_true",
        help="Also save a PDF alongside the PNG (close any open PDF viewer first)",
    )
    return p.parse_args()


if __name__ == "__main__":
    main()
