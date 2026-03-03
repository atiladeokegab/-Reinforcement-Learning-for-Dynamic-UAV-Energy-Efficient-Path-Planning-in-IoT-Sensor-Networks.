"""
Multi-Condition Fairness Sweep
==============================
Extends compare_agents.py to run all three agents across every combination of:
  - Grid size:     100x100 / 300x300 / 500x500 / 1000x1000
  - Sensor count:  10 / 20 / 30 / 40
  - Sensor layout: 3 independent seeds per condition

For each condition it captures Jain's fairness index and the full spatial
collection-rate profile, then produces four summary figures:

  fig_A  jains_summary_matrix.png
         3-panel heatmap (one per agent): rows=grid, cols=sensors, value=mean Jain's
         Tells you at a glance which conditions each agent handles well/badly.

  fig_B  jains_advantage_gap.png
         DQN minus best-greedy Jain's at every (grid, sensors) condition.
         Positive = DQN is fairer.  Shows where the learned policy wins.

  fig_C  jains_seed_variance.png
         For a fixed grid×sensors, box/violin of Jain's over 3 seeds.
         Shows how sensitive fairness is to the sensor layout.

  fig_D  spatial_fairness_grid_{gs}.png   (one per grid size)
         2D scatter of per-sensor collection rates for every sensor-count,
         one seed each (seed 42), all three agents side-by-side.
         The figure you already have, but systematically across all conditions.

Physics reference (grid unit = 10 m, UAV alt = 100 m, from iot_sensors.py):
  100x100  ->  SF7  dominant  (683 B/s,  max corner ~1.4 km)
  300x300  ->  SF7/SF9 mix    (220-683 B/s, ~4.2 km)
  500x500  ->  SF9/SF11       (55-220 B/s,  ~7.1 km)  <- training grid
  1000x1000 -> SF11/SF12      (31-55 B/s,  ~14 km, near link budget)

Author: ATILADE GABRIEL OKE
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import gymnasium
import json
from pathlib import Path
import time

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

# ==================== PATH SETUP ====================
script_dir         = Path(__file__).resolve().parent
src_dir            = script_dir.parent.parent.parent
script_dir_results = script_dir.parent

sys.path.insert(0, str(src_dir))

from environment.uav_env import UAVEnvironment
from greedy_agents import MaxThroughputGreedyV2, NearestSensorGreedy

# ==================== SWEEP CONFIGURATION ====================

SWEEP_SEEDS         = [42, 123, 256]           # 3 layouts per condition
SWEEP_SENSOR_COUNTS = [10, 20, 30, 40]
SWEEP_GRID_SIZES    = [(100, 100), (300, 300), (500, 500), (1000, 1000)]

# Physics annotation from iot_sensors.py
GRID_PHYSICS = {
    (100,  100): {"sf": "SF7",       "color": "#1B5E20", "label": "100x100\n(SF7)"},
    (300,  300): {"sf": "SF7/SF9",   "color": "#F9A825", "label": "300x300\n(SF7/SF9)"},
    (500,  500): {"sf": "SF9/SF11",  "color": "#E65100", "label": "500x500\n(SF9/SF11)"},
    (1000, 1000):{"sf": "SF11/SF12", "color": "#B71C1C", "label": "1000x1000\n(SF11/SF12)"},
}

BASE_CONFIG = {
    "max_steps":          2100,
    "path_loss_exponent": 3.8,
    "rssi_threshold":     -120.0,
    "sensor_duty_cycle":  10.0,
    "max_battery":        274.0,
}
EVAL_MAX_BATTERY = 274.0

# Paths — adjust if your layout differs
DQN_MODEL_PATH     = script_dir_results / "models" / "dqn_full_observability" / "dqn_final.zip"
DQN_CONFIG_PATH    = script_dir_results / "models" / "dqn_full_observability" / "training_config.json"
VEC_NORMALIZE_PATH = script_dir_results / "models" / "dqn_full_observability" / "vec_normalize.pkl"

OUTPUT_DIR = script_dir / "sweep_fairness_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AGENT_STYLES = {
    "DQN":     {"color": "#1565C0", "marker": "o", "label": "DQN Agent (Proposed)"},
    "SFGreedy":{"color": "#C62828", "marker": "s", "label": "SF-Aware Greedy V2"},
    "NNGreedy":{"color": "#555555", "marker": "^", "label": "Nearest Sensor Greedy"},
}

# ==================== ENVIRONMENT WRAPPERS ====================


class PaddedEnv(UAVEnvironment):
    """Zero-padding wrapper for DQN evaluation on any num_sensors <= max_sensors_limit."""
    def __init__(self, max_sensors_limit=50, **kwargs):
        self.max_sensors_limit = max_sensors_limit
        super().__init__(**kwargs)
        raw = self.observation_space.shape[0]
        self._fps = 0
        for uav_f in range(raw + 1):
            rem = raw - uav_f
            if rem > 0 and rem % self.num_sensors == 0:
                self._fps = rem // self.num_sensors
                break
        if self._fps == 0:
            raise ValueError("Cannot detect features_per_sensor from obs size {}".format(raw))
        padded = raw + (max_sensors_limit - self.num_sensors) * self._fps
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(padded,), dtype=np.float32
        )
        self._sensor_snapshot = None

    def _pad(self, obs):
        pad = np.zeros((self.max_sensors_limit - self.num_sensors) * self._fps, dtype=np.float32)
        return np.concatenate([obs, pad]).astype(np.float32)

    def reset(self, **kwargs):
        if hasattr(self, "sensors") and self.current_step > 0:
            self._sensor_snapshot = [
                {
                    "sensor_id":  int(s.sensor_id),
                    "position":   [float(x) for x in s.position],
                    "generated":  float(s.total_data_generated),
                    "transmitted":float(s.total_data_transmitted),
                }
                for s in self.sensors
            ]
        obs, info = super().reset(**kwargs)
        return self._pad(obs), info

    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        return self._pad(obs), r, term, trunc, info


# ==================== JAIN'S INDEX ====================

def jains_index(rates):
    n  = len(rates)
    s1 = sum(rates)
    s2 = sum(x**2 for x in rates)
    return (s1**2) / (n * s2) if n > 0 and s2 > 0 else 1.0


def collection_rates_from_env(env):
    rates = []
    for s in env.sensors:
        gen = float(s.total_data_generated)
        tx  = float(s.total_data_transmitted)
        rates.append((tx / gen * 100) if gen > 0 else 0.0)
    return rates


def collection_rates_from_snapshot(snapshot):
    rates = []
    for s in snapshot:
        gen = s["generated"]
        tx  = s["transmitted"]
        rates.append((tx / gen * 100) if gen > 0 else 0.0)
    return rates


def sensor_positions_from_env(env):
    return [(float(s.position[0]), float(s.position[1])) for s in env.sensors]


def sensor_positions_from_snapshot(snapshot):
    return [(s["position"][0], s["position"][1]) for s in snapshot]


# ==================== HELPERS ====================

def load_training_config(path):
    defaults = {"use_frame_stacking": True, "n_stack": 4, "max_sensors_limit": 50}
    try:
        with open(path) as f:
            return {**defaults, **json.load(f)}
    except FileNotFoundError:
        return defaults


def _unwrap(vec):
    inner = vec
    while hasattr(inner, "venv"):
        inner = inner.venv
    env = inner.envs[0]
    while hasattr(env, "env"):
        env = env.env
    return env


def _env_kwargs(grid_size, num_sensors):
    return {
        "grid_size":          grid_size,
        "num_sensors":        num_sensors,
        "max_steps":          BASE_CONFIG["max_steps"],
        "path_loss_exponent": BASE_CONFIG["path_loss_exponent"],
        "rssi_threshold":     BASE_CONFIG["rssi_threshold"],
        "sensor_duty_cycle":  BASE_CONFIG["sensor_duty_cycle"],
        "max_battery":        BASE_CONFIG["max_battery"],
        "render_mode":        None,
    }


def get_canonical_positions(seed, grid_size, num_sensors):
    env = UAVEnvironment(**_env_kwargs(grid_size, num_sensors))
    env.reset(seed=seed)
    pos = [(float(s.position[0]), float(s.position[1])) for s in env.sensors]
    env.close()
    return pos


# ==================== EPISODE RUNNERS ====================

def run_greedy(agent_class, fixed_pos, seed, grid_size, num_sensors):
    """Run one greedy episode on a fixed sensor layout. Returns (jains, rates, positions)."""
    kw  = _env_kwargs(grid_size, num_sensors)
    env = UAVEnvironment(**kw)
    obs, _ = env.reset(seed=seed)
    # Override positions
    for sensor, pos in zip(env.sensors, fixed_pos):
        sensor.position = np.array(pos, dtype=np.float32)
    agent = agent_class(env)
    done = False
    while not done:
        action = agent.select_action(obs)
        obs, _, done, trunc, _ = env.step(action)
        done = done or trunc
    rates = collection_rates_from_env(env)
    positions = sensor_positions_from_env(env)
    j = jains_index(rates)
    env.close()
    return j, rates, positions


def run_dqn(model, training_config, fixed_pos, seed, grid_size, num_sensors):
    """Run one DQN episode on a fixed sensor layout. Returns (jains, rates, positions)."""
    kw  = _env_kwargs(grid_size, num_sensors)
    kw["max_sensors_limit"] = training_config.get("max_sensors_limit", 50)

    fp = fixed_pos
    def _make():
        e = PaddedEnv(**kw)
        return e

    vec = DummyVecEnv([_make])
    if training_config.get("use_frame_stacking", True):
        vec = VecFrameStack(vec, n_stack=training_config.get("n_stack", 4))

    if VEC_NORMALIZE_PATH.exists():
        try:
            vec = VecNormalize.load(str(VEC_NORMALIZE_PATH), vec)
            vec.training   = False
            vec.norm_reward = False
        except AssertionError:
            pass  # old pkl, skip normalisation

    base = _unwrap(vec)
    # Override positions after reset
    obs = vec.reset()
    for sensor, pos in zip(base.sensors, fixed_pos):
        sensor.position = np.array(pos, dtype=np.float32)

    last_snapshot = None
    last_positions = None
    step = 0
    while True:
        action, _ = model.predict(obs, deterministic=True)
        av = int(action[0]) if hasattr(action, "__len__") else int(action)

        pre_snap = [
            {"sensor_id": int(s.sensor_id),
             "position":  [float(x) for x in s.position],
             "generated":  float(s.total_data_generated),
             "transmitted":float(s.total_data_transmitted)}
            for s in base.sensors
        ]
        pre_pos = [(float(s.position[0]), float(s.position[1])) for s in base.sensors]

        obs, _, dones, _ = vec.step([av])
        step += 1
        if bool(dones[0]):
            last_snapshot = pre_snap
            last_positions = pre_pos
            break

    vec.close()

    if last_snapshot:
        rates = collection_rates_from_snapshot(last_snapshot)
        positions = last_positions
    else:
        rates = [0.0] * num_sensors
        positions = fixed_pos

    return jains_index(rates), rates, positions


# ==================== MAIN SWEEP ====================

def run_sweep(dqn_model, training_config):
    """
    Run all conditions. Returns:
        results[gs][n][seed] = {
            "DQN":      (jains, rates, positions),
            "SFGreedy": (jains, rates, positions),
            "NNGreedy": (jains, rates, positions),
        }
    """
    total = len(SWEEP_GRID_SIZES) * len(SWEEP_SENSOR_COUNTS) * len(SWEEP_SEEDS)
    agents_per = 3 if dqn_model else 2
    run_idx = 0

    results = {}
    for gs in SWEEP_GRID_SIZES:
        results[gs] = {}
        for n in SWEEP_SENSOR_COUNTS:
            results[gs][n] = {}
            for seed in SWEEP_SEEDS:
                fixed_pos = get_canonical_positions(seed, gs, n)
                cond = {}

                if dqn_model is not None:
                    run_idx += 1
                    t0 = time.time()
                    print("  [{}/{}] DQN  gs={}x{}  n={}  seed={}".format(
                        run_idx, total * agents_per, gs[0], gs[1], n, seed))
                    try:
                        j, r, p = run_dqn(dqn_model, training_config, fixed_pos, seed, gs, n)
                    except Exception as e:
                        print("    DQN failed: {}  -- using zeros".format(e))
                        j, r, p = 0.0, [0.0]*n, fixed_pos
                    cond["DQN"] = (j, r, p)
                    print("    J={:.4f}  ({:.1f}s)".format(j, time.time()-t0))

                for agent_key, agent_class in [("SFGreedy", MaxThroughputGreedyV2),
                                               ("NNGreedy", NearestSensorGreedy)]:
                    run_idx += 1
                    t0 = time.time()
                    print("  [{}/{}] {}  gs={}x{}  n={}  seed={}".format(
                        run_idx, total * agents_per,
                        agent_key, gs[0], gs[1], n, seed))
                    j, r, p = run_greedy(agent_class, fixed_pos, seed, gs, n)
                    cond[agent_key] = (j, r, p)
                    print("    J={:.4f}  ({:.1f}s)".format(j, time.time()-t0))

                results[gs][n][seed] = cond

    return results


# ==================== FIGURE A: JAIN'S SUMMARY HEATMAP ====================

def plot_jains_summary_matrix(results, has_dqn):
    """
    3-panel heatmap.  Rows = grid sizes, Cols = sensor counts.
    Each cell = mean Jain's index across seeds.
    """
    sns.set_theme(style="white", font_scale=1.1)
    plt.rcParams["font.family"] = "serif"

    agents  = (["DQN", "SFGreedy", "NNGreedy"] if has_dqn else ["SFGreedy", "NNGreedy"])
    n_agents = len(agents)
    fig, axes = plt.subplots(1, n_agents, figsize=(6 * n_agents, 5))
    if n_agents == 1:
        axes = [axes]

    fig.suptitle(
        "Mean Jain's Fairness Index  —  Grid Size x Sensor Count\n"
        "(Mean over {} seeds;  SF regime from iot_sensors.py)".format(len(SWEEP_SEEDS)),
        fontsize=14, fontweight="bold"
    )

    row_labels = [GRID_PHYSICS[gs]["label"] for gs in SWEEP_GRID_SIZES]
    col_labels = [str(n) for n in SWEEP_SENSOR_COUNTS]

    for ax, agent_key in zip(axes, agents):
        matrix = np.zeros((len(SWEEP_GRID_SIZES), len(SWEEP_SENSOR_COUNTS)))
        for i, gs in enumerate(SWEEP_GRID_SIZES):
            for j, n in enumerate(SWEEP_SENSOR_COUNTS):
                jvals = [results[gs][n][s][agent_key][0]
                         for s in SWEEP_SEEDS if agent_key in results[gs][n][s]]
                matrix[i, j] = np.mean(jvals) if jvals else np.nan

        # Annotate with actual values
        annot = np.round(matrix, 3).astype(str)

        sns.heatmap(
            matrix, ax=ax,
            xticklabels=col_labels,
            yticklabels=row_labels,
            annot=annot, fmt="s",
            cmap="RdYlGn",
            vmin=0.5, vmax=1.0,
            linewidths=0.8, linecolor="white",
            cbar_kws={"label": "Jain's Index", "shrink": 0.8},
            annot_kws={"size": 10, "weight": "bold"},
        )
        ax.set_title(AGENT_STYLES[agent_key]["label"], fontsize=12,
                     fontweight="bold", color=AGENT_STYLES[agent_key]["color"], pad=10)
        ax.set_xlabel("Number of Sensors", fontsize=11, fontweight="bold")
        if ax == axes[0]:
            ax.set_ylabel("Grid Size  (dominant SF)", fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", rotation=0)
        ax.tick_params(axis="y", rotation=0)

        # Shade the training condition (500x500, 20 sensors)
        try:
            train_row = SWEEP_GRID_SIZES.index((500, 500))
            train_col = SWEEP_SENSOR_COUNTS.index(20)
            ax.add_patch(plt.Rectangle(
                (train_col, train_row), 1, 1,
                fill=False, edgecolor="blue", linewidth=3, zorder=5
            ))
            if ax == axes[0]:
                ax.text(train_col + 0.5, train_row - 0.15, "train",
                        ha="center", fontsize=7, color="blue", style="italic")
        except ValueError:
            pass

    plt.tight_layout()
    out = OUTPUT_DIR / "figA_jains_summary_matrix.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print("Saved: {}".format(out.name))
    plt.close()


# ==================== FIGURE B: DQN ADVANTAGE GAP ====================

def plot_jains_advantage_gap(results):
    """
    Heatmap: DQN Jain's - best_greedy Jain's at every (grid, sensors) condition.
    Green = DQN fairer, Red = greedy fairer.
    """
    sns.set_theme(style="white", font_scale=1.1)
    plt.rcParams["font.family"] = "serif"

    matrix = np.zeros((len(SWEEP_GRID_SIZES), len(SWEEP_SENSOR_COUNTS)))
    std_matrix = np.zeros_like(matrix)

    for i, gs in enumerate(SWEEP_GRID_SIZES):
        for j, n in enumerate(SWEEP_SENSOR_COUNTS):
            gaps = []
            for s in SWEEP_SEEDS:
                cond = results[gs][n][s]
                if "DQN" not in cond:
                    continue
                dqn_j  = cond["DQN"][0]
                best_g = max(cond["SFGreedy"][0], cond["NNGreedy"][0])
                gaps.append(dqn_j - best_g)
            matrix[i, j]     = np.mean(gaps) if gaps else 0.0
            std_matrix[i, j] = np.std(gaps)  if gaps else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "DQN Fairness Advantage  (DQN Jain's minus Best Greedy Jain's)\n"
        "Green = DQN fairer  |  Red = Greedy fairer  |  Blue box = training condition",
        fontsize=13, fontweight="bold"
    )

    row_labels = [GRID_PHYSICS[gs]["label"] for gs in SWEEP_GRID_SIZES]
    col_labels = [str(n) for n in SWEEP_SENSOR_COUNTS]
    lim = max(abs(matrix).max(), 0.05)

    for ax, (data, title) in zip(axes, [
        (matrix,     "Mean Gap (DQN - Best Greedy)"),
        (std_matrix, "Std Dev of Gap (layout sensitivity)"),
    ]):
        sns.heatmap(
            data, ax=ax,
            xticklabels=col_labels,
            yticklabels=row_labels,
            annot=np.round(data, 3), fmt=".3f",
            cmap="RdYlGn" if "Mean" in title else "YlOrBr",
            center=0 if "Mean" in title else None,
            vmin=-lim if "Mean" in title else 0,
            vmax= lim if "Mean" in title else None,
            linewidths=0.8, linecolor="white",
            cbar_kws={"label": "Jain's Index Difference", "shrink": 0.8},
            annot_kws={"size": 10, "weight": "bold"},
        )
        ax.set_title(title, fontsize=11, fontweight="bold", pad=10)
        ax.set_xlabel("Number of Sensors", fontsize=11, fontweight="bold")
        ax.set_ylabel("Grid Size  (dominant SF)", fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", rotation=0); ax.tick_params(axis="y", rotation=0)

        try:
            tr = SWEEP_GRID_SIZES.index((500, 500))
            tc = SWEEP_SENSOR_COUNTS.index(20)
            ax.add_patch(plt.Rectangle((tc, tr), 1, 1,
                fill=False, edgecolor="blue", linewidth=3, zorder=5))
        except ValueError:
            pass

    plt.tight_layout()
    out = OUTPUT_DIR / "figB_jains_advantage_gap.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print("Saved: {}".format(out.name))
    plt.close()


# ==================== FIGURE C: SEED VARIANCE ====================

def plot_jains_seed_variance(results, has_dqn):
    """
    For every (grid_size, sensor_count) condition, show a grouped box plot
    of Jain's index across seeds for all three agents.
    """
    sns.set_theme(style="whitegrid", font_scale=1.05)
    plt.rcParams["font.family"] = "serif"

    n_gs = len(SWEEP_GRID_SIZES)
    n_nc = len(SWEEP_SENSOR_COUNTS)

    fig, axes = plt.subplots(n_gs, n_nc, figsize=(4 * n_nc, 3.5 * n_gs), sharey=True)
    fig.suptitle(
        "Jain's Fairness Index — Layout Sensitivity  ({} seeds per condition)\n"
        "Box = distribution over sensor layouts; width shows variance".format(len(SWEEP_SEEDS)),
        fontsize=14, fontweight="bold"
    )

    agents  = (["DQN", "SFGreedy", "NNGreedy"] if has_dqn else ["SFGreedy", "NNGreedy"])

    for i, gs in enumerate(SWEEP_GRID_SIZES):
        for j, n in enumerate(SWEEP_SENSOR_COUNTS):
            ax = axes[i][j] if n_gs > 1 else axes[j]
            data_dict = {ak: [] for ak in agents}
            for seed in SWEEP_SEEDS:
                cond = results[gs][n][seed]
                for ak in agents:
                    if ak in cond:
                        data_dict[ak].append(cond[ak][0])

            x_pos  = np.arange(len(agents))
            for xi, ak in enumerate(agents):
                vals  = data_dict[ak]
                color = AGENT_STYLES[ak]["color"]
                if len(vals) > 1:
                    bp = ax.boxplot(vals, positions=[xi], widths=0.5, patch_artist=True,
                                    medianprops=dict(color="black", linewidth=2),
                                    whiskerprops=dict(linewidth=1.5),
                                    capprops=dict(linewidth=1.5))
                    for patch in bp["boxes"]:
                        patch.set_facecolor(color); patch.set_alpha(0.7)
                elif len(vals) == 1:
                    ax.scatter([xi], vals, color=color, s=80, zorder=4)
                # overlay seed points
                jitter = np.random.default_rng(xi).uniform(-0.1, 0.1, len(vals))
                ax.scatter(np.full(len(vals), xi) + jitter, vals,
                           color=color, s=30, zorder=5, edgecolors="white",
                           linewidths=0.5, alpha=0.9)

            ax.set_xticks(x_pos)
            ax.set_xticklabels([AGENT_STYLES[ak]["label"].split(" ")[0] for ak in agents],
                               fontsize=7, rotation=15)
            ax.set_ylim(0, 1.05)
            ax.axhline(1.0, color="green", linestyle=":", linewidth=1.0, alpha=0.5)
            ax.grid(axis="y", alpha=0.4, linestyle="--")

            # Row label (grid size + SF regime)
            if j == 0:
                phys = GRID_PHYSICS[gs]
                ax.set_ylabel(phys["label"] + "\n" + phys["sf"],
                              fontsize=8, fontweight="bold", color=phys["color"])
            # Column label (sensor count)
            if i == 0:
                ax.set_title("{} sensors".format(n), fontsize=9, fontweight="bold")
            # Mark training condition
            if gs == (500, 500) and n == 20:
                for spine in ax.spines.values():
                    spine.set_edgecolor("blue"); spine.set_linewidth(2.5)

    plt.tight_layout()
    out = OUTPUT_DIR / "figC_jains_seed_variance.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print("Saved: {}".format(out.name))
    plt.close()


# ==================== FIGURE D: SPATIAL FAIRNESS GRID ====================

def plot_spatial_fairness_grid(results, gs, has_dqn, seed=42):
    """
    For one fixed grid size, plot spatial collection-rate heatmaps for every
    sensor count (cols) x agent (rows).  Uses seed 42 for visual consistency.
    """
    agents  = (["DQN", "SFGreedy", "NNGreedy"] if has_dqn else ["SFGreedy", "NNGreedy"])
    n_agents = len(agents)
    n_nc     = len(SWEEP_SENSOR_COUNTS)
    phys     = GRID_PHYSICS[gs]

    sns.set_theme(style="white", font_scale=1.0)
    plt.rcParams["font.family"] = "serif"

    fig, axes = plt.subplots(n_agents, n_nc, figsize=(4.5 * n_nc, 4 * n_agents))
    fig.suptitle(
        "Spatial Fairness  —  Grid {}x{}  ({})\n"
        "Dot colour = collection rate (red=0% / green=100%);  "
        "dot size ∝ sensor count;  seed={}".format(
            gs[0], gs[1], phys["sf"], seed
        ),
        fontsize=13, fontweight="bold"
    )

    for row, agent_key in enumerate(agents):
        for col, n in enumerate(SWEEP_SENSOR_COUNTS):
            ax = axes[row][col] if n_agents > 1 else axes[col]
            cond = results[gs][n].get(seed, {})
            if agent_key not in cond:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
                continue

            j, rates, positions = cond[agent_key]
            xs = [p[0] for p in positions]
            ys = [p[1] for p in positions]

            sc = ax.scatter(
                xs, ys, c=rates, cmap="RdYlGn",
                s=max(80, 500 // n),   # smaller dots when more sensors
                edgecolors="black", linewidths=0.8,
                vmin=0, vmax=100, zorder=3
            )

            # Annotate each sensor with its rate
            for x, y, r in zip(xs, ys, rates):
                ax.annotate("{:.0f}".format(r), (x, y),
                            textcoords="offset points", xytext=(0, 6),
                            fontsize=5.5, ha="center", color="black")

            # Starved sensors (rate < 20%) in red ring
            starved_x = [x for x, r in zip(xs, rates) if r < 20]
            starved_y = [y for y, r in zip(ys, rates) if r < 20]
            if starved_x:
                ax.scatter(starved_x, starved_y, s=max(100, 600 // n),
                           facecolors="none", edgecolors="red",
                           linewidths=1.8, zorder=4)

            ax.set_xlim(0, gs[0])
            ax.set_ylim(0, gs[1])
            ax.set_aspect("equal")

            # Stats box
            starved = sum(1 for r in rates if r < 20)
            ax.text(0.02, 0.98,
                    "J={:.3f}\nStarved: {}/{}".format(j, starved, n),
                    transform=ax.transAxes, fontsize=7, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="#CCCCCC", alpha=0.85))

            # Axis labels
            if col == 0:
                ax.set_ylabel(AGENT_STYLES[agent_key]["label"] + "\n" + "Y (m)",
                              fontsize=8, fontweight="bold",
                              color=AGENT_STYLES[agent_key]["color"])
            if row == 0:
                ax.set_title("{} sensors".format(n), fontsize=9, fontweight="bold")
            if row == n_agents - 1:
                ax.set_xlabel("X (m)", fontsize=8)

            ax.grid(True, alpha=0.3, linestyle="--")
            ax.tick_params(labelsize=7)

    # Shared colorbar
    cbar_ax = fig.add_axes([1.01, 0.15, 0.015, 0.7])
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(vmin=0, vmax=100))
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax, label="Collection Rate (%)")

    plt.tight_layout()
    label = "{}x{}".format(gs[0], gs[1])
    out   = OUTPUT_DIR / "figD_spatial_fairness_{}.png".format(label)
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print("Saved: {}".format(out.name))
    plt.close()


# ==================== FIGURE E: JAIN'S TREND LINES ====================

def plot_jains_trend_lines(results, has_dqn):
    """
    Line plots: Jain's index vs sensor count, one line per grid size, separate panel per agent.
    Shows trends with SF regime colour coding.
    """
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams["font.family"] = "serif"

    agents   = (["DQN", "SFGreedy", "NNGreedy"] if has_dqn else ["SFGreedy", "NNGreedy"])
    n_agents = len(agents)

    fig, axes = plt.subplots(1, n_agents, figsize=(6 * n_agents, 5), sharey=True)
    if n_agents == 1:
        axes = [axes]

    fig.suptitle(
        "Jain's Fairness vs Sensor Count by Grid Size  (Mean +- Std over {} seeds)\n"
        "Each line = different grid size / LoRa SF regime".format(len(SWEEP_SEEDS)),
        fontsize=13, fontweight="bold"
    )

    for ax, agent_key in zip(axes, agents):
        for gs in SWEEP_GRID_SIZES:
            phys  = GRID_PHYSICS[gs]
            means = []
            stds  = []
            for n in SWEEP_SENSOR_COUNTS:
                jvals = [results[gs][n][s][agent_key][0]
                         for s in SWEEP_SEEDS if agent_key in results[gs][n][s]]
                means.append(np.mean(jvals) if jvals else np.nan)
                stds.append(np.std(jvals)   if jvals else 0.0)
            means = np.array(means); stds = np.array(stds)
            ax.plot(SWEEP_SENSOR_COUNTS, means,
                    color=phys["color"], linewidth=2.5, marker="o", markersize=8,
                    label=phys["label"].replace("\n", " "), zorder=4)
            ax.fill_between(SWEEP_SENSOR_COUNTS, means - stds, means + stds,
                            color=phys["color"], alpha=0.15, zorder=3)

        ax.set_xlabel("Number of Sensors", fontsize=11, fontweight="bold")
        ax.set_ylabel("Jain's Fairness Index", fontsize=11, fontweight="bold")
        ax.set_title(AGENT_STYLES[agent_key]["label"], fontsize=11, fontweight="bold",
                     color=AGENT_STYLES[agent_key]["color"], pad=8)
        ax.set_xticks(SWEEP_SENSOR_COUNTS)
        ax.set_ylim(0, 1.05)
        ax.axhline(1.0, color="green", linestyle=":", linewidth=1.2, alpha=0.5,
                   label="Perfect fairness")
        ax.legend(fontsize=8, framealpha=0.9)
        ax.grid(True, alpha=0.4, linestyle="--")

    plt.tight_layout()
    out = OUTPUT_DIR / "figE_jains_trend_lines.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print("Saved: {}".format(out.name))
    plt.close()


# ==================== SAVE RESULTS CSV ====================

def save_results_csv(results, has_dqn):
    rows = []
    agents = ["DQN", "SFGreedy", "NNGreedy"] if has_dqn else ["SFGreedy", "NNGreedy"]
    for gs in SWEEP_GRID_SIZES:
        for n in SWEEP_SENSOR_COUNTS:
            for seed in SWEEP_SEEDS:
                for ak in agents:
                    cond = results[gs][n][seed]
                    if ak not in cond:
                        continue
                    j, rates, _ = cond[ak]
                    starved = sum(1 for r in rates if r < 20)
                    rows.append({
                        "grid_w":        gs[0],
                        "grid_h":        gs[1],
                        "sf_regime":     GRID_PHYSICS[gs]["sf"],
                        "num_sensors":   n,
                        "seed":          seed,
                        "agent":         ak,
                        "agent_label":   AGENT_STYLES[ak]["label"],
                        "jains_index":   round(j, 5),
                        "mean_coll_rate":round(float(np.mean(rates)), 2),
                        "std_coll_rate": round(float(np.std(rates)), 2),
                        "starved_count": starved,
                        "starved_pct":   round(starved / n * 100, 1),
                    })
    df = pd.DataFrame(rows)
    out = OUTPUT_DIR / "sweep_fairness_results.csv"
    df.to_csv(out, index=False)
    print("Saved: {}".format(out.name))
    return df


# ==================== MAIN ====================

def main():
    print("=" * 70)
    print("MULTI-CONDITION FAIRNESS SWEEP")
    print("Grid sizes:    {}".format(["{}x{}".format(*g) for g in SWEEP_GRID_SIZES]))
    print("Sensor counts: {}".format(SWEEP_SENSOR_COUNTS))
    print("Seeds:         {}".format(SWEEP_SEEDS))
    n_conditions = len(SWEEP_GRID_SIZES) * len(SWEEP_SENSOR_COUNTS) * len(SWEEP_SEEDS)
    print("Conditions:    {}  x 3 agents = {} episodes".format(
        n_conditions, n_conditions * 3))
    print("Output:        {}".format(OUTPUT_DIR))
    print("=" * 70)

    # Load DQN model
    dqn_model = None
    training_config = {"use_frame_stacking": True, "n_stack": 4, "max_sensors_limit": 50}
    if DQN_MODEL_PATH.exists():
        print("\nLoading DQN from {}...".format(DQN_MODEL_PATH))
        dqn_model       = DQN.load(DQN_MODEL_PATH)
        training_config = load_training_config(DQN_CONFIG_PATH)
        print("  Loaded | max_sensors_limit={}".format(
            training_config.get("max_sensors_limit", 50)))
    else:
        print("  DQN model not found -- running greedy agents only.")
    has_dqn = dqn_model is not None

    # Run sweep
    t0 = time.time()
    results = run_sweep(dqn_model, training_config)
    print("\nSweep complete: {:.1f} min".format((time.time()-t0)/60))

    # Save raw CSV
    df = save_results_csv(results, has_dqn)

    # Generate figures
    print("\nGenerating figures...")
    plot_jains_summary_matrix(results, has_dqn)      # figA
    if has_dqn:
        plot_jains_advantage_gap(results)             # figB
    plot_jains_seed_variance(results, has_dqn)        # figC
    for gs in SWEEP_GRID_SIZES:                       # figD (one per grid)
        plot_spatial_fairness_grid(results, gs, has_dqn)
    plot_jains_trend_lines(results, has_dqn)          # figE

    # Console summary
    print("\n" + "=" * 70)
    print("JAIN'S INDEX SUMMARY  (mean over all seeds)")
    print("=" * 70)
    pivot = df.pivot_table(
        index=["grid_w", "sf_regime"],
        columns=["agent", "num_sensors"],
        values="jains_index",
        aggfunc="mean"
    )
    print(pivot.round(3).to_string())

    print("\n" + "=" * 70)
    print("DONE")
    print("  figA  jains_summary_matrix  -- which conditions each agent handles")
    print("  figB  jains_advantage_gap   -- where DQN is fairer than greedy")
    print("  figC  jains_seed_variance   -- layout sensitivity per condition")
    print("  figD  spatial_fairness_*    -- spatial scatter per grid size")
    print("  figE  jains_trend_lines     -- fairness vs sensor count by SF regime")
    print("  CSV   sweep_fairness_results.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()