"""
Multi-Seed x Multi-Sensor-Count x Multi-Grid-Size Evaluation
=============================================================
Two orthogonal sweeps using the SAME trained DQN model (zero-padding):

  Sweep A - Sensor Count  (grid fixed 500x500):
    10 / 20 / 30 / 40 sensors.  Output: sensors_10/ ... fig10-fig12.

  Sweep B - Grid Size  (sensors fixed at 20):
    Grid unit = 10 m, UAV altitude = 100 m.

    Grid        Area      Max corner   Dominant SF   Data rate
    ----------------------------------------------------------
    100x100     1 km2     ~1.4 km      SF7           683 B/s
    300x300     9 km2     ~4.2 km      SF7/SF9       220-683 B/s
    500x500     25 km2    ~7.1 km      SF9/SF11       55-220 B/s
    1000x1000  100 km2   ~14.1 km     SF11/SF12      31-55 B/s

    Output: grid_100x100/ ... fig13-fig15.

Author: ATILADE GABRIEL OKE
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import gymnasium
import seaborn as sns
import json
from pathlib import Path
import time
from scipy import stats
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

# ==================== PATH SETUP ====================
script_dir         = Path(__file__).resolve().parent
src_dir            = script_dir.parent.parent.parent
script_dir_results = script_dir.parent

sys.path.insert(0, str(src_dir))

from environment.uav_env import UAVEnvironment
from greedy_agents import MaxThroughputGreedyV2, NearestSensorGreedy

# ==================== CONFIGURATION ====================

SEEDS         = [42, 123, 256, 789, 1337, 2024, 999, 314, 555, 2048,
                 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
SENSOR_COUNTS = [10, 20, 30, 40]          # all must be <= max_sensors_limit (50)

# Grid sizes for Sweep B.
GRID_SIZES = [
    (100,  100),
    (300,  300),
    (500,  500),
    (1000, 1000),
]

# Physics annotations from iot_sensors.py (grid unit=10m, UAV alt=100m)
GRID_PHYSICS = {
    (100,  100): {"label": "100x100\n(1 km2)",   "sf": "SF7",        "color": "#1B5E20"},
    (300,  300): {"label": "300x300\n(9 km2)",   "sf": "SF7/SF9",    "color": "#F9A825"},
    (500,  500): {"label": "500x500\n(25 km2)",  "sf": "SF9/SF11",   "color": "#E65100"},
    (1000, 1000):{"label": "1000x1000\n(100km2)","sf": "SF11/SF12",  "color": "#B71C1C"},
}

GRID_SWEEP_NUM_SENSORS = 20  # fixed sensor count during grid sweep

PLOT_CONFIG = {
    "grid_size":          (500, 500),
    "max_steps":          2100,
    "path_loss_exponent": 3.8,
    "rssi_threshold":     -120.0,
    "sensor_duty_cycle":  10.0,
}

EVAL_MAX_BATTERY = 274.0

DQN_MODEL_PATH = (
    script_dir_results / "models" / "dqn_full_observability" / "dqn_final.zip"
)
DQN_CONFIG_PATH = (
    script_dir_results / "models" / "dqn_full_observability" / "training_config.json"
)
VEC_NORMALIZE_PATH = (
    script_dir_results / "models" / "dqn_full_observability" / "vec_normalize.pkl"
)

OUTPUT_DIR = script_dir / "multi_seed_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_EVERY = 50

# ==================== ENVIRONMENT WRAPPERS ====================


class FixedLayoutEnv(UAVEnvironment):
    """Forces identical sensor positions across all envs for a given seed."""
    def __init__(self, fixed_positions, **kwargs):
        self._fixed_positions = fixed_positions
        super().__init__(**kwargs)

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        for sensor, pos in zip(self.sensors, self._fixed_positions):
            sensor.position = np.array(pos, dtype=np.float32)
        return obs


class FixedLayoutSnapshotEnv(FixedLayoutEnv):
    """FixedLayout + snapshot + zero-padding. features_per_sensor auto-detected."""

    def __init__(self, fixed_positions, max_sensors_limit=50, **kwargs):
        self.max_sensors_limit = max_sensors_limit
        super().__init__(fixed_positions, **kwargs)

        raw_obs_size = self.observation_space.shape[0]
        self._features_per_sensor = 0
        for uav_f in range(raw_obs_size + 1):
            remainder = raw_obs_size - uav_f
            if remainder > 0 and remainder % self.num_sensors == 0:
                self._features_per_sensor = remainder // self.num_sensors
                break
        if self._features_per_sensor == 0:
            raise ValueError(
                "Cannot infer features_per_sensor: raw obs {} has no divisor "
                "matching num_sensors={}.".format(raw_obs_size, self.num_sensors)
            )

        padded = raw_obs_size + (max_sensors_limit - self.num_sensors) * self._features_per_sensor
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(padded,), dtype=np.float32,
        )
        self.last_sensor_data = None

    def _pad(self, raw):
        padding = np.zeros(
            (self.max_sensors_limit - self.num_sensors) * self._features_per_sensor,
            dtype=np.float32,
        )
        return np.concatenate([raw, padding]).astype(np.float32)

    def reset(self, **kwargs):
        if hasattr(self, "sensors") and self.current_step > 0:
            self.last_sensor_data = [
                {
                    "sensor_id":              int(s.sensor_id),
                    "total_data_generated":   float(s.total_data_generated),
                    "total_data_transmitted": float(s.total_data_transmitted),
                    "total_data_lost":        float(s.total_data_lost),
                }
                for s in self.sensors
            ]
        obs, info = super().reset(**kwargs)
        return self._pad(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return self._pad(obs), reward, terminated, truncated, info


# ==================== HELPERS ====================


def load_training_config(path):
    defaults = {
        "use_frame_stacking":  True,
        "n_stack":             4,
        "max_sensors_limit":   50,
        "features_per_sensor": None,
    }
    try:
        with open(path) as f:
            return {**defaults, **json.load(f)}
    except FileNotFoundError:
        print("Config not found at {} -- using defaults".format(path))
        return defaults


def _base_env_kwargs(num_sensors, grid_size=None):
    gs = grid_size if grid_size is not None else PLOT_CONFIG["grid_size"]
    return {
        "grid_size":          gs,
        "num_sensors":        num_sensors,
        "max_steps":          PLOT_CONFIG["max_steps"],
        "path_loss_exponent": PLOT_CONFIG["path_loss_exponent"],
        "rssi_threshold":     PLOT_CONFIG["rssi_threshold"],
        "sensor_duty_cycle":  PLOT_CONFIG["sensor_duty_cycle"],
        "max_battery":        274.0,
        "render_mode":        None,
    }


def get_canonical_positions(seed, num_sensors, grid_size=None):
    master = UAVEnvironment(**_base_env_kwargs(num_sensors, grid_size))
    master.reset(seed=seed)
    positions = [tuple(float(v) for v in s.position) for s in master.sensors]
    master.close()
    return positions


def compute_jains_index(collection_rates):
    n = len(collection_rates)
    if n == 0:
        return 0.0
    s1 = sum(collection_rates)
    s2 = sum(x ** 2 for x in collection_rates)
    return (s1 ** 2) / (n * s2) if s2 > 0 else 1.0


def compute_energy_efficiency(env):
    energy_used = EVAL_MAX_BATTERY - env.uav.battery
    return float(env.total_data_collected / energy_used) if energy_used > 0 else 0.0


def compute_jains_from_env(env):
    rates = []
    for s in env.sensors:
        gen = float(s.total_data_generated)
        tx  = float(s.total_data_transmitted)
        rates.append((tx / gen * 100) if gen > 0 else 0.0)
    return compute_jains_index(rates)


def _unwrap_base_env(vec):
    inner = vec
    while hasattr(inner, "venv"):
        inner = inner.venv
    env = inner.envs[0]
    while hasattr(env, "env"):
        env = env.env
    return env


# ==================== EPISODE RUNNERS ====================


def run_greedy_episode(agent_class, fixed_positions, seed, agent_name,
                       num_sensors, grid_size=None):
    env = FixedLayoutEnv(fixed_positions, **_base_env_kwargs(num_sensors, grid_size))
    obs, _ = env.reset(seed=seed)
    agent = agent_class(env)

    history = {"step": [], "cumulative_reward": [], "battery_percent": [],
               "coverage_percent": [], "total_data_collected": [],
               "energy_consumed_wh": []}
    cum_reward = 0.0

    while True:
        action = agent.select_action(obs)
        obs, reward, done, trunc, _ = env.step(action)
        cum_reward += reward
        if env.current_step % LOG_EVERY == 0 or done or trunc:
            history["step"].append(env.current_step)
            history["cumulative_reward"].append(cum_reward)
            history["battery_percent"].append(env.uav.get_battery_percentage())
            history["coverage_percent"].append(
                (len(env.sensors_visited) / env.num_sensors) * 100
            )
            history["total_data_collected"].append(env.total_data_collected)
            history["energy_consumed_wh"].append(EVAL_MAX_BATTERY - env.uav.battery)
        if done or trunc:
            break

    energy_used  = EVAL_MAX_BATTERY - env.uav.battery
    efficiencies = [
        (dc / ec) if ec > 0 else 0.0
        for dc, ec in zip(history["total_data_collected"], history["energy_consumed_wh"])
    ]
    summary = {
        "seed":               seed,
        "num_sensors":        num_sensors,
        "grid_size":          str(grid_size or PLOT_CONFIG["grid_size"]),
        "agent":              agent_name,
        "final_reward":       cum_reward,
        "final_coverage":     (len(env.sensors_visited) / env.num_sensors) * 100,
        "jains_index":        compute_jains_from_env(env),
        "energy_efficiency":  compute_energy_efficiency(env),
        "final_battery":      env.uav.get_battery_percentage(),
        "total_data_bytes":   env.total_data_collected,
        "energy_consumed_wh": energy_used,
        "peak_efficiency":    max(efficiencies) if efficiencies else 0.0,
        "avg_efficiency":     float(np.mean(efficiencies)) if efficiencies else 0.0,
    }
    env.close()
    return pd.DataFrame(history), summary


def run_dqn_episode(model, training_config, fixed_positions, seed,
                    num_sensors, grid_size=None):
    kw                = _base_env_kwargs(num_sensors, grid_size)
    max_sensors_limit = training_config.get("max_sensors_limit", 50)
    fp                = fixed_positions

    def _make():
        return FixedLayoutSnapshotEnv(fp, max_sensors_limit=max_sensors_limit, **kw)

    vec = DummyVecEnv([_make])
    if training_config.get("use_frame_stacking", True):
        stacked = VecFrameStack(vec, n_stack=training_config.get("n_stack", 4))
    else:
        stacked = vec

    if VEC_NORMALIZE_PATH.exists():
        try:
            stacked = VecNormalize.load(str(VEC_NORMALIZE_PATH), stacked)
            stacked.training   = False
            stacked.norm_reward = False
        except AssertionError as e:
            print("  vec_normalize.pkl shape mismatch -- skipping. ({})".format(e))

    base_env   = _unwrap_base_env(stacked)
    obs        = stacked.reset()
    history    = {"step": [], "cumulative_reward": [], "battery_percent": [],
                  "coverage_percent": [], "total_data_collected": [],
                  "energy_consumed_wh": []}
    cum_reward = 0.0
    step_count = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        av = int(action[0]) if hasattr(action, "__len__") else int(action)

        pre_battery  = base_env.uav.battery
        pre_coverage = (len(base_env.sensors_visited) / base_env.num_sensors) * 100
        pre_data     = base_env.total_data_collected

        obs, rwds, dones, _ = stacked.step([av])
        cum_reward  += float(rwds[0])
        step_count  += 1
        episode_done = bool(dones[0])

        if episode_done:
            history["step"].append(step_count)
            history["cumulative_reward"].append(cum_reward)
            history["battery_percent"].append((pre_battery / EVAL_MAX_BATTERY) * 100)
            history["coverage_percent"].append(pre_coverage)
            history["total_data_collected"].append(pre_data)
            history["energy_consumed_wh"].append(EVAL_MAX_BATTERY - pre_battery)
            break
        elif step_count % LOG_EVERY == 0:
            history["step"].append(step_count)
            history["cumulative_reward"].append(cum_reward)
            history["battery_percent"].append(base_env.uav.get_battery_percentage())
            history["coverage_percent"].append(
                (len(base_env.sensors_visited) / base_env.num_sensors) * 100
            )
            history["total_data_collected"].append(base_env.total_data_collected)
            history["energy_consumed_wh"].append(EVAL_MAX_BATTERY - base_env.uav.battery)

    energy_used    = history["energy_consumed_wh"][-1]  if history["energy_consumed_wh"]  else 0.0
    total_data     = history["total_data_collected"][-1] if history["total_data_collected"] else 0.0
    final_batt_pct = history["battery_percent"][-1]     if history["battery_percent"]      else 100.0
    final_coverage = history["coverage_percent"][-1]    if history["coverage_percent"]      else 0.0
    final_eff      = total_data / energy_used if energy_used > 0 else 0.0

    if base_env.last_sensor_data:
        rates  = []
        for s in base_env.last_sensor_data:
            gen = s["total_data_generated"]
            tx  = s["total_data_transmitted"]
            rates.append((tx / gen * 100) if gen > 0 else 0.0)
        jains = compute_jains_index(rates)
    else:
        jains = 0.0

    efficiencies = [
        (dc / ec) if ec > 0 else 0.0
        for dc, ec in zip(history["total_data_collected"], history["energy_consumed_wh"])
    ]

    stacked.reset()
    summary = {
        "seed":               seed,
        "num_sensors":        num_sensors,
        "grid_size":          str(grid_size or PLOT_CONFIG["grid_size"]),
        "agent":              "DQN",
        "final_reward":       cum_reward,
        "final_coverage":     final_coverage,
        "jains_index":        jains,
        "energy_efficiency":  final_eff,
        "final_battery":      final_batt_pct,
        "total_data_bytes":   total_data,
        "energy_consumed_wh": energy_used,
        "peak_efficiency":    max(efficiencies) if efficiencies else 0.0,
        "avg_efficiency":     float(np.mean(efficiencies)) if efficiencies else 0.0,
    }
    stacked.close()
    return pd.DataFrame(history), summary


# ==================== SWEEP RUNNERS ====================


def _run_sweep(dqn_model, training_config, sweep_keys, num_sensors_fn, grid_size_fn, key_label_fn):
    """
    Generic sweep runner.
    sweep_keys: list of sweep values (sensor counts or grid tuples)
    num_sensors_fn(key) -> int
    grid_size_fn(key) -> tuple or None
    key_label_fn(key) -> str
    """
    agent_names = ["DQN", "SF-Aware Greedy V2", "Nearest Sensor Greedy"]
    results = {}
    total_runs = len(sweep_keys) * len(SEEDS) * (3 if dqn_model else 2)
    run_idx = 0

    for key in sweep_keys:
        results[key] = {
            "histories": {n: [] for n in agent_names},
            "summaries": {n: [] for n in agent_names},
        }
        h = results[key]["histories"]
        s = results[key]["summaries"]
        n_sensors = num_sensors_fn(key)
        gs        = grid_size_fn(key)
        label     = key_label_fn(key)

        print("\n" + "#" * 70)
        print("  {} (sensors={}, grid={})".format(
            label, n_sensors,
            "{}x{}".format(gs[0], gs[1]) if gs else "500x500"
        ))
        print("#" * 70)

        for seed in SEEDS:
            print("\n  " + "=" * 60)
            print("  SEED {}  ({}/{})".format(seed, SEEDS.index(seed)+1, len(SEEDS)))
            print("  " + "=" * 60)

            fixed_positions = get_canonical_positions(seed, n_sensors, gs)

            if dqn_model is not None:
                run_idx += 1
                t0 = time.time()
                print("\n  [{}/{}] DQN".format(run_idx, total_runs))
                hist, summ = run_dqn_episode(
                    dqn_model, training_config, fixed_positions, seed, n_sensors, gs
                )
                h["DQN"].append(hist)
                s["DQN"].append(summ)
                print("    reward={:.0f}  cov={:.1f}%  J={:.4f}  ({:.1f}s)".format(
                    summ["final_reward"], summ["final_coverage"],
                    summ["jains_index"], time.time()-t0))

            for agent_name, agent_class in [
                ("SF-Aware Greedy V2",    MaxThroughputGreedyV2),
                ("Nearest Sensor Greedy", NearestSensorGreedy),
            ]:
                run_idx += 1
                t0 = time.time()
                print("\n  [{}/{}] {}".format(run_idx, total_runs, agent_name))
                hist, summ = run_greedy_episode(
                    agent_class, fixed_positions, seed, agent_name, n_sensors, gs
                )
                h[agent_name].append(hist)
                s[agent_name].append(summ)
                print("    reward={:.0f}  cov={:.1f}%  J={:.4f}  ({:.1f}s)".format(
                    summ["final_reward"], summ["final_coverage"],
                    summ["jains_index"], time.time()-t0))

    return results


def run_all(dqn_model, training_config):
    """Sweep A: vary sensor counts, grid fixed at 500x500."""
    return _run_sweep(
        dqn_model, training_config,
        sweep_keys    = SENSOR_COUNTS,
        num_sensors_fn = lambda n: n,
        grid_size_fn  = lambda n: None,
        key_label_fn  = lambda n: "SENSOR COUNT: {}".format(n),
    )


def run_grid_sweep(dqn_model, training_config):
    """Sweep B: vary grid size, sensors fixed at GRID_SWEEP_NUM_SENSORS."""
    return _run_sweep(
        dqn_model, training_config,
        sweep_keys    = GRID_SIZES,
        num_sensors_fn = lambda gs: GRID_SWEEP_NUM_SENSORS,
        grid_size_fn  = lambda gs: gs,
        key_label_fn  = lambda gs: "GRID: {}x{} ({})".format(
            gs[0], gs[1], GRID_PHYSICS[gs]["sf"]
        ),
    )


# ==================== INTERPOLATION ====================


def interpolate_to_common_steps(histories, n_points=200):
    result = {}
    for agent, hist_list in histories.items():
        if not hist_list:
            continue
        max_step     = max(df["step"].iloc[-1] for df in hist_list)
        common_steps = np.linspace(0, max_step, n_points)
        ir, ic, id_ = [], [], []
        for df in hist_list:
            steps = df["step"].values
            ir.append(np.interp(common_steps, steps, df["cumulative_reward"].values))
            ic.append(np.interp(common_steps, steps, df["coverage_percent"].values))
            id_.append(np.interp(common_steps, steps, df["total_data_collected"].values))
        result[agent] = {
            "steps":         common_steps,
            "mean_reward":   np.array(ir).mean(0),
            "std_reward":    np.array(ir).std(0),
            "mean_coverage": np.array(ic).mean(0),
            "std_coverage":  np.array(ic).std(0),
            "mean_data":     np.array(id_).mean(0),
            "std_data":      np.array(id_).std(0),
        }
    return result


# ==================== STYLE ====================

AGENT_STYLES = {
    "DQN":                  {"color": "#1565C0", "linestyle": "-",  "marker": "o",
                              "label": "DQN Agent (Proposed)"},
    "SF-Aware Greedy V2":   {"color": "#C62828", "linestyle": "--", "marker": "s",
                              "label": "SF-Aware Greedy V2"},
    "Nearest Sensor Greedy":{"color": "#555555", "linestyle": ":",  "marker": "^",
                              "label": "Nearest Sensor Greedy"},
}
AGENT_COLORS = {k: v["color"] for k, v in AGENT_STYLES.items()}


# ==================== PER-CONDITION PLOTS ====================


def _shaded_line(ax, steps, mean, std, style):
    ax.plot(steps, mean, color=style["color"], linewidth=2.5,
            linestyle=style["linestyle"], label=style["label"], zorder=3)
    ax.fill_between(steps, mean - std, mean + std,
                    color=style["color"], alpha=0.18, zorder=2)


def plot_shaded_reward(interp_data, seeds, out_dir):
    sns.set_theme(style="whitegrid", font_scale=1.3)
    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=(14, 8))
    for agent, d in interp_data.items():
        _shaded_line(ax, d["steps"], d["mean_reward"], d["std_reward"],
                     AGENT_STYLES[agent])
    ax.set_xlabel("Simulation Step (t)", fontsize=15, fontweight="bold")
    ax.set_ylabel("Cumulative Reward", fontsize=15, fontweight="bold")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.tick_params(axis="both", labelsize=13)
    ax.legend(loc="lower right", fontsize=13, framealpha=0.9)
    ax.set_title("Comparative Performance (Mean +- Std, n={} seeds)".format(len(seeds)),
                 fontsize=15, fontweight="bold", pad=12)
    plt.tight_layout()
    out = out_dir / "fig1_shaded_reward.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print("  {}".format(out.relative_to(OUTPUT_DIR)))
    plt.close()


def plot_shaded_coverage(interp_data, seeds, out_dir):
    sns.set_theme(style="whitegrid", font_scale=1.15)
    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=(12, 6))
    for agent, d in interp_data.items():
        _shaded_line(ax, d["steps"], d["mean_coverage"], d["std_coverage"],
                     AGENT_STYLES[agent])
    ax.set_xlabel("Simulation Step (t)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Sensor Coverage (%)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", fontsize=11)
    ax.set_title("Coverage Rate (n={} seeds, +- 1 std)".format(len(seeds)),
                 fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    out = out_dir / "fig2_shaded_coverage.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print("  {}".format(out.relative_to(OUTPUT_DIR)))
    plt.close()


def plot_summary_bars(all_summaries, seeds, out_dir):
    metrics = [
        ("final_reward",      "Cumulative Reward",             "1e6", True),
        ("final_coverage",    "Coverage (%)",                  "%",   True),
        ("jains_index",       "Jain's Fairness Index",         "",    True),
        ("energy_efficiency", "Energy Efficiency (bytes/Wh)",  "",    True),
    ]
    sns.set_theme(style="whitegrid", font_scale=1.3)
    plt.rcParams["font.family"] = "serif"
    fig, axes = plt.subplots(1, 4, figsize=(22, 7))
    fig.suptitle("Multi-Metric Summary (n={} seeds)".format(len(seeds)),
                 fontsize=16, fontweight="bold", y=1.02)
    for ax, (mk, ml, unit, hi) in zip(axes, metrics):
        means, stds, names, colors = [], [], [], []
        for agent, style in AGENT_STYLES.items():
            summs = all_summaries.get(agent, [])
            if not summs:
                continue
            vals = [s[mk] for s in summs]
            m, sv = np.mean(vals), np.std(vals)
            means.append(m); stds.append(sv)
            names.append(style["label"]); colors.append(style["color"])
        if not means:
            continue
        x    = np.arange(len(names))
        bars = ax.bar(x, means, color=colors, alpha=0.8, edgecolor="white",
                      linewidth=1.2, zorder=3)
        ax.errorbar(x, means, yerr=stds, fmt="none", color="black",
                    capsize=6, capthick=2, elinewidth=2, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=11)
        ax.set_ylabel(ml, fontsize=13, fontweight="bold")
        ax.set_title(ml, fontsize=13, fontweight="bold", pad=8)
        ax.yaxis.grid(True, alpha=0.5)
        if unit == "1e6":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        best = np.argmax(means) if hi else np.argmin(means)
        bars[best].set_edgecolor("gold"); bars[best].set_linewidth(3)
        ax.text(best, means[best] + stds[best] * 0.1,
                "*", ha="center", va="bottom", fontsize=18, color="goldenrod")
    plt.tight_layout()
    out = out_dir / "fig3_metric_bars.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print("  {}".format(out.relative_to(OUTPUT_DIR)))
    plt.close()
    rows = []
    for agent, style in AGENT_STYLES.items():
        summs = all_summaries.get(agent, [])
        if not summs:
            continue
        for mk, ml, _, _ in metrics:
            vals = [s[mk] for s in summs]
            rows.append({"Agent": style["label"], "Metric": ml,
                         "Mean": "{:.4f}".format(np.mean(vals)),
                         "Std":  "{:.4f}".format(np.std(vals))})
    pd.DataFrame(rows).to_csv(out_dir / "summary_table.csv", index=False)


def plot_per_seed_variance(all_histories, out_dir):
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=(13, 7))
    for agent, hist_list in all_histories.items():
        if not hist_list:
            continue
        style = AGENT_STYLES[agent]
        for df in hist_list:
            ax.plot(df["step"], df["cumulative_reward"],
                    color=style["color"], alpha=0.2, linewidth=1.0)
        common  = np.linspace(0, max(df["step"].iloc[-1] for df in hist_list), 300)
        interps = [np.interp(common, df["step"].values, df["cumulative_reward"].values)
                   for df in hist_list]
        ax.plot(common, np.mean(interps, 0),
                color=style["color"], linewidth=3, label=style["label"], zorder=5)
    ax.set_xlabel("Simulation Step (t)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Cumulative Reward", fontsize=13, fontweight="bold")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend(loc="lower right", fontsize=11)
    ax.set_title("Per-Seed Reward Traces (faint=individual, bold=mean)",
                 fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    out = out_dir / "fig4_per_seed_variance.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print("  {}".format(out.relative_to(OUTPUT_DIR)))
    plt.close()


def plot_shaded_data_throughput(interp_data, seeds, out_dir):
    sns.set_theme(style="whitegrid", font_scale=1.15)
    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=(12, 6))
    for agent, d in interp_data.items():
        _shaded_line(ax, d["steps"], d["mean_data"], d["std_data"], AGENT_STYLES[agent])
    ax.set_xlabel("Simulation Step (t)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Cumulative Data Collected (Bytes)", fontsize=13, fontweight="bold")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend(loc="lower right", fontsize=11)
    ax.set_title("Data Throughput (n={} seeds, +- 1 std)".format(len(seeds)),
                 fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    out = out_dir / "fig5_data_throughput.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print("  {}".format(out.relative_to(OUTPUT_DIR)))
    plt.close()


def plot_box_plots(all_summaries, seeds, out_dir):
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams["font.family"] = "serif"
    metrics = [
        ("final_reward",      "Cumulative Reward",        "1e"),
        ("final_coverage",    "Final Coverage (%)",       "%"),
        ("jains_index",       "Jain's Fairness Index",    ""),
        ("energy_efficiency", "Energy Efficiency (B/Wh)", ""),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    fig.suptitle("Seed-by-Seed Distributions (n={})".format(len(seeds)),
                 fontsize=13, fontweight="bold", y=1.02)
    for ax, (key, label, unit) in zip(axes, metrics):
        plot_data, plot_labels, plot_colors = [], [], []
        for agent, style in AGENT_STYLES.items():
            summs = all_summaries.get(agent, [])
            if not summs:
                continue
            plot_data.append([s[key] for s in summs])
            plot_labels.append(style["label"].replace(" ", "\n"))
            plot_colors.append(style["color"])
        if not plot_data:
            continue
        parts = ax.violinplot(plot_data, positions=range(len(plot_data)),
                              showmeans=False, showmedians=False, showextrema=False)
        for pc, color in zip(parts["bodies"], plot_colors):
            pc.set_facecolor(color); pc.set_alpha(0.35); pc.set_edgecolor(color)
        bp = ax.boxplot(plot_data, positions=range(len(plot_data)),
                        widths=0.25, patch_artist=True,
                        medianprops=dict(color="black", linewidth=2),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5),
                        flierprops=dict(marker="o", markersize=5, alpha=0.6))
        for patch, color in zip(bp["boxes"], plot_colors):
            patch.set_facecolor(color); patch.set_alpha(0.7)
        for i, (vals, color) in enumerate(zip(plot_data, plot_colors)):
            jitter = np.random.default_rng(0).uniform(-0.06, 0.06, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                       color=color, s=40, zorder=5, edgecolors="white",
                       linewidths=0.8, alpha=0.9)
        ax.set_xticks(range(len(plot_labels)))
        ax.set_xticklabels(plot_labels, fontsize=8.5)
        ax.set_ylabel(label, fontsize=10, fontweight="bold")
        ax.set_title(label, fontsize=10, fontweight="bold", pad=8)
        ax.yaxis.grid(True, alpha=0.5, linestyle="--")
        if unit == "1e":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.tight_layout()
    out = out_dir / "fig6_box_violin.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print("  {}".format(out.relative_to(OUTPUT_DIR)))
    plt.close()


def plot_convergence_stability(interp_data, seeds, out_dir):
    sns.set_theme(style="whitegrid", font_scale=1.15)
    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=(12, 6))
    for agent, d in interp_data.items():
        style = AGENT_STYLES[agent]
        ax.plot(d["steps"], d["std_reward"], color=style["color"],
                linewidth=2.5, linestyle=style["linestyle"], label=style["label"], zorder=3)
    ax.set_xlabel("Simulation Step (t)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Std Dev of Cumulative Reward", fontsize=13, fontweight="bold")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend(loc="upper left", fontsize=11)
    ax.set_title("Convergence Stability (lower = more robust, n={} seeds)".format(len(seeds)),
                 fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    out = out_dir / "fig7_convergence_stability.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print("  {}".format(out.relative_to(OUTPUT_DIR)))
    plt.close()


def plot_jains_over_time(all_histories, all_summaries, seeds, out_dir):
    sns.set_theme(style="whitegrid", font_scale=1.15)
    plt.rcParams["font.family"] = "serif"
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("Fairness & Coverage Dynamics", fontsize=13, fontweight="bold")

    ax = axes[0]
    for agent, hist_list in all_histories.items():
        if not hist_list:
            continue
        style  = AGENT_STYLES[agent]
        common = np.linspace(0, max(df["step"].iloc[-1] for df in hist_list), 200)
        interps = [np.interp(common, df["step"].values, df["coverage_percent"].values)
                   for df in hist_list]
        mean_cov = np.mean(interps, 0); std_cov = np.std(interps, 0)
        ax.plot(common, mean_cov, color=style["color"],
                linewidth=2.5, linestyle=style["linestyle"], label=style["label"])
        ax.fill_between(common, mean_cov - std_cov, mean_cov + std_cov,
                        color=style["color"], alpha=0.18)
    ax.set_xlabel("Simulation Step (t)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Sensor Coverage (%)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 105); ax.set_title("Coverage Accumulation Rate", fontsize=11, fontweight="bold")
    ax.legend(fontsize=10); ax.grid(True, alpha=0.4, linestyle="--")

    ax2 = axes[1]
    for i, (agent, style) in enumerate(AGENT_STYLES.items()):
        summs = all_summaries.get(agent, [])
        if not summs:
            continue
        jvals  = [s["jains_index"] for s in summs]
        jitter = np.random.default_rng(i).uniform(-0.08, 0.08, size=len(jvals))
        ax2.scatter(np.full(len(jvals), i) + jitter, jvals,
                    color=style["color"], s=80, zorder=4,
                    edgecolors="white", linewidths=1.0, alpha=0.9, label=style["label"])
        ax2.hlines(np.mean(jvals), i - 0.2, i + 0.2,
                   color=style["color"], linewidth=3, zorder=5)
    ax2.set_xticks(range(len(AGENT_STYLES)))
    ax2.set_xticklabels([v["label"].replace(" ", "\n") for v in AGENT_STYLES.values()], fontsize=9)
    ax2.set_ylabel("Jain's Fairness Index", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 1.05)
    ax2.axhline(1.0, color="green", linestyle=":", linewidth=1.2, alpha=0.6)
    ax2.set_title("Final Jain's Index per Seed", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9, loc="lower right"); ax2.grid(axis="y", alpha=0.4, linestyle="--")

    plt.tight_layout()
    out = out_dir / "fig8_jains_dynamics.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print("  {}".format(out.relative_to(OUTPUT_DIR)))
    plt.close()


def run_statistical_tests(all_summaries, out_dir):
    dqn_rewards = [s["final_reward"] for s in all_summaries.get("DQN", [])]
    if not dqn_rewards:
        return
    results = []
    for agent in ["SF-Aware Greedy V2", "Nearest Sensor Greedy"]:
        baseline = [s["final_reward"] for s in all_summaries.get(agent, [])]
        if not baseline:
            continue
        t_stat, p_val = stats.ttest_ind(dqn_rewards, baseline, equal_var=False)
        sig = "YES" if p_val < 0.05 else "NO"
        print("    DQN vs {}: t={:.4f} p={:.4f} -> {}".format(agent, t_stat, p_val, sig))
        results.append({
            "Comparison":    "DQN vs {}".format(agent),
            "DQN Mean":      "{:.0f}".format(np.mean(dqn_rewards)),
            "DQN Std":       "{:.0f}".format(np.std(dqn_rewards)),
            "Baseline Mean": "{:.0f}".format(np.mean(baseline)),
            "Baseline Std":  "{:.0f}".format(np.std(baseline)),
            "t-stat":        "{:.4f}".format(t_stat),
            "p-value":       "{:.4f}".format(p_val),
            "Significant":   sig,
        })
    if results:
        pd.DataFrame(results).to_csv(out_dir / "statistical_tests.csv", index=False)


def save_raw_data(all_summaries, seeds, out_dir, label):
    rows = []
    for agent, summs in all_summaries.items():
        rows.extend(summs)
    pd.DataFrame(rows).to_csv(out_dir / "summary_{}.csv".format(label), index=False)


# ==================== SENSOR-COUNT SCALABILITY PLOTS (fig10-fig12) ====================


def plot_scalability_metrics(results):
    sns.set_theme(style="whitegrid", font_scale=1.3)
    plt.rcParams["font.family"] = "serif"
    metrics = [
        ("final_reward",      "Cumulative Reward",             "sci"),
        ("final_coverage",    "Final Coverage (%)",            "plain"),
        ("jains_index",       "Jain's Fairness Index",         "plain"),
        ("energy_efficiency", "Energy Efficiency (bytes/Wh)",  "sci"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Scalability: Performance vs Sensor Count\n"
        "(Mean +- Std, {} seeds per point, grid=500x500)".format(len(SEEDS)),
        fontsize=16, fontweight="bold"
    )
    axes = axes.flatten()
    for ax, (mk, ml, fmt) in zip(axes, metrics):
        for agent, style in AGENT_STYLES.items():
            means, stds = [], []
            for n in SENSOR_COUNTS:
                summs = results[n]["summaries"].get(agent, [])
                if not summs:
                    means.append(np.nan); stds.append(np.nan); continue
                vals = [s[mk] for s in summs]
                means.append(np.mean(vals)); stds.append(np.std(vals))
            means = np.array(means, dtype=float)
            stds  = np.array(stds,  dtype=float)
            ax.plot(SENSOR_COUNTS, means, color=style["color"],
                    linestyle=style["linestyle"], marker=style["marker"],
                    markersize=8, linewidth=2.5, label=style["label"], zorder=3)
            ax.fill_between(SENSOR_COUNTS, means - stds, means + stds,
                            color=style["color"], alpha=0.15, zorder=2)
        ax.set_xlabel("Number of Sensors", fontsize=13, fontweight="bold")
        ax.set_ylabel(ml, fontsize=13, fontweight="bold")
        ax.set_title(ml, fontsize=13, fontweight="bold", pad=8)
        ax.set_xticks(SENSOR_COUNTS)
        ax.tick_params(axis="both", labelsize=12)
        ax.legend(fontsize=11); ax.grid(True, alpha=0.4, linestyle="--")
        if fmt == "sci":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.tight_layout()
    out = OUTPUT_DIR / "fig10_sensor_scalability.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print("\nSaved: {}".format(out.name)); plt.close()


def plot_dqn_advantage_gap(results):
    sns.set_theme(style="whitegrid", font_scale=1.3)
    plt.rcParams["font.family"] = "serif"
    metrics = [
        ("final_reward",   "Reward Advantage"),
        ("final_coverage", "Coverage Advantage (pp)"),
        ("jains_index",    "Fairness Advantage (Jain's)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    fig.suptitle(
        "DQN Advantage Over Best Greedy vs Sensor Count\n"
        "(Positive = DQN wins, grid=500x500)",
        fontsize=15, fontweight="bold"
    )
    for ax, (mk, ml) in zip(axes, metrics):
        gaps, errs = [], []
        for n in SENSOR_COUNTS:
            dqn_s = results[n]["summaries"].get("DQN", [])
            if not dqn_s:
                gaps.append(np.nan); errs.append(0); continue
            dqn_mean = np.mean([s[mk] for s in dqn_s])
            best_g   = max(
                np.mean([s[mk] for s in results[n]["summaries"].get(a, dqn_s)])
                for a in ["SF-Aware Greedy V2", "Nearest Sensor Greedy"]
            )
            gaps.append(dqn_mean - best_g)
            errs.append(np.std([s[mk] for s in dqn_s]))
        colors = ["#1565C0" if (g >= 0 if not np.isnan(g) else True) else "#C62828"
                  for g in gaps]
        ax.bar(SENSOR_COUNTS, gaps, color=colors, alpha=0.75, edgecolor="white",
               linewidth=1.2, width=4, zorder=3)
        ax.errorbar(SENSOR_COUNTS, gaps, yerr=errs, fmt="none", color="black",
                    capsize=5, capthick=1.5, elinewidth=1.5, zorder=4)
        ax.axhline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.5)
        ax.set_xlabel("Number of Sensors", fontsize=13, fontweight="bold")
        ax.set_ylabel("Delta {}".format(ml), fontsize=13, fontweight="bold")
        ax.set_title(ml, fontsize=13, fontweight="bold", pad=8)
        ax.set_xticks(SENSOR_COUNTS)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(axis="y", alpha=0.4, linestyle="--")
        if mk == "final_reward":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    plt.tight_layout()
    out = OUTPUT_DIR / "fig11_sensor_dqn_advantage.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print("Saved: {}".format(out.name)); plt.close()


def plot_scalability_heatmap(results):
    sns.set_theme(style="white", font_scale=1.05)
    plt.rcParams["font.family"] = "serif"
    metrics = [
        ("final_reward",      "Cumulative\nReward"),
        ("final_coverage",    "Coverage\n(%)"),
        ("jains_index",       "Jain's\nFairness"),
        ("energy_efficiency", "Energy Eff.\n(B/Wh)"),
    ]
    agent_list   = list(AGENT_STYLES.keys())
    agent_labels = ["DQN (Proposed)", "SF-Aware Greedy V2", "Nearest Sensor Greedy"]
    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(
        "Sensor-Count Heatmap (mean, {} seeds, grid=500x500)".format(len(SEEDS)),
        fontsize=13, fontweight="bold"
    )
    for ax, (mk, ml) in zip(axes, metrics):
        matrix = np.array([
            [np.mean([s[mk] for s in results[n]["summaries"].get(a, [])]) if results[n]["summaries"].get(a) else np.nan
             for n in SENSOR_COUNTS]
            for a in agent_list
        ])
        col_min = np.nanmin(matrix, 0, keepdims=True)
        col_max = np.nanmax(matrix, 0, keepdims=True)
        norm    = (matrix - col_min) / (col_max - col_min + 1e-9)
        sns.heatmap(
            norm, ax=ax,
            xticklabels=[str(n) for n in SENSOR_COUNTS],
            yticklabels=agent_labels if ax == axes[0] else [],
            annot=np.round(matrix, 2), fmt=".2g",
            cmap="YlGnBu", vmin=0, vmax=1,
            linewidths=0.5, linecolor="white",
            cbar=True, annot_kws={"size": 8},
        )
        ax.set_title(ml, fontsize=11, fontweight="bold", pad=8)
        ax.set_xlabel("Sensor Count", fontsize=10)
        if ax == axes[0]:
            ax.set_ylabel("Agent", fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", rotation=0); ax.tick_params(axis="y", rotation=0)
    plt.tight_layout()
    out = OUTPUT_DIR / "fig12_sensor_heatmap.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print("Saved: {}".format(out.name)); plt.close()


def save_scalability_csv(results):
    rows = []
    for n in SENSOR_COUNTS:
        for agent, summs in results[n]["summaries"].items():
            rows.extend(summs)
    df = pd.DataFrame(rows)
    out = OUTPUT_DIR / "all_results_sensor_sweep.csv"
    df.to_csv(out, index=False)
    print("Saved: {}".format(out.name))
    return df


# ==================== GRID-SIZE SCALABILITY PLOTS (fig13-fig15) ====================


def plot_grid_scalability_metrics(grid_results):
    sns.set_theme(style="whitegrid", font_scale=1.3)
    plt.rcParams["font.family"] = "serif"
    metrics = [
        ("final_reward",      "Cumulative Reward",             "sci"),
        ("final_coverage",    "Final Coverage (%)",            "plain"),
        ("jains_index",       "Jain's Fairness Index",         "plain"),
        ("energy_efficiency", "Energy Efficiency (bytes/Wh)",  "sci"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(
        "Scalability Across Grid Sizes (LoRa SF Regimes)\n"
        "Grid unit=10m | UAV alt=100m | {} sensors | {} seeds".format(
            GRID_SWEEP_NUM_SENSORS, len(SEEDS)
        ),
        fontsize=16, fontweight="bold"
    )
    axes = axes.flatten()
    x = np.arange(len(GRID_SIZES))
    x_labels = [GRID_PHYSICS[g]["label"] for g in GRID_SIZES]
    band_colors = ["#E8F5E9", "#FFFDE7", "#FFF3E0", "#FFEBEE"]

    for ax, (mk, ml, fmt) in zip(axes, metrics):
        for agent, style in AGENT_STYLES.items():
            means, stds = [], []
            for gs in GRID_SIZES:
                summs = grid_results[gs]["summaries"].get(agent, [])
                if not summs:
                    means.append(np.nan); stds.append(np.nan); continue
                vals = [s[mk] for s in summs]
                means.append(np.mean(vals)); stds.append(np.std(vals))
            means = np.array(means, dtype=float)
            stds  = np.array(stds,  dtype=float)
            ax.plot(x, means, color=style["color"],
                    linestyle=style["linestyle"], marker=style["marker"],
                    markersize=9, linewidth=2.5, label=style["label"], zorder=4)
            ax.fill_between(x, means - stds, means + stds,
                            color=style["color"], alpha=0.15, zorder=3)

        # SF regime background bands
        for i, (bc, gs) in enumerate(zip(band_colors, GRID_SIZES)):
            ax.axvspan(i - 0.45, i + 0.45, color=bc, alpha=0.45, zorder=1)
            ax.text(i, ax.get_ylim()[0], GRID_PHYSICS[gs]["sf"],
                    ha="center", va="bottom", fontsize=7, color="grey", style="italic")

        ax.set_xticks(x); ax.set_xticklabels(x_labels, fontsize=11)
        ax.set_xlabel("Grid Size  (SF regime)", fontsize=13, fontweight="bold")
        ax.set_ylabel(ml, fontsize=13, fontweight="bold")
        ax.set_title(ml, fontsize=13, fontweight="bold", pad=8)
        ax.tick_params(axis="y", labelsize=12)
        ax.legend(fontsize=11); ax.grid(True, alpha=0.35, linestyle="--")
        if fmt == "sci":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    plt.tight_layout()
    out = OUTPUT_DIR / "fig13_grid_scalability.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print("\nSaved: {}".format(out.name)); plt.close()


def plot_grid_dqn_advantage(grid_results):
    sns.set_theme(style="whitegrid", font_scale=1.3)
    plt.rcParams["font.family"] = "serif"
    metrics = [
        ("final_reward",   "Reward Advantage"),
        ("final_coverage", "Coverage Advantage (pp)"),
        ("jains_index",    "Fairness Advantage (Jain's)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.5))
    fig.suptitle(
        "DQN Advantage Over Best Greedy vs Grid Size\n"
        "SF regime from iot_sensors.py (grid unit=10m, UAV alt=100m)",
        fontsize=15, fontweight="bold"
    )
    x = np.arange(len(GRID_SIZES))
    x_labels = [GRID_PHYSICS[g]["label"] for g in GRID_SIZES]
    sf_labels = [GRID_PHYSICS[g]["sf"] for g in GRID_SIZES]

    for ax, (mk, ml) in zip(axes, metrics):
        gaps, errs, bar_colors = [], [], []
        for gs in GRID_SIZES:
            dqn_s = grid_results[gs]["summaries"].get("DQN", [])
            if not dqn_s:
                gaps.append(0); errs.append(0); bar_colors.append("#AAAAAA"); continue
            dqn_mean = np.mean([s[mk] for s in dqn_s])
            best_g   = max(
                np.mean([s[mk] for s in grid_results[gs]["summaries"].get(a, dqn_s)])
                for a in ["SF-Aware Greedy V2", "Nearest Sensor Greedy"]
            )
            gap = dqn_mean - best_g
            gaps.append(gap); errs.append(np.std([s[mk] for s in dqn_s]))
            bar_colors.append("#1565C0" if gap >= 0 else "#C62828")

        ax.bar(x, gaps, color=bar_colors, alpha=0.75, edgecolor="white",
               linewidth=1.2, width=0.55, zorder=3)
        ax.errorbar(x, gaps, yerr=errs, fmt="none", color="black",
                    capsize=5, capthick=1.5, elinewidth=1.5, zorder=4)
        ax.axhline(0, color="black", linewidth=1.2, linestyle="--", alpha=0.5)

        for i, (gap, sl) in enumerate(zip(gaps, sf_labels)):
            yoff = errs[i] * 0.15
            va   = "bottom" if gap >= 0 else "top"
            ax.text(i, gap + (yoff if gap >= 0 else -yoff),
                    sl, ha="center", va=va, fontsize=7.5, color="grey", style="italic")

        ax.set_xticks(x); ax.set_xticklabels(x_labels, fontsize=11)
        ax.set_xlabel("Grid Size", fontsize=13, fontweight="bold")
        ax.set_ylabel("Delta {}".format(ml), fontsize=13, fontweight="bold")
        ax.set_title(ml, fontsize=13, fontweight="bold", pad=8)
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(axis="y", alpha=0.4, linestyle="--")
        if mk == "final_reward":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    plt.tight_layout()
    out = OUTPUT_DIR / "fig14_grid_dqn_advantage.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print("Saved: {}".format(out.name)); plt.close()


def plot_grid_heatmap(grid_results):
    sns.set_theme(style="white", font_scale=1.05)
    plt.rcParams["font.family"] = "serif"
    metrics = [
        ("final_reward",      "Cumulative\nReward"),
        ("final_coverage",    "Coverage\n(%)"),
        ("jains_index",       "Jain's\nFairness"),
        ("energy_efficiency", "Energy Eff.\n(B/Wh)"),
    ]
    agent_list   = list(AGENT_STYLES.keys())
    agent_labels = ["DQN (Proposed)", "SF-Aware Greedy V2", "Nearest Sensor Greedy"]
    col_labels   = ["{}x{}\n({})".format(gs[0], gs[1], GRID_PHYSICS[gs]["sf"])
                    for gs in GRID_SIZES]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    fig.suptitle(
        "Grid-Size Heatmap | {} sensors | {} seeds | grid unit=10m, UAV alt=100m".format(
            GRID_SWEEP_NUM_SENSORS, len(SEEDS)
        ),
        fontsize=12, fontweight="bold"
    )
    for ax, (mk, ml) in zip(axes, metrics):
        matrix = np.array([
            [np.mean([s[mk] for s in grid_results[gs]["summaries"].get(a, [])]) if grid_results[gs]["summaries"].get(a) else np.nan
             for gs in GRID_SIZES]
            for a in agent_list
        ])
        col_min = np.nanmin(matrix, 0, keepdims=True)
        col_max = np.nanmax(matrix, 0, keepdims=True)
        norm    = (matrix - col_min) / (col_max - col_min + 1e-9)
        sns.heatmap(
            norm, ax=ax,
            xticklabels=col_labels,
            yticklabels=agent_labels if ax == axes[0] else [],
            annot=np.round(matrix, 2), fmt=".2g",
            cmap="YlGnBu", vmin=0, vmax=1,
            linewidths=0.5, linecolor="white",
            cbar=True, annot_kws={"size": 8},
        )
        ax.set_title(ml, fontsize=10, fontweight="bold", pad=8)
        ax.set_xlabel("Grid Size (SF regime)", fontsize=9)
        if ax == axes[0]:
            ax.set_ylabel("Agent", fontsize=10, fontweight="bold")
        ax.tick_params(axis="x", rotation=0, labelsize=8)
        ax.tick_params(axis="y", rotation=0)

    plt.tight_layout()
    out = OUTPUT_DIR / "fig15_grid_heatmap.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print("Saved: {}".format(out.name)); plt.close()


def save_grid_csv(grid_results):
    rows = []
    for gs in GRID_SIZES:
        for agent, summs in grid_results[gs]["summaries"].items():
            rows.extend(summs)
    df = pd.DataFrame(rows)
    out = OUTPUT_DIR / "all_results_grid_sweep.csv"
    df.to_csv(out, index=False)
    print("Saved: {}".format(out.name))
    return df


# ==================== MAIN ====================


def main():
    print("=" * 70)
    print("MULTI-SEED x MULTI-SENSOR x MULTI-GRID EVALUATION")
    print("Seeds:         {}".format(SEEDS))
    print("Sensor counts: {}  (Sweep A, grid=500x500)".format(SENSOR_COUNTS))
    print("Grid sizes:    {}  (Sweep B, sensors={})".format(
        ["{}x{}".format(g[0], g[1]) for g in GRID_SIZES], GRID_SWEEP_NUM_SENSORS
    ))
    print("Episodes/agent: {} (A) + {} (B)".format(
        len(SEEDS) * len(SENSOR_COUNTS), len(SEEDS) * len(GRID_SIZES)
    ))
    print("Output: {}".format(OUTPUT_DIR))
    print("=" * 70)

    dqn_model       = None
    training_config = {"use_frame_stacking": True, "n_stack": 4, "max_sensors_limit": 50}

    if DQN_MODEL_PATH.exists():
        print("\nLoading DQN model from {}...".format(DQN_MODEL_PATH))
        dqn_model       = DQN.load(DQN_MODEL_PATH)
        training_config = load_training_config(DQN_CONFIG_PATH)
        max_limit       = training_config.get("max_sensors_limit", 50)
        print("  Loaded | n_stack={} | max_sensors_limit={}".format(
            training_config.get("n_stack", 4), max_limit
        ))
        if any(n > max_limit for n in SENSOR_COUNTS):
            bad = [n for n in SENSOR_COUNTS if n > max_limit]
            print("  WARNING: sensor counts {} exceed max_sensors_limit={}".format(bad, max_limit))
        if VEC_NORMALIZE_PATH.exists():
            print("  vec_normalize.pkl found (auto-skipped if shape changed)")
        else:
            print("  vec_normalize.pkl not found -- obs won't be normalised")
    else:
        print("  DQN model not found -- running greedy baselines only.")

    # ── Sweep A: Sensor count ───────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SWEEP A -- SENSOR COUNT  (grid=500x500)")
    print("=" * 70)
    t0 = time.time()
    results = run_all(dqn_model, training_config)
    print("Sweep A done: {:.1f} min".format((time.time()-t0)/60))

    print("\n" + "=" * 70)
    print("GENERATING PER-SENSOR-COUNT FIGURES")
    print("=" * 70)
    for n_sensors in SENSOR_COUNTS:
        sub_dir = OUTPUT_DIR / "sensors_{}".format(n_sensors)
        sub_dir.mkdir(exist_ok=True)
        histories = results[n_sensors]["histories"]
        summaries = results[n_sensors]["summaries"]
        interp    = interpolate_to_common_steps(histories)
        print("\n-- {} sensors -> {}/".format(n_sensors, sub_dir.name))
        plot_shaded_reward(interp, SEEDS, sub_dir)
        plot_shaded_coverage(interp, SEEDS, sub_dir)
        plot_summary_bars(summaries, SEEDS, sub_dir)
        plot_per_seed_variance(histories, sub_dir)
        plot_shaded_data_throughput(interp, SEEDS, sub_dir)
        plot_box_plots(summaries, SEEDS, sub_dir)
        plot_convergence_stability(interp, SEEDS, sub_dir)
        plot_jains_over_time(histories, summaries, SEEDS, sub_dir)
        run_statistical_tests(summaries, sub_dir)
        save_raw_data(summaries, SEEDS, sub_dir, "n{}".format(n_sensors))

    print("\n" + "=" * 70)
    print("SENSOR-COUNT SCALABILITY FIGURES (fig10-fig12)")
    print("=" * 70)
    plot_scalability_metrics(results)
    plot_dqn_advantage_gap(results)
    plot_scalability_heatmap(results)
    df_sensors = save_scalability_csv(results)

    # ── Sweep B: Grid size ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SWEEP B -- GRID SIZE  (sensors={})".format(GRID_SWEEP_NUM_SENSORS))
    print("=" * 70)
    t0 = time.time()
    grid_results = run_grid_sweep(dqn_model, training_config)
    print("Sweep B done: {:.1f} min".format((time.time()-t0)/60))

    print("\n" + "=" * 70)
    print("GENERATING PER-GRID-SIZE FIGURES")
    print("=" * 70)
    for gs in GRID_SIZES:
        sub_dir = OUTPUT_DIR / "grid_{}x{}".format(gs[0], gs[1])
        sub_dir.mkdir(exist_ok=True)
        histories = grid_results[gs]["histories"]
        summaries = grid_results[gs]["summaries"]
        interp    = interpolate_to_common_steps(histories)
        phys      = GRID_PHYSICS[gs]
        print("\n-- {}x{} ({}) -> {}/".format(gs[0], gs[1], phys["sf"], sub_dir.name))
        plot_shaded_reward(interp, SEEDS, sub_dir)
        plot_shaded_coverage(interp, SEEDS, sub_dir)
        plot_summary_bars(summaries, SEEDS, sub_dir)
        plot_per_seed_variance(histories, sub_dir)
        plot_shaded_data_throughput(interp, SEEDS, sub_dir)
        plot_box_plots(summaries, SEEDS, sub_dir)
        plot_convergence_stability(interp, SEEDS, sub_dir)
        plot_jains_over_time(histories, summaries, SEEDS, sub_dir)
        run_statistical_tests(summaries, sub_dir)
        save_raw_data(summaries, SEEDS, sub_dir, "{}x{}".format(gs[0], gs[1]))

    print("\n" + "=" * 70)
    print("GRID-SIZE SCALABILITY FIGURES (fig13-fig15)")
    print("=" * 70)
    plot_grid_scalability_metrics(grid_results)
    plot_grid_dqn_advantage(grid_results)
    plot_grid_heatmap(grid_results)
    df_grids = save_grid_csv(grid_results)

    # ── Summary tables ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("SWEEP A -- mean reward per agent x sensor count")
    print("=" * 70)
    print(df_sensors.pivot_table(
        index="agent", columns="num_sensors", values="final_reward", aggfunc="mean"
    ).to_string())

    print("\n" + "=" * 70)
    print("SWEEP B -- mean reward per agent x grid size")
    print("=" * 70)
    print(df_grids.pivot_table(
        index="agent", columns="grid_size", values="final_reward", aggfunc="mean"
    ).to_string())

    print("\n" + "=" * 70)
    print("DONE")
    print("  Sweep A:  {}/sensors_{{10,20,30,40}}/  +  fig10-fig12".format(OUTPUT_DIR))
    print("  Sweep B:  {}/grid_{{100x100,...}}/  +  fig13-fig15".format(OUTPUT_DIR))
    print("  CSVs: all_results_sensor_sweep.csv  |  all_results_grid_sweep.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()