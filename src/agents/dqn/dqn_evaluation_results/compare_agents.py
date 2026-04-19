"""
Dissertation Comparison Plotter - FAIR VERSION WITH FRAME STACKING + SENSOR SNAPSHOTS
Generates the final 'DQN vs. Greedy' performance graph with FRESH DQN evaluation.
All agents run on the EXACT SAME ENVIRONMENT with the same seed.
CRITICAL: DQN uses VecFrameStack to match training conditions.
NEW: Automatically saves sensor-level data for fairness & heatmap analysis.
NEW: Battery / Steps / Data collected panel plot added.
UPDATED: Zero-padding support so the trained model can run on any num_sensors ≤ max_sensors_limit.

Author: ATILADE GABRIEL OKE
Modified: 27 February 2026
"""

import sys
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gymnasium
import json
from pathlib import Path
import time
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize
import ieee_style
ieee_style.apply()

# ==================== CRITICAL FIX: Correct Import Paths ====================
script_dir = Path(__file__).resolve().parent  # dqn_evaluation_results/
src_dir = script_dir.parent.parent.parent  # Go up 3 levels to src
script_dir_results = Path(__file__).resolve().parent.parent  # dqn/

print(f"Script location: {script_dir}")
print(f"Source directory: {src_dir}")
print(f"Results directory: {script_dir_results}")

sys.path.insert(0, str(src_dir))

from environment.uav_env import UAVEnvironment
from greedy_agents import MaxThroughputGreedyV2, NearestSensorGreedy

# ==================== ANALYSIS ENV WRAPPER (WITH ZERO PADDING) ====================


class AnalysisUAVEnv(UAVEnvironment):
    """
    Smart wrapper that:
      1. Saves the final state of the simulation the instant before the env resets.
      2. Zero-pads the observation vector to a fixed length so the DQN neural
         network (trained with max_sensors_limit) can evaluate on any num_sensors
         that is ≤ max_sensors_limit without reloading or retraining.

    features_per_sensor is AUTO-DETECTED from the parent observation_space at
    init time — no hardcoded constants that can silently mismatch the env layout.
    """

    def __init__(self, max_sensors_limit: int = 50, **kwargs):
        self.max_sensors_limit = max_sensors_limit
        super().__init__(**kwargs)

        # ── Auto-detect features_per_sensor ───────────────────────────────
        raw_obs_size = self.observation_space.shape[0]
        self._features_per_sensor: int = 0
        for uav_f in range(raw_obs_size + 1):
            remainder = raw_obs_size - uav_f
            if remainder > 0 and remainder % self.num_sensors == 0:
                self._features_per_sensor = remainder // self.num_sensors
                break
        if self._features_per_sensor == 0:
            raise ValueError(
                f"[AnalysisUAVEnv] Cannot infer features_per_sensor: "
                f"raw obs {raw_obs_size} has no divisor matching num_sensors={self.num_sensors}."
            )

        padded_obs_size = (
            raw_obs_size
            + (self.max_sensors_limit - self.num_sensors) * self._features_per_sensor
        )
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(padded_obs_size,), dtype=np.float32,
        )

        self.last_episode_sensor_data = None
        self.last_episode_info = None

        print(
            f"  [AnalysisUAVEnv] raw={raw_obs_size} → padded={padded_obs_size} "
            f"(active={self.num_sensors}, max={self.max_sensors_limit}, "
            f"{self._features_per_sensor} features/sensor)"
        )

    # ── Padding helper ───────────────────────────────────────────────────────

    def _pad(self, raw: np.ndarray) -> np.ndarray:
        padding = np.zeros(
            (self.max_sensors_limit - self.num_sensors) * self._features_per_sensor,
            dtype=np.float32,
        )
        return np.concatenate([raw, padding]).astype(np.float32)

    # ── Data-persistence fix: snapshot before auto-reset ───────────────────

    def reset(self, **kwargs):
        if hasattr(self, "sensors") and self.current_step > 0:
            self.last_episode_sensor_data = [
                {
                    "sensor_id":              sensor.sensor_id,
                    "position":               tuple(sensor.position),
                    "total_data_generated":   float(sensor.total_data_generated),
                    "total_data_transmitted": float(sensor.total_data_transmitted),
                    "total_data_lost":        float(sensor.total_data_lost),
                    "data_buffer":            float(sensor.data_buffer),
                    "max_buffer_size":        float(sensor.max_buffer_size),
                }
                for sensor in self.sensors
            ]
            self.last_episode_info = {
                "battery":            self.uav.battery,
                "battery_percent":    self.uav.get_battery_percentage(),
                "coverage_percentage": (len(self.sensors_visited) / self.num_sensors) * 100,
            }

        obs, info = super().reset(**kwargs)
        return self._pad(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        return self._pad(obs), reward, terminated, truncated, info


# ==================== TRAJECTORY TRACKING ====================


class TrajectoryTracker:
    def __init__(self):
        self.positions = []

    def record(self, x, y):
        self.positions.append((x, y))

    def get_array(self):
        if not self.positions:
            return np.array([])
        return np.array(self.positions)


# ==================== SENSOR SNAPSHOT SAVING ====================


def convert_to_python_types(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, dict):
        return {key: convert_to_python_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_to_python_types(item) for item in obj]
    else:
        return obj


def save_sensor_snapshot(env, filepath, agent_name, is_wrapper=False,
                         override_sensor_data=None):
    """
    Save a sensor snapshot to JSON.

    override_sensor_data: if provided (list of dicts), use this directly instead
    of reading from env. Used by the DQN path to pass in the pre-step sensor
    snapshot captured before the VecEnv auto-reset fires (FIX 3).
    """
    print(f"\n{'=' * 80}")
    print(f"Saving sensor snapshot: {agent_name}")
    print(f"{'=' * 80}")

    if override_sensor_data is not None:
        sensor_data_list  = override_sensor_data
        total_generated   = sum(s["total_data_generated"]  for s in sensor_data_list)
        total_transmitted = sum(s["total_data_transmitted"] for s in sensor_data_list)
        total_lost        = sum(s["total_data_lost"]        for s in sensor_data_list)
        uav_info = {}
        print(f"✓ Using pre-step sensor snapshot (FIX 3 - DQN)")

    elif (
        is_wrapper
        and hasattr(env, "last_episode_sensor_data")
        and env.last_episode_sensor_data
    ):
        sensor_data_list  = env.last_episode_sensor_data
        uav_info          = env.last_episode_info if env.last_episode_info else {}
        total_generated   = sum(s["total_data_generated"]  for s in sensor_data_list)
        total_transmitted = sum(s["total_data_transmitted"] for s in sensor_data_list)
        total_lost        = sum(s["total_data_lost"]        for s in sensor_data_list)
        print(f"✓ Using preserved snapshot data")

    else:
        if not hasattr(env, "sensors"):
            print(f"⚠ WARNING: Environment has no sensors attribute")
            return None
        sensor_data_list = [
            {
                "sensor_id":              int(sensor.sensor_id),
                "position":               tuple(float(x) for x in sensor.position),
                "total_data_generated":   float(sensor.total_data_generated),
                "total_data_transmitted": float(sensor.total_data_transmitted),
                "total_data_lost":        float(sensor.total_data_lost),
                "data_buffer":            float(sensor.data_buffer),
            }
            for sensor in env.sensors
        ]
        total_generated   = sum(s["total_data_generated"]  for s in sensor_data_list)
        total_transmitted = sum(s["total_data_transmitted"] for s in sensor_data_list)
        total_lost        = sum(s["total_data_lost"]        for s in sensor_data_list)
        uav_info = {
            "battery":            float(env.uav.battery),
            "battery_percent":    float(env.uav.get_battery_percentage()),
            "coverage_percentage": float(
                (len(env.sensors_visited) / env.num_sensors) * 100
            ),
        }
        print(f"✓ Using current environment state")

    snapshot = {
        "timestamp":   time.strftime("%Y-%m-%d %H:%M:%S"),
        "agent_name":  agent_name,
        "environment_config": {
            "grid_size":   list(env.grid_size) if hasattr(env, "grid_size") else [100, 100],
            "num_sensors": int(len(sensor_data_list)),
            "max_steps":   int(env.max_steps) if hasattr(env, "max_steps") else 0,
        },
        "uav_state":    convert_to_python_types(uav_info),
        "mission_stats": {
            "total_data_collected":  float(total_transmitted),
            "total_data_lost":       float(total_lost),
            "sensors_visited":       int(len([s for s in sensor_data_list if s["total_data_transmitted"] > 0])),
            "coverage_percentage":   float(
                len([s for s in sensor_data_list if s["total_data_transmitted"] > 0])
                / len(sensor_data_list) * 100
            ),
        },
        "sensor_data": [],
    }

    for sensor_data in sensor_data_list:
        generated   = float(sensor_data["total_data_generated"])
        transmitted = float(sensor_data["total_data_transmitted"])
        lost        = float(sensor_data["total_data_lost"])
        max_buf = float(sensor_data.get("max_buffer_size", 1000.0))
        snapshot["sensor_data"].append({
            "sensor_id":              int(sensor_data["sensor_id"]),
            "position":               [float(x) for x in sensor_data["position"]],
            "total_data_generated":   generated,
            "total_data_transmitted": transmitted,
            "total_data_lost":        lost,
            "data_buffer":            float(sensor_data["data_buffer"]),
            "buffer_occupancy":       float(sensor_data["data_buffer"]) / max_buf,
            "collection_rate": float((transmitted / generated * 100) if generated > 0 else 0.0),
            "loss_rate":       float((lost / generated * 100)        if generated > 0 else 0.0),
        })

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w") as f:
        json.dump(snapshot, f, indent=2)

    print(f"✓ Sensor snapshot saved: {filepath.name}")
    print(f"  • Sensors: {len(snapshot['sensor_data'])}")
    print(f"  • Data collected: {snapshot['mission_stats']['total_data_collected']:.0f} bytes")
    print(f"  • NDR: {snapshot['mission_stats']['coverage_percentage']:.1f}%")

    collection_rates = [s["collection_rate"] for s in snapshot["sensor_data"]]
    n = len(collection_rates)
    s2 = sum(x**2 for x in collection_rates)
    jains_index = (sum(collection_rates)**2 / (n * s2)) if n > 0 and s2 > 0 else 0
    starved_count = sum(1 for rate in collection_rates if rate < 20.0)
    min_rate  = min(collection_rates) if collection_rates else 0.0
    buf_levels = [s.get("buffer_occupancy", 0.0) for s in snapshot["sensor_data"]]
    peak_aoi  = max(buf_levels) if buf_levels else 0.0
    print(f"  • Jain's Fairness Index: {jains_index:.4f}")
    print(f"  • Min collection rate (Max-Min): {min_rate:.1f}%")
    print(f"  • Peak AoI proxy (max buffer occ.): {peak_aoi:.3f}")
    print(f"  • Starved sensors (<20%): {starved_count}/{n}")
    print(f"{'=' * 80}")
    return snapshot


# ==================== CONFIGURATION ====================
BASELINES_DIR = src_dir / "agents" / "baselines"
OUTPUT_DIR = script_dir / "baseline_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DQN_MODEL_PATH = (
    script_dir.parent.parent.parent.parent / "models" / "dqn_v4a" / "dqn_final.zip"
)
DQN_CONFIG_PATH = (
    script_dir.parent.parent.parent.parent / "models" / "dqn_v4a" / "training_config.json"
)
VEC_NORMALIZE_PATH = (
    script_dir.parent.parent.parent.parent / "models" / "dqn_v4a" / "vec_normalize.pkl"
)

PLOT_CONFIG = {
    "grid_size":          (500, 500),
    "num_sensors":        20,
    "max_steps":          2100,
    "path_loss_exponent": 3.8,
    "rssi_threshold":     -85.0,
    "sensor_duty_cycle":  10.0,
    "seed":               42,
}

EVAL_MAX_BATTERY = 274.0

AGENT_STYLES = {
    "DQN Agent":       {"color": ieee_style.AGENT_COLORS["DQN Agent"],       "marker": "o"},
    "Smart Greedy V2": {"color": ieee_style.AGENT_COLORS["Smart Greedy V2"], "marker": "s"},
    "Nearest Greedy":  {"color": ieee_style.AGENT_COLORS["Nearest Greedy"],  "marker": "^"},
}

print(f"Output directory: {OUTPUT_DIR}")
print(f"DQN Model path: {DQN_MODEL_PATH}")
print()

# ==================== HELPER FUNCTIONS ====================


def load_training_config(config_path):
    """
    Load training_config.json.
    Falls back gracefully if the file is the old frame_stacking_config.json format
    (missing max_sensors_limit) so existing model dirs still work.
    """
    defaults = {
        "use_frame_stacking": True,
        "n_stack": 4,
        "max_sensors_limit": 50,
        # features_per_sensor is auto-detected at runtime from the env obs space;
        # it is NOT stored as a class constant any more. The value in the JSON
        # (written by train_dqn_zero_padding.py) is used when available; the
        # default here is only a placeholder and is never read by AnalysisUAVEnv.
        "features_per_sensor": None,
    }
    try:
        with open(config_path, "r") as f:
            loaded = json.load(f)
        # Merge: loaded values override defaults
        return {**defaults, **loaded}
    except FileNotFoundError:
        print(f"⚠ Config not found at {config_path} — using defaults")
        return defaults


def _unwrap_base_env(vec):
    inner = vec
    while hasattr(inner, 'venv'):
        inner = inner.venv
    env = inner.envs[0]
    while hasattr(env, 'env'):
        env = env.env
    return env


def _seed_base_env_directly(base_env, seed):
    try:
        import gymnasium
        base_env.np_random, _ = gymnasium.utils.seeding.np_random(seed)
        print(f"  ✓ Seeded via gymnasium.utils.seeding (seed={seed})")
        np.random.seed(seed)
        random.seed(seed)
        return
    except Exception:
        pass
    try:
        import gym
        base_env.np_random, _ = gym.utils.seeding.np_random(seed)
        print(f"  ✓ Seeded via gym.utils.seeding (seed={seed})")
        np.random.seed(seed)
        random.seed(seed)
        return
    except Exception:
        pass
    base_env.np_random = np.random.RandomState(seed)
    print(f"  ✓ Seeded via np.random.RandomState fallback (seed={seed})")
    np.random.seed(seed)
    random.seed(seed)


def create_stacked_dqn_env(env_kwargs, training_config, seed=PLOT_CONFIG["seed"]):
    """
    Build the stacked/normalised VecEnv for DQN evaluation.
    env_kwargs must include max_sensors_limit so AnalysisUAVEnv can pad correctly.
    """
    # Seed global RNG before env construction so sensor positions match greedy envs
    np.random.seed(seed)
    random.seed(seed)
    vec_env = DummyVecEnv([lambda: AnalysisUAVEnv(**env_kwargs)])

    if training_config.get("use_frame_stacking", True):
        n_stack = training_config.get("n_stack", 4)
        vec_env = VecFrameStack(vec_env, n_stack=n_stack)
        print(f"✓ Frame stacking enabled (n_stack={n_stack})")
    else:
        print(f"✓ Frame stacking disabled")

    if VEC_NORMALIZE_PATH.exists():
        # Guard against stale vec_normalize.pkl from a previous training run whose
        # obs shape no longer matches (e.g. saved before zero-padding was added).
        # If the shapes disagree we skip normalisation rather than crashing.
        try:
            vec_env = VecNormalize.load(str(VEC_NORMALIZE_PATH), vec_env)
            vec_env.training = False
            vec_env.norm_reward = False
            print(f"✓ VecNormalize loaded from {VEC_NORMALIZE_PATH.name}")
        except AssertionError as e:
            print(
                f"⚠ vec_normalize.pkl shape mismatch — skipping normalisation.\n"
                f"  (The .pkl was saved with a different obs size: {e})\n"
                f"  Delete {VEC_NORMALIZE_PATH.name} if it belongs to an old run."
            )
    else:
        print(f"⚠ vec_normalize.pkl not found — observations will NOT be normalised")

    true_base_env = _unwrap_base_env(vec_env)
    print(f"  Seeding DQN base env directly (seed={seed})...")
    _seed_base_env_directly(true_base_env, seed)
    print(
        f"  [DEBUG] base_env type : {type(true_base_env).__name__}"
        f" | battery={true_base_env.uav.battery:.1f}"
    )
    return vec_env, true_base_env


def save_baseline_data(agent_name, history_df):
    output_file = OUTPUT_DIR / f"{agent_name}_results.csv"
    history_df.to_csv(output_file, index=False)
    print(f"✓ Saved {agent_name} data to {output_file}")
    return output_file


def save_comparison_metadata(agents_config):
    metadata = {
        "timestamp":        time.strftime("%Y-%m-%d %H:%M:%S"),
        "grid_size":        PLOT_CONFIG["grid_size"],
        "num_sensors":      PLOT_CONFIG["num_sensors"],
        "max_steps":        PLOT_CONFIG["max_steps"],
        "seed":             PLOT_CONFIG["seed"],
        "agents_evaluated": agents_config,
        "output_directory": str(OUTPUT_DIR),
        "note":             "All agents evaluated on identical environment with same seed",
    }
    metadata_file = OUTPUT_DIR / "comparison_metadata.json"
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=4)
    print(f"✓ Saved metadata to {metadata_file}")


# ==================== AGENT EVALUATION ====================


def run_greedy_agent_for_plot(agent, env, name="Agent", seed=PLOT_CONFIG["seed"]):
    print(f"\nRunning {name}...")
    obs, info = env.reset(seed=seed)
    done = False
    trajectory = TrajectoryTracker()

    history = {
        "step": [], "cumulative_reward": [], "battery_percent": [],
        "battery_wh": [], "coverage_percent": [], "sensors_visited": [],
        "total_data_collected": [], "efficiency": [],
    }

    cumulative_reward = 0
    step_count = 0

    while not done:
        action = agent.select_action(obs)
        obs, reward, done, truncated, info = env.step(action)
        cumulative_reward += reward
        step_count += 1
        trajectory.record(env.uav.position[0], env.uav.position[1])

        if env.current_step % 50 == 0 or done or truncated:
            battery_pct  = env.uav.get_battery_percentage()
            coverage_pct = (
                (len(env.sensors_visited) / env.num_sensors) * 100
                if hasattr(env, "sensors_visited") else 0
            )
            energy_consumed = EVAL_MAX_BATTERY - env.uav.battery
            efficiency = (env.total_data_collected / energy_consumed) if energy_consumed > 0 else 0
            history["step"].append(env.current_step)
            history["cumulative_reward"].append(cumulative_reward)
            history["battery_percent"].append(battery_pct)
            history["battery_wh"].append(env.uav.battery)
            history["coverage_percent"].append(coverage_pct)
            history["sensors_visited"].append(len(env.sensors_visited))
            history["total_data_collected"].append(env.total_data_collected)
            history["efficiency"].append(efficiency)
            print(
                f"  Step {env.current_step:>4}: Reward={cumulative_reward:>10.1f}, "
                f"Battery={battery_pct:>5.1f}%, NDR={coverage_pct:>5.1f}%, "
                f"Data={env.total_data_collected:>8.0f}bytes"
            )

        if done or truncated:
            break

    return pd.DataFrame(history), step_count, trajectory.get_array()


def run_dqn_agent_for_plot(
    model, stacked_env, base_env, name="DQN Agent", seed=PLOT_CONFIG["seed"]
):
    """
    FIX 3: Captures a full sensor snapshot on every step BEFORE calling
    stacked_env.step(). The VecEnv auto-reset fires inside step() the instant
    done=True, replacing base_env.sensors before any post-step code runs.
    """
    print(f"\nRunning {name}...")
    obs = stacked_env.reset()
    trajectory = TrajectoryTracker()

    history = {
        "step": [], "cumulative_reward": [], "battery_percent": [],
        "battery_wh": [], "coverage_percent": [], "sensors_visited": [],
        "total_data_collected": [], "efficiency": [],
    }

    cumulative_reward     = 0
    step_count            = 0
    final_sensor_snapshot = None

    while True:
        action, _ = model.predict(obs, deterministic=True)
        action = (
            int(action[0]) if isinstance(action, np.ndarray) and action.ndim > 0
            else int(action)
        )

        # Pre-step snapshot of UAV state
        pre_battery   = base_env.uav.battery
        pre_coverage  = (len(base_env.sensors_visited) / base_env.num_sensors) * 100 \
                        if hasattr(base_env, "sensors_visited") else 0.0
        pre_data      = base_env.total_data_collected
        pre_n_visited = len(base_env.sensors_visited) \
                        if hasattr(base_env, "sensors_visited") else 0
        pre_pos       = tuple(base_env.uav.position)

        # FIX 3: Pre-step sensor snapshot (before VecEnv auto-reset fires)
        pre_sensor_snapshot = [
            {
                "sensor_id":              int(s.sensor_id),
                "position":               [float(x) for x in s.position],
                "total_data_generated":   float(s.total_data_generated),
                "total_data_transmitted": float(s.total_data_transmitted),
                "total_data_lost":        float(s.total_data_lost),
                "data_buffer":            float(s.data_buffer),
            }
            for s in base_env.sensors
        ]

        obs, rewards, dones, infos = stacked_env.step([action])
        reward = float(rewards[0]) if isinstance(rewards, np.ndarray) else float(rewards)
        done   = bool(dones[0])    if isinstance(dones,   np.ndarray) else bool(dones)

        cumulative_reward += reward
        step_count += 1
        trajectory.record(pre_pos[0], pre_pos[1])

        if done:
            final_sensor_snapshot = pre_sensor_snapshot
            energy_consumed = EVAL_MAX_BATTERY - pre_battery
            efficiency      = (pre_data / energy_consumed) if energy_consumed > 0 else 0.0

            history["step"].append(step_count)
            history["cumulative_reward"].append(cumulative_reward)
            history["battery_percent"].append((pre_battery / EVAL_MAX_BATTERY) * 100)
            history["battery_wh"].append(pre_battery)
            history["coverage_percent"].append(pre_coverage)
            history["sensors_visited"].append(pre_n_visited)
            history["total_data_collected"].append(pre_data)
            history["efficiency"].append(efficiency)

            print(
                f"  Step {step_count:>4}: Reward={cumulative_reward:>10.1f}, "
                f"Battery={(pre_battery/EVAL_MAX_BATTERY*100):>5.1f}%, "
                f"NDR={pre_coverage:>5.1f}%, "
                f"Data={pre_data:>8.0f}bytes  [FINAL - pre-reset snapshot]"
            )
            break

        elif step_count % 50 == 0:
            battery_pct     = base_env.uav.get_battery_percentage()
            coverage_pct    = (len(base_env.sensors_visited) / base_env.num_sensors) * 100 \
                              if hasattr(base_env, "sensors_visited") else 0
            energy_consumed = EVAL_MAX_BATTERY - base_env.uav.battery
            efficiency      = (base_env.total_data_collected / energy_consumed) \
                              if energy_consumed > 0 else 0.0

            history["step"].append(step_count)
            history["cumulative_reward"].append(cumulative_reward)
            history["battery_percent"].append(battery_pct)
            history["battery_wh"].append(base_env.uav.battery)
            history["coverage_percent"].append(coverage_pct)
            history["sensors_visited"].append(len(base_env.sensors_visited))
            history["total_data_collected"].append(base_env.total_data_collected)
            history["efficiency"].append(efficiency)

            print(
                f"  Step {step_count:>4}: Reward={cumulative_reward:>10.1f}, "
                f"Battery={battery_pct:>5.1f}%, NDR={coverage_pct:>5.1f}%, "
                f"Data={base_env.total_data_collected:>8.0f}bytes"
            )

    print(f"\n✓ Final sensor snapshot captured: {len(final_sensor_snapshot)} sensors")
    return pd.DataFrame(history), step_count, trajectory.get_array(), final_sensor_snapshot


# ==================== PLOTTING ====================


def plot_trajectories(env, dqn_trajectory, greedy_smart_trajectory, greedy_dumb_trajectory):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    sensor_positions = np.array([sensor.position for sensor in env.sensors])
    grid_size = PLOT_CONFIG["grid_size"][0]

    trajectories = [
        (dqn_trajectory, "DQN Agent (Proposed)", "#1b9e77", axes[0]),
        (greedy_smart_trajectory, "SF-Aware Greedy V2", "#d95f02", axes[1]),
        (greedy_dumb_trajectory, "Nearest Sensor Greedy", "#7570b3", axes[2]),
    ]

    for trajectory, title, color, ax in trajectories:
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_aspect("equal")
        ax.scatter(
            sensor_positions[:, 0], sensor_positions[:, 1],
            s=100, c="#d95f02", marker="s", edgecolors="#a03a00",
            linewidth=1.0, label="Sensor Locations", zorder=3,
        )
        for i, pos in enumerate(sensor_positions):
            ax.annotate(
                f"S{i}", (pos[0], pos[1]), fontsize=7,
                bbox=dict(boxstyle="round,pad=0.2", fc="lightyellow", alpha=0.7),
                xytext=(4, 4), textcoords="offset points",
            )

        if trajectory is not None and len(trajectory) > 0:
            traj_array = (
                trajectory if isinstance(trajectory, np.ndarray) else np.array(trajectory)
            )
            if len(traj_array) > 0:
                # Step drawstyle shows discrete grid movements
                ax.step(traj_array[:, 0], traj_array[:, 1], where="post",
                        color=color, linewidth=1.2, alpha=0.65,
                        label="UAV Path", zorder=2)
                # Waypoint dots — hover points at low alpha
                ax.scatter(traj_array[:, 0], traj_array[:, 1],
                           c=color, s=8, alpha=0.30, zorder=1)
                ax.scatter(traj_array[0, 0], traj_array[0, 1], c="#2ca02c", s=150,
                           marker="^", edgecolors="darkgreen", linewidth=1.2,
                           label="Start", zorder=4)
                ax.scatter(traj_array[-1, 0], traj_array[-1, 1], c="#1b9e77", s=150,
                           marker="*", edgecolors="#0d5c44", linewidth=1.2,
                           label="End", zorder=4)
                total_distance = np.sum(
                    np.sqrt(np.sum(np.diff(traj_array, axis=0) ** 2, axis=1))
                )
                ax.text(0.02, 0.98,
                        f"Path: {total_distance:.0f} m\nWaypoints: {len(traj_array)}",
                        transform=ax.transAxes, fontsize=8, verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="white",
                                  edgecolor="#CCCCCC", alpha=0.85))

        ax.set_xlabel("$x$ Position (m)", fontweight="bold")
        ax.set_ylabel("$y$ Position (m)", fontweight="bold")
        ax.set_title(title, fontweight="bold")
        ax.legend(loc="lower right", fontsize=8)
        ieee_style.clean_axes(ax)

    plt.tight_layout()
    output_file = OUTPUT_DIR / "agent_trajectories"
    ieee_style.save(fig, str(output_file))
    print(f"  Trajectory plots saved to {output_file}.pdf / .eps")
    plt.close()


def plot_comparative_analysis(dqn_df, greedy_smart_df, greedy_dumb_df):
    fig, ax1 = plt.subplots(figsize=(12, 5.5))

    ax1.set_xlabel("Simulation Step (t)", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Cumulative Reward", fontsize=12, fontweight="bold", color="black")
    ax1.tick_params(axis="y", labelcolor="black")

    if dqn_df is not None and not dqn_df.empty:
        ax1.plot(dqn_df["step"], dqn_df["cumulative_reward"],
                 color="#1b9e77", linewidth=3, label="DQN Agent (Proposed)",
                 marker="o", markersize=5, linestyle="-")
        dqn_sat = dqn_df[dqn_df["battery_percent"] < 30]
        if not dqn_sat.empty:
            sat_idx = dqn_sat.index[0]
            ax1.scatter(dqn_df.loc[sat_idx, "step"],
                        dqn_df.loc[sat_idx, "cumulative_reward"],
                        color="#1b9e77", s=200, zorder=5, marker="*")

    ax1.plot(greedy_smart_df["step"], greedy_smart_df["cumulative_reward"],
             color="#d95f02", linewidth=2.5, linestyle="--",
             label="SF-Aware Greedy (V2)", marker="s", markersize=4)
    ax1.plot(greedy_dumb_df["step"], greedy_dumb_df["cumulative_reward"],
             color="#7570b3", linewidth=2, linestyle=":", alpha=0.8,
             label="Nearest Sensor Greedy", marker="^", markersize=4)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Battery Level (%)", fontsize=12, fontweight="bold", color="black")
    ax2.tick_params(axis="y", labelcolor="black")
    ax2.plot(greedy_smart_df["step"], greedy_smart_df["battery_percent"],
             color="black", linewidth=1.5, linestyle="--", alpha=0.4,
             label="Battery Reference")
    ax2.set_ylim(0, 105)

    sat_smart = greedy_smart_df[greedy_smart_df["battery_percent"] < 30].head(1)
    if not sat_smart.empty:
        step_val   = sat_smart["step"].values[0]
        reward_val = sat_smart["cumulative_reward"].values[0]
        ax1.annotate(
            f"Greedy Saturation\n(t={int(step_val)})",
            xy=(step_val, reward_val),
            xytext=(step_val - 300, reward_val + 500000),
            arrowprops=dict(facecolor="#d95f02", shrink=0.05, width=2),
            fontsize=10, fontweight="bold", color="#d95f02",
            bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#d95f02", alpha=0.9),
        )

    ax1.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=9, ncol=2)

    ax1.set_title("Comparative Performance: DRL vs. Heuristics",
                  fontweight="bold", pad=10)
    ieee_style.clean_axes(ax1)
    ax2.spines["top"].set_visible(False)
    ax2.grid(False)

    plt.tight_layout()
    output_file = OUTPUT_DIR / "final_comparison_graph"
    ieee_style.save(fig, str(output_file))
    print(f"\n  Graph saved: {output_file}.pdf / .eps")
    plt.close()


def plot_efficiency_table(dqn_df, smart_df, dumb_df):
    AGENT_COLORS = {
        "DQN Agent":       "#1b9e77",
        "Smart Greedy V2": "#d95f02",
        "Nearest Greedy":  "#7570b3",
    }

    def get_stats(df, name):
        if df is None or df.empty:
            return name, 0.0, 0, 0.0, 0.0, 0.0
        energy    = EVAL_MAX_BATTERY - df["battery_wh"].iloc[-1]
        data      = df["total_data_collected"].iloc[-1]
        final_eff = data / energy if energy > 0 else 0.0
        avg_eff   = df["efficiency"].mean() if "efficiency" in df.columns else 0.0
        peak_eff  = df["efficiency"].max()  if "efficiency" in df.columns else 0.0
        return name, energy, int(data), final_eff, avg_eff, peak_eff

    rows_raw = [
        get_stats(dqn_df,   "DQN Agent"),
        get_stats(smart_df, "Smart Greedy V2"),
        get_stats(dumb_df,  "Nearest Greedy"),
    ]
    columns = [
        "Agent", "Final Energy\nConsumed (Wh)", "Final Data\nCollected (Bytes)",
        "Final Efficiency\n(Bytes/Wh)", "Average Efficiency\n(Bytes/Wh)",
        "Peak Efficiency\n(Bytes/Wh)",
    ]
    rows_display = [
        [n, f"{e:.2f}", f"{d}", f"{fe:.2f}", f"{ae:.2f}", f"{pe:.2f}"]
        for n, e, d, fe, ae, pe in rows_raw
    ]
    best_higher = [True, False, True, True, True, True]
    best_idx = {}
    for col_i in range(1, len(columns)):
        vals = [rows_raw[r][col_i] for r in range(len(rows_raw))]
        best_idx[col_i] = int(np.argmax(vals)) if best_higher[col_i] else int(np.argmin(vals))

    fig, ax = plt.subplots(figsize=(13, 3.2))
    ax.axis("off")
    tbl = ax.table(cellText=rows_display, colLabels=columns, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 2.2)

    for j in range(len(columns)):
        cell = tbl[0, j]
        cell.set_facecolor("#1b9e77")
        cell.set_text_props(color="white", fontweight="bold", fontsize=9)
        cell.set_edgecolor("white")

    for i, (name, *_) in enumerate(rows_raw):
        row_bg = "white" if i % 2 == 0 else "#F5F5F5"
        for j in range(len(columns)):
            cell = tbl[i + 1, j]
            cell.set_facecolor(row_bg)
            cell.set_edgecolor("#DDDDDD")
            if j == 0:
                cell.set_text_props(color=AGENT_COLORS.get(name, "black"), fontweight="bold")
        for col_i, best_row in best_idx.items():
            if best_row == i:
                cell = tbl[i + 1, col_i]
                cell.set_facecolor("#A5D6A7")
                cell.set_text_props(fontweight="bold")

    ax.set_title("Efficiency Metrics Summary", fontweight="bold", pad=14)
    plt.tight_layout()
    ieee_style.save(fig, str(OUTPUT_DIR / "efficiency_metrics_table"))
    print(f"  Saved: efficiency_metrics_table.pdf / .eps")
    plt.close()


def plot_fairness_heatmap(dqn_snapshot_path, smart_snapshot_path, dumb_snapshot_path):
    paths = {
        "DQN Agent":       dqn_snapshot_path,
        "Smart Greedy V2": smart_snapshot_path,
        "Nearest Greedy":  dumb_snapshot_path,
    }
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Spatial Fairness Heatmap — Collection Rate per Sensor (%)",
                 fontsize=13, fontweight="bold", y=1.02)

    for ax, (agent_name, path) in zip(axes, paths.items()):
        if not Path(path).exists():
            ax.text(0.5, 0.5, f"No snapshot\n{agent_name}", ha="center", va="center",
                    transform=ax.transAxes, fontsize=11, color="gray")
            ax.set_title(agent_name, fontsize=11, fontweight="bold")
            continue
        with open(path) as f:
            data = json.load(f)
        sensors = data["sensor_data"]
        x     = [s["position"][0] for s in sensors]
        y_pos = [s["position"][1] for s in sensors]
        rates = [s["collection_rate"] for s in sensors]
        sc = ax.scatter(x, y_pos, c=rates, cmap="RdYlGn", s=250,
                        edgecolors="black", linewidths=1.2, vmin=0, vmax=100, zorder=3)
        plt.colorbar(sc, ax=ax, label="Collection Rate (%)", fraction=0.046, pad=0.04)
        for s in sensors:
            ax.annotate(f"{s['collection_rate']:.0f}%",
                        (s["position"][0], s["position"][1]),
                        textcoords="offset points", xytext=(0, 8),
                        fontsize=6.5, ha="center", color="black")
        n  = len(rates)
        s2 = sum(r**2 for r in rates)
        jains   = (sum(rates)**2 / (n * s2)) if n > 0 and s2 > 0 else 0
        starved = sum(1 for r in rates if r < 20)
        ax.set_xlim(0, PLOT_CONFIG["grid_size"][0])
        ax.set_ylim(0, PLOT_CONFIG["grid_size"][1])
        ax.set_xlabel("X (m)", fontsize=10)
        ax.set_ylabel("Y (m)", fontsize=10)
        ax.set_title(agent_name, fontsize=11, fontweight="bold",
                     color=AGENT_STYLES[agent_name]["color"])
        ax.text(0.02, 0.98, f"Jain's Index: {jains:.3f}\nStarved (<20%): {starved}/{n}",
                transform=ax.transAxes, fontsize=9, va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                          edgecolor="#CCCCCC", alpha=0.9))
    plt.tight_layout()
    ieee_style.save(fig, str(OUTPUT_DIR / "fairness_heatmap"))
    print(f"  Saved: fairness_heatmap.pdf / .eps")
    plt.close()


def plot_pareto_scatter(dqn_df, smart_df, dumb_df):
    import matplotlib.patheffects as pe
    fig, ax = plt.subplots(figsize=(9, 6))
    for name, df in [("DQN Agent", dqn_df), ("Smart Greedy V2", smart_df), ("Nearest Greedy", dumb_df)]:
        if df is None or df.empty:
            continue
        energy = EVAL_MAX_BATTERY - df["battery_wh"].iloc[-1]
        data   = df["total_data_collected"].iloc[-1]
        style  = AGENT_STYLES[name]
        ax.scatter(energy, data, s=350, color=style["color"], marker=style["marker"],
                   zorder=5, edgecolors="white", linewidths=2, label=name)
        ax.annotate(name, (energy, data), textcoords="offset points", xytext=(10, 6),
                    fontsize=10, fontweight="bold", color=style["color"],
                    path_effects=[pe.withStroke(linewidth=2, foreground="white")])
    ax.set_xlabel("Total Energy Consumed (Wh)", fontweight="bold")
    ax.set_ylabel("Total Data Collected (Bytes)", fontweight="bold")
    ax.set_title("Energy--Data Pareto Frontier\n(top-left = more efficient)",
                 fontweight="bold", pad=10)
    ax.legend()
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ieee_style.clean_axes(ax)
    plt.tight_layout()
    out = OUTPUT_DIR / "pareto_scatter"
    ieee_style.save(fig, str(out))
    print(f"  Saved: {out}.pdf / .eps")
    plt.close()


def plot_per_sensor_bar(dqn_snapshot_path, smart_snapshot_path, dumb_snapshot_path):
    paths = {
        "DQN Agent":       dqn_snapshot_path,
        "Smart Greedy V2": smart_snapshot_path,
        "Nearest Greedy":  dumb_snapshot_path,
    }
    all_data = {}
    sensor_ids = None
    for name, path in paths.items():
        if not Path(path).exists():
            continue
        with open(path) as f:
            snap = json.load(f)
        rates = {s["sensor_id"]: s["collection_rate"] for s in snap["sensor_data"]}
        all_data[name] = rates
        if sensor_ids is None:
            sensor_ids = sorted(rates.keys())
    if not all_data or sensor_ids is None:
        print("  ⚠ No sensor snapshots found for per-sensor bar chart")
        return
    n_sensors = len(sensor_ids)
    n_agents  = len(all_data)
    bar_h     = 0.25
    y         = np.arange(n_sensors)
    fig, ax = plt.subplots(figsize=(11, max(6, n_sensors * 0.45)))
    for i, (name, rates) in enumerate(all_data.items()):
        vals   = [rates.get(sid, 0) for sid in sensor_ids]
        offset = (i - n_agents / 2 + 0.5) * bar_h
        ax.barh(y + offset, vals, bar_h * 0.9, label=name,
                color=AGENT_STYLES[name]["color"], alpha=0.85, edgecolor="white")
    ax.axvline(20, color="red", linestyle="--", linewidth=1.2, alpha=0.7,
               label="Starvation threshold (20%)")
    ax.set_yticks(y)
    ax.set_yticklabels([f"Sensor {sid}" for sid in sensor_ids], fontsize=8)
    ax.set_xlabel("Collection Rate (%)", fontsize=11, fontweight="bold")
    ax.set_xlim(0, 110)
    ax.set_title("Per-Sensor Collection Rate by Agent\n(red dashed = starvation threshold)",
                 fontweight="bold", pad=10)
    ax.legend(loc="lower right")
    ieee_style.clean_axes(ax)
    plt.tight_layout()
    ieee_style.save(fig, str(OUTPUT_DIR / "per_sensor_bar"))
    print(f"  Saved: per_sensor_bar.pdf / .eps")
    plt.close()


def plot_radar_chart(dqn_df, smart_df, dumb_df,
                     dqn_snapshot_path, smart_snapshot_path, dumb_snapshot_path):
    def load_jains(path):
        if not Path(path).exists():
            return 0.0
        with open(path) as f:
            snap = json.load(f)
        rates = [s["collection_rate"] for s in snap["sensor_data"]]
        n  = len(rates)
        s2 = sum(r**2 for r in rates)
        return (sum(rates)**2 / (n * s2)) if n > 0 and s2 > 0 else 0.0

    def get_metrics(df, snapshot_path):
        if df is None or df.empty:
            return [0.0] * 5
        energy     = EVAL_MAX_BATTERY - df["battery_wh"].iloc[-1]
        data       = df["total_data_collected"].iloc[-1]
        batt       = df["battery_percent"].iloc[-1]
        coverage   = df["coverage_percent"].iloc[-1]
        efficiency = data / energy if energy > 0 else 0.0
        jains      = load_jains(snapshot_path) * 100
        return [data, jains, batt, coverage, efficiency]

    agents = [
        ("DQN Agent",       dqn_df,   dqn_snapshot_path),
        ("Smart Greedy V2", smart_df, smart_snapshot_path),
        ("Nearest Greedy",  dumb_df,  dumb_snapshot_path),
    ]
    raw_metrics = {name: get_metrics(df, sp) for name, df, sp in agents}
    categories  = ["Data\nThroughput", "Jain's\nFairness", "Battery\nRemaining",
                   "Sensor\nCoverage", "Energy\nEfficiency"]
    N = len(categories)

    all_vals = np.array(list(raw_metrics.values()))
    col_max  = all_vals.max(axis=0)
    col_max[col_max == 0] = 1
    norm_metrics = {name: (np.array(vals) / col_max * 100).tolist()
                    for name, vals in raw_metrics.items()}

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    for name, vals in norm_metrics.items():
        vals_plot = vals + vals[:1]
        style = AGENT_STYLES[name]
        ax.plot(angles, vals_plot, color=style["color"], linewidth=2.5, label=name)
        ax.fill(angles, vals_plot, color=style["color"], alpha=0.12)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight="bold")
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8, color="gray")
    ax.set_ylim(0, 110)
    ax.grid(color="gray", linestyle="--", linewidth=0.6, alpha=0.5)
    ax.set_title("Multi-Objective Agent Comparison\n(Normalised — larger area = better)",
                 fontsize=12, fontweight="bold", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.1), fontsize=10, framealpha=0.9)

    raw_lines = ["  Agent               Data(B)   Jain   Batt%  Cov%   Eff(B/Wh)",
                 "  " + "-" * 58]
    for name, df, sp in agents:
        m = raw_metrics[name]
        raw_lines.append(
            f"  {name:<20} {m[0]:>7.0f}  {m[1]/100:>5.3f}  {m[2]:>5.1f}"
            f"  {m[3]:>5.1f}  {m[4]:>8.2f}"
        )
    fig.text(0.5, -0.04, "\n".join(raw_lines), ha="center", fontsize=8,
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5", facecolor="#F5F5F5",
                       edgecolor="#CCCCCC", alpha=0.9))
    plt.tight_layout()
    ieee_style.save(fig, str(OUTPUT_DIR / "radar_chart"))
    print(f"  Saved: radar_chart.pdf / .eps")
    plt.close()


def plot_buffer_dynamics(dqn_df, smart_df, dumb_df):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("Network Data Backlog Clearance Over Time", fontsize=13, fontweight="bold")
    agents = [("DQN Agent", dqn_df), ("Smart Greedy V2", smart_df), ("Nearest Greedy", dumb_df)]

    ax = axes[0]
    for name, df in agents:
        if df is None or df.empty:
            continue
        style = AGENT_STYLES[name]
        ax.plot(df["step"], df["total_data_collected"], color=style["color"],
                linewidth=2.5, label=name, marker=style["marker"], markersize=4, markevery=5)
    ax.set_xlabel("Simulation Step (t)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Cumulative Data Collected (Bytes)", fontsize=11, fontweight="bold")
    ax.set_title("Cumulative Data Collected", fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.grid(True, alpha=0.4, linestyle="--")

    ax2 = axes[1]
    for name, df in agents:
        if df is None or df.empty:
            continue
        style   = AGENT_STYLES[name]
        data    = df["total_data_collected"].values
        steps   = df["step"].values
        d_data  = np.diff(data,  prepend=data[0])
        d_steps = np.diff(steps, prepend=max(steps[0], 1))
        d_steps[d_steps == 0] = 1
        ax2.plot(steps, d_data / d_steps, color=style["color"], linewidth=2,
                 label=name, alpha=0.85)
    ax2.set_xlabel("Simulation Step (t)", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Collection Rate (Bytes / Step)", fontsize=11, fontweight="bold")
    ax2.set_title("Instantaneous Data Collection Rate", fontsize=11, fontweight="bold")
    ax2.legend(fontsize=10)
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax2.grid(True, alpha=0.4, linestyle="--")

    ieee_style.clean_figure(fig)
    plt.tight_layout()
    ieee_style.save(fig, str(OUTPUT_DIR / "buffer_dynamics"))
    print(f"  Saved: buffer_dynamics.pdf / .eps")
    plt.close()


def plot_battery_steps_data(dqn_df, smart_df, dumb_df):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "Battery Depletion, Data Throughput & Combined Overview by Agent",
        fontsize=14, fontweight="bold", y=1.02,
    )

    agents = [
        ("DQN Agent (Proposed)",  dqn_df,   AGENT_STYLES["DQN Agent"]),
        ("SF-Aware Greedy (V2)",  smart_df, AGENT_STYLES["Smart Greedy V2"]),
        ("Nearest Sensor Greedy", dumb_df,  AGENT_STYLES["Nearest Greedy"]),
    ]

    ax = axes[0]
    for name, df, style in agents:
        if df is None or df.empty:
            continue
        ax.plot(df["step"], df["battery_percent"], color=style["color"],
                linewidth=2.5, label=name, marker=style["marker"], markersize=4, markevery=5)
        low = df[df["battery_percent"] < 30]
        if not low.empty:
            sx = low["step"].iloc[0]
            sy = low["battery_percent"].iloc[0]
            ax.annotate(f"t={int(sx)}", xy=(sx, sy), xytext=(sx + 40, sy + 6),
                        fontsize=8, color=style["color"], fontweight="bold",
                        arrowprops=dict(arrowstyle="->", color=style["color"], lw=1.2))
    ax.axhline(30, color="red", linestyle="--", linewidth=1.4, alpha=0.75,
               label="Low battery threshold (30%)")
    ax.set_xlabel("Simulation Step (t)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Battery Level (%)", fontsize=11, fontweight="bold")
    ax.set_title("Battery Depletion", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, alpha=0.4, linestyle="--")

    ax = axes[1]
    for name, df, style in agents:
        if df is None or df.empty:
            continue
        ax.plot(df["step"], df["total_data_collected"], color=style["color"],
                linewidth=2.5, label=name, marker=style["marker"], markersize=4, markevery=5)
        final_step = df["step"].iloc[-1]
        final_data = df["total_data_collected"].iloc[-1]
        ax.annotate(f"{final_data:.2e}", xy=(final_step, final_data),
                    xytext=(final_step - 200, final_data * 0.92),
                    fontsize=8, color=style["color"], fontweight="bold",
                    arrowprops=dict(arrowstyle="->", color=style["color"], lw=1.0))
    ax.set_xlabel("Simulation Step (t)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Cumulative Data Collected (Bytes)", fontsize=11, fontweight="bold")
    ax.set_title("Data Throughput", fontsize=12, fontweight="bold")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.4, linestyle="--")

    ax3  = axes[2]
    ax3b = ax3.twinx()
    for name, df, style in agents:
        if df is None or df.empty:
            continue
        ax3.plot(df["step"], df["battery_percent"], color=style["color"],
                 linewidth=1.8, linestyle="--", alpha=0.55)
        ax3b.plot(df["step"], df["total_data_collected"], color=style["color"],
                  linewidth=2.5, label=name, marker=style["marker"], markersize=4, markevery=5)
    ax3.axhline(30, color="red", linestyle=":", linewidth=1.2, alpha=0.6)
    ax3.set_ylabel("Battery Level (%) — dashed", fontsize=10, color="gray")
    ax3.set_ylim(0, 105)
    ax3.tick_params(axis="y", labelcolor="gray")
    ax3b.set_ylabel("Data Collected (Bytes) — solid", fontsize=10, fontweight="bold")
    ax3b.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax3.set_xlabel("Simulation Step (t)", fontsize=11, fontweight="bold")
    ax3.set_title("Combined: Battery vs Data Collected", fontsize=12, fontweight="bold")
    lines, labels = ax3b.get_legend_handles_labels()
    ax3.legend(lines, labels, fontsize=9, loc="upper left")
    ax3.grid(True, alpha=0.4, linestyle="--")

    ieee_style.clean_figure(fig)
    plt.tight_layout()
    ieee_style.save(fig, str(OUTPUT_DIR / "battery_steps_data"))
    print(f"  Saved: battery_steps_data.pdf / .eps")
    plt.close()


# ==================== MAIN ====================


def main():
    print("=" * 100)
    print("FAIR BASELINE AGENTS EVALUATION & COMPARISON + SENSOR SNAPSHOTS")
    print("(All agents evaluated on identical environment with same seed)")
    print("=" * 100)
    print(f"\nOutput Directory: {OUTPUT_DIR}")
    print(f"Seed: {PLOT_CONFIG['seed']}\n")

    # Load config — now reads training_config.json which includes max_sensors_limit
    training_config = load_training_config(DQN_CONFIG_PATH)
    max_sensors_limit = training_config["max_sensors_limit"]
    print(f"  max_sensors_limit from config: {max_sensors_limit}")
    print(f"  frame_stacking: n_stack={training_config.get('n_stack', 4)}")

    env_kwargs = {
        "grid_size":          PLOT_CONFIG["grid_size"],
        "num_sensors":        PLOT_CONFIG["num_sensors"],
        "max_sensors_limit":  max_sensors_limit,   # ← zero-padding limit
        "max_steps":          PLOT_CONFIG["max_steps"],
        "path_loss_exponent": PLOT_CONFIG["path_loss_exponent"],
        "rssi_threshold":     PLOT_CONFIG["rssi_threshold"],
        "sensor_duty_cycle":  PLOT_CONFIG["sensor_duty_cycle"],
        "max_battery":        274.0,
        "render_mode":        "human",
    }

    agents_config = []

    # ========== STEP 1: Run DQN Agent ==========
    print("-" * 100)
    df_dqn          = None
    dqn_trajectory  = None
    dqn_base_env    = None
    dqn_sensor_snap = None

    if DQN_MODEL_PATH.exists():
        try:
            print(f"Loading DQN model from {DQN_MODEL_PATH}...")
            model = DQN.load(DQN_MODEL_PATH)
            print("✓ DQN model loaded successfully")

            dqn_stacked_env, dqn_base_env = create_stacked_dqn_env(
                env_kwargs, training_config, seed=PLOT_CONFIG["seed"]
            )
            df_dqn, steps_dqn, dqn_trajectory, dqn_sensor_snap = run_dqn_agent_for_plot(
                model, dqn_stacked_env, dqn_base_env, "DQN Agent",
                seed=PLOT_CONFIG["seed"],
            )
            save_baseline_data("dqn_agent_fresh", df_dqn)
            save_sensor_snapshot(
                dqn_base_env,
                OUTPUT_DIR / "dqn_sensor_snapshot.json",
                "DQN Agent",
                is_wrapper=True,
                override_sensor_data=dqn_sensor_snap,
            )
            agents_config.append({
                "name":           "DQN (Proposed)",
                "steps":          steps_dqn,
                "final_reward":   float(df_dqn["cumulative_reward"].iloc[-1]),
                "final_coverage": float(df_dqn["coverage_percent"].iloc[-1]),
            })
            print("✓ DQN evaluation complete")
            dqn_stacked_env.close()
        except Exception as e:
            print(f"⚠ WARNING: Could not run DQN agent: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"⚠ WARNING: DQN model not found at {DQN_MODEL_PATH}")

    # ========== STEP 2 & 3: Run Greedy Agents ==========
    # Greedy agents use the base UAVEnvironment — no padding needed
    print("\n" + "-" * 100)
    print("Setting up environment for greedy agents...")
    greedy_env_kwargs = {k: v for k, v in env_kwargs.items()
                         if k not in ("max_sensors_limit", "render_mode")}
    greedy_env_kwargs["render_mode"] = "human"
    # Seed global RNG to match DQN env so sensor positions are identical
    np.random.seed(PLOT_CONFIG["seed"])
    random.seed(PLOT_CONFIG["seed"])
    env = UAVEnvironment(**greedy_env_kwargs)
    print("✓ Environment created\n")

    print("-" * 100)
    agent_smart = MaxThroughputGreedyV2(env)
    df_smart, steps_smart, smart_trajectory = run_greedy_agent_for_plot(
        agent_smart, env, "SF-Aware Greedy V2", seed=PLOT_CONFIG["seed"]
    )
    save_baseline_data("greedy_smart_v2", df_smart)
    save_sensor_snapshot(env, OUTPUT_DIR / "greedy_smart_sensor_snapshot.json",
                         "Smart Greedy V2", is_wrapper=False)
    agents_config.append({
        "name":           "SF-Aware Greedy V2",
        "steps":          steps_smart,
        "final_reward":   float(df_smart["cumulative_reward"].iloc[-1]),
        "final_coverage": float(df_smart["coverage_percent"].iloc[-1]),
    })

    print("\n" + "-" * 100)
    agent_dumb = NearestSensorGreedy(env)
    df_dumb, steps_dumb, dumb_trajectory = run_greedy_agent_for_plot(
        agent_dumb, env, "Nearest Sensor Greedy", seed=PLOT_CONFIG["seed"]
    )
    save_baseline_data("greedy_nearest", df_dumb)
    save_sensor_snapshot(env, OUTPUT_DIR / "greedy_nearest_sensor_snapshot.json",
                         "Nearest Sensor Greedy", is_wrapper=False)
    agents_config.append({
        "name":           "Nearest Sensor Greedy",
        "steps":          steps_dumb,
        "final_reward":   float(df_dumb["cumulative_reward"].iloc[-1]),
        "final_coverage": float(df_dumb["coverage_percent"].iloc[-1]),
    })

    save_comparison_metadata(agents_config)

    # ========== PLOTTING ==========
    print("\n" + "-" * 100)
    plot_trajectories(env, dqn_trajectory, smart_trajectory, dumb_trajectory)
    print("\n" + "-" * 100)
    plot_comparative_analysis(df_dqn, df_smart, df_dumb)
    print("\n" + "-" * 100)
    plot_efficiency_table(df_dqn, df_smart, df_dumb)
    print("\n" + "-" * 100)
    plot_fairness_heatmap(
        OUTPUT_DIR / "dqn_sensor_snapshot.json",
        OUTPUT_DIR / "greedy_smart_sensor_snapshot.json",
        OUTPUT_DIR / "greedy_nearest_sensor_snapshot.json",
    )
    print("\n" + "-" * 100)
    plot_pareto_scatter(df_dqn, df_smart, df_dumb)
    print("\n" + "-" * 100)
    plot_per_sensor_bar(
        OUTPUT_DIR / "dqn_sensor_snapshot.json",
        OUTPUT_DIR / "greedy_smart_sensor_snapshot.json",
        OUTPUT_DIR / "greedy_nearest_sensor_snapshot.json",
    )
    print("\n" + "-" * 100)
    plot_radar_chart(
        df_dqn, df_smart, df_dumb,
        OUTPUT_DIR / "dqn_sensor_snapshot.json",
        OUTPUT_DIR / "greedy_smart_sensor_snapshot.json",
        OUTPUT_DIR / "greedy_nearest_sensor_snapshot.json",
    )
    print("\n" + "-" * 100)
    plot_buffer_dynamics(df_dqn, df_smart, df_dumb)
    print("\n" + "-" * 100)
    plot_battery_steps_data(df_dqn, df_smart, df_dumb)

    # ========== SUMMARY ==========
    print("\n" + "=" * 100)
    print("COMPARISON SUMMARY STATISTICS")
    print("=" * 100)

    if df_dqn is not None and not df_dqn.empty:
        print(f"\nDQN Agent:")
        print(f"  Final Reward:   {df_dqn['cumulative_reward'].iloc[-1]:>15.1f}")
        print(f"  Final Battery:  {df_dqn['battery_percent'].iloc[-1]:>13.1f}%")
        print(f"  Final NDR: {df_dqn['coverage_percent'].iloc[-1]:>12.1f}%")
        print(f"  Data Collected: {df_dqn['total_data_collected'].iloc[-1]:>11.0f} bytes")
        print(f"  Efficiency:     {df_dqn['efficiency'].iloc[-1]:>21.4f} bytes/Wh")

    for label, df in [("Smart Greedy V2", df_smart), ("Nearest Sensor Greedy", df_dumb)]:
        print(f"\n{label}:")
        print(f"  Final Reward:   {df['cumulative_reward'].iloc[-1]:>15.1f}")
        print(f"  Final Battery:  {df['battery_percent'].iloc[-1]:>13.1f}%")
        print(f"  Final NDR: {df['coverage_percent'].iloc[-1]:>12.1f}%")
        print(f"  Data Collected: {df['total_data_collected'].iloc[-1]:>11.0f} bytes")
        print(f"  Efficiency:     {df['efficiency'].iloc[-1]:>21.4f} bytes/Wh")

    env.close()
    print("\n" + "=" * 100)
    print("✓ EVALUATION COMPLETE")
    print("=" * 100 + "\n")


if __name__ == "__main__":
    main()