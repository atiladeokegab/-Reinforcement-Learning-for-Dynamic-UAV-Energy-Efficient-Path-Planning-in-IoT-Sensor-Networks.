"""
DQN Evaluation & Visualization with Frame Stacking Support

Evaluates trained DQN agent with detailed fairness analysis.
Automatically handles frame-stacked environments and PERSISTS data after reset.
Saves detailed CSV file for comparison with greedy baselines.

Output: src/agents/dqn/dqn_evaluation_results

Author: ATILADE GABRIEL OKE
Date: 09 November 2025
Modified: Added CSV data persistence and detailed metrics logging
"""

import sys
from pathlib import Path

# ==================== CRITICAL: Set up paths ====================
script_dir = Path(__file__).resolve().parent  # dqn/
src_dir = script_dir.parent.parent  # Go up to src/

sys.path.insert(0, str(src_dir))

import numpy as np
import json
import time
import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize  # ← ADDED VecNormalize
from environment.uav_env import UAVEnvironment

# ==================== CONFIGURATION ====================

MODEL_PATH = script_dir / "models/dqn_full_observability/dqn_final.zip"
CONFIG_PATH = script_dir / "models/dqn_full_observability/frame_stacking_config.json"
VEC_NORMALIZE_PATH = script_dir / "models/dqn_full_observability/vec_normalize.pkl"  # ← ADDED
OUTPUT_DIR = script_dir / "dqn_evaluation_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"Script location: {script_dir}")
print(f"Output directory: {OUTPUT_DIR}")
print(f"Model path: {MODEL_PATH}")
print()

EVAL_CONFIG = {
    "grid_size": (500, 500),
    "uav_start_position": (50, 50),
    "num_sensors": 20,
    "max_battery": 600.0,
    "max_steps": 2100,
    "sensor_duty_cycle": 10.0,
    "penalty_data_loss": -1.0,       # ← FIXED from -500.0
    "reward_urgency_reduction": 20.0,
    "render_mode": None,
}

VIZ_CONFIG = {
    "step_delay": 0.05,
    "progress_interval": 50,
}


# ==================== SMART ENVIRONMENT WRAPPER ====================


class AnalysisUAVEnv(UAVEnvironment):
    """
    A smart wrapper that saves the final state of the simulation
    the instant before the environment resets.

    This ensures we can capture all metrics even after environment.reset()
    wipes the internal state.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.last_episode_sensor_data = None
        self.last_episode_info = None

    def reset(self, **kwargs):
        # 1. SNAPSHOT: If we have run a simulation (step > 0), save the data now!
        if hasattr(self, "sensors") and self.current_step > 0:
            self.last_episode_sensor_data = []
            for sensor in self.sensors:
                self.last_episode_sensor_data.append(
                    {
                        "sensor_id": sensor.sensor_id,
                        "position": tuple(sensor.position),
                        "total_data_generated": float(sensor.total_data_generated),
                        "total_data_transmitted": float(sensor.total_data_transmitted),
                        "total_data_lost": float(sensor.total_data_lost),
                        "data_buffer": float(sensor.data_buffer),
                    }
                )

            # Save final global stats
            self.last_episode_info = {
                "battery": self.uav.battery,
                "battery_percent": self.uav.get_battery_percentage(),
                "coverage_percentage": (len(self.sensors_visited) / self.num_sensors)
                * 100,
            }

        # 2. PROCEED: Now allow the standard reset to wipe everything
        return super().reset(**kwargs)


# ==================== HELPER FUNCTIONS ====================


def load_frame_stacking_config(config_path):
    """Load frame stacking configuration from JSON file."""
    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠ Config file not found at {config_path}, using defaults")
        return {"use_frame_stacking": False, "n_stack": 4}


def _unwrap_base_env(vec):
    """Drill through VecNormalize → VecFrameStack → DummyVecEnv → Monitor
    to get the raw AnalysisUAVEnv instance."""
    inner = vec
    while hasattr(inner, 'venv'):
        inner = inner.venv
    env = inner.envs[0]
    while hasattr(env, 'env'):
        env = env.env
    return env


def create_eval_env(config, frame_stacking_config):
    """Create evaluation environment with optional frame stacking + VecNormalize.
    Returns (vec_env, true_base_env) — base_env resolved AFTER all wrappers applied."""
    base_env = AnalysisUAVEnv(**config)
    vec_env = DummyVecEnv([lambda: base_env])

    if frame_stacking_config.get("use_frame_stacking", False):
        n_stack = frame_stacking_config.get("n_stack", 4)
        vec_env = VecFrameStack(vec_env, n_stack=n_stack)
        print(f"✓ Frame stacking enabled (n_stack={n_stack})")
    else:
        print(f"✓ Frame stacking disabled")

    if VEC_NORMALIZE_PATH.exists():
        vec_env = VecNormalize.load(str(VEC_NORMALIZE_PATH), vec_env)
        vec_env.training = False
        vec_env.norm_reward = False
        print(f"✓ VecNormalize loaded from {VEC_NORMALIZE_PATH.name}")
    else:
        print(f"⚠ vec_normalize.pkl not found — observations will NOT be normalised")

    # ← Unwrap AFTER all wrappers applied so reference is correct
    true_base_env = _unwrap_base_env(vec_env)
    print(f"  [DEBUG] base_env type: {type(true_base_env).__name__} | battery={true_base_env.uav.battery:.1f}")
    return vec_env, true_base_env


def calculate_fairness_metrics(sensor_collections):
    """Calculate fairness statistics from sensor collection rates."""
    if not sensor_collections:
        return {}
    return {
        "mean": np.mean(sensor_collections),
        "std": np.std(sensor_collections),
        "min": np.min(sensor_collections),
        "max": np.max(sensor_collections),
        "range": np.max(sensor_collections) - np.min(sensor_collections),
    }


def get_fairness_level(std_dev):
    """Classify fairness level based on standard deviation."""
    if std_dev < 15:
        return "EXCELLENT", "+++"
    elif std_dev < 25:
        return "GOOD", "++"
    elif std_dev < 35:
        return "MODERATE", "+"
    else:
        return "POOR", "x"


def save_detailed_csv(history_data, output_dir, filename="dqn_results.csv"):
    """Save detailed step-by-step metrics to CSV."""
    df = pd.DataFrame(history_data)
    filepath = output_dir / filename
    df.to_csv(filepath, index=False)
    print(f"✓ Saved detailed metrics to {filepath}")
    return filepath


def save_sensor_fairness_analysis(
    sensor_data, output_dir, filename="dqn_sensor_fairness.csv"
):
    """Save per-sensor fairness analysis to CSV."""
    df = pd.DataFrame(sensor_data)
    filepath = output_dir / filename
    df.to_csv(filepath, index=False)
    print(f"✓ Saved sensor fairness analysis to {filepath}")
    return filepath


def save_summary_json(summary, output_dir, filename="dqn_summary.json"):
    """Save overall summary statistics as JSON."""
    filepath = output_dir / filename
    with open(filepath, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"✓ Saved summary statistics to {filepath}")
    return filepath


# ==================== MAIN EVALUATION ====================


def main():
    print("=" * 100)
    print("DQN AGENT EVALUATION (PERSISTENT DATA MODE WITH CSV EXPORT)")
    print("=" * 100)
    print(f"Output Directory: {OUTPUT_DIR.absolute()}\n")

    frame_stacking_config = load_frame_stacking_config(CONFIG_PATH)

    try:
        print(f"Loading model from: {MODEL_PATH}")
        model = DQN.load(MODEL_PATH)
        print("✓ Model loaded successfully")
        eval_env, base_env = create_eval_env(EVAL_CONFIG, frame_stacking_config)
        print("✓ Environment created\n")
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return

    print("Starting Evaluation... (Press Ctrl+C to stop)\n")

    obs = eval_env.reset()

    # Debug: Check observation shape
    print(f"Observation shape: {obs.shape if hasattr(obs, 'shape') else type(obs)}")
    print(f"Model expects observation space: {model.observation_space}\n")

    done = False
    step = 0
    total_reward = 0
    start_time = time.time()

    # Initialize tracking lists for CSV
    history_data = {
        "step": [],
        "cumulative_reward": [],
        "battery_percent": [],
        "battery_wh": [],
        "coverage_percent": [],
        "instant_reward": [],
    }

    # These will hold the true final-step values captured before auto-reset
    final_battery_wh = base_env.uav.battery
    final_coverage_pct = 0.0

    try:
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            if isinstance(action, np.ndarray):
                action = action[0]
            action = int(action)

            # ← CRITICAL FIX: snapshot BEFORE step() — VecEnv auto-resets inside step()
            pre_battery    = base_env.uav.battery
            pre_battery_pct = base_env.uav.get_battery_percentage()
            pre_coverage   = (len(base_env.sensors_visited) / base_env.num_sensors) * 100 \
                             if hasattr(base_env, "sensors_visited") else 0.0

            obs, rewards, dones, infos = eval_env.step([action])

            reward = float(rewards[0]) if isinstance(rewards, np.ndarray) else float(rewards)
            done   = bool(dones[0])   if isinstance(dones,   np.ndarray) else bool(dones)

            total_reward += reward
            step += 1

            if done:
                # Use pre-step snapshot — base_env is already reset at this point
                final_battery_wh   = pre_battery
                final_coverage_pct = pre_coverage
                history_data["step"].append(step)
                history_data["cumulative_reward"].append(total_reward)
                history_data["battery_percent"].append(pre_battery_pct)
                history_data["battery_wh"].append(pre_battery)
                history_data["coverage_percent"].append(pre_coverage)
                history_data["instant_reward"].append(reward)
                print(
                    f"Step {step:>4}: Cov={pre_coverage:>5.1f}% "
                    f"Bat={pre_battery_pct:>5.1f}% Rew={total_reward:>7.1f} "
                    f"InstRew={reward:>7.1f}  [FINAL - pre-reset snapshot]"
                )
            elif step % VIZ_CONFIG["progress_interval"] == 0:
                # Mid-episode — base_env not yet reset, safe to read directly
                coverage_pct = (len(base_env.sensors_visited) / base_env.num_sensors) * 100 \
                               if hasattr(base_env, "sensors_visited") else 0
                battery_pct  = base_env.uav.get_battery_percentage()
                battery_wh   = base_env.uav.battery

                history_data["step"].append(step)
                history_data["cumulative_reward"].append(total_reward)
                history_data["battery_percent"].append(battery_pct)
                history_data["battery_wh"].append(battery_wh)
                history_data["coverage_percent"].append(coverage_pct)
                history_data["instant_reward"].append(reward)

                print(
                    f"Step {step:>4}: Cov={coverage_pct:>5.1f}% "
                    f"Bat={battery_pct:>5.1f}% Rew={total_reward:>7.1f} "
                    f"InstRew={reward:>7.1f}"
                )

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"\n✗ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()

    elapsed_time = time.time() - start_time

    if base_env.last_episode_sensor_data:
        print("\n✓ Successfully recovered pre-reset simulation data!")
        saved_sensor_data = base_env.last_episode_sensor_data
        final_info = base_env.last_episode_info
    else:
        print("\n⚠ Warning: No snapshot found. Using pre-step captured values.")
        saved_sensor_data = []
        # Use values we captured during the loop before auto-reset
        final_info = {
            "battery": final_battery_wh,
            "battery_percent": (final_battery_wh / 600.0) * 100,
            "coverage_percentage": final_coverage_pct,
        }

    print("\n" + "=" * 100)
    print("EPISODE COMPLETE")
    print("=" * 100)

    total_generated = (
        sum(s["total_data_generated"] for s in saved_sensor_data)
        if saved_sensor_data else 0
    )
    total_collected = (
        sum(s["total_data_transmitted"] for s in saved_sensor_data)
        if saved_sensor_data else 0
    )
    total_lost = (
        sum(s["total_data_lost"] for s in saved_sensor_data)
        if saved_sensor_data else 0
    )

    efficiency = (total_collected / total_generated * 100) if total_generated > 0 else 0
    loss_rate = (total_lost / total_generated * 100) if total_generated > 0 else 0

    battery_used = 600.0 - final_info["battery"]
    bytes_per_watt = total_collected / battery_used if battery_used > 0 else 0

    print(f"\nOverall Performance:")
    print(f"  Total Reward:          {total_reward:>12.1f}")
    print(f"  Steps:                 {step:>12}")
    print(f"  Elapsed Time:          {elapsed_time:>12.1f}s")
    print(f"  Coverage:              {final_info['coverage_percentage']:>12.1f}%")
    print(f"  Collection Efficiency: {efficiency:>12.1f}%")
    print(f"  Data Loss Rate:        {loss_rate:>12.1f}%")
    print(f"  Battery Used:          {battery_used:>12.1f} Wh")
    print(f"  Bytes per Watt:        {bytes_per_watt:>12.1f} B/Wh")

    # ========== PER-SENSOR FAIRNESS ANALYSIS ==========
    print("\n" + "=" * 100)
    print("PER-SENSOR FAIRNESS ANALYSIS")
    print("=" * 100)

    print(f"\n{'Sensor':<8} {'Gen':<8} {'Col':<8} {'Eff %':<8} {'Visual':<50}")
    print("-" * 100)

    sensor_collections = []
    sensor_fairness_data = []

    for s in saved_sensor_data:
        gen = s["total_data_generated"]
        col = s["total_data_transmitted"]
        lost = s["total_data_lost"]
        pct = (col / gen * 100) if gen > 0 else 0
        sensor_collections.append(pct)

        sensor_fairness_data.append(
            {
                "sensor_id": s["sensor_id"],
                "position_x": s["position"][0],
                "position_y": s["position"][1],
                "data_generated": gen,
                "data_transmitted": col,
                "data_lost": lost,
                "efficiency_percent": pct,
                "buffer_remaining": s["data_buffer"],
            }
        )

        bar_len = int(pct / 2.5)
        bar = "█" * bar_len
        print(f"S{s['sensor_id']:<7} {gen:<8.0f} {col:<8.0f} {pct:<8.1f} {bar}")

    fairness_stats = {}
    if sensor_collections:
        stats = calculate_fairness_metrics(sensor_collections)
        lvl, sym = get_fairness_level(stats["std"])

        print("\n" + "-" * 100)
        print(f"Fairness Level: {lvl} {sym}")
        print(
            f"Std Dev: {stats['std']:.1f}% (Min: {stats['min']:.1f}%, Max: {stats['max']:.1f}%)"
        )
        fairness_stats = stats

    # ========== SAVE DATA TO CSV FILES ==========
    print("\n" + "=" * 100)
    print("SAVING DATA TO CSV FILES")
    print("=" * 100)

    save_detailed_csv(history_data, OUTPUT_DIR, "dqn_results.csv")
    save_sensor_fairness_analysis(
        sensor_fairness_data, OUTPUT_DIR, "dqn_sensor_fairness.csv"
    )

    summary_dict = {
        "total_reward": float(total_reward),
        "steps": int(step),
        "elapsed_time_seconds": float(elapsed_time),
        "coverage_percentage": float(final_info["coverage_percentage"]),
        "collection_efficiency_percent": float(efficiency),
        "data_loss_rate_percent": float(loss_rate),
        "battery_used_wh": float(battery_used),
        "bytes_per_watt": float(bytes_per_watt),
        "total_data_generated": float(total_generated),
        "total_data_transmitted": float(total_collected),
        "total_data_lost": float(total_lost),
        "fairness_metrics": {
            "mean": float(fairness_stats.get("mean", 0)),
            "std": float(fairness_stats.get("std", 0)),
            "min": float(fairness_stats.get("min", 0)),
            "max": float(fairness_stats.get("max", 0)),
            "range": float(fairness_stats.get("range", 0)),
        },
        "fairness_level": (
            get_fairness_level(fairness_stats.get("std", 0))[0]
            if fairness_stats else "N/A"
        ),
    }

    save_summary_json(summary_dict, OUTPUT_DIR, "dqn_summary.json")

    eval_env.close()
    print("\n✓ Evaluation and data export complete.")
    print(f"✓ All results saved to: {OUTPUT_DIR.absolute()}")


if __name__ == "__main__":
    main()