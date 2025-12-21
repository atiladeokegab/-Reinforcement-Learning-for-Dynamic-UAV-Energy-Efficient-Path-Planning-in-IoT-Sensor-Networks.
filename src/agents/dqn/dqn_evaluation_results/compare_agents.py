"""
Dissertation Comparison Plotter - FAIR VERSION WITH FRAME STACKING
Generates the final 'DQN vs. Greedy' performance graph with FRESH DQN evaluation.
All agents run on the EXACT SAME ENVIRONMENT with the same seed.
CRITICAL: DQN uses VecFrameStack to match training conditions.
if link quality if 0 uav does not move
if 0.8 moves from 0 to 200
Author: ATILADE GABRIEL OKE
Modified: December 2025
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
from pathlib import Path
import time
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# ==================== CRITICAL FIX: Correct Import Paths ====================
script_dir = Path(__file__).resolve().parent  # dqn_evaluation_results/
src_dir = script_dir.parent.parent.parent     # Go up 3 levels to src
script_dir_results = Path(__file__).resolve().parent.parent  # dqn/

print(f"Script location: {script_dir}")
print(f"Source directory: {src_dir}")
print(f"Results directory: {script_dir_results}")

sys.path.insert(0, str(src_dir))

from environment.uav_env import UAVEnvironment
from agents.baselines.greedy_agents import MaxThroughputGreedyV2, NearestSensorGreedy

# ==================== ANALYSIS ENV WRAPPER ====================

class AnalysisUAVEnv(UAVEnvironment):
    """
    Smart wrapper that saves the final state of the simulation
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
        if hasattr(self, 'sensors') and self.current_step > 0:
            self.last_episode_sensor_data = []
            for sensor in self.sensors:
                self.last_episode_sensor_data.append({
                    'sensor_id': sensor.sensor_id,
                    'position': tuple(sensor.position),
                    'total_data_generated': float(sensor.total_data_generated),
                    'total_data_transmitted': float(sensor.total_data_transmitted),
                    'total_data_lost': float(sensor.total_data_lost),
                    'data_buffer': float(sensor.data_buffer),
                })

            # Save final global stats
            self.last_episode_info = {
                'battery': self.uav.battery,
                'battery_percent': self.uav.get_battery_percentage(),
                'coverage_percentage': (len(self.sensors_visited) / self.num_sensors) * 100
            }

        # 2. PROCEED: Now allow the standard reset to wipe everything
        return super().reset(**kwargs)

# ==================== TRAJECTORY TRACKING ====================

class TrajectoryTracker:
    """Tracks UAV trajectory during evaluation."""
    def __init__(self):
        self.positions = []

    def record(self, x, y):
        """Record UAV position."""
        self.positions.append((x, y))

    def get_array(self):
        """Return positions as numpy array."""
        if not self.positions:
            return np.array([])
        return np.array(self.positions)

# ==================== CONFIGURATION ====================
BASELINES_DIR = src_dir / "agents" / "baselines"
OUTPUT_DIR = script_dir / "baseline_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DQN_MODEL_PATH = script_dir_results / "models" / "dqn_full_observability" / "dqn_final.zip"
DQN_CONFIG_PATH = script_dir_results / "models" / "dqn_full_observability" / "frame_stacking_config.json"

PLOT_CONFIG = {
    'grid_size': (500, 500),
    'num_sensors': 20,
    'max_steps': 2100,
    'path_loss_exponent': 3.8,
    'rssi_threshold': -120.0,
    'sensor_duty_cycle': 10.0,
    'seed': 42,
}

EVAL_MAX_BATTERY = 274.0

print(f"Output directory: {OUTPUT_DIR}")
print(f"DQN Model path: {DQN_MODEL_PATH}")
print()

# ==================== HELPER FUNCTIONS ====================

def load_frame_stacking_config(config_path):
    """Load frame stacking configuration from JSON file."""
    try:
        with open(config_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"⚠ Config file not found at {config_path}, using defaults")
        return {'use_frame_stacking': True, 'n_stack': 4}

def create_stacked_dqn_env(env_kwargs, frame_stacking_config):
    """Create environment with VecFrameStack and Analysis Wrapper for DQN evaluation."""
    # Use the Analysis Wrapper (Snapshot Logic)
    base_env = AnalysisUAVEnv(**env_kwargs)
    vec_env = DummyVecEnv([lambda: base_env])

    if frame_stacking_config.get('use_frame_stacking', True):
        n_stack = frame_stacking_config.get('n_stack', 4)
        vec_env = VecFrameStack(vec_env, n_stack=n_stack)
        print(f"✓ Frame stacking enabled (n_stack={n_stack})")
    else:
        print(f"✓ Frame stacking disabled")

    return vec_env, base_env

def save_baseline_data(agent_name, history_df):
    """Saves baseline agent data to CSV file."""
    output_file = OUTPUT_DIR / f"{agent_name}_results.csv"
    history_df.to_csv(output_file, index=False)
    print(f"✓ Saved {agent_name} data to {output_file}")
    return output_file

def save_comparison_metadata(agents_config):
    """Saves metadata about the comparison run."""
    metadata = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'grid_size': PLOT_CONFIG['grid_size'],
        'num_sensors': PLOT_CONFIG['num_sensors'],
        'max_steps': PLOT_CONFIG['max_steps'],
        'seed': PLOT_CONFIG['seed'],
        'agents_evaluated': agents_config,
        'output_directory': str(OUTPUT_DIR),
        'note': 'All agents evaluated on identical environment with same seed'
    }

    metadata_file = OUTPUT_DIR / "comparison_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"✓ Saved metadata to {metadata_file}")

# ==================== AGENT EVALUATION ====================

def run_greedy_agent_for_plot(agent, env, name="Agent", seed=PLOT_CONFIG['seed']):
    """
    Runs a greedy agent for one episode and returns step-by-step history + trajectory.
    Returns: (dataframe, steps, trajectory_array)
    """
    print(f"\nRunning {name}...")
    obs, info = env.reset(seed=seed)
    done = False

    trajectory = TrajectoryTracker()

    history = {
        'step': [],
        'cumulative_reward': [],
        'battery_percent': [],
        'battery_wh': [],
        'coverage_percent': [],
        'sensors_visited': [],
        'total_data_collected': [],
        'efficiency': []
    }

    cumulative_reward = 0
    step_count = 0

    while not done:
        action = agent.select_action(obs)
        obs, reward, done, truncated, info = env.step(action)
        cumulative_reward += reward
        step_count += 1

        # Record trajectory
        trajectory.record(env.uav.position[0], env.uav.position[1])

        # Log data every 50 steps
        if env.current_step % 50 == 0 or done or truncated:
            battery_pct = env.uav.get_battery_percentage()
            coverage_pct = (len(env.sensors_visited) / env.num_sensors) * 100 if hasattr(env, 'sensors_visited') else 0
            energy_consumed = EVAL_MAX_BATTERY - env.uav.battery
            efficiency = (env.total_data_collected / energy_consumed) if energy_consumed > 0 else 0

            history['step'].append(env.current_step)
            history['cumulative_reward'].append(cumulative_reward)
            history['battery_percent'].append(battery_pct)
            history['battery_wh'].append(env.uav.battery)
            history['coverage_percent'].append(coverage_pct)
            history['sensors_visited'].append(len(env.sensors_visited))
            history['total_data_collected'].append(env.total_data_collected)
            history['efficiency'].append(efficiency)

            print(f"  Step {env.current_step:>4}: Reward={cumulative_reward:>10.1f}, "
                  f"Battery={battery_pct:>5.1f}%, Coverage={coverage_pct:>5.1f}%, "
                  f"Data={env.total_data_collected:>8.0f}bytes")

        if done or truncated:
            break

    return pd.DataFrame(history), step_count, trajectory.get_array()

def run_dqn_agent_for_plot(model, stacked_env, base_env, name="DQN Agent", seed=PLOT_CONFIG['seed']):
    """
    Runs a trained DQN agent using the STACKED environment.
    Uses AnalysisUAVEnv snapshot to preserve final metrics.
    Tracks trajectory throughout episode.
    Returns: (dataframe, steps, trajectory_array)
    """
    print(f"\nRunning {name}...")

    # Reset the STACKED environment with seed
    # Reset the STACKED environment (VecFrameStack doesn't accept seed parameter)
    obs = stacked_env.reset()
    # Set seed on base_env separately if needed
    if hasattr(base_env, 'reset'):
        base_env.reset(seed=seed)

    trajectory = TrajectoryTracker()

    done = False

    history = {
        'step': [],
        'cumulative_reward': [],
        'battery_percent': [],
        'battery_wh': [],
        'coverage_percent': [],
        'sensors_visited': [],
        'total_data_collected': [],
        'efficiency': []
    }

    cumulative_reward = 0
    step_count = 0

    while not done:
        # Predict uses the stacked observation (shape: [1, 4*obs_dim])
        action, _ = model.predict(obs, deterministic=True)

        # Extract scalar action from array
        action = int(action[0]) if isinstance(action, np.ndarray) and action.ndim > 0 else int(action)

        # Step the STACKED environment (returns vectorized output)
        obs, rewards, dones, infos = stacked_env.step([action])

        # Extract scalar values from vectorized outputs
        reward = float(rewards[0]) if isinstance(rewards, np.ndarray) else float(rewards)
        done = bool(dones[0]) if isinstance(dones, np.ndarray) else bool(dones)
        info = infos[0] if isinstance(infos, list) else infos

        cumulative_reward += reward
        step_count += 1

        # Record trajectory
        trajectory.record(base_env.uav.position[0], base_env.uav.position[1])

        # Access base_env for exact state data
        unwrapped_env = base_env

        # Log data every 50 steps
        if unwrapped_env.current_step % 50 == 0 or done:
            battery_pct = unwrapped_env.uav.get_battery_percentage()
            coverage_pct = (len(unwrapped_env.sensors_visited) / unwrapped_env.num_sensors) * 100 if hasattr(unwrapped_env, 'sensors_visited') else 0
            energy_consumed = EVAL_MAX_BATTERY - unwrapped_env.uav.battery
            efficiency = (unwrapped_env.total_data_collected / energy_consumed) if energy_consumed > 0 else 0

            history['step'].append(unwrapped_env.current_step)
            history['cumulative_reward'].append(cumulative_reward)
            history['battery_percent'].append(battery_pct)
            history['battery_wh'].append(unwrapped_env.uav.battery)
            history['coverage_percent'].append(coverage_pct)
            history['sensors_visited'].append(len(unwrapped_env.sensors_visited))
            history['total_data_collected'].append(unwrapped_env.total_data_collected)
            history['efficiency'].append(efficiency)

            print(f"  Step {unwrapped_env.current_step:>4}: Reward={cumulative_reward:>10.1f}, "
                  f"Battery={battery_pct:>5.1f}%, Coverage={coverage_pct:>5.1f}%, "
                  f"Data={unwrapped_env.total_data_collected:>8.0f}bytes")

        if done:
            break

    # CRITICAL: Call reset to trigger snapshot logic BEFORE we access the final data
    stacked_env.reset()

    # Now retrieve the snapshot data that was preserved by AnalysisUAVEnv
    if base_env.last_episode_info:
        final_info = base_env.last_episode_info
        print(f"\n✓ Episode snapshot captured: Battery={final_info['battery_percent']:.1f}%, Coverage={final_info['coverage_percentage']:.1f}%")

    return pd.DataFrame(history), step_count, trajectory.get_array()

# ==================== PLOTTING ====================

def plot_trajectories(env, dqn_trajectory, greedy_smart_trajectory, greedy_dumb_trajectory):
    """
    Generates trajectory plot showing agent paths overlaid on sensor positions.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Extract sensor positions from environment
    sensor_positions = np.array([sensor.position for sensor in env.sensors])

    # Common plot settings
    grid_size = PLOT_CONFIG['grid_size'][0]

    trajectories = [
        (dqn_trajectory, "DQN Agent (Proposed)", '#1f77b4', axes[0]),
        (greedy_smart_trajectory, "SF-Aware Greedy V2", '#d62728', axes[1]),
        (greedy_dumb_trajectory, "Nearest Sensor Greedy", '#808080', axes[2])
    ]

    for trajectory, title, color, ax in trajectories:
        # Plot grid background
        ax.set_xlim(0, grid_size)
        ax.set_ylim(0, grid_size)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3, linestyle='--')

        # Plot sensor positions
        ax.scatter(sensor_positions[:, 0], sensor_positions[:, 1],
                  s=150, c='red', marker='s', edgecolors='darkred', linewidth=2,
                  label='Sensor Locations', zorder=3)

        # Add sensor labels
        for i, pos in enumerate(sensor_positions):
            ax.annotate(f'S{i}', (pos[0], pos[1]), fontsize=8,
                       bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7),
                       xytext=(5, 5), textcoords='offset points')

        # Plot trajectory
        if trajectory is not None and len(trajectory) > 0:
            traj_array = trajectory if isinstance(trajectory, np.ndarray) else np.array(trajectory)
            if len(traj_array) > 0:
                # Plot path as line
                ax.plot(traj_array[:, 0], traj_array[:, 1],
                       color=color, linewidth=2, alpha=0.7, label='UAV Path', zorder=2)

                # Plot waypoints
                ax.scatter(traj_array[:, 0], traj_array[:, 1],
                          c=color, s=20, alpha=0.5, zorder=1)

                # Mark start position
                ax.scatter(traj_array[0, 0], traj_array[0, 1],
                          c='green', s=200, marker='o', edgecolors='darkgreen', linewidth=2,
                          label='Start Position', zorder=4)

                # Mark end position
                ax.scatter(traj_array[-1, 0], traj_array[-1, 1],
                          c='blue', s=200, marker='*', edgecolors='darkblue', linewidth=2,
                          label='End Position', zorder=4)

                # Path statistics
                total_distance = np.sum(np.sqrt(np.sum(np.diff(traj_array, axis=0)**2, axis=1)))
                ax.text(0.02, 0.98, f'Path Length: {total_distance:.1f}m\nWaypoints: {len(traj_array)}',
                       transform=ax.transAxes, fontsize=10, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        ax.set_xlabel('X Position (m)', fontsize=11, fontweight='bold')
        ax.set_ylabel('Y Position (m)', fontsize=11, fontweight='bold')
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9)

    plt.tight_layout()
    output_file = OUTPUT_DIR / 'agent_trajectories.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"✓ Trajectory plot saved to {output_file}")
    plt.show()

def plot_comparative_analysis(dqn_df, greedy_smart_df, greedy_dumb_df):
    """Generates the multi-agent comparison graph."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # --- LEFT Y-AXIS: CUMULATIVE REWARD ---
    ax1.set_xlabel('Simulation Step (t)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cumulative Reward', fontsize=12, fontweight='bold', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    # 1. Plot DQN (The Hero)
    if dqn_df is not None and not dqn_df.empty:
        ax1.plot(dqn_df['step'], dqn_df['cumulative_reward'],
                 color='#1f77b4', linewidth=3, label='DQN Agent (Proposed)', marker='o', markersize=5)
        dqn_sat = dqn_df[dqn_df['battery_percent'] < 30]
        if not dqn_sat.empty:
            sat_idx = dqn_sat.index[0]
            ax1.scatter(dqn_df.loc[sat_idx, 'step'],
                       dqn_df.loc[sat_idx, 'cumulative_reward'],
                       color='#1f77b4', s=200, zorder=5, marker='*')

    # 2. Plot Smart Greedy (The Competitor)
    ax1.plot(greedy_smart_df['step'], greedy_smart_df['cumulative_reward'],
             color='#d62728', linewidth=2.5, linestyle='-', label='SF-Aware Greedy (V2)', marker='s', markersize=4)

    # 3. Plot Dumb Greedy (The Baseline)
    ax1.plot(greedy_dumb_df['step'], greedy_dumb_df['cumulative_reward'],
             color='gray', linewidth=2, linestyle=':', alpha=0.8, label='Nearest Sensor Greedy', marker='^', markersize=4)

    # --- RIGHT Y-AXIS: BATTERY (Reference) ---
    ax2 = ax1.twinx()
    ax2.set_ylabel('Battery Level (%)', fontsize=12, fontweight='bold', color='black')
    ax2.tick_params(axis='y', labelcolor='black')

    ax2.plot(greedy_smart_df['step'], greedy_smart_df['battery_percent'],
             color='black', linewidth=1.5, linestyle='--', alpha=0.4, label='Battery Reference')
    ax2.set_ylim(0, 105)

    # --- ANNOTATIONS ---
    sat_smart = greedy_smart_df[greedy_smart_df['battery_percent'] < 30].head(1)

    if not sat_smart.empty:
        step_val = sat_smart['step'].values[0]
        reward_val = sat_smart['cumulative_reward'].values[0]

        ax1.annotate(f'Greedy Saturation\n(t={int(step_val)})',
                     xy=(step_val, reward_val),
                     xytext=(step_val - 300, reward_val + 500000),
                     arrowprops=dict(facecolor='#d62728', shrink=0.05, width=2),
                     fontsize=10, fontweight='bold', color='#d62728',
                     bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#d62728", alpha=0.9))

    # --- FORMATTING ---
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax1.grid(True, linestyle='--', alpha=0.6)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right',
               fancybox=True, shadow=True, fontsize=11, ncol=2)

    plt.title('Comparative Performance: Deep Reinforcement Learning vs. Heuristics',
              fontsize=14, fontweight='bold', pad=15)

    plt.tight_layout()
    output_file = OUTPUT_DIR / 'final_comparison_graph.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Graph saved to {output_file}")
    plt.show()

# ==================== MAIN ====================

def main():
    print("=" * 100)
    print("FAIR BASELINE AGENTS EVALUATION & COMPARISON")
    print("(All agents evaluated on identical environment with same seed)")
    print("=" * 100)
    print(f"\nOutput Directory: {OUTPUT_DIR}")
    print(f"Seed: {PLOT_CONFIG['seed']}\n")

    # Load frame stacking config
    frame_stacking_config = load_frame_stacking_config(DQN_CONFIG_PATH)

    env_kwargs = {
        'grid_size': PLOT_CONFIG['grid_size'],
        'num_sensors': PLOT_CONFIG['num_sensors'],
        'max_steps': PLOT_CONFIG['max_steps'],
        'path_loss_exponent': PLOT_CONFIG['path_loss_exponent'],
        'rssi_threshold': PLOT_CONFIG['rssi_threshold'],
        'sensor_duty_cycle': PLOT_CONFIG['sensor_duty_cycle'],
        'render_mode': None
    }

    agents_config = []

    # ========== STEP 1: Run DQN Agent with VecFrameStack ==========
    print("-" * 100)
    df_dqn = None
    dqn_trajectory = None
    if DQN_MODEL_PATH.exists():
        try:
            print(f"Loading DQN model from {DQN_MODEL_PATH}...")
            model = DQN.load(DQN_MODEL_PATH)
            print("✓ DQN model loaded successfully")

            # Create STACKED environment for DQN
            print("Creating stacked environment for DQN...")
            dqn_stacked_env, dqn_base_env = create_stacked_dqn_env(env_kwargs, frame_stacking_config)

            # Run DQN with stacked environment
            df_dqn, steps_dqn, dqn_trajectory = run_dqn_agent_for_plot(
                model, dqn_stacked_env, dqn_base_env, "DQN Agent", seed=PLOT_CONFIG['seed']
            )
            save_baseline_data("dqn_agent_fresh", df_dqn)
            agents_config.append({
                'name': 'DQN (Proposed)',
                'steps': steps_dqn,
                'final_reward': float(df_dqn['cumulative_reward'].iloc[-1]),
                'final_coverage': float(df_dqn['coverage_percent'].iloc[-1])
            })
            print("✓ DQN evaluation complete")
            dqn_stacked_env.close()
        except Exception as e:
            print(f"⚠ WARNING: Could not run DQN agent: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"⚠ WARNING: DQN model not found at {DQN_MODEL_PATH}")

    # ========== STEP 2: Run Greedy Agents on FRESH environment ==========
    print("\n" + "-" * 100)
    print("Setting up environment for greedy agents...")
    env = UAVEnvironment(**env_kwargs)
    print("✓ Environment created\n")

    print("-" * 100)
    agent_smart = MaxThroughputGreedyV2(env)
    df_smart, steps_smart, smart_trajectory = run_greedy_agent_for_plot(agent_smart, env, "SF-Aware Greedy V2", seed=PLOT_CONFIG['seed'])
    save_baseline_data("greedy_smart_v2", df_smart)
    agents_config.append({
        'name': 'SF-Aware Greedy V2',
        'steps': steps_smart,
        'final_reward': float(df_smart['cumulative_reward'].iloc[-1]),
        'final_coverage': float(df_smart['coverage_percent'].iloc[-1])
    })

    # ========== STEP 3: Run Nearest Sensor Greedy ==========
    print("\n" + "-" * 100)
    agent_dumb = NearestSensorGreedy(env)
    df_dumb, steps_dumb, dumb_trajectory = run_greedy_agent_for_plot(agent_dumb, env, "Nearest Sensor Greedy", seed=PLOT_CONFIG['seed'])
    save_baseline_data("greedy_nearest", df_dumb)
    agents_config.append({
        'name': 'Nearest Sensor Greedy',
        'steps': steps_dumb,
        'final_reward': float(df_dumb['cumulative_reward'].iloc[-1]),
        'final_coverage': float(df_dumb['coverage_percent'].iloc[-1])
    })

    # 5. Save Metadata
    save_comparison_metadata(agents_config)

    # 6. Generate Trajectory Plot
    print("\n" + "-" * 100)
    print("Generating trajectory visualization...")
    plot_trajectories(env, dqn_trajectory, smart_trajectory, dumb_trajectory)

    # 7. Generate Performance Comparison Plot
    print("\n" + "-" * 100)
    print("Generating performance comparison plot...")
    plot_comparative_analysis(df_dqn, df_smart, df_dumb)

    # 8. Summary Statistics
    print("\n" + "=" * 100)
    print("COMPARISON SUMMARY STATISTICS")
    print("=" * 100)

    if df_dqn is not None and not df_dqn.empty:
        print(f"\nDQN Agent:")
        print(f"  Final Reward: {df_dqn['cumulative_reward'].iloc[-1]:>15.1f}")
        print(f"  Final Battery: {df_dqn['battery_percent'].iloc[-1]:>13.1f}%")
        print(f"  Final Coverage: {df_dqn['coverage_percent'].iloc[-1]:>12.1f}%")
        print(f"  Data Collected: {df_dqn['total_data_collected'].iloc[-1]:>11.0f} bytes")
        print(f"  Efficiency: {df_dqn['efficiency'].iloc[-1]:>21.4f} bytes/Wh")

    print(f"\nSmart Greedy V2:")
    print(f"  Final Reward: {df_smart['cumulative_reward'].iloc[-1]:>15.1f}")
    print(f"  Final Battery: {df_smart['battery_percent'].iloc[-1]:>13.1f}%")
    print(f"  Final Coverage: {df_smart['coverage_percent'].iloc[-1]:>12.1f}%")
    print(f"  Data Collected: {df_smart['total_data_collected'].iloc[-1]:>11.0f} bytes")
    print(f"  Efficiency: {df_smart['efficiency'].iloc[-1]:>21.4f} bytes/Wh")

    print(f"\nNearest Sensor Greedy:")
    print(f"  Final Reward: {df_dumb['cumulative_reward'].iloc[-1]:>15.1f}")
    print(f"  Final Battery: {df_dumb['battery_percent'].iloc[-1]:>13.1f}%")
    print(f"  Final Coverage: {df_dumb['coverage_percent'].iloc[-1]:>12.1f}%")
    print(f"  Data Collected: {df_dumb['total_data_collected'].iloc[-1]:>11.0f} bytes")
    print(f"  Efficiency: {df_dumb['efficiency'].iloc[-1]:>21.4f} bytes/Wh")

    env.close()
    print("\n✓ Evaluation complete.\n")

if __name__ == "__main__":
    main()