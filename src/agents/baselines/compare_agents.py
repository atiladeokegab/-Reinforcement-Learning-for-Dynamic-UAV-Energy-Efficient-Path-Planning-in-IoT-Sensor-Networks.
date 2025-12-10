"""
Dissertation Comparison Plotter
Generates the final 'DQN vs. Greedy' performance graph.
Saves baseline data to the baselines directory.

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

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from environment.uav_env import UAVEnvironment
from agents.baselines.greedy_agents import MaxThroughputGreedyV2, NearestSensorGreedy

# ==================== CONFIGURATION ====================
BASELINES_DIR = Path(__file__).parent  # C:\Users\okeat\final_year\src\agents\baselines
OUTPUT_DIR = BASELINES_DIR / "baseline_results"
OUTPUT_DIR.mkdir(exist_ok=True)

PLOT_CONFIG = {
    'grid_size': (50, 50),       # MUST match your DRL training
    'num_sensors': 20,
    'max_steps': 2100,
    'path_loss_exponent': 3.8,
    'rssi_threshold': -120.0,
    'dqn_csv_path': '../dqn/dqn_evaluation_results/dqn_results.csv',  # Path to your DRL data
    'seed': 42
}

# ==================== DATA SAVING FUNCTIONS ====================

def save_baseline_data(agent_name, history_df):
    """Saves baseline agent data to CSV file in the baselines directory."""
    output_file = OUTPUT_DIR / f"{agent_name}_results.csv"
    history_df.to_csv(output_file, index=False)
    print(f"✓ Saved {agent_name} data to {output_file}")
    return output_file

def save_comparison_metadata():
    """Saves metadata about the comparison run."""
    metadata = {
        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
        'grid_size': PLOT_CONFIG['grid_size'],
        'num_sensors': PLOT_CONFIG['num_sensors'],
        'max_steps': PLOT_CONFIG['max_steps'],
        'seed': PLOT_CONFIG['seed'],
        'output_directory': str(OUTPUT_DIR)
    }

    metadata_file = OUTPUT_DIR / "comparison_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=4)

    print(f"✓ Saved metadata to {metadata_file}")

# ==================== AGENT EVALUATION ====================

def run_agent_for_plot(agent, env, name="Agent", seed=42):
    """Runs a single episode and returns step-by-step history."""
    print(f"\nRunning {name}...")
    obs, _ = env.reset(seed=seed)
    done = False

    history = {
        'step': [],
        'cumulative_reward': [],
        'battery_percent': [],
        'battery_wh': [],
        'coverage_percent': [],
        'episode_step': 0
    }

    cumulative_reward = 0
    step_count = 0

    while not done:
        action = agent.select_action(obs)
        obs, reward, done, truncated, info = env.step(action)
        cumulative_reward += reward
        step_count += 1

        # Log data every 50 steps (to match DRL log frequency)
        if env.current_step % 50 == 0 or done or truncated:
            battery_pct = env.uav.get_battery_percentage()
            coverage_pct = (len(env.sensors_visited) / env.num_sensors) * 100 if hasattr(env, 'sensors_visited') else 0

            history['step'].append(env.current_step)
            history['cumulative_reward'].append(cumulative_reward)
            history['battery_percent'].append(battery_pct)
            history['battery_wh'].append(env.uav.battery)
            history['coverage_percent'].append(coverage_pct)

            print(f"  Step {env.current_step:>4}: Reward={cumulative_reward:>10.1f}, "
                  f"Battery={battery_pct:>5.1f}%, Coverage={coverage_pct:>5.1f}%")

        if done or truncated:
            break

    history['episode_step'] = step_count
    return pd.DataFrame({k: v for k, v in history.items() if k != 'episode_step'}), step_count

# ==================== PLOTTING ====================

def plot_comparative_analysis(dqn_df, greedy_smart_df, greedy_dumb_df):
    """Generates the multi-agent comparison graph."""
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(14, 8))

    # --- LEFT Y-AXIS: CUMULATIVE REWARD ---
    ax1.set_xlabel('Simulation Step (t)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Cumulative Reward', fontsize=12, fontweight='bold', color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    # 1. Plot DQN (The Hero)
    if dqn_df is not None:
        ax1.plot(dqn_df['step'], dqn_df['cumulative_reward'],
                 color='#1f77b4', linewidth=3, label='DQN Agent (Proposed)', marker='o', markersize=5)
        # Mark DQN's saturation point if visible
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

    # Plot Battery depletion for reference
    ax2.plot(greedy_smart_df['step'], greedy_smart_df['battery_percent'],
             color='black', linewidth=1.5, linestyle='--', alpha=0.4, label='Battery Reference')
    ax2.set_ylim(0, 105)

    # --- ANNOTATIONS ---
    # Find Saturation Points (where battery < 30%)
    sat_smart = greedy_smart_df[greedy_smart_df['battery_percent'] < 30].head(1)

    if not sat_smart.empty:
        step_val = sat_smart['step'].values[0]
        reward_val = sat_smart['cumulative_reward'].values[0]

        # Draw Arrow for Greedy Failure
        ax1.annotate(f'Greedy Saturation\n(t={int(step_val)})',
                     xy=(step_val, reward_val),
                     xytext=(step_val - 300, reward_val + 500000),
                     arrowprops=dict(facecolor='#d62728', shrink=0.05, width=2),
                     fontsize=10, fontweight='bold', color='#d62728',
                     bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#d62728", alpha=0.9))

    # --- FORMATTING ---
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(0,0))
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Combined Legend
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
    print("BASELINE AGENTS EVALUATION & COMPARISON")
    print("=" * 100)
    print(f"\nOutput Directory: {OUTPUT_DIR}\n")

    # 1. Setup Environment
    print("Setting up environment...")
    env = UAVEnvironment(
        grid_size=PLOT_CONFIG['grid_size'],
        num_sensors=PLOT_CONFIG['num_sensors'],
        max_steps=PLOT_CONFIG['max_steps'],
        path_loss_exponent=PLOT_CONFIG['path_loss_exponent'],
        rssi_threshold=PLOT_CONFIG['rssi_threshold'],
        render_mode=None
    )

    # 2. Run Smart Greedy Baseline
    agent_smart = MaxThroughputGreedyV2(env)
    df_smart, steps_smart = run_agent_for_plot(agent_smart, env, "SF-Aware Greedy V2", seed=PLOT_CONFIG['seed'])
    save_baseline_data("greedy_smart_v2", df_smart)

    # 3. Run Nearest Sensor Greedy Baseline
    agent_dumb = NearestSensorGreedy(env)
    df_dumb, steps_dumb = run_agent_for_plot(agent_dumb, env, "Nearest Sensor Greedy", seed=PLOT_CONFIG['seed'])
    save_baseline_data("greedy_nearest", df_dumb)

    # 4. Load DRL Data
    df_dqn = None
    try:
        df_dqn = pd.read_csv(PLOT_CONFIG['dqn_csv_path'])
        print(f"\n✓ Loaded DQN data from {PLOT_CONFIG['dqn_csv_path']}")
    except FileNotFoundError:
        print(f"\n⚠ WARNING: '{PLOT_CONFIG['dqn_csv_path']}' not found. Plotting without DRL line.")

    # 5. Save Metadata
    save_comparison_metadata()

    # 6. Generate Plot
    print("\nGenerating comparison plot...")
    plot_comparative_analysis(df_dqn, df_smart, df_dumb)

    # 7. Summary Statistics
    print("\n" + "=" * 100)
    print("BASELINE SUMMARY STATISTICS")
    print("=" * 100)

    print(f"\nSmart Greedy V2:")
    print(f"  Final Reward: {df_smart['cumulative_reward'].iloc[-1]:>12.1f}")
    print(f"  Final Battery: {df_smart['battery_percent'].iloc[-1]:>10.1f}%")
    print(f"  Final Coverage: {df_smart['coverage_percent'].iloc[-1]:>9.1f}%")

    print(f"\nNearest Sensor Greedy:")
    print(f"  Final Reward: {df_dumb['cumulative_reward'].iloc[-1]:>12.1f}")
    print(f"  Final Battery: {df_dumb['battery_percent'].iloc[-1]:>10.1f}%")
    print(f"  Final Coverage: {df_dumb['coverage_percent'].iloc[-1]:>9.1f}%")

    if df_dqn is not None:
        print(f"\nDQN Agent:")
        print(f"  Final Reward: {df_dqn['cumulative_reward'].iloc[-1]:>12.1f}")
        print(f"  Final Battery: {df_dqn['battery_percent'].iloc[-1]:>10.1f}%")
        print(f"  Final Coverage: {df_dqn['coverage_percent'].iloc[-1]:>9.1f}%")

        improvement = ((df_dqn['cumulative_reward'].iloc[-1] - df_smart['cumulative_reward'].iloc[-1])
                      / abs(df_smart['cumulative_reward'].iloc[-1])) * 100
        print(f"\n  DQN Improvement over Smart Greedy: {improvement:>6.2f}%")

    env.close()
    print("\n✓ Evaluation complete.\n")

if __name__ == "__main__":
    main()