import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from environment.uav_env import UAVEnvironment

# ==================== CONFIGURATION ====================
MODEL_PATH = "models/dqn_fairness_framestack/dqn_final.zip"  # <--- CHECK THIS PATH
ENV_CONFIG = {
    'grid_size': (50, 50),
    'num_sensors': 20,
    'max_steps': 2100,  # 2100 steps
    'path_loss_exponent': 3.8,  # Urban
    'rssi_threshold': -120.0,
    'render_mode': 'human'  # Enable visualization window
}


def evaluate_and_plot():
    # 1. Create Environment (Same config as training)
    # Note: We don't need the Analysis wrapper here because we will
    # manually extract data step-by-step for the graph.
    env = UAVEnvironment(**ENV_CONFIG)

    # 2. Vectorize and Stack Frames (CRITICAL: Must match training)
    env = DummyVecEnv([lambda: env])
    env = VecFrameStack(env, n_stack=4)

    # 3. Load Model
    print(f"Loading model from: {MODEL_PATH}")
    model = DQN.load(MODEL_PATH)

    # 4. Run One Episode
    obs = env.reset()
    done = False

    # storage for plotting
    history = {
        'step': [], 'reward': [], 'battery': [],
        'coverage': [], 'data_collected': []
    }

    cumulative_reward = 0
    step_count = 0

    print("Starting Evaluation Run...")

    try:
        while not done:
            # Predict action (Deterministic = True means no random exploration)
            action, _ = model.predict(obs, deterministic=True)

            obs, reward, done, info = env.step(action)

            # Access the underlying environment to get stats
            # env is VecFrameStack -> DummyVecEnv -> UAVEnvironment
            real_env = env.envs[0]

            cumulative_reward += reward[0]
            step_count += 1

            # Store data every 50 steps to keep graph clean
            if step_count % 50 == 0 or done:
                history['step'].append(step_count)
                history['reward'].append(cumulative_reward)
                history['battery'].append(real_env.uav.get_battery_percentage())

                coverage_pct = (len(real_env.sensors_visited) / real_env.num_sensors) * 100
                history['coverage'].append(coverage_pct)

                print(
                    f"Step {step_count}: Rew={cumulative_reward:.0f} Bat={history['battery'][-1]:.1f}% Cov={coverage_pct:.1f}%")

            # Render the checkmarks/rings
            real_env.render()

    except KeyboardInterrupt:
        print("Stopping evaluation...")

    print("Episode Finished.")

    # 5. Generate the Dissertation Graph
    plot_results(history)


def plot_results(history):
    df = pd.DataFrame(history)

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Reward (Blue)
    color = 'tab:blue'
    ax1.set_xlabel('Time Steps')
    ax1.set_ylabel('Cumulative Reward', color=color, fontweight='bold')
    ax1.plot(df['step'], df['reward'], color=color, linewidth=2.5)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.7)

    # Battery (Red)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Battery Level (%)', color=color, fontweight='bold')
    ax2.plot(df['step'], df['battery'], color=color, linewidth=2.5, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.set_ylim(0, 100)

    # Saturation Line
    # Find step where battery dropped < 30%
    saturation_point = df[df['battery'] < 30].head(1)
    if not saturation_point.empty:
        step_val = saturation_point['step'].values[0]
        plt.axvline(x=step_val, color='black', linestyle=':', label='Saturation Point')
        plt.text(step_val + 50, 50, 'Saturation Point\n(Battery < 30%)', fontsize=10)

    plt.title('DQN Agent Evaluation: Reward vs. Battery Sustainability')
    fig.tight_layout()
    plt.savefig('evaluation_result_graph.png', dpi=300)
    print("Graph saved to evaluation_result_graph.png")
    plt.show()


if __name__ == "__main__":
    evaluate_and_plot()