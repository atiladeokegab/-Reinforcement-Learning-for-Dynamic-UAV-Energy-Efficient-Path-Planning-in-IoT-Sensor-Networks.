"""
Training Script for Q-Learning UAV Path Planning

Trains a Q-Learning agent to navigate a UAV through a grid environment
to collect data from IoT sensors efficiently.

Usage:
    python train.py --episodes 1000 --render False

Author: ATILADE GABRIEL OKE
Date: October 2025
Project: Reinforcement Learning for Dynamic UAV Energy-Efficient Path Planning
         in IoT Sensor Networks
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / 'src'))

import numpy as np
import argparse
from datetime import datetime
import json
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

from src.environment.uav_env import UAVEnvironment
from src.agents.q_learning_implementation.q_learning_agent import QLearningAgent


class TrainingLogger:
    """Logs training metrics and statistics."""

    def __init__(self, log_dir: str = "logs"):
        """Initialize logger."""
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Metrics storage
        self.episode_rewards = []
        self.episode_steps = []
        self.episode_success = []
        self.episode_coverage = []
        self.episode_battery_used = []
        self.episode_data_collected = []
        self.epsilon_history = []

        # Running averages (last 100 episodes)
        self.window_size = 100

    def log_episode(self, episode: int, metrics: dict):
        """Log metrics for one episode."""
        self.episode_rewards.append(metrics['total_reward'])
        self.episode_steps.append(metrics['steps'])
        self.episode_success.append(1 if metrics['success'] else 0)
        self.episode_coverage.append(metrics['coverage_percent'])
        self.episode_battery_used.append(metrics['battery_used'])
        self.episode_data_collected.append(metrics['data_collected'])
        self.epsilon_history.append(metrics['epsilon'])

    def get_running_average(self, data, window=None):
        """Calculate running average."""
        if window is None:
            window = self.window_size

        if len(data) < window:
            return np.mean(data) if data else 0

        return np.mean(data[-window:])

    def print_progress(self, episode: int, metrics: dict):
        """Print training progress."""
        avg_reward = self.get_running_average(self.episode_rewards)
        avg_steps = self.get_running_average(self.episode_steps)
        avg_success = self.get_running_average(self.episode_success)

        print(f"Episode {episode:4d} | "
              f"Reward: {metrics['total_reward']:7.2f} (avg: {avg_reward:7.2f}) | "
              f"Steps: {metrics['steps']:3d} (avg: {avg_steps:5.1f}) | "
              f"Success: {metrics['success']} (rate: {avg_success:.2%}) | "
              f"ε: {metrics['epsilon']:.3f}")

    def save_metrics(self, filename: str = None):
        """Save all metrics to JSON file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.log_dir}/training_metrics_{timestamp}.json"

        metrics = {
            'episode_rewards': self.episode_rewards,
            'episode_steps': self.episode_steps,
            'episode_success': self.episode_success,
            'episode_coverage': self.episode_coverage,
            'episode_battery_used': self.episode_battery_used,
            'episode_data_collected': self.episode_data_collected,
            'epsilon_history': self.epsilon_history,
            'total_episodes': len(self.episode_rewards)
        }

        with open(filename, 'w') as f:
            json.dump(metrics, f, indent=2)

        print(f"\n✓ Metrics saved to {filename}")
        return filename

    def plot_training_curves(self, save_path: str = None):
        """Plot training curves."""
        if save_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"{self.log_dir}/training_curves_{timestamp}.png"

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Q-Learning Training Progress', fontsize=16, fontweight='bold')

        episodes = range(1, len(self.episode_rewards) + 1)

        # Plot 1: Rewards
        ax = axes[0, 0]
        ax.plot(episodes, self.episode_rewards, alpha=0.3, label='Episode Reward')
        if len(self.episode_rewards) >= self.window_size:
            running_avg = [self.get_running_average(self.episode_rewards[:i + 1])
                           for i in range(len(self.episode_rewards))]
            ax.plot(episodes, running_avg, linewidth=2, label=f'{self.window_size}-Episode Average')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('Episode Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Steps per Episode
        ax = axes[0, 1]
        ax.plot(episodes, self.episode_steps, alpha=0.3, label='Episode Steps')
        if len(self.episode_steps) >= self.window_size:
            running_avg = [self.get_running_average(self.episode_steps[:i + 1])
                           for i in range(len(self.episode_steps))]
            ax.plot(episodes, running_avg, linewidth=2, label=f'{self.window_size}-Episode Average')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.set_title('Steps per Episode')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Success Rate
        ax = axes[0, 2]
        if len(self.episode_success) >= self.window_size:
            success_rate = [self.get_running_average(self.episode_success[:i + 1]) * 100
                            for i in range(len(self.episode_success))]
            ax.plot(episodes, success_rate, linewidth=2, color='green')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Success Rate (%)')
        ax.set_title(f'Success Rate ({self.window_size}-Episode Average)')
        ax.set_ylim(0, 105)
        ax.grid(True, alpha=0.3)

        # Plot 4: Coverage
        ax = axes[1, 0]
        ax.plot(episodes, self.episode_coverage, alpha=0.3, label='Coverage')
        if len(self.episode_coverage) >= self.window_size:
            running_avg = [self.get_running_average(self.episode_coverage[:i + 1])
                           for i in range(len(self.episode_coverage))]
            ax.plot(episodes, running_avg, linewidth=2, label=f'{self.window_size}-Episode Average')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Coverage (%)')
        ax.set_title('Sensor Coverage')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 5: Epsilon Decay
        ax = axes[1, 1]
        ax.plot(episodes, self.epsilon_history, linewidth=2, color='orange')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Epsilon')
        ax.set_title('Exploration Rate (ε)')
        ax.grid(True, alpha=0.3)

        # Plot 6: Battery Usage
        ax = axes[1, 2]
        ax.plot(episodes, self.episode_battery_used, alpha=0.3, label='Battery Used')
        if len(self.episode_battery_used) >= self.window_size:
            running_avg = [self.get_running_average(self.episode_battery_used[:i + 1])
                           for i in range(len(self.episode_battery_used))]
            ax.plot(episodes, running_avg, linewidth=2, label=f'{self.window_size}-Episode Average')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Battery Used (Wh)')
        ax.set_title('Battery Consumption')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ Training curves saved to {save_path}")

        return save_path


def train_agent(args):
    """Main training function."""

    print("=" * 70)
    print("Q-Learning Training for UAV Path Planning")
    print("=" * 70)
    print()

    # Create directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    # Create environment
    print("Creating environment...")
    env = UAVEnvironment(
        grid_size=args.grid_size,
        num_sensors=args.num_sensors,
        max_steps=args.max_steps,
        rssi_threshold=args.rssi_threshold,
        render_mode='human' if args.render else None
    )

    print(f"✓ Environment created")
    print(f"  Grid: {args.grid_size[0]}×{args.grid_size[1]}")
    print(f"  Sensors: {args.num_sensors}")
    print(f"  Max steps: {args.max_steps}")
    print(f"  Action space: {env.action_space}")
    print(f"  Observation space: {env.observation_space.shape}")
    print()

    # Create agent
    print("Creating Q-Learning agent...")
    agent = QLearningAgent(
        num_actions=env.action_space.n,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        epsilon=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay
    )

    print(f"✓ Agent created: {agent}")
    print()

    # Load checkpoint if specified
    if args.load_checkpoint:
        print(f"Loading checkpoint from {args.load_checkpoint}...")
        agent.load(args.load_checkpoint)
        print()

    # Create logger
    logger = TrainingLogger(log_dir="logs")

    # Training loop
    print("=" * 70)
    print(f"Starting training for {args.episodes} episodes...")
    print("=" * 70)
    print()

    best_reward = -float('inf')

    try:
        for episode in tqdm(range(1, args.episodes + 1), desc="Training"):
            # Reset environment
            obs, info = env.reset()
            state = agent.discretize_state(obs, grid_size=args.grid_size)

            episode_reward = 0.0
            episode_steps = 0
            done = False

            # Episode loop
            while not done:
                # Select action
                action = agent.select_action(state, training=True)

                # Take step
                next_obs, reward, terminated, truncated, info = env.step(action)
                next_state = agent.discretize_state(next_obs, grid_size=args.grid_size)

                done = terminated or truncated

                # Update Q-values
                agent.update(state, action, reward, next_state, done)

                # Render if enabled
                if args.render:
                    env.render()

                # Update tracking
                episode_reward += reward
                episode_steps += 1
                state = next_state

            # Decay epsilon
            agent.decay_epsilon()

            # Log episode metrics
            metrics = {
                'total_reward': episode_reward,
                'steps': episode_steps,
                'success': terminated,  # True if mission complete
                'coverage_percent': info['coverage_percentage'],
                'battery_used': 274.0 - info['battery'],
                'data_collected': info['total_data_collected'],
                'epsilon': agent.epsilon
            }

            logger.log_episode(episode, metrics)

            # Print progress every N episodes
            if episode % args.log_interval == 0:
                logger.print_progress(episode, metrics)

            # Save best model
            if episode_reward > best_reward:
                best_reward = episode_reward
                agent.save(f"checkpoints/best_agent.pkl")

            # Save checkpoint periodically
            if episode % args.save_interval == 0:
                agent.save(f"checkpoints/agent_episode_{episode}.pkl")
                logger.save_metrics(f"logs/metrics_episode_{episode}.json")

    except KeyboardInterrupt:
        print("\n\n⏸️  Training interrupted by user")

    # Close environment
    env.close()

    # Final statistics
    print()
    print("=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print()

    final_stats = agent.get_statistics()
    print(f"Final Statistics:")
    print(f"  Total episodes: {episode}")
    print(f"  Q-table size: {final_stats['q_table_size']:,}")
    print(f"  Total updates: {final_stats['total_updates']:,}")
    print(f"  Final epsilon: {final_stats['epsilon']:.4f}")
    print(f"  Best reward: {best_reward:.2f}")
    print()

    avg_reward = logger.get_running_average(logger.episode_rewards)
    avg_success = logger.get_running_average(logger.episode_success)
    avg_coverage = logger.get_running_average(logger.episode_coverage)

    print(f"Last 100 Episodes Average:")
    print(f"  Reward: {avg_reward:.2f}")
    print(f"  Success rate: {avg_success:.2%}")
    print(f"  Coverage: {avg_coverage:.1f}%")
    print()

    # Save final results
    print("Saving results...")
    agent.save("checkpoints/final_agent.pkl")
    logger.save_metrics("logs/final_metrics.json")
    logger.plot_training_curves("results/training_curves.png")

    print()
    print("=" * 70)
    print("✓ All results saved!")
    print("=" * 70)


def main():
    """Parse arguments and start training."""
    parser = argparse.ArgumentParser(description='Train Q-Learning agent for UAV path planning')

    # Environment parameters
    parser.add_argument('--grid-size', type=int, nargs=2, default=[10, 10],
                        help='Grid size (width height)')
    parser.add_argument('--num-sensors', type=int, default=20,
                        help='Number of sensors')
    parser.add_argument('--max-steps', type=int, default=200,
                        help='Maximum steps per episode')
    parser.add_argument('--rssi-threshold', type=float, default=-75.0,
                        help='RSSI threshold for LoRa communication (dBm)')

    # Agent parameters
    parser.add_argument('--learning-rate', type=float, default=0.1,
                        help='Learning rate (alpha)')
    parser.add_argument('--discount-factor', type=float, default=0.95,
                        help='Discount factor (gamma)')
    parser.add_argument('--epsilon-start', type=float, default=1.0,
                        help='Initial exploration rate')
    parser.add_argument('--epsilon-min', type=float, default=0.01,
                        help='Minimum exploration rate')
    parser.add_argument('--epsilon-decay', type=float, default=0.995,
                        help='Epsilon decay rate per episode')

    # Training parameters
    parser.add_argument('--episodes', type=int, default=1000,
                        help='Number of training episodes')
    parser.add_argument('--log-interval', type=int, default=10,
                        help='Log progress every N episodes')
    parser.add_argument('--save-interval', type=int, default=100,
                        help='Save checkpoint every N episodes')

    # Other
    parser.add_argument('--render', action='store_true',
                        help='Render environment during training')
    parser.add_argument('--load-checkpoint', type=str, default=None,
                        help='Load agent from checkpoint')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Set random seeds
    np.random.seed(args.seed)

    # Start training
    train_agent(args)


if __name__ == "__main__":
    main()