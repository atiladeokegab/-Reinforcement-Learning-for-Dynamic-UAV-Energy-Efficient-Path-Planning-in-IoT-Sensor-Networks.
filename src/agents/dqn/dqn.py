"""
Complete DQN Training Script with Fairness Constraints

Trains DQN agent for UAV data collection with SF-aware optimization
and fairness constraints.

Author: ATILADE GABRIEL OKE
Date: November 2025
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import torch
from datetime import datetime

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
    BaseCallback
)

from environment.uav_env import UAVEnvironment


# ==================== CUSTOM CALLBACK ====================

class FairnessMetricsCallback(BaseCallback):
    """
    Custom callback to track fairness and SF-awareness metrics during training.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_coverages = []
        self.episode_efficiencies = []
        self.episode_data_losses = []
        self.episode_max_urgencies = []
        self.episode_fairness_stds = []
        self.n_episodes = 0

    def _on_step(self) -> bool:
        # Check if episode ended
        if self.locals.get('dones')[0]:
            self.n_episodes += 1

            # Get environment
            env = self.training_env.envs[0]
            if hasattr(env, 'unwrapped'):
                env = env.unwrapped

            # Get info
            info = self.locals.get('infos')[0]

            # Basic metrics
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])

            self.episode_coverages.append(info.get('coverage_percentage', 0))

            # Calculate collection efficiency
            total_generated = sum(s.total_data_generated for s in env.sensors)
            total_collected = sum(s.total_data_transmitted for s in env.sensors)
            efficiency = (total_collected / total_generated * 100) if total_generated > 0 else 0
            self.episode_efficiencies.append(efficiency)

            # Fairness metrics
            total_data_loss = sum(s.total_data_lost for s in env.sensors)
            self.episode_data_losses.append(total_data_loss)

            self.episode_max_urgencies.append(info.get('max_urgency', 0))

            # Calculate fairness (std of per-sensor collection rates)
            sensor_collection_rates = []
            for s in env.sensors:
                if s.total_data_generated > 0:
                    rate = (s.total_data_transmitted / s.total_data_generated) * 100
                    sensor_collection_rates.append(rate)

            if sensor_collection_rates:
                fairness_std = np.std(sensor_collection_rates)
                self.episode_fairness_stds.append(fairness_std)

            # Print progress every 10 episodes
            if self.n_episodes % 10 == 0:
                recent_rewards = self.episode_rewards[-10:] if len(self.episode_rewards) >= 10 else self.episode_rewards
                recent_efficiency = self.episode_efficiencies[-10:] if len(
                    self.episode_efficiencies) >= 10 else self.episode_efficiencies
                recent_coverage = self.episode_coverages[-10:] if len(
                    self.episode_coverages) >= 10 else self.episode_coverages
                recent_loss = self.episode_data_losses[-10:] if len(
                    self.episode_data_losses) >= 10 else self.episode_data_losses

                print(f"\n📊 Episode {self.n_episodes} | "
                      f"Reward: {np.mean(recent_rewards):.1f} | "
                      f"Coverage: {np.mean(recent_coverage):.1f}% | "
                      f"Efficiency: {np.mean(recent_efficiency):.1f}% | "
                      f"Loss: {np.mean(recent_loss):.1f}B")

        return True

    def get_metrics(self):
        """Return collected metrics."""
        return {
            'rewards': self.episode_rewards,
            'coverages': self.episode_coverages,
            'efficiencies': self.episode_efficiencies,
            'data_losses': self.episode_data_losses,
            'max_urgencies': self.episode_max_urgencies,
            'fairness_stds': self.episode_fairness_stds,
        }


# ==================== CONFIGURATION ====================

# A. Environment Configuration
ENV_CONFIG = {
    'grid_size': (50, 50),  #  Manageable for RTX 3050 Ti
    'num_sensors': 20,  #  Full problem
    'max_steps': 500,  #  Reasonable episode length
    'sensor_duty_cycle': 10.0,  #  10% duty cycle
    'penalty_data_loss': -5000.0,  #  FAIRNESS CONSTRAINT
    'reward_urgency_reduction': 20.0,  #  FAIRNESS BONUS
    'render_mode': None,  # No rendering during training
}

# B. DQN Hyperparameters (Optimized for RTX 3050 Ti)
HYPERPARAMS = {
    'policy': 'MlpPolicy',
    'learning_rate': 1e-4,  #  Good starting point
    'buffer_size': 50_000,  #  Fits in 4GB VRAM
    'batch_size': 32,  #  Optimal for RTX 3050 Ti
    'gamma': 0.99,  #  Balance short/long-term
    'learning_starts': 1000,  #  Collect data before learning
    'train_freq': 4,  #  Train every 4 steps
    'target_update_interval': 1000,  #  Update target network
    'exploration_initial_eps': 1.0,  #  Start with full exploration
    'exploration_fraction': 0.6,  #  Decay over 30% of training
    'exploration_final_eps': 0.01,  #  Minimum exploration
    'tau': 1.0,  # Hard target updates
    'gradient_steps': 1,  #  One gradient step per train
    'policy_kwargs': {
        'net_arch': [512, 256, 128] #  Neural network: 2 hidden layers, 256 units each
    },
}

# C. Training Parameters
TRAINING_CONFIG = {
    'total_timesteps': 100_000,  # Start with 100K (5-10 min on GPU)
    'eval_freq': 5000,  # Evaluate every 5K steps
    'n_eval_episodes': 5,  # 5 episodes for evaluation
    'save_freq': 10_000,  #  Save checkpoint every 10K
    'log_interval': 10,  # Log every 10 updates
}

# D. Paths
SAVE_DIR = Path("models/dqn_fairness")
LOG_DIR = Path("logs/dqn_fairness")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ==================== MAIN TRAINING ====================

def main():
    print("DQN TRAINING WITH FAIRNESS CONSTRAINTS")

    # Check GPU
    print("\nHardware Configuration:")
    print(f"  PyTorch version: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Print configuration
    print("\nEnvironment Configuration:")
    for key, value in ENV_CONFIG.items():
        print(f"  {key}: {value}")

    print("\nDQN Hyperparameters:")
    for key, value in HYPERPARAMS.items():
        if key != 'policy_kwargs':
            print(f"  {key}: {value}")
    print(f"  Neural Network: {HYPERPARAMS['policy_kwargs']['net_arch']}")

    print("\nTraining Configuration:")
    for key, value in TRAINING_CONFIG.items():
        if isinstance(value, int):
            print(f"  {key}: {value:,}")
        else:
            print(f"  {key}: {value}")

    print("\nSave Paths:")
    print(f"  Models: {SAVE_DIR}")
    print(f"  Logs: {LOG_DIR}")

    # ==================== CREATE ENVIRONMENTS ====================

    print("\n" + "=" * 100)
    print("CREATING ENVIRONMENTS")
    print("=" * 100)

    def make_env():
        """Factory function to create environment."""
        env = UAVEnvironment(**ENV_CONFIG)
        env = Monitor(env)  # Wrap for monitoring
        return env

    # Training environment
    train_env = DummyVecEnv([make_env])
    print("✓ Training environment created")

    # Evaluation environment
    eval_env = DummyVecEnv([make_env])
    print("✓ Evaluation environment created")

    # ==================== CREATE DQN MODEL ====================

    print("\n" + "=" * 100)
    print("CREATING DQN MODEL")
    print("=" * 100)

    model = DQN(
        HYPERPARAMS['policy'],
        train_env,
        learning_rate=HYPERPARAMS['learning_rate'],
        buffer_size=HYPERPARAMS['buffer_size'],
        batch_size=HYPERPARAMS['batch_size'],
        gamma=HYPERPARAMS['gamma'],
        learning_starts=HYPERPARAMS['learning_starts'],
        train_freq=HYPERPARAMS['train_freq'],
        target_update_interval=HYPERPARAMS['target_update_interval'],
        exploration_initial_eps=HYPERPARAMS['exploration_initial_eps'],
        exploration_fraction=HYPERPARAMS['exploration_fraction'],
        exploration_final_eps=HYPERPARAMS['exploration_final_eps'],
        tau=HYPERPARAMS['tau'],
        gradient_steps=HYPERPARAMS['gradient_steps'],
        policy_kwargs=HYPERPARAMS['policy_kwargs'],
        verbose=1,
        tensorboard_log=str(LOG_DIR),
        device='auto',  #  Automatically use GPU if available
    )

    print(f"✓ DQN model created")
    print(f"  Device: {model.device}")
    print(f"  Policy: {model.policy}")

    # ==================== SETUP CALLBACKS ====================

    print("\n" + "=" * 100)
    print("SETTING UP CALLBACKS")
    print("=" * 100)

    # Checkpoint callback
    checkpoint_callback = CheckpointCallback(
        save_freq=TRAINING_CONFIG['save_freq'],
        save_path=str(SAVE_DIR),
        name_prefix="dqn_checkpoint",
        save_replay_buffer=True,
        save_vecnormalize=True,
    )
    print("✓ Checkpoint callback configured")

    # Evaluation callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=str(SAVE_DIR),
        log_path=str(LOG_DIR),
        eval_freq=TRAINING_CONFIG['eval_freq'],
        n_eval_episodes=TRAINING_CONFIG['n_eval_episodes'],
        deterministic=True,
        render=False,
    )
    print("✓ Evaluation callback configured")

    # Fairness metrics callback
    fairness_callback = FairnessMetricsCallback()
    print("✓ Fairness metrics callback configured")

    # Combine callbacks
    callback = CallbackList([checkpoint_callback, eval_callback, fairness_callback])

    # ==================== TRAIN ====================

    print("\n" + "=" * 100)
    print("STARTING TRAINING")
    print("=" * 100)
    print(f"\ Monitor training in real-time:")
    print(f"   tensorboard --logdir {LOG_DIR}")
    print(f"\nTraining for {TRAINING_CONFIG['total_timesteps']:,} timesteps...")
    print(f"   Estimated time: ~5-10 minutes on RTX 3050 Ti")
    print("=" * 100)

    start_time = datetime.now()

    try:
        model.learn(
            total_timesteps=TRAINING_CONFIG['total_timesteps'],
            callback=callback,
            log_interval=TRAINING_CONFIG['log_interval'],
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n\n Training interrupted by user!")

    training_time = (datetime.now() - start_time).total_seconds()

    # ==================== SAVE FINAL MODEL ====================

    print("\n" + "=" * 100)
    print("SAVING FINAL MODEL")
    print("=" * 100)

    final_model_path = SAVE_DIR / "dqn_final"
    model.save(str(final_model_path))
    print(f"✓ Final model saved: {final_model_path}.zip")

    # Save metrics
    metrics = fairness_callback.get_metrics()
    import json
    metrics_path = SAVE_DIR / "training_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in metrics.items()}, f, indent=2)
    print(f"✓ Training metrics saved: {metrics_path}")

    # ==================== TRAINING SUMMARY ====================

    print("\n" + "=" * 100)
    print("TRAINING COMPLETE")
    print("=" * 100)
    print(f"\nTraining Statistics:")
    print(f"  Total time: {training_time / 60:.1f} minutes")
    print(f"  Episodes completed: {len(metrics['rewards'])}")
    print(f"  Timesteps: {TRAINING_CONFIG['total_timesteps']:,}")

    if len(metrics['rewards']) > 0:
        print(f"\nPerformance Metrics (Last 50 episodes):")
        last_n = min(50, len(metrics['rewards']))
        print(f"  Average Reward: {np.mean(metrics['rewards'][-last_n:]):.2f}")
        print(f"  Average Coverage: {np.mean(metrics['coverages'][-last_n:]):.2f}%")
        print(f"  Average Efficiency: {np.mean(metrics['efficiencies'][-last_n:]):.2f}%")
        print(f"  Average Data Loss: {np.mean(metrics['data_losses'][-last_n:]):.2f} bytes")

        if len(metrics['fairness_stds']) > 0:
            print(f"  Fairness (σ): {np.mean(metrics['fairness_stds'][-last_n:]):.2f}% (lower = fairer)")

    print(f"\nSaved Files:")
    print(f"  Best model: {SAVE_DIR}/best_model.zip")
    print(f"  Final model: {final_model_path}.zip")
    print(f"  Checkpoints: {SAVE_DIR}/dqn_checkpoint_*.zip")
    print(f"  Metrics: {metrics_path}")

    print(f"\nTensorBoard Logs:")
    print(f"  {LOG_DIR}")

    print("\n" + "=" * 100)
    print("ALL DONE!")
    print("=" * 100)

    print(f"\n Next Steps:")
    print(f"  1. View logs: tensorboard --logdir {LOG_DIR}")
    print(f"  2. Load model: model = DQN.load('{final_model_path}.zip')")
    print(f"  3. Evaluate model: python evaluate_dqn.py")
    print(f"  4. Compare with baselines")

    # Close environments
    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()