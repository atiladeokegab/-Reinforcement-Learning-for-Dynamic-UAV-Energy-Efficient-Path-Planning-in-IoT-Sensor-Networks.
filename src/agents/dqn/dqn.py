"""
Complete DQN Training Script with Data Persistence Fix

Solves the issue where metrics are lost when the environment auto-resets.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import json
from datetime import datetime

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
    CallbackList,
    BaseCallback
)

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from environment.uav_env import UAVEnvironment

# ==================== 1. THE SMART WRAPPER (FIX) ====================

class AnalysisUAVEnv(UAVEnvironment):
    """
    Smart Wrapper: Saves the final state of the simulation
    the instant before the environment resets.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Persistent storage for the last finished episode
        self.last_episode_stats = None

    def reset(self, **kwargs):
        # 1. SNAPSHOT: If we have run a simulation (step > 0), save the data now!
        # This runs BEFORE the super().reset() wipes everything.
        if hasattr(self, 'sensors') and self.current_step > 0:

            # Calculate metrics on the 'finished' environment
            total_generated = sum(s.total_data_generated for s in self.sensors)
            total_collected = sum(s.total_data_transmitted for s in self.sensors)
            total_lost = sum(s.total_data_lost for s in self.sensors)

            sensor_rates = []
            for s in self.sensors:
                if s.total_data_generated > 0:
                    rate = (s.total_data_transmitted / s.total_data_generated) * 100
                    sensor_rates.append(rate)

            # Save to persistent variable
            self.last_episode_stats = {
                'total_generated': total_generated,
                'total_collected': total_collected,
                'total_lost': total_lost,
                'battery_remaining': self.uav.battery,
                'coverage': (len(self.sensors_visited) / self.num_sensors) * 100,
                'fairness_std': np.std(sensor_rates) if sensor_rates else 0.0
            }

        # 2. PROCEED: Now allow the standard reset to wipe everything
        return super().reset(**kwargs)

# ==================== 2. THE CALLBACK (READS SNAPSHOT) ====================

class FairnessMetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_coverages = []
        self.episode_efficiencies = []
        self.episode_data_losses = []
        self.episode_fairness_stds = []
        self.episode_battery_efficiency = []
        self.n_episodes = 0

    def _on_step(self) -> bool:
        if self.locals.get('dones')[0]:
            self.n_episodes += 1

            # Get the underlying environment
            env = self.training_env.envs[0]
            if hasattr(env, 'unwrapped'):
                env = env.unwrapped

            # === CRITICAL: Use the SNAPSHOT data, not live sensors ===
            if not hasattr(env, 'last_episode_stats') or env.last_episode_stats is None:
                return True

            stats = env.last_episode_stats
            info = self.locals.get('infos')[0]

            # 1. Record Metrics
            if 'episode' in info:
                self.episode_rewards.append(info['episode']['r'])

            self.episode_coverages.append(stats['coverage'])

            efficiency = (stats['total_collected'] / stats['total_generated'] * 100) if stats['total_generated'] > 0 else 0
            self.episode_efficiencies.append(efficiency)

            battery_used = 274.0 - stats['battery_remaining']
            batt_eff = (stats['total_collected'] / battery_used) if battery_used > 0 else 0
            self.episode_battery_efficiency.append(batt_eff)

            self.episode_data_losses.append(stats['total_lost'])
            self.episode_fairness_stds.append(stats['fairness_std'])

            # 2. Print Progress (Every 10 episodes)
            if self.n_episodes % 10 == 0:
                recent_rewards = self.episode_rewards[-10:]
                recent_cov = self.episode_coverages[-10:]
                recent_eff = self.episode_efficiencies[-10:]

                print(f"Episode {self.n_episodes} | "
                      f"Rew: {np.mean(recent_rewards):.0f} | "
                      f"Cov: {np.mean(recent_cov):.1f}% | "
                      f"Eff: {np.mean(recent_eff):.1f}% | "
                      f"Fairness(std): {stats['fairness_std']:.1f}%")

        return True

    def get_metrics(self):
        return {
            'rewards': self.episode_rewards,
            'coverages': self.episode_coverages,
            'efficiencies': self.episode_efficiencies,
            'data_losses': self.episode_data_losses,
            'fairness_stds': self.episode_fairness_stds,
            'battery_efficiency': self.episode_battery_efficiency
        }

# ==================== CONFIGURATION ====================

ENV_CONFIG = {
    'grid_size': (100, 100),
    'num_sensors': 20,
    'max_steps': 2100,
    'sensor_duty_cycle': 10.0,
    'penalty_data_loss': -500.0,
    'reward_urgency_reduction': 20.0,
    'render_mode': None,
}

FRAME_STACKING_CONFIG = {'use_frame_stacking': True, 'n_stack': 4}

HYPERPARAMS = {
    'policy': 'MlpPolicy',
    'learning_rate': 1e-4,
    'buffer_size': 50_000,
    'batch_size': 32,
    'gamma': 0.95,
    'learning_starts': 1000,
    'train_freq': 4,
    'target_update_interval': 1000,
    'exploration_fraction': 0.95,
    'gradient_steps': 1,
    'policy_kwargs': {'net_arch': [512, 512, 256]},
}

TRAINING_CONFIG = {
    'total_timesteps': 1_000_000,
    'eval_freq': 5000,
    'save_freq': 10_000,
    'log_interval': 10,
}

SAVE_DIR = Path("models/dqn_fairness_framestack")
LOG_DIR = Path("logs/dqn_fairness_framestack")
SAVE_DIR.mkdir(parents=True, exist_ok=True)


# ==================== MAIN TRAINING ====================

def main():
    print("STARTING TRAINING (WITH PERSISTENCE FIX)")

    # 1. Environment Factory (Uses Wrapper)
    def make_env():
        # Use AnalysisUAVEnv here!
        env = AnalysisUAVEnv(**ENV_CONFIG)
        env = Monitor(env)
        return env

    # 2. Vectorize & Stack
    train_env = DummyVecEnv([make_env])
    train_env = VecFrameStack(train_env, n_stack=4)

    eval_env = DummyVecEnv([make_env])
    eval_env = VecFrameStack(eval_env, n_stack=4)

    # 3. Model
    model = DQN(
        HYPERPARAMS['policy'],
        train_env,
        tensorboard_log=str(LOG_DIR),
        **{k:v for k,v in HYPERPARAMS.items() if k != 'policy'}
    )

    # 4. Callbacks
    checkpoint_callback = CheckpointCallback(save_freq=10000, save_path=str(SAVE_DIR), name_prefix="dqn")
    fairness_callback = FairnessMetricsCallback()

    # 5. Train
    model.learn(
        total_timesteps=TRAINING_CONFIG['total_timesteps'],
        callback=CallbackList([checkpoint_callback, fairness_callback]),
        progress_bar=True
    )

    # 6. Save & Finish
    model.save(str(SAVE_DIR / "dqn_final"))

    # Save Metrics
    metrics = fairness_callback.get_metrics()
    with open(SAVE_DIR / "training_metrics.json", 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in metrics.items()}, f)

    with open(SAVE_DIR / "frame_stacking_config.json", 'w') as f:
        json.dump(FRAME_STACKING_CONFIG, f)

    print("DONE! Metrics Saved.")

if __name__ == "__main__":
    main()