"""
A4 Training — No Domain Randomisation
======================================
Trains a DQN policy on a FIXED (500×500, N=20) configuration with no
curriculum and no grid-size randomisation. This is the control condition
for Ablation A4: it shows how much the domain randomisation regime
contributes to generalisation and overall performance.

Configuration mirrors the main dqn.py training (same hyperparameters,
same observation/reward structure) but locks the environment to the
primary evaluation condition throughout training.

Output:
  models/dqn_no_dr/dqn_final.zip   ← load this in ablation_study.py

Estimated training time on RTX 3050 Ti: ~1.5–2 hours for 1M steps.

Author: ATILADE GABRIEL OKE
"""

import sys
from pathlib import Path
import numpy as np
import torch
import gymnasium
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from environment.uav_env import UAVEnvironment

# ==================== CONFIGURATION ====================

FIXED_GRID_SIZE = (500, 500)
FIXED_N_SENSORS = 20
MAX_SENSORS_LIMIT = 50   # keep identical to main model so policy weights match

BASE_ENV_CONFIG = {
    "grid_size":          FIXED_GRID_SIZE,
    "num_sensors":        FIXED_N_SENSORS,
    "max_steps":          2100,
    "path_loss_exponent": 3.8,
    "rssi_threshold":     -85.0,
    "sensor_duty_cycle":  10.0,
    "max_battery":        274.0,
    "render_mode":        None,
}

HYPERPARAMS = {
    "policy":                 "MlpPolicy",
    "learning_rate":          3e-4,
    "buffer_size":            150_000,
    "batch_size":             128,
    "gamma":                  0.99,
    "learning_starts":        10_000,
    "exploration_fraction":   0.25,
    "exploration_final_eps":  0.05,
    "target_update_interval": 1000,
    "train_freq":             4,
    "policy_kwargs": {
        "net_arch": [512, 512, 256],
    },
}

TOTAL_TIMESTEPS = 1_000_000   # 1M steps — sufficient for a fixed condition
SAVE_FREQ       = 50_000
N_STACK         = 4
N_ENVS          = 4           # parallel workers (all see same fixed config)

SAVE_DIR = Path(__file__).parent / "models" / "dqn_no_dr"
LOG_DIR  = Path(__file__).parent / "logs"  / "dqn_no_dr"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True,  exist_ok=True)


# ==================== ENVIRONMENT WRAPPER ====================

class FixedConditionEnv(UAVEnvironment):
    """
    Zero-padded env pinned to FIXED_GRID_SIZE and FIXED_N_SENSORS.
    No domain randomisation — same config every episode.
    Observation is zero-padded to MAX_SENSORS_LIMIT to keep the
    same network input size as the main DR model.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._build_padded_obs_space()

    def _build_padded_obs_space(self):
        raw = self.observation_space.shape[0]
        self._fps = 0
        for uav_f in range(raw + 1):
            rem = raw - uav_f
            if rem > 0 and rem % self.num_sensors == 0:
                self._fps = rem // self.num_sensors
                break
        if self._fps == 0:
            raise ValueError(
                f"Cannot infer features_per_sensor: raw={raw}, "
                f"num_sensors={self.num_sensors}"
            )
        self._raw_obs_size = raw
        padded = raw + (MAX_SENSORS_LIMIT - self.num_sensors) * self._fps
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(padded,), dtype=np.float32
        )

    def _pad(self, obs: np.ndarray) -> np.ndarray:
        n_pad = (MAX_SENSORS_LIMIT - self.num_sensors) * self._fps
        return np.concatenate([obs, np.zeros(n_pad, dtype=np.float32)]).astype(
            np.float32
        )

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        return self._pad(obs), info

    def step(self, action):
        obs, reward, term, trunc, info = super().step(action)
        return self._pad(obs), reward, term, trunc, info


# ==================== MAIN ====================

def get_device():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1024 ** 3
        print(f"GPU: {name} ({mem:.1f} GB) | CUDA {torch.version.cuda}")
        return "cuda"
    print("No GPU — using CPU")
    return "cpu"


def main():
    print("=" * 65)
    print("A4 Training — No Domain Randomisation")
    print(f"  Fixed config: {FIXED_GRID_SIZE}, N={FIXED_N_SENSORS}")
    print(f"  Total steps : {TOTAL_TIMESTEPS:,}")
    print(f"  N workers   : {N_ENVS}")
    print(f"  Save dir    : {SAVE_DIR}")
    print("=" * 65)

    device = get_device()

    def _make_env():
        env = FixedConditionEnv(**BASE_ENV_CONFIG)
        env = Monitor(env)
        return env

    vec = DummyVecEnv([_make_env] * N_ENVS)
    stacked = VecFrameStack(vec, n_stack=N_STACK)

    model = DQN(
        env    = stacked,
        device = device,
        verbose= 1,
        tensorboard_log=str(LOG_DIR),
        **HYPERPARAMS,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq   = SAVE_FREQ // N_ENVS,
        save_path   = str(SAVE_DIR),
        name_prefix = "dqn_no_dr",
    )

    print("\nStarting training…")
    model.learn(
        total_timesteps      = TOTAL_TIMESTEPS,
        callback             = checkpoint_cb,
        progress_bar         = True,
        reset_num_timesteps  = True,
    )

    final_path = SAVE_DIR / "dqn_final.zip"
    model.save(str(final_path))
    print(f"\nFinal model saved to {final_path}")
    print("Now run ablation_study.py to include A4 results.")

    stacked.close()


if __name__ == "__main__":
    main()
