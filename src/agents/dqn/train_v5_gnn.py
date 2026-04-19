"""
DQN v5 — GNN Feature Extractor + k=10 Frame Stack
===================================================
Key changes from v4b:
  - GNNExtractor: sensor self-attention (complete-graph GNN) + GRU across 10 frames
  - n_stack=10 (up from 4) for ADR convergence window (~1/λ = 10 steps)
  - Same 3-feature obs space as v4b (buffer, urgency, link_quality)
  - Saves to models/dqn_v5_gnn/

Author: ATILADE GABRIEL OKE
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
import torch
import gymnasium

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback, CallbackList, EvalCallback,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from environment.uav_env import UAVEnvironment

sys.path.insert(0, str(Path(__file__).parent))
from dqn import (
    DomainRandEnv,
    CurriculumCallback,
    CURRICULUM_STAGES,
    WORKER_SENSOR_COUNTS,
    get_device,
)
from gnn_extractor import GNNExtractor

# ==================== CONFIG ====================

MAX_SENSORS_LIMIT = 50
N_ENVS            = 4

EVAL_GRID      = (500, 500)
EVAL_N_SENSORS = 20
N_STACK        = 10   # GRU temporal window ≈ 1/λ (ADR convergence)

SENSOR_FEATURES = 3   # buffer, urgency, link_quality
FRAME_DIM       = 3 + MAX_SENSORS_LIMIT * SENSOR_FEATURES  # 153

BASE_ENV_CONFIG = {
    "max_steps":          2100,
    "path_loss_exponent": 3.8,
    "rssi_threshold":     -85.0,
    "sensor_duty_cycle":  10.0,
    "max_battery":        274.0,
    "render_mode":        None,
    "penalty_battery":    0.0,
    "reward_movement":    10.0,
}

SAVE_DIR = Path(__file__).parent.parent.parent.parent / "models" / "dqn_v5_gnn"
LOG_DIR  = Path(__file__).parent.parent.parent.parent / "logs"  / "dqn_v5_gnn"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# Use DomainRandEnv directly — obs space is unchanged (3 features per sensor)
DomainRandEnvV5 = DomainRandEnv


# ==================== MAIN ====================

def main():
    print("=" * 70)
    print("DQN v5 — GNN + k=10 (self-attention + GRU)")
    print("=" * 70)
    print("SENSOR_FEATURES: {}  (buffer, urgency, link_quality)".format(SENSOR_FEATURES))
    print("FRAME_DIM:       {}  (3 UAV + 50×3 sensors)".format(FRAME_DIM))
    print("N_STACK:         {}  (temporal GRU window)".format(N_STACK))
    print("Total obs:       {}".format(N_STACK * FRAME_DIM))
    print()

    device = get_device()

    # ── Environment factories ────────────────────────────────────────────
    def make_train_env(rank=0):
        fixed_n = WORKER_SENSOR_COUNTS[rank % len(WORKER_SENSOR_COUNTS)]
        def _init():
            env = DomainRandEnvV5(
                fixed_num_sensors = fixed_n,
                max_sensors_limit = MAX_SENSORS_LIMIT,
                curriculum_stage  = 0,
                base_config       = BASE_ENV_CONFIG,
            )
            return Monitor(env)
        return _init

    def make_eval_env():
        from types import MethodType
        env = UAVEnvironment(
            grid_size   = EVAL_GRID,
            num_sensors = EVAL_N_SENSORS,
            **BASE_ENV_CONFIG,
        )
        fps = SENSOR_FEATURES  # 3 — matches DomainRandEnv padding
        raw = env.observation_space.shape[0]
        padded_size = raw + (MAX_SENSORS_LIMIT - EVAL_N_SENSORS) * fps
        env.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(padded_size,), dtype=np.float32
        )
        _fps = fps
        _orig_reset = env.reset
        _orig_step  = env.step

        def _pad(obs):
            pad = np.zeros((MAX_SENSORS_LIMIT - EVAL_N_SENSORS) * _fps, dtype=np.float32)
            return np.concatenate([obs, pad]).astype(np.float32)

        def patched_reset(self, **kwargs):
            obs, info = _orig_reset(**kwargs)
            return _pad(obs), info

        def patched_step(self, action):
            obs, r, term, trunc, info = _orig_step(action)
            return _pad(obs), r, term, trunc, info

        env.reset = MethodType(patched_reset, env)
        env.step  = MethodType(patched_step, env)
        return Monitor(env)

    train_vec = DummyVecEnv([make_train_env(i) for i in range(N_ENVS)])
    train_vec = VecFrameStack(train_vec, n_stack=N_STACK)

    print("Obs shape (stacked): {}".format(train_vec.observation_space.shape))
    assert train_vec.observation_space.shape[0] == N_STACK * FRAME_DIM, (
        "Expected obs dim {}, got {}. "
        "Check MAX_SENSORS_LIMIT and SENSOR_FEATURES.".format(
            N_STACK * FRAME_DIM, train_vec.observation_space.shape[0]
        )
    )

    # ── GNN policy kwargs ────────────────────────────────────────────────
    policy_kwargs = {
        "features_extractor_class": GNNExtractor,
        "features_extractor_kwargs": {
            "features_dim":    256,
            "k":               N_STACK,
            "max_sensors":     MAX_SENSORS_LIMIT,
            "sensor_features": SENSOR_FEATURES,
            "embed_dim":       64,
            "n_heads":         4,
            "gru_hidden":      128,
        },
        "net_arch": [256, 256],
    }

    model = DQN(
        "MlpPolicy",
        train_vec,
        tensorboard_log          = str(LOG_DIR),
        device                   = device,
        verbose                  = 0,
        learning_rate            = lambda p: 3e-4 * max(0.1, 1.0 - p * 0.8),
        buffer_size              = 150_000,
        batch_size               = 256,
        gamma                    = 0.99,
        learning_starts          = 25_000,
        exploration_fraction     = 0.25,
        exploration_final_eps    = 0.10,
        target_update_interval   = 5_000,
        train_freq               = 4,
        policy_kwargs            = policy_kwargs,
    )
    print("Model device: {}".format(next(model.policy.parameters()).device))
    n_params = sum(p.numel() for p in model.policy.parameters())
    print("Policy params: {:,}".format(n_params))
    print()

    # ── Callbacks ────────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq   = 25_000,
        save_path   = str(SAVE_DIR),
        name_prefix = "dqn_v5",
    )
    curriculum_cb = CurriculumCallback(n_envs=N_ENVS, verbose=1)

    eval_env_vec = DummyVecEnv([make_eval_env])
    eval_env_vec = VecFrameStack(eval_env_vec, n_stack=N_STACK)
    eval_cb = EvalCallback(
        eval_env_vec,
        best_model_save_path = str(SAVE_DIR / "best_model"),
        log_path             = str(LOG_DIR / "eval"),
        eval_freq            = 25_000,
        n_eval_episodes      = 5,
        deterministic        = True,
        verbose              = 1,
    )

    # ── Train ────────────────────────────────────────────────────────────
    total_ts = 10_000_000
    print("Starting training ({:,} steps)...".format(total_ts))
    t0 = time.time()
    model.learn(
        total_timesteps = total_ts,
        callback        = CallbackList([checkpoint_cb, curriculum_cb, eval_cb]),
        progress_bar    = True,
    )
    print("\nTraining complete: {:.1f} min".format((time.time() - t0) / 60))

    # ── Save ────────────────────────────────────────────────────────────
    model.save(str(SAVE_DIR / "dqn_final"))
    print("Model saved: {}".format(SAVE_DIR / "dqn_final"))

    training_config = {
        "use_frame_stacking":       True,
        "n_stack":                  N_STACK,
        "max_sensors_limit":        MAX_SENSORS_LIMIT,
        "features_per_sensor":      SENSOR_FEATURES,
        "include_sensor_positions": False,
        "domain_randomisation":     True,
        "curriculum_stages":        [s[2] for s in CURRICULUM_STAGES],
        "eval_grid":                list(EVAL_GRID),
        "eval_n_sensors":           EVAL_N_SENSORS,
        "trained_timesteps":        total_ts,
        "extractor":                "GNNExtractor",
        "net_arch":                 [256, 256],
    }
    with open(SAVE_DIR / "training_config.json", "w") as f:
        json.dump(training_config, f, indent=2)
    print("Config saved.")


if __name__ == "__main__":
    main()
