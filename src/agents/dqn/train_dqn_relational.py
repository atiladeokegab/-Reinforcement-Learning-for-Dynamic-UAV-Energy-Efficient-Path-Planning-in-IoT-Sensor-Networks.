"""
DQN + relational encoder (UAVAttentionExtractor) — 2x2 design matrix cell.

Trains the DQN algorithm with the permutation-invariant cross-attention feature
extractor, completing the 2x2 (algorithm x architecture) decomposition table
alongside DQN(flat-MLP), PPO(flat-MLP), and PPO(relational).

Training recipe is identical to dqn.py — same domain randomisation distribution,
same competence-gated curriculum, same hyperparameters — with one controlled change:
    policy_kwargs swaps  net_arch=[512,512,256]
    for                  UAVAttentionExtractor (embed_dim=64, n_heads=4, features_dim=128)
                         + shallow head net_arch=[256,128]

Usage (RunPod / any CUDA box):
    PYTHONIOENCODING=utf-8 uv run python src/agents/dqn/train_dqn_relational.py

Model saved to: models/dqn_relational/dqn_final.zip
Config saved to: models/dqn_relational/training_config.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC  = _HERE.parent.parent
for _p in (str(_SRC / "environment"), str(_SRC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
# Make dqn.py importable as a module
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback, CallbackList, EvalCallback,
)

import gymnasium

from environment.uav_env import UAVEnvironment

# Import shared infrastructure — importing dqn creates models/dqn_v3/ as a side
# effect but that is harmless; we override SAVE_DIR and LOG_DIR below.
from dqn import (
    get_device,
    DomainRandEnv,
    CurriculumCallback,
    BestByMetricCallback,
    UAVAttentionExtractor,
    CURRICULUM_STAGES,
    COMPETENCE_GATE,
    DEMOTION_GATE,
    GREEDY_BENCHMARK,
    BASE_ENV_CONFIG,
    MAX_SENSORS_LIMIT,
    WORKER_SENSOR_COUNTS,
    TRAINING_CONFIG,
    EVAL_GRID,
    EVAL_N_SENSORS,
    EVAL_FREQ,
    N_EVAL_EPISODES,
    N_ENVS,
)

# ==================== OVERRIDES ====================

SAVE_DIR = Path("models/dqn_relational")
LOG_DIR  = Path("logs/dqn_relational")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Only policy_kwargs changes — everything else is identical to dqn.py.
HYPERPARAMS = {
    "policy": "MlpPolicy",
    "learning_rate":          lambda progress: 3e-4 * max(0.1, 1.0 - progress * 0.8),
    "buffer_size":            150_000,
    "batch_size":             256,
    "gamma":                  0.99,
    "learning_starts":        25_000,
    "exploration_fraction":   0.25,
    "exploration_final_eps":  0.03,
    "target_update_interval": 5_000,
    "train_freq":             4,
    "policy_kwargs": {
        "features_extractor_class":  UAVAttentionExtractor,
        "features_extractor_kwargs": {
            "embed_dim":    64,
            "n_heads":      4,
            "features_dim": 128,
        },
        # Shallow head — the extractor already compresses to features_dim=128.
        "net_arch": [256, 128],
    },
}


# ==================== MAIN ====================

def main():
    print("=" * 70)
    print("DQN + RELATIONAL ENCODER — 2x2 DESIGN MATRIX")
    print("Algorithm: DQN  |  Encoder: UAVAttentionExtractor (cross-attention)")
    print("=" * 70)
    print("Curriculum stages:")
    for i, (grids, sensors, desc) in enumerate(CURRICULUM_STAGES):
        print("  Stage {}  {}".format(i, desc))
    print()
    print("Competence Gate (same as dqn.py — controlled comparison):")
    print("  NDR   >= {:.1f}%  (rolling window = {} episodes)".format(
        COMPETENCE_GATE["ndr_pct"], COMPETENCE_GATE["window"]))
    print("  Jain's >= {:.2f}  (min dwell = {:,} steps)".format(
        COMPETENCE_GATE["jains"], COMPETENCE_GATE["min_steps"]))
    print()
    print("MAX_SENSORS_LIMIT: {}".format(MAX_SENSORS_LIMIT))
    print("N_ENVS:            {}  workers, sensor counts: {}".format(
        N_ENVS, WORKER_SENSOR_COUNTS))
    print("Total timesteps:   {:,}".format(TRAINING_CONFIG["total_timesteps"]))
    print()

    device = get_device()
    print()

    # ── Environment factories ────────────────────────────────────────────
    def make_train_env(rank=0):
        fixed_n = WORKER_SENSOR_COUNTS[rank % len(WORKER_SENSOR_COUNTS)]
        def _init():
            env = DomainRandEnv(
                fixed_num_sensors = fixed_n,
                max_sensors_limit = MAX_SENSORS_LIMIT,
                curriculum_stage  = 0,
                base_config       = BASE_ENV_CONFIG,
            )
            env = Monitor(env)
            return env
        return _init

    def make_eval_env():
        env = UAVEnvironment(
            grid_size   = EVAL_GRID,
            num_sensors = EVAL_N_SENSORS,
            **BASE_ENV_CONFIG,
        )
        from types import MethodType

        raw = env.observation_space.shape[0]
        fps = 0
        for uav_f in range(raw + 1):
            rem = raw - uav_f
            if rem > 0 and rem % EVAL_N_SENSORS == 0:
                fps = rem // EVAL_N_SENSORS
                break
        padded = raw + (MAX_SENSORS_LIMIT - EVAL_N_SENSORS) * fps
        env.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(padded,), dtype=np.float32
        )
        _fps        = fps
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
        env = Monitor(env)
        return env

    train_vec = DummyVecEnv([make_train_env(i) for i in range(N_ENVS)])
    print("Using DummyVecEnv with {} workers".format(N_ENVS))
    train_vec = VecFrameStack(train_vec, n_stack=TRAINING_CONFIG["n_stack"])

    # ── Model ────────────────────────────────────────────────────────────
    model = DQN(
        HYPERPARAMS["policy"],
        train_vec,
        tensorboard_log = str(LOG_DIR),
        device          = device,
        verbose         = 0,
        **{k: v for k, v in HYPERPARAMS.items() if k != "policy"},
    )
    print("Model device: {}".format(next(model.policy.parameters()).device))
    print("Obs shape:    {}".format(train_vec.observation_space.shape))

    n_params = sum(p.numel() for p in model.policy.parameters() if p.requires_grad)
    print("Policy params: {:,}".format(n_params))
    print()

    # ── Callbacks ────────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq   = TRAINING_CONFIG["save_freq"],
        save_path   = str(SAVE_DIR),
        name_prefix = "dqn_rel",
    )
    curriculum_cb = CurriculumCallback(n_envs=N_ENVS, verbose=1)

    eval_env = DummyVecEnv([make_eval_env])
    eval_env = VecFrameStack(eval_env, n_stack=TRAINING_CONFIG["n_stack"])

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = str(SAVE_DIR / "best_model"),
        log_path             = str(LOG_DIR / "eval"),
        eval_freq            = EVAL_FREQ,
        n_eval_episodes      = N_EVAL_EPISODES,
        deterministic        = True,
        verbose              = 1,
    )
    metric_cb = BestByMetricCallback(
        eval_env        = eval_env,
        save_path       = SAVE_DIR / "best_metric_model",
        eval_freq       = EVAL_FREQ,
        n_eval_episodes = N_EVAL_EPISODES,
        verbose         = 1,
    )

    # ── Train ────────────────────────────────────────────────────────────
    print("Starting training...")
    t_start = __import__("time").time()
    model.learn(
        total_timesteps = TRAINING_CONFIG["total_timesteps"],
        callback        = CallbackList([checkpoint_cb, curriculum_cb, eval_cb, metric_cb]),
        progress_bar    = True,
    )
    elapsed = (__import__("time").time() - t_start) / 60
    print("\nTraining complete: {:.1f} min".format(elapsed))

    # ── Save ─────────────────────────────────────────────────────────────
    model.save(str(SAVE_DIR / "dqn_final"))
    print("Model saved: {}".format(SAVE_DIR / "dqn_final"))

    _tmp = DomainRandEnv(
        fixed_num_sensors = WORKER_SENSOR_COUNTS[0],
        max_sensors_limit = MAX_SENSORS_LIMIT,
        curriculum_stage  = 0,
        base_config       = BASE_ENV_CONFIG,
    )
    training_config = {
        "use_frame_stacking":    True,
        "n_stack":               TRAINING_CONFIG["n_stack"],
        "max_sensors_limit":     MAX_SENSORS_LIMIT,
        "features_per_sensor":   _tmp._features_per_sensor,
        "encoder":               "UAVAttentionExtractor",
        "encoder_kwargs":        HYPERPARAMS["policy_kwargs"]["features_extractor_kwargs"],
        "domain_randomisation":  True,
        "curriculum_type":       "competence_based",
        "competence_gate":       COMPETENCE_GATE,
        "demotion_gate":         DEMOTION_GATE,
        "curriculum_stages":     [s[2] for s in CURRICULUM_STAGES],
        "eval_grid":             list(EVAL_GRID),
        "eval_n_sensors":        EVAL_N_SENSORS,
        "trained_timesteps":     TRAINING_CONFIG["total_timesteps"],
    }
    _tmp.close()
    with open(SAVE_DIR / "training_config.json", "w") as f:
        json.dump(training_config, f, indent=2)
    print("Config saved: {}".format(SAVE_DIR / "training_config.json"))

    condition_summary = curriculum_cb.get_condition_summary()
    with open(SAVE_DIR / "condition_summary.json", "w") as f:
        json.dump(condition_summary, f, indent=2)

    graduation_log = curriculum_cb.get_graduation_log()
    with open(SAVE_DIR / "graduation_log.json", "w") as f:
        json.dump(graduation_log, f, indent=2)

    # ── Summary ──────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Per-condition Jain's index during training:")
    print("=" * 70)
    for cond, stats in sorted(condition_summary.items()):
        print("  {}  ->  J={:.4f} +/- {:.4f}  ({} episodes)".format(
            cond, stats["mean_jains"], stats["std_jains"], stats["n_episodes"]
        ))

    print("\n" + "=" * 70)
    print("Curriculum Graduation Log:")
    print("=" * 70)
    if graduation_log:
        for entry in graduation_log:
            print(
                "  Stage {} -> Stage {}  |  step {:,}  |  steps_in_stage {:,}  "
                "|  NDR={:.1f}%  Jain={:.3f}".format(
                    entry["from_stage"], entry["to_stage"],
                    entry["ts"], entry["steps_in_stage"],
                    entry["rolling_ndr"], entry["rolling_jains"],
                )
            )
    else:
        final_stage = curriculum_cb._current_stage
        print("  Reached Stage {} but no further graduations.".format(final_stage))

    print("\nDone. Evaluate with:")
    print("  DQN_MODEL_PATH:  {}".format(SAVE_DIR / "dqn_final.zip"))
    print("  DQN_CONFIG_PATH: {}".format(SAVE_DIR / "training_config.json"))


if __name__ == "__main__":
    main()
