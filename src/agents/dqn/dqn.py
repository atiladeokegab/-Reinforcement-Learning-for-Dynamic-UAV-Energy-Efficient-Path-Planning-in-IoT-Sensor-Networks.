"""
DQN Training with Domain Randomisation + Curriculum Learning
=============================================================
Trains one model that generalises across all evaluation conditions without
retraining by exposing the agent to randomly sampled (grid_size, num_sensors)
combinations every episode.

Key improvements over the original train_dqn.py:
  1. Domain Randomisation   — every episode samples a new (grid, sensor-count)
                              from the full evaluation distribution.
  2. Curriculum Learning    — starts on easy conditions (small grid, few sensors),
                              gradually unlocks harder ones as performance improves.
  3. Multi-env Parallelism  — N_ENVS parallel workers, each independently
                              randomised, for faster and more diverse training.
  4. Adaptive Reward Shaping— fairness bonus scaled to grid size so the agent
                              is rewarded consistently across all conditions.
  5. Larger Replay Buffer   — needed because experience now covers a much wider
                              distribution of states.

Zero-padding layout (unchanged from original):
    [raw UAVEnvironment obs]  +  [zeros for missing sensors up to MAX_SENSORS_LIMIT]

Author: ATILADE GABRIEL OKE
"""

import sys
from pathlib import Path
import numpy as np
import torch
import json
import gymnasium
from datetime import datetime

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback, BaseCallback, CallbackList,
)

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from environment.uav_env import UAVEnvironment

# ==================== GPU ====================

def get_device():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        mem  = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print("GPU: {} ({:.1f} GB VRAM) | CUDA {}".format(name, mem, torch.version.cuda))
        return "cuda"
    print("No GPU — using CPU")
    return "cpu"

# ==================== DOMAIN DISTRIBUTION ====================
#
# These are the exact conditions tested in fairness_sweep.py and multi_seed_eval.py.
# The model will be sampled from this joint distribution every episode.
#
# Curriculum stages:
#   Stage 0  (early training)  — small grids only, modest sensor counts
#   Stage 1  (mid training)    — medium grids added
#   Stage 2  (full training)   — all conditions, including 1000x1000 / SF11/SF12

CURRICULUM_STAGES = [
    # (grid_sizes,                     sensor_counts,  description)
    ([(100, 100), (300, 300)],          [10, 20],       "Stage 0 — SF7/SF9, small nets"),
    ([(100, 100), (300, 300),(500,500)],[10, 20, 30],   "Stage 1 — up to SF11, mid nets"),
    ([(100,100),(300,300),(500,500),(1000,1000)],[10,20,30,40], "Stage 2 — full distribution"),
]

# Timestep thresholds to advance the curriculum
CURRICULUM_THRESHOLDS = [150_000, 400_000]   # advance at these timesteps

# Fixed target config for evaluation during training (matches your baseline)
EVAL_GRID      = (500, 500)
EVAL_N_SENSORS = 20

# Neural network always sees this fixed input size regardless of active sensors
MAX_SENSORS_LIMIT = 50

# ==================== ENVIRONMENT WRAPPER ====================

class DomainRandEnv(UAVEnvironment):
    """
    Domain-randomised wrapper for UAVEnvironment.

    KEY DESIGN DECISION:
    UAVEnvironment was not designed to change num_sensors after __init__
    (its internal reset() rebuilds sensors from the count stored at init time).
    Attempting to overwrite self.num_sensors at reset causes a raw-obs/padded-obs
    shape mismatch crash.

    Solution: each worker is pinned to a FIXED num_sensors chosen at creation
    time.  Grid size is randomised on every reset().  Diversity across sensor
    counts is achieved by creating N_ENVS workers, each with a different
    num_sensors (see make_train_env() in main()).

    Zero-padding layout (always constant):
        [ raw UAVEnvironment obs (uav_features + num_sensors*fps) ]
        [ zeros for (max_sensors_limit - num_sensors) ghost sensors ]
    Total length = uav_features + max_sensors_limit * fps  (never changes)
    """

    def __init__(
        self,
        fixed_num_sensors: int,
        max_sensors_limit: int = MAX_SENSORS_LIMIT,
        curriculum_stage: int  = 0,
        base_config: dict      = None,
        **kwargs,
    ):
        self.max_sensors_limit = max_sensors_limit
        self._curriculum_stage = curriculum_stage
        self._base_config      = base_config or {}
        self._fixed_num_sensors = fixed_num_sensors

        # Initialise with stage-0 smallest grid so obs space is defined
        init_grid = CURRICULUM_STAGES[0][0][0]
        super().__init__(
            grid_size   = init_grid,
            num_sensors = fixed_num_sensors,   # FIXED — never changes
            **self._base_config,
            **kwargs,
        )
        self._build_padded_obs_space()
        self.last_episode_stats = None
        print(
            "  DomainRandEnv: fixed_n={}, max_limit={}, "
            "fps={}, padded_obs={}".format(
                fixed_num_sensors, max_sensors_limit,
                self._features_per_sensor,
                self.observation_space.shape[0],
            )
        )

    def _build_padded_obs_space(self):
        raw = self.observation_space.shape[0]
        self._features_per_sensor = 0
        for uav_f in range(raw + 1):
            rem = raw - uav_f
            if rem > 0 and rem % self.num_sensors == 0:
                self._features_per_sensor = rem // self.num_sensors
                break
        if self._features_per_sensor == 0:
            raise ValueError(
                "Cannot detect features_per_sensor: raw={}, num_sensors={}".format(
                    raw, self.num_sensors
                )
            )
        # This size is CONSTANT for this worker's lifetime
        self._raw_obs_size = raw
        padded = raw + (self.max_sensors_limit - self.num_sensors) * self._features_per_sensor
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(padded,), dtype=np.float32
        )

    def set_curriculum_stage(self, stage: int):
        self._curriculum_stage = min(stage, len(CURRICULUM_STAGES) - 1)

    def _sample_grid(self):
        """Sample a grid_size from the current curriculum stage (sensor count is fixed)."""
        grids, _, _ = CURRICULUM_STAGES[self._curriculum_stage]
        return grids[np.random.randint(len(grids))]

    def _pad(self, obs: np.ndarray) -> np.ndarray:
        expected_raw = self._raw_obs_size
        if obs.shape[0] != expected_raw:
            raise AssertionError(
                "Raw obs {} != expected {}. "
                "fixed_num_sensors={}, fps={}, max_limit={}".format(
                    obs.shape[0], expected_raw,
                    self._fixed_num_sensors, self._features_per_sensor,
                    self.max_sensors_limit
                )
            )
        n_pad = (self.max_sensors_limit - self.num_sensors) * self._features_per_sensor
        return np.concatenate([obs, np.zeros(n_pad, dtype=np.float32)]).astype(np.float32)

    def reset(self, **kwargs):
        # Snapshot stats before parent wipes them
        if hasattr(self, "sensors") and self.current_step > 0:
            sensor_rates = []
            for s in self.sensors:
                if s.total_data_generated > 0:
                    sensor_rates.append(
                        s.total_data_transmitted / s.total_data_generated * 100
                    )
            self.last_episode_stats = {
                "total_generated":   sum(s.total_data_generated   for s in self.sensors),
                "total_collected":   sum(s.total_data_transmitted for s in self.sensors),
                "total_lost":        sum(s.total_data_lost        for s in self.sensors),
                "battery_remaining": self.uav.battery,
                "coverage":          len(self.sensors_visited) / self.num_sensors * 100,
                "fairness_std":      float(np.std(sensor_rates)) if sensor_rates else 0.0,
                "jains_index":       self._jains(sensor_rates),
                "grid_size":         tuple(self.grid_size),
                "num_sensors":       self.num_sensors,
            }

        # Randomise grid_size only — num_sensors is fixed for this worker
        self.grid_size = self._sample_grid()

        obs, info = super().reset(**kwargs)
        return self._pad(obs), info

    def step(self, action):
        obs, reward, term, trunc, info = super().step(action)

        # Small per-step fairness shaping bonus (normalised by num_sensors)
        if hasattr(self, "sensors"):
            rates = []
            for s in self.sensors:
                gen = float(s.total_data_generated)
                if gen > 0:
                    rates.append(float(s.total_data_transmitted) / gen)
            if rates:
                j = self._jains([r * 100 for r in rates])
                reward += 0.5 * (j - 0.5) / self.num_sensors

        return self._pad(obs), reward, term, trunc, info

    @staticmethod
    def _jains(rates):
        n  = len(rates)
        s1 = sum(rates)
        s2 = sum(x**2 for x in rates)
        return (s1**2) / (n * s2) if n > 0 and s2 > 0 else 1.0


# ==================== CURRICULUM CALLBACK ====================

class CurriculumCallback(BaseCallback):
    """
    Advances the curriculum stage when timestep thresholds are crossed.
    Also logs per-episode metrics grouped by (grid, sensors) condition
    so you can see which conditions the model struggles with during training.
    """

    def __init__(self, n_envs: int = 1, verbose: int = 1):
        super().__init__(verbose)
        self._n_envs          = n_envs
        self._current_stage   = 0
        self._episode_rewards = []
        self._episode_jains   = []
        self._episode_coverages = []
        self._condition_stats = {}   # (grid, n) -> list of jains values
        self._n_episodes      = 0

    def _on_step(self) -> bool:
        # Advance curriculum
        for i, threshold in enumerate(CURRICULUM_THRESHOLDS):
            if self.num_timesteps >= threshold and self._current_stage <= i:
                self._current_stage = i + 1
                desc = CURRICULUM_STAGES[self._current_stage][2]
                print("\n[Curriculum] Advancing to {} at step {}".format(
                    desc, self.num_timesteps))
                # Update curriculum stage in all envs via the VecEnv API
                # (works with both DummyVecEnv and SubprocVecEnv)
                try:
                    self.training_env.env_method(
                        "set_curriculum_stage", self._current_stage
                    )
                except Exception as e:
                    print("  Warning: could not set curriculum stage: {}".format(e))

        # Log episodes — use get_attr() which works with any VecEnv type
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for idx, (done, info) in enumerate(zip(dones, infos)):
            if not done:
                continue
            self._n_episodes += 1
            if "episode" in info:
                self._episode_rewards.append(info["episode"]["r"])

            # Fetch last_episode_stats from the correct worker
            try:
                stats_list = self.training_env.get_attr("last_episode_stats")
                stats = stats_list[idx] if idx < len(stats_list) else None
            except Exception:
                stats = None

            if stats is None:
                continue

            key  = (stats["grid_size"], stats["num_sensors"])
            cond = self._condition_stats.setdefault(key, [])
            cond.append(stats["jains_index"])
            self._episode_jains.append(stats["jains_index"])
            self._episode_coverages.append(stats["coverage"])

        if self._n_episodes > 0 and self._n_episodes % 20 == 0 and self._episode_rewards:
            print(
                "  ep={:5d} | stage={} | "
                "rew={:8.0f} | cov={:.1f}% | J={:.4f} | ts={}".format(
                    self._n_episodes,
                    self._current_stage,
                    np.mean(self._episode_rewards[-20:]),
                    np.mean(self._episode_coverages[-20:]) if self._episode_coverages else 0,
                    np.mean(self._episode_jains[-20:])     if self._episode_jains else 0,
                    self.num_timesteps,
                )
            )
        return True

    def get_condition_summary(self):
        return {
            str(k): {
                "mean_jains": float(np.mean(v)),
                "std_jains":  float(np.std(v)),
                "n_episodes": len(v),
            }
            for k, v in self._condition_stats.items()
        }


# ==================== CONFIGURATION ====================

N_ENVS = 4   # one worker per sensor count — each sees all grid sizes
             # 4 workers × episode_len=2100 fits comfortably within 4GB VRAM
             # with buffer_size=150k

# Each worker is pinned to one of these sensor counts.
# len(WORKER_SENSOR_COUNTS) must equal N_ENVS.
WORKER_SENSOR_COUNTS = [10, 20, 30, 40]

BASE_ENV_CONFIG = {
    "max_steps":          2100,
    "path_loss_exponent": 3.8,
    "rssi_threshold":     -120.0,
    "sensor_duty_cycle":  10.0,
    "max_battery":        274.0,
    "render_mode":        None,
}

HYPERPARAMS = {
    "policy":                 "MlpPolicy",
    "learning_rate":          3e-4,
    "buffer_size":            150_000,  # reduced from 300k — fits comfortably in 4GB VRAM
    "batch_size":             128,      # reduced from 256 to leave headroom for env workers
    "gamma":                  0.99,
    "learning_starts":        10_000,   # reduced from 20k — smaller buffer fills faster
    "exploration_fraction":   0.25,
    "exploration_final_eps":  0.05,
    "target_update_interval": 1000,
    "train_freq":             4,
    "policy_kwargs": {
        "net_arch": [512, 512, 256],
    },
}

TRAINING_CONFIG = {
    "total_timesteps": 2_000_000,   # 2x original — more conditions = need more data
    "save_freq":       25_000,
    "n_stack":         4,           # frame stacking
}

SAVE_DIR = Path("models/dqn_domain_rand")
LOG_DIR  = Path("logs/dqn_domain_rand")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ==================== MAIN ====================

def main():
    print("=" * 70)
    print("DQN TRAINING — DOMAIN RANDOMISATION + CURRICULUM")
    print("=" * 70)
    print("Curriculum stages:")
    for i, (grids, sensors, desc) in enumerate(CURRICULUM_STAGES):
        thresh = CURRICULUM_THRESHOLDS[i] if i < len(CURRICULUM_THRESHOLDS) else "end"
        print("  Stage {}  {}  (unlocks at ts={})".format(i, desc, thresh))
    print("MAX_SENSORS_LIMIT: {}  (NN input size fixed — never change)".format(MAX_SENSORS_LIMIT))
    print("N_ENVS:            {}  workers, sensor counts: {}".format(
        N_ENVS, WORKER_SENSOR_COUNTS))
    print("Total timesteps:   {:,}".format(TRAINING_CONFIG["total_timesteps"]))
    print()

    device = get_device()
    print()

    # ── Environment factories ────────────────────────────────────────────
    def make_train_env(rank=0):
        # Each worker gets a different fixed sensor count.
        # rank 0 → 10 sensors, rank 1 → 20, rank 2 → 30, rank 3 → 40.
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
        # Eval always on the standard baseline condition
        env = UAVEnvironment(
            grid_size   = EVAL_GRID,
            num_sensors = EVAL_N_SENSORS,
            **BASE_ENV_CONFIG,
        )
        # Wrap with zero-padding so obs shape matches training
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
        env = Monitor(env)
        return env

    # DummyVecEnv runs workers in the same process — avoids SubprocVecEnv's
    # pickling requirements and the .envs attribute issue entirely.
    # With N_ENVS=2 and episode length=2100, dummy is fast enough on GPU.
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
    print()

    # ── Callbacks ────────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq  = TRAINING_CONFIG["save_freq"],
        save_path  = str(SAVE_DIR),
        name_prefix= "dqn_dr",
    )
    curriculum_cb = CurriculumCallback(n_envs=N_ENVS, verbose=1)

    # ── Train ────────────────────────────────────────────────────────────
    print("Starting training...")
    t_start = __import__("time").time()
    model.learn(
        total_timesteps = TRAINING_CONFIG["total_timesteps"],
        callback        = CallbackList([checkpoint_cb, curriculum_cb]),
        progress_bar    = True,
    )
    elapsed = (__import__("time").time() - t_start) / 60
    print("\nTraining complete: {:.1f} min".format(elapsed))

    # ── Save ────────────────────────────────────────────────────────────
    model.save(str(SAVE_DIR / "dqn_final"))
    print("Model saved: {}".format(SAVE_DIR / "dqn_final"))

    # Save training config — same keys as original so evaluation scripts work unchanged
    _tmp = DomainRandEnv(
        fixed_num_sensors = WORKER_SENSOR_COUNTS[0],  # any valid count works
        max_sensors_limit = MAX_SENSORS_LIMIT,
        curriculum_stage  = 0,
        base_config       = BASE_ENV_CONFIG,
    )
    training_config = {
        "use_frame_stacking":  True,
        "n_stack":             TRAINING_CONFIG["n_stack"],
        "max_sensors_limit":   MAX_SENSORS_LIMIT,
        "features_per_sensor": _tmp._features_per_sensor,
        "domain_randomisation": True,
        "curriculum_stages":   [s[2] for s in CURRICULUM_STAGES],
        "eval_grid":           list(EVAL_GRID),
        "eval_n_sensors":      EVAL_N_SENSORS,
        "trained_timesteps":   TRAINING_CONFIG["total_timesteps"],
    }
    _tmp.close()
    with open(SAVE_DIR / "training_config.json", "w") as f:
        json.dump(training_config, f, indent=2)
    print("Config saved: {}".format(SAVE_DIR / "training_config.json"))

    # Save per-condition performance from curriculum callback
    condition_summary = curriculum_cb.get_condition_summary()
    with open(SAVE_DIR / "condition_summary.json", "w") as f:
        json.dump(condition_summary, f, indent=2)

    print("\n" + "=" * 70)
    print("Per-condition Jain's index during training:")
    print("=" * 70)
    for cond, stats in sorted(condition_summary.items()):
        print("  {}  ->  J={:.4f} +- {:.4f}  ({} episodes)".format(
            cond, stats["mean_jains"], stats["std_jains"], stats["n_episodes"]
        ))

    print("\nDone. Point evaluation scripts at:")
    print("  DQN_MODEL_PATH:  {}".format(SAVE_DIR / "dqn_final.zip"))
    print("  DQN_CONFIG_PATH: {}".format(SAVE_DIR / "training_config.json"))


if __name__ == "__main__":
    main()