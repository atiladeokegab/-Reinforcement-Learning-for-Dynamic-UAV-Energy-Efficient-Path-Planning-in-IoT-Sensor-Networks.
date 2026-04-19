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
import torch.nn as nn
import torch.nn.functional as F
import json
import gymnasium
from datetime import datetime

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback, BaseCallback, CallbackList, EvalCallback,
)
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from environment.uav_env import UAVEnvironment
from environment.iot_sensors import IoTSensor

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
    # 1000×1000 removed — physically infeasible (TSP 44,700m vs d_max 15,782m),
    # no policy can achieve >12% NDR there so it contributes only noise.
    ([(100,100), (200,200), (300,300)],                     [10,20,30,40], "Stage 0 — SF7/SF9, small nets"),
    ([(100,100), (200,200), (300,300), (400,400)],          [10,20,30,40], "Stage 1 — up to SF11, mid nets"),
    ([(100,100), (200,200), (300,300), (400,400),(500,500)],[10,20,30,40], "Stage 2 — feasible distribution"),
]

# Timestep thresholds to advance the curriculum (fallback — performance gates below take priority)
CURRICULUM_THRESHOLDS = [1_000_000, 2_000_000]

# Performance-based curriculum gates: agent must sustain these metrics over
# PERF_WINDOW consecutive episodes before the stage advances.
# Stage N criteria must be met to unlock Stage N+1.
PERF_THRESHOLDS = [
    {"ndr": 90.0, "jains": 0.60},  # Stage 0 → Stage 1
    {"ndr": 80.0, "jains": 0.55},  # Stage 1 → Stage 2 (lower bar: harder env)
]
PERF_WINDOW        = 50       # rolling episode window for performance gate
MIN_STEPS_PER_STAGE = 400_000  # minimum steps before any stage can advance

# Fixed target config for evaluation during training (matches your baseline)
EVAL_GRID      = (500, 500)
EVAL_N_SENSORS = 20

# Neural network always sees this fixed input size regardless of active sensors
MAX_SENSORS_LIMIT = 50

# ==================== LAYOUT GENERATORS ====================
# Used to diversify sensor placement during training so the agent generalises
# beyond the uniform-random layout it was originally trained on.

def _layout_uniform(rng, W, H, n):
    """Standard training distribution — uniform random placement."""
    return [(float(rng.uniform(0, W)), float(rng.uniform(0, H))) for _ in range(n)]


def _layout_two_clusters(rng, W, H, n):
    """Dense cluster near centre + peripheral outlier group (the cross-layout failure case)."""
    half = n // 2
    c1 = [
        (float(np.clip(rng.normal(W * 0.50, W * 0.05), 0, W)),
         float(np.clip(rng.normal(H * 0.50, H * 0.05), 0, H)))
        for _ in range(half)
    ]
    c2 = [
        (float(np.clip(rng.normal(W * 0.84, W * 0.05), 0, W)),
         float(np.clip(rng.normal(H * 0.84, H * 0.05), 0, H)))
        for _ in range(n - half)
    ]
    return c1 + c2


def _layout_four_corners(rng, W, H, n):
    """Sensors concentrated in four corner regions."""
    per_corner = n // 4
    corners = [
        (W * 0.15, H * 0.15), (W * 0.85, H * 0.15),
        (W * 0.15, H * 0.85), (W * 0.85, H * 0.85),
    ]
    positions = []
    for cx, cy in corners:
        for _ in range(per_corner):
            positions.append((
                float(np.clip(rng.normal(cx, W * 0.06), 0, W)),
                float(np.clip(rng.normal(cy, H * 0.06), 0, H)),
            ))
    while len(positions) < n:
        positions.append(_layout_uniform(rng, W, H, 1)[0])
    return positions


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
        # Reset per-episode coverage timer
        self._first_full_coverage_step = None

        # Snapshot stats before parent wipes them
        if hasattr(self, "sensors") and self.current_step > 0:
            sensor_rates = []
            for s in self.sensors:
                if s.total_data_generated > 0:
                    sensor_rates.append(
                        s.total_data_transmitted / s.total_data_generated * 100
                    )
            total_generated = sum(s.total_data_generated   for s in self.sensors)
            total_collected = sum(s.total_data_transmitted for s in self.sensors)
            battery_used    = self.uav.max_battery - self.uav.battery

            self.last_episode_stats = {
                "total_generated":   total_generated,
                "total_collected":   total_collected,
                "total_lost":        sum(s.total_data_lost for s in self.sensors),
                "battery_remaining": self.uav.battery,
                "ndr":               len(self.sensors_visited) / self.num_sensors * 100,
                "fairness_std":      float(np.std(sensor_rates)) if sensor_rates else 0.0,
                "jains_index":       self._jains(sensor_rates),
                "grid_size":         tuple(self.grid_size),
                "num_sensors":       self.num_sensors,
                # Data efficiency: fraction of generated data actually collected
                "data_efficiency":   (total_collected / total_generated * 100)
                                     if total_generated > 0 else 0.0,
                # Energy efficiency: bytes delivered per Wh consumed
                "bytes_per_wh":      (total_collected / battery_used)
                                     if battery_used > 0 else 0.0,
                # Time-to-full-coverage: step when last new sensor was first visited
                # (stored during episode via _first_full_coverage_step)
                "time_to_coverage":  getattr(self, "_first_full_coverage_step", None),
            }

        # Randomise grid_size only — num_sensors is fixed for this worker
        self.grid_size = self._sample_grid()

        # Randomize UAV start position so the agent learns a grid-wide policy,
        # not a corner-biased one. Keep 10% margin from boundaries.
        W, H = float(self.grid_size[0]), float(self.grid_size[1])
        self.uav.start_position = np.array([
            float(np.random.uniform(0.1 * W, 0.9 * W)),
            float(np.random.uniform(0.1 * H, 0.9 * H)),
        ], dtype=np.float32)

        obs, info = super().reset(**kwargs)

        # --- Layout diversification ---
        # Sample a sensor layout every episode so the agent sees clustered and
        # bimodal arrangements during training, not just uniform random placement.
        # Probability mix: 70% uniform, 15% two-clusters, 15% four-corners.
        rng = np.random.default_rng()
        r = rng.random()
        W, H = float(self.grid_size[0]), float(self.grid_size[1])
        n = self.num_sensors

        if r < 0.70:
            new_positions = _layout_uniform(rng, W, H, n)
        elif r < 0.85:
            new_positions = _layout_two_clusters(rng, W, H, n)
        else:
            new_positions = _layout_four_corners(rng, W, H, n)

        # Copy sensor config from existing sensor (avoids storing redundant attrs on env)
        s0 = self.sensors[0]
        self.sensor_positions = new_positions
        self.sensors = [
            IoTSensor(
                sensor_id=i,
                position=pos,
                data_generation_rate=s0.data_generation_rate,
                max_buffer_size=s0.max_buffer_size,
                spreading_factor=s0.spreading_factor,
                path_loss_exponent=s0.path_loss_exponent,
                rssi_threshold=s0.rssi_threshold,
                duty_cycle=s0.duty_cycle,
            )
            for i, pos in enumerate(new_positions)
        ]
        # Recompute observation with updated sensor positions
        obs = self._get_observation()

        return self._pad(obs), info

    def step(self, action):
        prev_visited = len(self.sensors_visited)
        obs, reward, term, trunc, info = super().step(action)

        # Track the step at which all sensors were visited for the first time
        if (not hasattr(self, "_first_full_coverage_step") or
                self._first_full_coverage_step is None):
            if len(self.sensors_visited) == self.num_sensors:
                self._first_full_coverage_step = self.current_step

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


# ==================== ATTENTION FEATURE EXTRACTOR ====================

class UAVAttentionExtractor(BaseFeaturesExtractor):
    """
    Relational feature extractor for UAV IoT collection (DQN-7).

    Observation layout after VecFrameStack(k=4) — 612-dim flat vector:
        [frame_0 | frame_1 | frame_2 | frame_3]
    Each 153-dim frame:
        [uav_x, uav_y, battery  |  s0_buffer, s0_urgency, s0_link_quality  | ... | s49_...]
         ^^--- UAV_FEATURES=3 ---^^  ^^------------- 50 × SENSOR_FEATURES=3 ---------------^^

    Architecture:
        1. Temporal UAV encoder   — MLP over all 4 frames' UAV states → embed_dim
        2. Sensor entity encoder  — Linear(3 → embed_dim) per sensor slot
        3. Cross-attention        — UAV (1 query token) attends to sensor (50 key/value tokens)
        4. Masking                — zero-padded ghost sensors AND out-of-range sensors
                                    (link_quality == 0) are excluded from attention
        5. Fusion                 — cat(uav_embed, attn_context) → features_dim

    VRAM estimate (RTX 3050 Ti, 4 GB):
        Model params  ≈ 55k  (vs 1.3M for flat MLP)
        Replay buffer ≈ 0.69 GB  (unchanged)
        Batch forward ≈ 1.2 MB   (batch=256 × 612 × 4 B × 2 sides)
    """

    UAV_FEATURES    = 3
    SENSOR_FEATURES = 3
    N_SENSORS       = 50
    N_STACK         = 10
    FRAME_SIZE      = UAV_FEATURES + N_SENSORS * SENSOR_FEATURES  # 153

    def __init__(
        self,
        observation_space,
        embed_dim: int    = 64,
        n_heads: int      = 4,
        features_dim: int = 128,
    ):
        super().__init__(observation_space, features_dim)
        self.embed_dim = embed_dim

        # All 4 frames' UAV states → single temporal context vector
        self.uav_encoder = nn.Sequential(
            nn.Linear(self.UAV_FEATURES * self.N_STACK, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )

        # Project each sensor slot's 3 features into embed space
        self.sensor_proj = nn.Linear(self.SENSOR_FEATURES, embed_dim)

        # Cross-attention: 1 UAV query over 50 sensor key/value tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            batch_first=True,
            dropout=0.0,
        )
        self.attn_norm = nn.LayerNorm(embed_dim)

        # Fuse UAV temporal embed + attention context → features_dim output
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]

        # --- Parse frame stack -------------------------------------------------
        frames = obs.view(B, self.N_STACK, self.FRAME_SIZE)           # (B, 4, 153)

        # Temporal UAV context: concatenate UAV state across all 4 frames
        uav_states  = frames[:, :, :self.UAV_FEATURES]                # (B, 4, 3)
        uav_flat    = uav_states.reshape(B, -1)                       # (B, 12)
        uav_embed   = self.uav_encoder(uav_flat)                      # (B, embed_dim)

        # Sensor entities from the most recent frame only
        current     = frames[:, -1, :]                                # (B, 153)
        sensor_flat = current[:, self.UAV_FEATURES:]                  # (B, 150)
        sensors     = sensor_flat.view(B, self.N_SENSORS,
                                       self.SENSOR_FEATURES)          # (B, 50, 3)

        # --- Attention mask ----------------------------------------------------
        # True  → this key slot is ignored by attention (softmax → −∞)
        # Mask 1: zero-padded ghost sensor slots (all three features == 0)
        is_ghost    = (sensors.abs().sum(dim=-1) < 1e-6)              # (B, 50)
        # Mask 2: out-of-range sensors (link_quality feature, index 2, == 0)
        is_oor      = (sensors[:, :, 2] < 1e-6)                      # (B, 50)
        key_pad_mask = is_ghost | is_oor                              # (B, 50)

        # Safety: if ALL slots are masked (can happen on tiny grids at step 0),
        # unmask everything to avoid NaN from softmax(−∞ everywhere).
        all_masked   = key_pad_mask.all(dim=1, keepdim=True)          # (B, 1)
        key_pad_mask = key_pad_mask & ~all_masked                     # at least 1 visible

        # --- Encode sensor entities -------------------------------------------
        sensor_embed = F.relu(self.sensor_proj(sensors))              # (B, 50, embed_dim)

        # --- Cross-attention --------------------------------------------------
        query    = uav_embed.unsqueeze(1)                             # (B, 1, embed_dim)
        attn_out, _ = self.cross_attn(
            query           = query,
            key             = sensor_embed,
            value           = sensor_embed,
            key_padding_mask= key_pad_mask,
        )                                                             # (B, 1, embed_dim)
        attn_context = self.attn_norm(attn_out.squeeze(1))           # (B, embed_dim)

        # --- Fuse and return --------------------------------------------------
        combined = torch.cat([uav_embed, attn_context], dim=-1)      # (B, embed_dim*2)
        return self.fusion(combined)                                  # (B, features_dim)


# ==================== CURRICULUM CALLBACK ====================

class CurriculumCallback(BaseCallback):
    """
    Advances the curriculum stage when timestep thresholds are crossed.
    Also logs per-episode metrics grouped by (grid, sensors) condition
    so you can see which conditions the model struggles with during training.
    """

    def __init__(self, n_envs: int = 1, verbose: int = 1):
        super().__init__(verbose)
        self._n_envs            = n_envs
        self._current_stage     = 0
        self._stage_start_step  = 0   # timestep when current stage began
        self._episode_rewards   = []
        self._episode_jains     = []
        self._episode_ndrs = []
        self._condition_stats   = {}   # (grid, n) -> list of jains values
        self._n_episodes        = 0
        self._last_logged_ep    = 0

    def _try_advance_curriculum(self):
        """
        Advance curriculum when ALL of the following are true:
          (a) minimum steps in current stage elapsed (prevents instant skip-through), AND
          (b) EITHER performance gate OR fallback timestep threshold crossed.

        On advancement the rolling window is cleared so the next gate is
        evaluated on data collected under the harder conditions only.
        """
        if self._current_stage >= len(CURRICULUM_STAGES) - 1:
            return

        perf_gate  = PERF_THRESHOLDS[self._current_stage]
        ts_gate    = CURRICULUM_THRESHOLDS[self._current_stage]
        min_steps  = MIN_STEPS_PER_STAGE

        # Enforce minimum dwell time in current stage
        steps_in_stage = self.num_timesteps - self._stage_start_step
        if steps_in_stage < min_steps:
            return

        # --- Performance gate (primary) ---
        if len(self._episode_ndrs) >= PERF_WINDOW:
            mean_cov  = np.mean(self._episode_ndrs[-PERF_WINDOW:])
            mean_jain = np.mean(self._episode_jains[-PERF_WINDOW:])
            perf_ok   = (mean_cov >= perf_gate["ndr"] and
                         mean_jain >= perf_gate["jains"])
        else:
            perf_ok = False

        # --- Fallback timestep gate ---
        ts_ok = self.num_timesteps >= ts_gate

        if perf_ok or ts_ok:
            reason = "performance" if perf_ok else "timestep fallback"
            self._current_stage += 1
            self._stage_start_step = self.num_timesteps
            desc = CURRICULUM_STAGES[self._current_stage][2]
            print("\n[Curriculum] ({}) Advancing to {} at step {}".format(
                reason, desc, self.num_timesteps))
            # Clear window — next gate must be earned under harder conditions
            self._episode_ndrs.clear()
            self._episode_jains.clear()
            try:
                self.training_env.env_method(
                    "set_curriculum_stage", self._current_stage
                )
            except Exception as e:
                print("  Warning: could not set curriculum stage: {}".format(e))

    def _on_step(self) -> bool:
        self._try_advance_curriculum()

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
            self._episode_ndrs.append(stats["ndr"])

        if self._n_episodes >= self._last_logged_ep + 20 and self._episode_rewards:
            # Collect recent data_efficiency and bytes_per_wh from all workers
            try:
                all_stats = self.training_env.get_attr("last_episode_stats")
                recent_eff = [s["data_efficiency"] for s in all_stats if s and "data_efficiency" in s]
                recent_bpwh = [s["bytes_per_wh"] for s in all_stats if s and "bytes_per_wh" in s]
            except Exception:
                recent_eff, recent_bpwh = [], []

            eff_str  = "{:.1f}%".format(np.mean(recent_eff))   if recent_eff  else "n/a"
            bpwh_str = "{:.0f}".format(np.mean(recent_bpwh))   if recent_bpwh else "n/a"
            print(
                "  ep={:5d} | stage={} | "
                "rew={:8.0f} | NDR={:.1f}% | J={:.4f} | "
                "d_eff={} | B/Wh={} | ts={}".format(
                    self._n_episodes,
                    self._current_stage,
                    np.mean(self._episode_rewards[-20:]),
                    np.mean(self._episode_ndrs[-20:]) if self._episode_ndrs else 0,
                    np.mean(self._episode_jains[-20:])     if self._episode_jains else 0,
                    eff_str,
                    bpwh_str,
                    self.num_timesteps,
                )
            )
            self._last_logged_ep = self._n_episodes
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
    "rssi_threshold":     -85.0,
    "sensor_duty_cycle":  10.0,
    "max_battery":        274.0,
    "render_mode":        None,
}

HYPERPARAMS = {
    "policy":                 "MlpPolicy",
    # Cosine-like decay: 3e-4 → 3e-5 over training. High LR early for fast learning,
    # low LR late so the Q-values settle rather than oscillating near the minimum.
    "learning_rate":          lambda progress: 3e-4 * max(0.1, 1.0 - progress * 0.8),
    "buffer_size":            150_000,  # 612-dim × 4B × 2 × 150k ≈ 0.69 GB — fits in 1.32 GB RAM
    "batch_size":             256,
    "gamma":                  0.99,
    "learning_starts":        25_000,   # let buffer populate across all conditions first
    "exploration_fraction":   0.25,
    "exploration_final_eps":  0.10,     # keep some exploration into Stage 2
    "target_update_interval": 5_000,    # stable Q-targets — was 1000 (too frequent)
    "train_freq":             4,
    "policy_kwargs": {
        "net_arch": [512, 512, 256],
    },
}

TRAINING_CONFIG = {
    "total_timesteps": 3_000_000,
    "save_freq":       25_000,
    "n_stack":         4,
}

SAVE_DIR = Path("models/dqn_v3")
LOG_DIR  = Path("logs/dqn_v3")
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

    eval_env = DummyVecEnv([make_eval_env])
    eval_env = VecFrameStack(eval_env, n_stack=TRAINING_CONFIG["n_stack"])
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path = str(SAVE_DIR / "best_model"),
        log_path             = str(LOG_DIR / "eval"),
        eval_freq            = 25_000,   # evaluate every 25k steps
        n_eval_episodes      = 5,
        deterministic        = True,
        verbose              = 1,
    )

    # ── Train ────────────────────────────────────────────────────────────
    print("Starting training...")
    t_start = __import__("time").time()
    model.learn(
        total_timesteps = TRAINING_CONFIG["total_timesteps"],
        callback        = CallbackList([checkpoint_cb, curriculum_cb, eval_cb]),
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