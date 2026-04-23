"""
DQN Training with Domain Randomisation + Competence-Based Curriculum Learning
==============================================================================
Trains one model that generalises across all evaluation conditions without
retraining by exposing the agent to randomly sampled (grid_size, num_sensors)
combinations every episode.

Key improvements over the time-based curriculum:
  1. Competence Gate       — graduation requires BOTH NDR > 95 % AND Jain's > 0.85
                             sustained over a rolling window of N episodes.
  2. Demotion Gate         — if performance collapses after advancing, the agent
                             is demoted one stage to stabilise.
  3. Min-Dwell Enforcement — prevents instant skip-through at training start.
  4. Domain Randomisation  — every episode samples a new grid_size from the
                             current curriculum stage; sensor count is fixed
                             per worker.
  5. Multi-env Parallelism — N_ENVS parallel workers, each independently
                             randomised, for faster and more diverse training.
  6. Adaptive Reward Shaping — fairness bonus scaled to grid size so the agent
                              is rewarded consistently across all conditions.
  7. Larger Replay Buffer  — needed because experience covers a wider
                             distribution of states.

Zero-padding layout (unchanged from original):
    [raw UAVEnvironment obs]  +  [zeros for missing sensors up to MAX_SENSORS_LIMIT]

Author: ATILADE GABRIEL OKE
"""

import sys
from pathlib import Path

# Must be first: add src/ and src/environment/ to sys.path before any
# third-party import can register a stale 'environment' package.
_SRC = Path(__file__).resolve().parent.parent.parent  # …/src
_ENV = _SRC / "environment"
for _p in [str(_ENV), str(_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

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
# The model is sampled from this joint distribution every episode.
#
# Curriculum stages:
#   Stage 0  — 100×100 only (agent learns basic navigation)
#   Stage 1  — adds 200×200 (agent learns to scale)
#   Stage 2  — adds 300×300 (medium-range generalisation)
#   Stage 3  — adds 400×400 (long-range generalisation)
#   Stage 4  — full range including 500×500 (deployment conditions)

CURRICULUM_STAGES = [
    ([(100, 100)],                                             [20, 30, 40], "Stage 0 — 100×100 only"),
    ([(100, 100), (200, 200)],                                 [20, 30, 40], "Stage 1 — up to 200×200"),
    ([(100, 100), (200, 200), (300, 300)],                     [20, 30, 40], "Stage 2 — up to 300×300"),
    ([(100, 100), (200, 200), (300, 300), (400, 400)],         [10, 30, 40], "Stage 3 — up to 400×400"),
    ([(100, 100), (200, 200), (300, 300), (400, 400), (500, 500)], [10, 20, 30, 40], "Stage 4 — full feasible range"),
]

# ── Competence Gate ───────────────────────────────────────────────────────────
# Both NDR and Jain's milestones must be sustained over `window` episodes AND
# a minimum dwell time must have elapsed before graduation is allowed.
# Override these values in your train script to tune difficulty.
COMPETENCE_GATE = {
    "ndr_pct":   95.0,   # New Discovery Rate: % of sensors visited per episode
    "jains":     0.85,   # Jain's fairness index
    "window":    50,     # rolling average over the last N episodes
    "min_steps": 50_000, # minimum timesteps in current stage before graduation
}

# Optional per-stage overrides. Keys are stage indices, values override
# (ndr_pct, jains) from COMPETENCE_GATE for that stage only. Used as the
# fallback path in _get_stage_thresholds when GREEDY_BENCHMARK is disabled.
# Larger grids are harder to cover uniformly in a fixed 2,100-step budget,
# so later stages can relax Jain's here without touching the global gate.
STAGE_GATES: dict[int, dict[str, float]] = {}

# ── Demotion Gate ─────────────────────────────────────────────────────────────
# Reverts one stage if performance collapses after an advancement.
# Both thresholds must be breached simultaneously to trigger demotion,
# and only after `min_episodes` have been collected in the new stage.
DEMOTION_GATE = {
    "ndr_pct":      70.0,  # rolling NDR below this ...
    "jains":        0.60,  # ... AND rolling Jain's below this triggers demotion
    "min_episodes": 20,    # minimum episodes in new stage before demotion is eligible
}

# ── Greedy Benchmark Gate ──────────────────────────────────────────────────────
# Before graduating, run MaxThroughputGreedyV2 on the hardest grid in the
# current stage.  The DQN must beat the greedy mean by the margin values below.
# Set enabled=False to fall back to the fixed COMPETENCE_GATE thresholds.
GREEDY_BENCHMARK = {
    "enabled":      True,
    "n_episodes":   20,    # greedy episodes per stage (more → more accurate baseline)
    "sensor_count": 20,    # fixed sensor count used for all greedy benchmarks
    "margin_ndr":   5.0,   # DQN rolling NDR must exceed greedy mean by this %
    "margin_jains": 0.05,  # DQN rolling Jain's must exceed greedy mean by this
    # Hard floor: even if greedy is weak, DQN must still pass these minimums.
    "floor_ndr":    50.0,
    "floor_jains":  0.40,
}

# Kept for backward-compat so any import that references these does not crash.
# They are NOT used by the competence-based callback.
CURRICULUM_THRESHOLDS = []
MIN_STEPS_PER_STAGE   = 50_000  # fallback; overridden by COMPETENCE_GATE["min_steps"]

# Fixed target config for evaluation during training (matches your baseline)
EVAL_GRID      = (500, 500)
EVAL_N_SENSORS = 20

# Neural network always sees this fixed input size regardless of active sensors
MAX_SENSORS_LIMIT = 50

# ── Navigation fixes ───────────────────────────────────────────────────────────
# Fix 1: rejection-sample UAV start position to guarantee it begins far from
#         all sensors, forcing the agent to learn navigation in every episode.
# Fix 2: potential-based proximity shaping — reward per unit of distance
#         reduction toward the nearest sensor that still has data in its buffer.
NAV_CONFIG = {
    "min_start_dist":   50.0,   # UAV must start ≥ this many units from every sensor
    "max_start_tries":  200,    # rejection sampling attempts; falls back to furthest found
    "prox_eta":         2.0,    # shaping gain η (reward per unit moved closer to sensor)
}


# ==================== LAYOUT GENERATORS ====================

def _layout_uniform(rng, W, H, n):
    """Standard training distribution — uniform random placement."""
    return [(float(rng.uniform(0, W)), float(rng.uniform(0, H))) for _ in range(n)]


def _layout_two_clusters(rng, W, H, n):
    """Dense cluster near centre + peripheral outlier group."""
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

    set_curriculum_stage() updates only the stage index — no internal state is
    reset and no observation spaces are rebuilt, so it is safe to call at any
    point during training via env_method().

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
        self.max_sensors_limit  = max_sensors_limit
        self._curriculum_stage  = min(curriculum_stage, len(CURRICULUM_STAGES) - 1)
        self._base_config       = base_config or {}
        self._fixed_num_sensors = fixed_num_sensors

        # Initialise with stage-0 smallest grid so obs space is defined correctly
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

    # ------------------------------------------------------------------
    def _build_padded_obs_space(self):
        """Compute and set the padded (constant-size) observation space."""
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
        self._raw_obs_size = raw
        padded = raw + (self.max_sensors_limit - self.num_sensors) * self._features_per_sensor
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(padded,), dtype=np.float32
        )

    # ------------------------------------------------------------------
    def set_curriculum_stage(self, stage: int):
        """
        Advance or demote the curriculum stage for this worker.

        Safe to call at any time via VecEnv.env_method() — only the stage index
        is updated; observation space and internal sensor state are untouched.
        """
        new_stage = int(np.clip(stage, 0, len(CURRICULUM_STAGES) - 1))
        if new_stage != self._curriculum_stage:
            direction = "→" if new_stage > self._curriculum_stage else "←"
            # Print only on stage changes to avoid log spam
            print(
                "  [Worker n={}] Curriculum {} Stage {} {} Stage {}".format(
                    self._fixed_num_sensors,
                    "advance" if new_stage > self._curriculum_stage else "demotion",
                    self._curriculum_stage,
                    direction,
                    new_stage,
                )
            )
        self._curriculum_stage = new_stage

    # ------------------------------------------------------------------
    def _sample_grid(self):
        """Sample a grid_size from the current curriculum stage (sensor count is fixed)."""
        grids, _, _ = CURRICULUM_STAGES[self._curriculum_stage]
        return grids[np.random.randint(len(grids))]

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    def reset(self, **kwargs):
        self._first_full_coverage_step = None

        # Snapshot episode stats before the parent wipes them
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
                "data_efficiency":   (total_collected / total_generated * 100)
                                     if total_generated > 0 else 0.0,
                "bytes_per_wh":      (total_collected / battery_used)
                                     if battery_used > 0 else 0.0,
                "time_to_coverage":  getattr(self, "_first_full_coverage_step", None),
            }

        # Randomise grid_size — num_sensors is fixed for this worker
        self.grid_size = self._sample_grid()

        # Randomise UAV start position across the full grid
        W, H = float(self.grid_size[0]), float(self.grid_size[1])

        obs, info = super().reset(**kwargs)

        # Re-randomise sensor layout with uniform placement
        rng = np.random.default_rng()
        n   = self.num_sensors
        new_positions = _layout_uniform(rng, W, H, n)

        s0 = self.sensors[0]
        self.sensor_positions = new_positions
        self.sensors = [
            IoTSensor(
                sensor_id           = i,
                position            = pos,
                data_generation_rate= s0.data_generation_rate,
                max_buffer_size     = s0.max_buffer_size,
                spreading_factor    = s0.spreading_factor,
                path_loss_exponent  = s0.path_loss_exponent,
                rssi_threshold      = s0.rssi_threshold,
                duty_cycle          = s0.duty_cycle,
            )
            for i, pos in enumerate(new_positions)
        ]

        # Fix 1 — rejection-sample UAV start so it begins far from all sensors.
        # Forces navigation as a mandatory first phase of every episode.
        self.uav.start_position = self._sample_far_start(W, H, new_positions)
        self.uav.position       = self.uav.start_position.copy()

        obs = self._get_observation()

        # Fix 2 — initialise proximity shaping baseline distance
        self._prev_dist_nearest = self._dist_to_nearest_with_data()

        return self._pad(obs), info

    # ------------------------------------------------------------------
    def _sample_far_start(self, W, H, sensor_positions) -> np.ndarray:
        """
        Rejection-sample a UAV start position at least NAV_CONFIG['min_start_dist']
        units from every sensor.  Falls back to the furthest candidate found if
        the threshold cannot be satisfied within max_start_tries attempts.
        """
        min_d    = NAV_CONFIG["min_start_dist"]
        max_tries = NAV_CONFIG["max_start_tries"]
        s_pos    = [np.array(p, dtype=np.float32) for p in sensor_positions]

        best_pos  = None
        best_dist = -1.0

        for _ in range(max_tries):
            candidate = np.array(
                [float(np.random.uniform(0.05 * W, 0.95 * W)),
                 float(np.random.uniform(0.05 * H, 0.95 * H))],
                dtype=np.float32,
            )
            if not s_pos:
                return candidate
            d = float(min(np.linalg.norm(candidate - sp) for sp in s_pos))
            if d > best_dist:
                best_dist = d
                best_pos  = candidate
            if d >= min_d:
                return candidate

        return best_pos   # fallback: furthest position found in max_tries attempts

    # ------------------------------------------------------------------
    def _dist_to_nearest_with_data(self) -> float:
        """Distance from UAV to the nearest sensor that still has data buffered."""
        candidates = [s for s in self.sensors if s.data_buffer > 0]
        if not candidates:
            return 0.0
        uav = self.uav.position
        return float(min(np.linalg.norm(s.position - uav) for s in candidates))

    # ------------------------------------------------------------------
    def step(self, action):
        # Fix 2 — snapshot distance before environment step
        prev_dist = self._prev_dist_nearest

        obs, reward, term, trunc, info = super().step(action)

        # Fix 2 — proximity shaping: Φ(s') - Φ(s) = η*(d_{t-1} - d_t)
        curr_dist = self._dist_to_nearest_with_data()
        if prev_dist > 0:
            reward += NAV_CONFIG["prox_eta"] * (prev_dist - curr_dist)
        self._prev_dist_nearest = curr_dist

        # Track the step at which all sensors were first visited
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
        s2 = sum(x ** 2 for x in rates)
        return (s1 ** 2) / (n * s2) if n > 0 and s2 > 0 else 1.0


# ==================== GREEDY BENCHMARK ====================

def _run_greedy_benchmark(stage: int, base_config: dict) -> dict:
    """
    Run MaxThroughputGreedyV2 on the hardest grid in `stage` for
    GREEDY_BENCHMARK["n_episodes"] episodes and return mean NDR + Jain's.

    Called once per stage transition and the result is cached inside
    CurriculumCallback._greedy_cache so it never re-runs for the same stage.
    """
    # Lazy import — greedy_agents lives in a subdirectory
    _eval_dir = str(Path(__file__).parent / "dqn_evaluation_results")
    if _eval_dir not in sys.path:
        sys.path.insert(0, _eval_dir)
    from greedy_agents import MaxThroughputGreedyV2  # noqa: PLC0415

    cfg        = GREEDY_BENCHMARK
    n_episodes = cfg["n_episodes"]
    n_sensors  = cfg["sensor_count"]
    grids      = CURRICULUM_STAGES[stage][0]
    grid       = grids[-1]  # hardest grid in the stage

    print(
        "\n[GreedyBenchmark] Stage {} — running {} episodes of "
        "MaxThroughputGreedyV2 on {}×{} grid, {} sensors ...".format(
            stage, n_episodes, grid[0], grid[1], n_sensors
        )
    )

    ndrs, jains_list = [], []
    min_dist = NAV_CONFIG["min_start_dist"]
    for ep in range(n_episodes):
        env = UAVEnvironment(grid_size=grid, num_sensors=n_sensors, **base_config)
        obs, _ = env.reset(seed=ep)

        # Match the distant-start condition used during DQN training so the
        # greedy baseline is measured under the same difficulty (Fix 1 parity).
        W, H = float(grid[0]), float(grid[1])
        rng  = np.random.default_rng(ep + 99991)
        s_pos = [s.position for s in env.sensors]
        best_pos, best_d = env.uav.position.copy(), -1.0
        for _ in range(NAV_CONFIG["max_start_tries"]):
            candidate = np.array(
                [float(rng.uniform(0.05 * W, 0.95 * W)),
                 float(rng.uniform(0.05 * H, 0.95 * H))],
                dtype=np.float32,
            )
            d = float(min(np.linalg.norm(candidate - sp) for sp in s_pos))
            if d > best_d:
                best_d, best_pos = d, candidate
            if d >= min_dist:
                break
        env.uav.position       = best_pos
        env.uav.start_position = best_pos
        obs = env._get_observation()

        agent  = MaxThroughputGreedyV2(env)
        done   = False
        while not done:
            action = agent.select_action(obs)
            obs, _, term, trunc, _ = env.step(action)
            done = term or trunc

        ndr = len(env.sensors_visited) / env.num_sensors * 100
        rates = [
            s.total_data_transmitted / s.total_data_generated * 100
            for s in env.sensors if s.total_data_generated > 0
        ]
        ndrs.append(ndr)
        jains_list.append(DomainRandEnv._jains(rates))
        env.close()

    result = {
        "ndr":      float(np.mean(ndrs)),
        "jains":    float(np.mean(jains_list)),
        "grid":     grid,
        "n_sensors": n_sensors,
        "n_episodes": n_episodes,
    }
    target_ndr   = min(100.0, max(result["ndr"]   + GREEDY_BENCHMARK["margin_ndr"],
                                  GREEDY_BENCHMARK["floor_ndr"]))
    target_jains = min(1.0,   max(result["jains"] + GREEDY_BENCHMARK["margin_jains"],
                                  GREEDY_BENCHMARK["floor_jains"]))
    print(
        "[GreedyBenchmark] Stage {} baseline — NDR={:.1f}%  Jain={:.3f}  "
        "(graduation target: NDR≥{:.1f}%  Jain≥{:.3f})".format(
            stage, result["ndr"], result["jains"], target_ndr, target_jains,
        )
    )
    return result


# ==================== ATTENTION FEATURE EXTRACTOR ====================

class UAVAttentionExtractor(BaseFeaturesExtractor):
    """
    Relational feature extractor for UAV IoT collection (DQN-7).

    Observation layout after VecFrameStack(k=N_STACK) — flat vector:
        [frame_0 | frame_1 | ... | frame_{N-1}]
    Each 153-dim frame:
        [uav_x, uav_y, battery  |  s0_buffer, s0_urgency, s0_link_quality  | ...]
         ^^--- UAV_FEATURES=3 ---^^  ^^--- 50 × SENSOR_FEATURES=3 ---^^

    Architecture:
        1. Temporal UAV encoder   — MLP over all N frames' UAV states → embed_dim
        2. Sensor entity encoder  — Linear(3 → embed_dim) per sensor slot
        3. Cross-attention        — UAV (1 query) attends to sensor (50 key/value tokens)
        4. Masking                — ghost sensors (zero-padded) AND out-of-range sensors
                                    (link_quality == 0) are excluded from attention
        5. Fusion                 — cat(uav_embed, attn_context) → features_dim
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

        # All N frames' UAV states → single temporal context vector
        self.uav_encoder = nn.Sequential(
            nn.Linear(self.UAV_FEATURES * self.N_STACK, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )

        # Project each sensor slot's 3 features into embed space
        self.sensor_proj = nn.Linear(self.SENSOR_FEATURES, embed_dim)

        # Cross-attention: 1 UAV query over 50 sensor key/value tokens
        self.cross_attn = nn.MultiheadAttention(
            embed_dim   = embed_dim,
            num_heads   = n_heads,
            batch_first = True,
            dropout     = 0.0,
        )
        self.attn_norm = nn.LayerNorm(embed_dim)

        # Fuse UAV temporal embed + attention context → features_dim output
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]

        # Parse frame stack
        frames = obs.view(B, self.N_STACK, self.FRAME_SIZE)           # (B, N, 153)

        # Temporal UAV context: concatenate UAV state across all N frames
        uav_states = frames[:, :, :self.UAV_FEATURES]                 # (B, N, 3)
        uav_flat   = uav_states.reshape(B, -1)                        # (B, N*3)
        uav_embed  = self.uav_encoder(uav_flat)                       # (B, embed_dim)

        # Sensor entities from the most recent frame only
        current     = frames[:, -1, :]                                # (B, 153)
        sensor_flat = current[:, self.UAV_FEATURES:]                  # (B, 150)
        sensors     = sensor_flat.view(B, self.N_SENSORS,
                                       self.SENSOR_FEATURES)          # (B, 50, 3)

        # Attention mask
        # True → this key slot is ignored (softmax → −∞)
        is_ghost     = (sensors.abs().sum(dim=-1) < 1e-6)             # padded slots
        is_oor       = (sensors[:, :, 2] < 1e-6)                      # out-of-range
        key_pad_mask = is_ghost | is_oor                              # (B, 50)

        # Safety: unmask everything if all slots are masked (avoids NaN in softmax)
        all_masked   = key_pad_mask.all(dim=1, keepdim=True)
        key_pad_mask = key_pad_mask & ~all_masked

        # Encode sensor entities
        sensor_embed = F.relu(self.sensor_proj(sensors))              # (B, 50, embed_dim)

        # Cross-attention
        query    = uav_embed.unsqueeze(1)                             # (B, 1, embed_dim)
        attn_out, _ = self.cross_attn(
            query            = query,
            key              = sensor_embed,
            value            = sensor_embed,
            key_padding_mask = key_pad_mask,
        )                                                             # (B, 1, embed_dim)
        attn_context = self.attn_norm(attn_out.squeeze(1))           # (B, embed_dim)

        # Fuse and return
        combined = torch.cat([uav_embed, attn_context], dim=-1)      # (B, embed_dim*2)
        return self.fusion(combined)                                  # (B, features_dim)


# ==================== COMPETENCE-BASED CURRICULUM CALLBACK ====================

class CurriculumCallback(BaseCallback):
    """
    Self-adaptive (competence-based) curriculum controller.

    Graduation logic
    ----------------
    The agent advances to the next stage when ALL of the following hold:
      (a) at least COMPETENCE_GATE["min_steps"] timesteps have elapsed in the
          current stage (prevents instant skip-through at initialisation), AND
      (b) the rolling mean NDR  over the last `window` episodes exceeds
          COMPETENCE_GATE["ndr_pct"]  (default 95 %), AND
      (c) the rolling mean Jain's index over the last `window` episodes exceeds
          COMPETENCE_GATE["jains"]    (default 0.85).

    Demotion logic
    --------------
    After advancing, if BOTH of the following drop below their demotion
    thresholds AND at least DEMOTION_GATE["min_episodes"] have been collected
    in the new stage, the agent is sent back one stage:
      • rolling NDR   < DEMOTION_GATE["ndr_pct"]  (default 70 %)
      • rolling Jain's < DEMOTION_GATE["jains"]   (default 0.60)

    Per-condition tracking
    ----------------------
    Jain's index is logged per (grid_size, num_sensors) condition so you can
    inspect which conditions the model struggles with during training.

    Stage graduation timestamps
    ---------------------------
    self.graduation_log is a list of dicts:
        {"stage": int, "ts": int, "steps_in_stage": int}
    This is used by train_v3_full_retrain.py to report "Time to Graduation".
    """

    def __init__(self, n_envs: int = 1, verbose: int = 1):
        super().__init__(verbose)
        self._n_envs            = n_envs
        self._current_stage     = 0
        self._stage_start_step  = 0      # timestep when current stage began
        self._episodes_in_stage = 0      # episode count since last stage change

        # Rolling performance windows (cleared on each stage change)
        self._window_ndrs   = []
        self._window_jains  = []

        # Full-run history (never cleared — used for logging and final summary)
        self._episode_rewards  = []
        self._all_ndrs         = []
        self._all_jains        = []

        # Per-condition stats: (grid_size, n_sensors) → list[jains]
        self._condition_stats  = {}

        # Per-worker rolling windows — graduation gates on the hardest worker,
        # not the pool mean (easy workers inflate the pooled signal otherwise).
        self._worker_ndrs:  dict[int, list] = {}
        self._worker_jains: dict[int, list] = {}

        self._n_episodes       = 0
        self._last_logged_ep   = 0

        # Graduation log: one entry per stage transition
        self.graduation_log: list[dict] = []

        # Greedy benchmark cache: stage_index -> {"ndr", "jains", ...}
        # Populated lazily the first time _get_stage_thresholds() is called
        # for a given stage so the benchmark never re-runs for the same stage.
        self._greedy_cache: dict = {}

    # ------------------------------------------------------------------
    def _rolling_mean(self, values: list, key: str) -> float:
        """Return the mean of the last `window` values, or 0 if window is empty."""
        w = COMPETENCE_GATE["window"]
        recent = values[-w:]
        return float(np.mean(recent)) if recent else 0.0

    # ------------------------------------------------------------------
    def _get_stage_thresholds(self, stage: int) -> tuple[float, float]:
        """
        Return (ndr_threshold, jains_threshold) for graduating out of `stage`.

        If GREEDY_BENCHMARK["enabled"] is True, thresholds are computed as:
            greedy_mean_NDR   + margin_ndr
            greedy_mean_Jain's + margin_jains
        clamped to the hard floors in GREEDY_BENCHMARK.

        Falls back to COMPETENCE_GATE fixed values when benchmarking is disabled
        or if the greedy runner raises an exception.
        """
        if not GREEDY_BENCHMARK["enabled"]:
            override = STAGE_GATES.get(stage, {})
            return (
                override.get("ndr_pct", COMPETENCE_GATE["ndr_pct"]),
                override.get("jains",   COMPETENCE_GATE["jains"]),
            )

        if stage not in self._greedy_cache:
            try:
                self._greedy_cache[stage] = _run_greedy_benchmark(
                    stage, BASE_ENV_CONFIG
                )
            except Exception as exc:
                print(
                    "[GreedyBenchmark] WARNING: benchmark failed for stage {} "
                    "({}) — falling back to fixed COMPETENCE_GATE.".format(stage, exc)
                )
                self._greedy_cache[stage] = None

        result = self._greedy_cache[stage]
        if result is None:
            return COMPETENCE_GATE["ndr_pct"], COMPETENCE_GATE["jains"]

        # Cap NDR at 98% — greedy often hits 100% so +margin would require
        # perfect coverage every episode, which is unreachable under ε-greedy.
        ndr_thresh   = min(98.0, max(
            result["ndr"]   + GREEDY_BENCHMARK["margin_ndr"],
            GREEDY_BENCHMARK["floor_ndr"],
        ))
        # Cap Jain's at 0.97 — high-variance greedy estimates (small n_episodes)
        # can push the target above what the DQN can reliably sustain.
        jains_thresh = min(0.97, max(
            result["jains"] + GREEDY_BENCHMARK["margin_jains"],
            GREEDY_BENCHMARK["floor_jains"],
        ))
        return ndr_thresh, jains_thresh

    # ------------------------------------------------------------------
    def _try_advance_stage(self):
        """
        Attempt to graduate to the next curriculum stage.

        Conditions (all must be true):
          1. Current stage is not the final stage.
          2. Minimum dwell time has elapsed.
          3. Rolling NDR   >= greedy baseline NDR + margin (or fixed gate).
          4. Rolling Jain's >= greedy baseline Jain's + margin (or fixed gate).
          5. Rolling window is fully populated (at least `window` episodes).
        """
        if self._current_stage >= len(CURRICULUM_STAGES) - 1:
            return  # Already at the final stage

        steps_in_stage = self.num_timesteps - self._stage_start_step
        min_steps      = COMPETENCE_GATE["min_steps"]
        window         = COMPETENCE_GATE["window"]

        # Enforce minimum dwell
        if steps_in_stage < min_steps:
            return

        # Gate on the hardest worker: every worker must have a full window and
        # the *minimum* rolling mean across workers must clear the threshold.
        # This prevents easy workers (small N, small grid) from pulling the
        # pooled average above the gate while hard workers are still struggling.
        w_roll_ndrs, w_roll_jains = [], []
        for w_id in sorted(self._worker_ndrs):
            w_ndrs  = self._worker_ndrs[w_id]
            w_jains = self._worker_jains[w_id]
            if len(w_ndrs) < window or len(w_jains) < window:
                return  # this worker hasn't accumulated a full window yet
            w_roll_ndrs.append(float(np.mean(w_ndrs[-window:])))
            w_roll_jains.append(float(np.mean(w_jains[-window:])))

        if not w_roll_ndrs:
            return

        rolling_ndr   = min(w_roll_ndrs)   # hardest worker drives the gate
        rolling_jains = min(w_roll_jains)

        ndr_gate, jains_gate = self._get_stage_thresholds(self._current_stage)

        if rolling_ndr >= ndr_gate and rolling_jains >= jains_gate:
            prev_stage = self._current_stage
            self._current_stage += 1
            desc = CURRICULUM_STAGES[self._current_stage][2]

            greedy = self._greedy_cache.get(prev_stage)
            self.graduation_log.append({
                "from_stage":      prev_stage,
                "to_stage":        self._current_stage,
                "ts":              self.num_timesteps,
                "steps_in_stage":  steps_in_stage,
                "rolling_ndr":     rolling_ndr,
                "rolling_jains":   rolling_jains,
                "threshold_ndr":   ndr_gate,
                "threshold_jains": jains_gate,
                "greedy_ndr":      greedy["ndr"]   if greedy else None,
                "greedy_jains":    greedy["jains"] if greedy else None,
            })
            greedy_str = (
                "  greedy_baseline: NDR={:.1f}%  Jain={:.3f}".format(
                    greedy["ndr"], greedy["jains"]
                ) if greedy else ""
            )
            print(
                "\n[Curriculum] ✓ Advancing to {} at step {:,} "
                "(NDR={:.1f}% ≥ {:.1f}%, Jain={:.3f} ≥ {:.3f}, "
                "steps_in_stage={:,}){}".format(
                    desc, self.num_timesteps,
                    rolling_ndr, ndr_gate,
                    rolling_jains, jains_gate,
                    steps_in_stage,
                    greedy_str,
                )
            )

            # Reset counters for the new stage
            self._stage_start_step  = self.num_timesteps
            self._episodes_in_stage = 0
            self._window_ndrs.clear()
            self._window_jains.clear()
            self._worker_ndrs.clear()
            self._worker_jains.clear()

            # Push new stage to all environment workers
            self._set_stage_on_envs(self._current_stage)

    # ------------------------------------------------------------------
    def _try_demote_stage(self):
        """
        Demote one stage if performance has collapsed since the last advancement.

        Triggered when rolling NDR AND Jain's simultaneously fall below the
        demotion thresholds, after enough episodes in the current stage.
        """
        if self._current_stage == 0:
            return  # Cannot demote below Stage 0

        min_episodes  = DEMOTION_GATE["min_episodes"]
        window        = COMPETENCE_GATE["window"]

        # Require a minimum episode count and a full rolling window
        if self._episodes_in_stage < min_episodes:
            return
        if len(self._window_ndrs) < window or len(self._window_jains) < window:
            return

        rolling_ndr   = self._rolling_mean(self._window_ndrs,  "ndr")
        rolling_jains = self._rolling_mean(self._window_jains, "jains")

        ndr_thresh   = DEMOTION_GATE["ndr_pct"]
        jains_thresh = DEMOTION_GATE["jains"]

        if rolling_ndr < ndr_thresh and rolling_jains < jains_thresh:
            prev_stage = self._current_stage
            self._current_stage -= 1
            desc = CURRICULUM_STAGES[self._current_stage][2]
            print(
                "\n[Curriculum] ✗ Demoting to {} at step {:,} "
                "(NDR={:.1f}% < {:.1f}%, Jain={:.3f} < {:.3f})".format(
                    desc, self.num_timesteps,
                    rolling_ndr, ndr_thresh,
                    rolling_jains, jains_thresh,
                )
            )

            # Reset stage counters
            self._stage_start_step  = self.num_timesteps
            self._episodes_in_stage = 0
            self._window_ndrs.clear()
            self._window_jains.clear()
            self._worker_ndrs.clear()
            self._worker_jains.clear()

            self._set_stage_on_envs(self._current_stage)

    # ------------------------------------------------------------------
    def _set_stage_on_envs(self, stage: int):
        """Broadcast a new curriculum stage index to all VecEnv workers."""
        try:
            self.training_env.env_method("set_curriculum_stage", stage)
        except Exception as e:
            print("  Warning: could not broadcast curriculum stage to envs: {}".format(e))

    # ------------------------------------------------------------------
    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        for idx, (done, info) in enumerate(zip(dones, infos)):
            if not done:
                continue

            self._n_episodes       += 1
            self._episodes_in_stage += 1

            # Standard SB3 episode reward from Monitor wrapper
            if "episode" in info:
                self._episode_rewards.append(info["episode"]["r"])

            # Fetch per-episode stats from the correct worker
            try:
                stats_list = self.training_env.get_attr("last_episode_stats")
                stats = stats_list[idx] if idx < len(stats_list) else None
            except Exception:
                stats = None

            if stats is None:
                continue

            # Update rolling performance windows
            self._window_ndrs.append(stats["ndr"])
            self._window_jains.append(stats["jains_index"])

            # Keep windows bounded to avoid unbounded memory growth
            max_window = COMPETENCE_GATE["window"] * 4
            if len(self._window_ndrs) > max_window:
                self._window_ndrs  = self._window_ndrs[-max_window:]
                self._window_jains = self._window_jains[-max_window:]

            # Per-worker tracking — used by _try_advance_stage to gate on hardest worker
            if idx not in self._worker_ndrs:
                self._worker_ndrs[idx]  = []
                self._worker_jains[idx] = []
            self._worker_ndrs[idx].append(stats["ndr"])
            self._worker_jains[idx].append(stats["jains_index"])
            if len(self._worker_ndrs[idx]) > max_window:
                self._worker_ndrs[idx]  = self._worker_ndrs[idx][-max_window:]
                self._worker_jains[idx] = self._worker_jains[idx][-max_window:]

            # Full-run history
            self._all_ndrs.append(stats["ndr"])
            self._all_jains.append(stats["jains_index"])

            # Per-condition tracking
            key  = (stats["grid_size"], stats["num_sensors"])
            self._condition_stats.setdefault(key, []).append(stats["jains_index"])

        # Check graduation and demotion gates every step
        self._try_advance_stage()
        self._try_demote_stage()

        # Periodic console logging every 20 completed episodes
        if self._n_episodes >= self._last_logged_ep + 20 and self._episode_rewards:
            try:
                all_stats    = self.training_env.get_attr("last_episode_stats")
                recent_eff   = [s["data_efficiency"]  for s in all_stats if s and "data_efficiency"  in s]
                recent_bpwh  = [s["bytes_per_wh"]     for s in all_stats if s and "bytes_per_wh"     in s]
                recent_coll  = [s["total_collected"]  for s in all_stats if s and "total_collected"  in s]
            except Exception:
                recent_eff, recent_bpwh, recent_coll = [], [], []

            eff_str  = "{:.1f}%".format(np.mean(recent_eff))   if recent_eff   else "n/a"
            bpwh_str = "{:.0f}".format(np.mean(recent_bpwh))   if recent_bpwh  else "n/a"
            coll_str = "{:.0f}B".format(np.mean(recent_coll))  if recent_coll  else "n/a"

            roll_ndr = self._rolling_mean(self._window_ndrs,  "ndr")
            roll_j   = self._rolling_mean(self._window_jains, "jains")

            # Show gate distance using hardest-worker rolling means when available
            # (pool mean can be inflated by easy workers — hardest worker is the real signal)
            _w = COMPETENCE_GATE["window"]
            _hw_ndrs  = [float(np.mean(v[-_w:])) for v in self._worker_ndrs.values()  if len(v) >= _w]
            _hw_jains = [float(np.mean(v[-_w:])) for v in self._worker_jains.values() if len(v) >= _w]
            gate_ndr   = min(_hw_ndrs)  if _hw_ndrs  else roll_ndr
            gate_jains = min(_hw_jains) if _hw_jains else roll_j

            dyn_ndr, dyn_j = self._get_stage_thresholds(self._current_stage)
            ndr_gap   = dyn_ndr - gate_ndr
            jains_gap = dyn_j   - gate_jains
            gate_str  = (
                "GATE MET ✓" if ndr_gap <= 0 and jains_gap <= 0
                else "hw_NDR={:.1f}% hw_J={:.3f} gap={:+.1f}%/{:+.3f} (tgt {:.1f}%/{:.3f})".format(
                    gate_ndr, gate_jains, ndr_gap, jains_gap, dyn_ndr, dyn_j
                )
            )

            print(
                "  ep={:5d} | stage={} | "
                "rew={:8.0f} | NDR={:.1f}% | J={:.4f} | "
                "collected={} | d_eff={} | B/Wh={} | {} | ts={}".format(
                    self._n_episodes,
                    self._current_stage,
                    np.mean(self._episode_rewards[-20:]),
                    roll_ndr,
                    roll_j,
                    coll_str,
                    eff_str,
                    bpwh_str,
                    gate_str,
                    self.num_timesteps,
                )
            )
            self._last_logged_ep = self._n_episodes

        return True

    # ------------------------------------------------------------------
    def get_condition_summary(self) -> dict:
        """Return per-(grid, sensors) Jain's index statistics collected during training."""
        return {
            str(k): {
                "mean_jains": float(np.mean(v)),
                "std_jains":  float(np.std(v)),
                "n_episodes": len(v),
            }
            for k, v in self._condition_stats.items()
        }

    # ------------------------------------------------------------------
    def get_graduation_log(self) -> list:
        """Return the list of stage graduation events (timestep, steps-in-stage, metrics)."""
        return self.graduation_log


# ==================== CONFIGURATION ====================

N_ENVS = 4   # One worker per sensor count — each sees all grid sizes.
             # 4 workers × episode_len=2100 fits within 4 GB VRAM with buffer=150k.

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
    "policy": "MlpPolicy",
    # Cosine-like LR decay: 3e-4 → 3e-5 over training. High LR early for fast
    # learning, low LR late so Q-values settle rather than oscillating.
    "optimize_memory_usage": True,
    "learning_rate":          lambda progress: 3e-4 * max(0.1, 1.0 - progress * 0.8),
    "buffer_size":            150_000,   # 612-dim × 4B × 2 × 150k ≈ 0.69 GB
    "batch_size":             256,
    "gamma":                  0.99,
    "learning_starts":        25_000,    # let buffer populate across all conditions first
    "exploration_fraction":   0.25,
    "exploration_final_eps":  0.03,      # low final ε forces policy to carry its own weight
    "target_update_interval": 5_000,     # stable Q-targets
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

EVAL_FREQ       = 5_000
N_EVAL_EPISODES = 25   # overridden to 100 in train_v3_full_retrain.py

SAVE_DIR = Path("models/dqn_v3")
LOG_DIR  = Path("logs/dqn_v3")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)


# ==================== BEST-BY-METRIC CALLBACK ====================

class BestByMetricCallback(BaseCallback):
    """
    Saves the model that maximises BWH × Jain's-fairness rather than raw reward.

    BWH   = total_bytes_collected / energy_used_Wh
    Jain  = (Σ CR_i)² / (N × Σ CR_i²),  CR_i = transmitted_i / generated_i
    Score = BWH × Jain  (multiplicative: both must be good to win)

    Saved to: <save_path>/best_metric_model.zip
    """

    BATTERY_FULL_WH = 274.0

    def __init__(self, eval_env, save_path: Path, eval_freq: int = 25_000,
                 n_eval_episodes: int = 5, verbose: int = 1):
        super().__init__(verbose)
        self.eval_env        = eval_env
        self.save_path       = Path(save_path)
        self.eval_freq       = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.best_score      = -np.inf
        self.save_path.mkdir(parents=True, exist_ok=True)

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq != 0:
            return True

        bwhs, jains = [], []
        obs = self.eval_env.reset()

        for _ in range(self.n_eval_episodes):
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, _, dones, infos = self.eval_env.step(action)
                done = dones[0]

            info            = infos[0]
            bytes_collected = float(info.get("total_data_collected", 0.0))
            battery_left    = float(info.get("battery", self.BATTERY_FULL_WH))
            energy_used     = max(self.BATTERY_FULL_WH - battery_left, 1e-6)
            bwh             = bytes_collected / energy_used

            crs  = np.array(info.get("sensor_collection_ratios", [1.0]), dtype=np.float64)
            n    = len(crs)
            jain = (crs.sum() ** 2) / (n * (crs ** 2).sum() + 1e-12)

            bwhs.append(bwh)
            jains.append(jain)

        mean_bwh  = float(np.mean(bwhs))
        mean_jain = float(np.mean(jains))
        score     = mean_bwh * mean_jain

        if self.verbose:
            print(
                "[BestByMetric] step={:,}  BWH={:.1f}  Jain={:.3f}  "
                "score={:.2f}  best={:.2f}".format(
                    self.num_timesteps, mean_bwh, mean_jain, score, self.best_score
                )
            )

        if score > self.best_score:
            self.best_score = score
            path = str(self.save_path / "best_metric_model")
            self.model.save(path)
            if self.verbose:
                print("  ✓ New best saved → {}.zip".format(path))

        return True


# ==================== MAIN ====================

def main():
    print("=" * 70)
    print("DQN TRAINING — DOMAIN RANDOMISATION + COMPETENCE-BASED CURRICULUM")
    print("=" * 70)
    print("Curriculum stages:")
    for i, (grids, sensors, desc) in enumerate(CURRICULUM_STAGES):
        print("  Stage {}  {}".format(i, desc))
    print()
    print("Competence Gate (graduation requires both):")
    print("  NDR   >= {:.1f}%  (rolling window = {} episodes)".format(
        COMPETENCE_GATE["ndr_pct"], COMPETENCE_GATE["window"]))
    print("  Jain's >= {:.2f}  (min dwell = {:,} steps)".format(
        COMPETENCE_GATE["jains"], COMPETENCE_GATE["min_steps"]))
    if STAGE_GATES and not GREEDY_BENCHMARK["enabled"]:
        print("Per-stage gate overrides:")
        for stage in sorted(STAGE_GATES.keys()):
            ov = STAGE_GATES[stage]
            print("  Stage {}: NDR >= {:.1f}%  Jain >= {:.2f}".format(
                stage,
                ov.get("ndr_pct", COMPETENCE_GATE["ndr_pct"]),
                ov.get("jains",   COMPETENCE_GATE["jains"]),
            ))
    print()
    print("Demotion Gate (triggers if both drop below):")
    print("  NDR   < {:.1f}%  |  Jain's < {:.2f}  (after {} episodes in stage)".format(
        DEMOTION_GATE["ndr_pct"], DEMOTION_GATE["jains"], DEMOTION_GATE["min_episodes"]))
    print()
    print("MAX_SENSORS_LIMIT: {}  (NN input size fixed — never change)".format(MAX_SENSORS_LIMIT))
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
    print()

    # ── Callbacks ────────────────────────────────────────────────────────
    checkpoint_cb = CheckpointCallback(
        save_freq   = TRAINING_CONFIG["save_freq"],
        save_path   = str(SAVE_DIR),
        name_prefix = "dqn_dr",
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

    # ── Save model ───────────────────────────────────────────────────────
    model.save(str(SAVE_DIR / "dqn_final"))
    print("Model saved: {}".format(SAVE_DIR / "dqn_final"))

    # Save training config
    _tmp = DomainRandEnv(
        fixed_num_sensors = WORKER_SENSOR_COUNTS[0],
        max_sensors_limit = MAX_SENSORS_LIMIT,
        curriculum_stage  = 0,
        base_config       = BASE_ENV_CONFIG,
    )
    training_config = {
        "use_frame_stacking":  True,
        "n_stack":             TRAINING_CONFIG["n_stack"],
        "max_sensors_limit":   MAX_SENSORS_LIMIT,
        "features_per_sensor": _tmp._features_per_sensor,
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

    # Save per-condition performance
    condition_summary = curriculum_cb.get_condition_summary()
    with open(SAVE_DIR / "condition_summary.json", "w") as f:
        json.dump(condition_summary, f, indent=2)

    # Save graduation log
    graduation_log = curriculum_cb.get_graduation_log()
    with open(SAVE_DIR / "graduation_log.json", "w") as f:
        json.dump(graduation_log, f, indent=2)

    # ── Final summary ────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Per-condition Jain's index during training:")
    print("=" * 70)
    for cond, stats in sorted(condition_summary.items()):
        print("  {}  ->  J={:.4f} ± {:.4f}  ({} episodes)".format(
            cond, stats["mean_jains"], stats["std_jains"], stats["n_episodes"]
        ))

    print("\n" + "=" * 70)
    print("Curriculum Graduation Log (Time to Graduation per Stage):")
    print("=" * 70)
    if graduation_log:
        for entry in graduation_log:
            print(
                "  Stage {} → Stage {}  |  graduated at step {:,}  "
                "|  steps in stage: {:,}  "
                "|  NDR={:.1f}%  Jain={:.3f}".format(
                    entry["from_stage"],
                    entry["to_stage"],
                    entry["ts"],
                    entry["steps_in_stage"],
                    entry["rolling_ndr"],
                    entry["rolling_jains"],
                )
            )
    else:
        final_stage = curriculum_cb._current_stage
        print("  Agent reached Stage {} but no further graduations occurred.".format(final_stage))
        print("  Consider increasing total_timesteps or relaxing COMPETENCE_GATE thresholds.")

    print("\nDone. Point evaluation scripts at:")
    print("  DQN_MODEL_PATH:  {}".format(SAVE_DIR / "dqn_final.zip"))
    print("  DQN_CONFIG_PATH: {}".format(SAVE_DIR / "training_config.json"))


if __name__ == "__main__":
    main()