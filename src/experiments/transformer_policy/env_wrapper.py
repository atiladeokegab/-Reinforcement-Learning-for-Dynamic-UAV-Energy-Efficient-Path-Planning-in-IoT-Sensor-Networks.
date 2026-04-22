"""
env_wrapper.py
==============
Gymnasium wrapper that adapts UAVEnvironment to the fixed observation layout
required by the GTrXLNet transformer policy.

Changes from the base UAVEnvironment
-------------------------------------
1. Sensor positions (dx, dy) are always included so that the multi-head
   attention mechanism can learn to sort sensors by both urgency *and*
   spatial proximity simultaneously.  Without relative positions the
   Transformer can still rank sensors by buffer fill / loss rate, but
   cannot weigh reachability.

2. Observations are zero-padded to a fixed ceiling of MAX_SENSORS=50.
   Padding slots are written as 0.0 across all 5 feature channels so they
   produce zero-valued keys and queries in the attention layers and are
   effectively masked out by the softmax denominator.

3. Frame-stacking is removed.  GTrXL's 50-step recurrent memory replaces
   the k=4 FrameStack used by the DQN baseline.  This avoids aliasing the
   temporal dimension twice and reduces the input dimensionality from
   4 × (3 + 5N) to (3 + 5N).

Resulting flat observation (253 floats, all in [-1, 1])
-------------------------------------------------------
  [ uav_x_norm, uav_y_norm, battery_norm,          ← indices 0-2
    buf_0, urg_0, lq_0, dx_0, dy_0,                ← indices 3-7
    buf_1, urg_1, lq_1, dx_1, dy_1,                ← indices 8-12
    ...
    buf_49, urg_49, lq_49, dx_49, dy_49 ]          ← indices 248-252

Padding: sensors [num_sensors, 49] are all zeros.

RLlib env registration
-----------------------
    import ray
    from env_wrapper import TransformerObsWrapper
    ray.tune.register_env("UAVTransformerEnv", lambda cfg: TransformerObsWrapper(cfg))
"""

from __future__ import annotations

import sys
import pathlib
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Ensure environment package is importable regardless of working directory.
_SRC = pathlib.Path(__file__).resolve().parents[2]   # src/ (not project root)
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from environment.uav_env import UAVEnvironment   # noqa: E402

# ---------------------------------------------------------------------------
# Constants (must stay in sync with transformer_model.py)
# ---------------------------------------------------------------------------
MAX_SENSORS = 50
UAV_FEATURES = 3        # [x_norm, y_norm, battery_norm]
SENSOR_FEATURES = 5     # [buffer, urgency, link_quality, dx, dy]
OBS_DIM = UAV_FEATURES + SENSOR_FEATURES * MAX_SENSORS   # 253

# Reward scale.  Episode returns from the base UAVEnvironment are ~10^6 because
# of the +5000 sensor-visit bonus, +1000 urgency-reduction, and +100 per byte ×
# urgency.  Ray RLlib clips PPO value loss to vf_clip_param=10 by default and
# silently ignores PPOConfig overrides on the new API stack.  This makes
# vf_explained_var ≈ 0 (the value head learns nothing) and reduces PPO to
# REINFORCE with a zero baseline.  Scaling rewards by REWARD_SCALE brings
# returns into a range where vf_clip=10 is non-binding.  PPO is invariant to
# reward scale × learning rate, so the policy gradient direction is unchanged.
REWARD_SCALE = 1e-5      # divide reward by 10^5; episode returns ~12 instead of ~10^6


class TransformerObsWrapper(gym.Wrapper):
    """
    Wraps UAVEnvironment with a fixed-size, padded observation space for
    the GTrXLNet policy.

    Parameters (passed as env_config dict by RLlib)
    -----------------------------------------------
    grid_size     : tuple[int, int]   – (W, H) of the grid, default (500, 500)
    num_sensors   : int               – active sensors, must be ≤ MAX_SENSORS
    max_steps     : int               – episode horizon, default 2100
    Any remaining keys are forwarded to UAVEnvironment.__init__.
    """

    def __init__(self, env_config: dict[str, Any] | None = None) -> None:
        cfg = dict(env_config or {})

        grid_size = cfg.pop("grid_size", (500, 500))
        num_sensors = int(cfg.pop("num_sensors", 20))

        if num_sensors > MAX_SENSORS:
            raise ValueError(
                f"num_sensors={num_sensors} exceeds MAX_SENSORS={MAX_SENSORS}. "
                "Increase MAX_SENSORS and retrain the network."
            )

        base_env = UAVEnvironment(
            grid_size=grid_size,
            num_sensors=num_sensors,
            include_sensor_positions=True,  # provides dx, dy per sensor
            **cfg,
        )
        super().__init__(base_env)

        self._num_sensors: int = num_sensors
        # Base obs size before padding: 3 + 5*num_sensors
        self._base_obs_len: int = UAV_FEATURES + SENSOR_FEATURES * num_sensors

        # Override observation space with padded, fixed-size spec.
        self.observation_space = spaces.Box(
            low=np.full(OBS_DIM, -1.0, dtype=np.float32),
            high=np.ones(OBS_DIM, dtype=np.float32),
            dtype=np.float32,
        )

    # ------------------------------------------------------------------
    # Public API for curriculum learning (called from training script)
    # ------------------------------------------------------------------

    def reconfigure(self, grid_size: tuple[int, int], num_sensors: int) -> None:
        """
        Replace the underlying environment with a new curriculum stage config.

        Called by the training loop via
            algo.workers.foreach_env(lambda e: e.reconfigure(...))
        so all rollout workers advance simultaneously.
        """
        if num_sensors > MAX_SENSORS:
            raise ValueError(f"num_sensors={num_sensors} > MAX_SENSORS={MAX_SENSORS}")

        self.env = UAVEnvironment(
            grid_size=grid_size,
            num_sensors=num_sensors,
            include_sensor_positions=True,
        )
        self._num_sensors = num_sensors
        self._base_obs_len = UAV_FEATURES + SENSOR_FEATURES * num_sensors

    # ------------------------------------------------------------------
    # gym.Wrapper overrides
    # ------------------------------------------------------------------

    def reset(self, **kwargs) -> tuple[np.ndarray, dict]:
        obs, info = self.env.reset(**kwargs)
        return self._pad(obs), info

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Inject episode-end metrics so the MetricsCallback can read them from
        # info[].  Without this, NDR/Jain are always 0 because the base
        # UAVEnvironment does not populate "sensor_collection_ratios".
        if terminated or truncated:
            ndr, jains, eff = self._compute_episode_metrics()
            info["ndr"]        = ndr
            info["jains"]      = jains
            info["efficiency"] = eff

        return self._pad(obs), reward * REWARD_SCALE, terminated, truncated, info

    def _compute_episode_metrics(self) -> tuple[float, float, float]:
        """
        NDR  = Σ transmitted / Σ generated   (fraction of generated data collected)
        Jain = (Σ CR_i)² / (N · Σ CR_i²)     (fairness over per-sensor CRs)
        Eff  = total bytes collected / battery used (Wh)
        """
        sensors    = self.env.sensors
        total_gen  = sum(s.total_data_generated   for s in sensors)
        total_tx   = sum(s.total_data_transmitted for s in sensors)
        ndr        = float(total_tx / total_gen) if total_gen > 0 else 0.0

        cr_list = [
            s.total_data_transmitted / max(s.total_data_generated, 1e-9)
            for s in sensors
        ]
        n       = len(cr_list)
        sum_cr  = sum(cr_list)
        sum_cr2 = sum(c * c for c in cr_list)
        jains   = float((sum_cr ** 2) / (n * sum_cr2)) if sum_cr2 > 0 else 0.0

        battery_used_wh = self.env.uav.max_battery - self.env.uav.battery
        efficiency = (
            float(self.env.total_data_collected / battery_used_wh)
            if battery_used_wh > 0 else 0.0
        )
        return ndr, jains, efficiency

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pad(self, obs: np.ndarray) -> np.ndarray:
        """Zero-pad a variable-length obs to OBS_DIM."""
        if obs.shape[0] == OBS_DIM:
            return obs.astype(np.float32)
        padded = np.zeros(OBS_DIM, dtype=np.float32)
        padded[: obs.shape[0]] = obs
        return padded
