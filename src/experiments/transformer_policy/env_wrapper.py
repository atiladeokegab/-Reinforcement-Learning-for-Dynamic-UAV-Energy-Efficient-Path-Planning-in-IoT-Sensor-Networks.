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
        return self._pad(obs), reward, terminated, truncated, info

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
