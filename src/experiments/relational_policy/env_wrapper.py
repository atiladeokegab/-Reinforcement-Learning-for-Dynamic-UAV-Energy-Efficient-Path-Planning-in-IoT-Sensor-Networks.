"""
Relational observation wrapper for UAVEnvironment.

Converts the flat Box observation into a Dict with:
  "uav"     : (3,)       — normalised [x_norm, y_norm, battery_norm]
  "sensors" : (N_MAX, 5) — per-slot [rel_x, rel_y, buffer, urgency, visited]
  "mask"    : (N_MAX,)   — 1=real sensor slot, 0=zero-padding

Also adds two reward supplements:
  1. Potential-based shaping  — penalises detours from the nearest unvisited
     sensor.  Uses Φ(s) = −dist_nearest_unvisited / grid_diagonal so that
     γΦ(s') − Φ(s) > 0 whenever the UAV approaches an unvisited sensor.
     Ng et al. (1999) guarantees this leaves the optimal policy unchanged.

  2. Cluster dwell bonus      — encourages the UAV to hover near ≥2 sensors
     for up to DWELL_MAX_STEPS consecutive steps before repositioning.
     Rationale: EMA-ADR (λ=0.1) needs ~10 steps to converge; staying in a
     cluster during that window maximises per-bit throughput.

Grid: 100×100 to 500×500 units (10 m/unit, UAV at 100 m altitude).
Battery: 274 Wh.  Flight power: 500 W.
"""

from __future__ import annotations

import sys
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces

# ── Make the existing src package importable regardless of CWD ────────────────
_HERE = Path(__file__).resolve().parent   # src/experiments/relational_policy/
_SRC  = _HERE.parent.parent              # src/
_ROOT = _SRC.parent                      # project root
for _p in (str(_SRC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from environment.uav_env import UAVEnvironment  # noqa: E402

# ── Constants ─────────────────────────────────────────────────────────────────
N_MAX            = 50    # Fixed observation tensor width (pad unused slots)
GAMMA            = 0.99  # Must equal PPO gamma for shaping to be policy-invariant
DWELL_BONUS_MAX  = 25.0  # Reward at dwell step 0; decays linearly to 0 at step 10
DWELL_MAX_STEPS  = 10    # ADR convergence horizon for EMA λ=0.1
MIN_CLUSTER_SIZE = 2     # Minimum in-range sensors to trigger dwell bonus
METRICS_WINDOW   = 20    # Rolling window for curriculum gating (episodes)

# Reward scale.  Episode returns from the base UAVEnvironment are ~10^6 because
# of the +5000 sensor-visit bonus, +1000 urgency-reduction, and +100 per byte ×
# urgency.  Ray RLlib's new API stack ignores PPOConfig(vf_clip_param=1e8) and
# clips value loss to 10.0, which makes vf_explained_var ≈ 0 and reduces PPO
# to REINFORCE with a zero baseline.  Scaling rewards by REWARD_SCALE brings
# returns into a range where vf_clip=10 is non-binding.  PPO is invariant to
# reward scale × learning rate, so the policy gradient direction is unchanged.
REWARD_SCALE = 1e-5      # divide reward by 10^5; episode returns ~12 instead of ~10^6


class EpisodeMetricsStore:
    """
    Lightweight process-local store for episode-end metrics.

    The training loop reads `rolling_means()` after each `algo.train()` call
    to decide whether to advance the curriculum.  Works correctly with
    `num_env_runners=0` (single-process) where the env and training loop share
    the same Python process and therefore the same class state.

    Metrics tracked (all normalised to [0, 1] or natural units):
      ndr        — Normalised Data Rate: Σ transmitted / Σ generated  ∈ [0,1]
      jains      — Jain's Fairness Index over per-sensor collection ratios ∈ [0,1]
      efficiency — bytes collected per Wh of battery consumed (B/Wh)
    """

    _ndr:        deque = deque(maxlen=METRICS_WINDOW)
    _jains:      deque = deque(maxlen=METRICS_WINDOW)
    _efficiency: deque = deque(maxlen=METRICS_WINDOW)

    @classmethod
    def reset(cls) -> None:
        """Clear history between curriculum stages."""
        cls._ndr.clear()
        cls._jains.clear()
        cls._efficiency.clear()

    @classmethod
    def record(cls, ndr: float, jains: float, efficiency: float) -> None:
        cls._ndr.append(ndr)
        cls._jains.append(jains)
        cls._efficiency.append(efficiency)

    @classmethod
    def rolling_means(cls) -> dict[str, float]:
        return {
            "ndr":        float(np.mean(cls._ndr))        if cls._ndr        else 0.0,
            "jains":      float(np.mean(cls._jains))      if cls._jains      else 0.0,
            "efficiency": float(np.mean(cls._efficiency)) if cls._efficiency else 0.0,
            "n_episodes": len(cls._ndr),
        }

    @classmethod
    def ready(cls) -> bool:
        """True once the window is full enough to be statistically meaningful."""
        return len(cls._ndr) >= max(5, METRICS_WINDOW // 4)


class RelationalUAVEnv(UAVEnvironment):
    """
    UAVEnvironment subclass with relational Dict observations and reward
    supplements.  All physics, LoRaWAN, and sensor logic is inherited
    unchanged — only the obs format and reward are modified.

    Parameters
    ----------
    grid_size   : (W, H) in grid units.  Capped at 500×500 per dissertation scope.
    num_sensors : Number of IoT sensors (10–N_MAX=50).
    n_max       : Observation tensor width.  Must be ≥ num_sensors.
    **kwargs    : Forwarded to UAVEnvironment (max_battery, max_steps, …).
    """

    def __init__(
        self,
        grid_size: tuple[int, int] = (500, 500),
        num_sensors: int = 20,
        n_max: int = N_MAX,
        **kwargs: Any,
    ):
        assert num_sensors <= n_max, f"num_sensors={num_sensors} exceeds n_max={n_max}"
        assert grid_size[0] <= 500 and grid_size[1] <= 500, (
            "Grid capped at 500×500 per dissertation scope"
        )

        self.n_max = n_max

        super().__init__(
            grid_size=grid_size,
            num_sensors=num_sensors,
            include_sensor_positions=True,  # needed so parent builds full flat obs
            **kwargs,
        )

        # Override the parent's flat Box observation space
        self.observation_space = spaces.Dict({
            "uav": spaces.Box(
                low=-1.0, high=1.0, shape=(3,), dtype=np.float32
            ),
            "sensors": spaces.Box(
                low=-1.0, high=1.0, shape=(n_max, 5), dtype=np.float32
            ),
            "mask": spaces.Box(
                low=0.0, high=1.0, shape=(n_max,), dtype=np.float32
            ),
        })

        self._dwell_steps: int   = 0
        self._prev_potential: float = 0.0

    # ── Gymnasium interface ───────────────────────────────────────────────────

    def reset(self, **kwargs):
        _, info = super().reset(**kwargs)
        self._dwell_steps = 0
        self._prev_potential = self._compute_potential()
        return self._build_relational_obs(), info

    def step(self, action: int):
        phi_s = self._prev_potential

        # Parent handles physics, LoRaWAN, base reward
        _, base_reward, terminated, truncated, info = super().step(action)

        phi_s_prime = self._compute_potential()
        self._prev_potential = phi_s_prime

        shaping     = GAMMA * phi_s_prime - phi_s
        dwell_bonus = self._cluster_dwell_bonus(action)

        # At episode end, compute and record coverage metrics for curriculum gating
        if terminated or truncated:
            ndr, jains, eff = self._compute_episode_metrics()
            EpisodeMetricsStore.record(ndr, jains, eff)
            info["ndr"]        = ndr
            info["jains"]      = jains
            info["efficiency"] = eff

        scaled_reward = (base_reward + shaping + dwell_bonus) * REWARD_SCALE
        return (
            self._build_relational_obs(),
            scaled_reward,
            terminated,
            truncated,
            info,
        )

    # ── Observation builder ───────────────────────────────────────────────────

    def _build_relational_obs(self) -> dict[str, np.ndarray]:
        W = float(self.grid_size[0])
        H = float(self.grid_size[1])
        uav_x = float(self.uav.position[0])
        uav_y = float(self.uav.position[1])

        uav_obs = np.array(
            [uav_x / W, uav_y / H, self.uav.battery / self.uav.max_battery],
            dtype=np.float32,
        )

        sensor_obs = np.zeros((self.n_max, 5), dtype=np.float32)
        mask_obs   = np.zeros(self.n_max,      dtype=np.float32)

        for i, sensor in enumerate(self.sensors):
            if i >= self.n_max:
                break
            sx = float(sensor.position[0])
            sy = float(sensor.position[1])
            sensor_obs[i] = [
                np.clip((sx - uav_x) / W, -1.0, 1.0),  # rel_x
                np.clip((sy - uav_y) / H, -1.0, 1.0),  # rel_y
                sensor.data_buffer / sensor.max_buffer_size,           # buffer fill [0,1]
                self._calculate_urgency(sensor),                        # urgency   [0,1]
                1.0 if sensor.sensor_id in self.sensors_visited else 0.0,  # visited
            ]
            mask_obs[i] = 1.0

        return {"uav": uav_obs, "sensors": sensor_obs, "mask": mask_obs}

    # ── Episode metrics ───────────────────────────────────────────────────────

    def _compute_episode_metrics(self) -> tuple[float, float, float]:
        """
        Compute NDR, Jain's Fairness Index, and Efficiency at episode end.

        NDR  = Σ_i transmitted_i / Σ_i generated_i
               Fraction of all generated sensor data that was collected.
               Same definition used as primary metric in the DQN evaluation.

        Jain = (Σ CR_i)² / (N · Σ CR_i²)   where CR_i = transmitted_i / generated_i
               1.0 = perfectly fair; approaches 1/N when one sensor dominates.

        Eff  = total_data_collected [bytes] / battery_used [Wh]
               Measures how efficiently the UAV extracts data per unit energy.
               Target ≥ 200 B/Wh at Stage 4 per dissertation energy analysis.
        """
        total_gen  = sum(s.total_data_generated  for s in self.sensors)
        total_tx   = sum(s.total_data_transmitted for s in self.sensors)
        ndr = float(total_tx / total_gen) if total_gen > 0 else 0.0

        cr_list = [
            s.total_data_transmitted / max(s.total_data_generated, 1e-9)
            for s in self.sensors
        ]
        n = len(cr_list)
        sum_cr  = sum(cr_list)
        sum_cr2 = sum(c * c for c in cr_list)
        jains = float((sum_cr ** 2) / (n * sum_cr2)) if sum_cr2 > 0 else 0.0

        battery_used_wh = self.uav.max_battery - self.uav.battery
        efficiency = (
            float(self.total_data_collected / battery_used_wh)
            if battery_used_wh > 0 else 0.0
        )

        return ndr, jains, efficiency

    # ── Reward helpers ────────────────────────────────────────────────────────

    def _compute_potential(self) -> float:
        """
        Φ(s) = −nearest_unvisited_distance / grid_diagonal.

        Bounded in [−1, 0].  Zero when all sensors have been visited.
        The negative sign means γΦ(s') − Φ(s) > 0 iff UAV moves closer to
        the next unvisited sensor (the desired behaviour).
        """
        unvisited = [
            s for s in self.sensors
            if s.sensor_id not in self.sensors_visited
        ]
        if not unvisited:
            return 0.0

        uav_pos   = self.uav.position
        dists     = [np.linalg.norm(np.array(s.position) - uav_pos) for s in unvisited]
        grid_diag = float(np.sqrt(self.grid_size[0] ** 2 + self.grid_size[1] ** 2))
        return -float(min(dists)) / grid_diag

    def _cluster_dwell_bonus(self, action: int) -> float:
        """
        Bonus for hovering (action=4) near ≥2 sensors for the first
        DWELL_MAX_STEPS consecutive steps.

        The bonus decays linearly to zero over the dwell window so the agent
        is incentivised to reposition once ADR has converged.
        """
        if action != 4:
            self._dwell_steps = 0
            return 0.0

        uav_pos   = (float(self.uav.position[0]), float(self.uav.position[1]))
        n_in_range = sum(1 for s in self.sensors if s.is_in_range(uav_pos))

        if n_in_range >= MIN_CLUSTER_SIZE and self._dwell_steps < DWELL_MAX_STEPS:
            frac  = 1.0 - self._dwell_steps / DWELL_MAX_STEPS
            bonus = DWELL_BONUS_MAX * frac
            self._dwell_steps += 1
            return float(bonus)

        if n_in_range < MIN_CLUSTER_SIZE:
            self._dwell_steps = 0
        return 0.0
