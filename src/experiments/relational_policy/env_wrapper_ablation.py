"""
Ablation env wrapper for the Relational RL retraining experiment.

Differences from env_wrapper.py (RelationalUAVEnv):
  - NO cluster dwell bonus: any hovering near sensors is purely emergent behaviour
    driven by the underlying LoRaWAN physics (EMA-ADR convergence), not a shaped
    reward.  This eliminates an artefactual confound when comparing architectures.
  - Step penalty: -STEP_PENALTY / max_steps per timestep, applied AFTER reward
    scaling.  Over a full 2100-step episode the total contribution is -1.0 in
    scaled reward units (episode return ~12), i.e. ~8% of return.  Mild enough
    not to discourage necessary movement, large enough to penalise jitter.
  - Extended episode metrics: Gini coefficient and min/max collection ratio over
    per-sensor CR, logged to info dict and available to the training callback.

Potential-based shaping (Ng et al., 1999) is kept unchanged — it is policy-
invariant and does not introduce any architectural confound.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np

_HERE = Path(__file__).resolve().parent
_SRC  = _HERE.parent.parent
_ROOT = _SRC.parent
for _p in (str(_SRC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from environment.uav_env import UAVEnvironment
from experiments.relational_policy.env_wrapper import (
    RelationalUAVEnv,
    EpisodeMetricsStore,
    GAMMA,
    REWARD_SCALE,
    N_MAX,
)

# Per-step penalty magnitude.  Total episode contribution = -1.0 in scaled
# reward space over 2100 steps (~8% of a typical scaled episode return of 12).
STEP_PENALTY = 1.0


def _gini(values: list[float]) -> float:
    """
    Gini coefficient over a list of non-negative values.

    Returns 0.0 for perfect equality, approaching 1.0 as one value dominates.
    Formula: G = (2 * Σ_i rank_i * x_i) / (n * Σ x_i) - (n+1)/n
    where ranks are 1-indexed over values sorted ascending.
    """
    n = len(values)
    if n == 0:
        return 0.0
    total = sum(values)
    if total == 0.0:
        return 0.0
    sorted_v = sorted(values)
    weighted = sum((i + 1) * v for i, v in enumerate(sorted_v))
    return float((2 * weighted) / (n * total) - (n + 1) / n)


class AblationUAVEnv(RelationalUAVEnv):
    """
    RelationalUAVEnv variant for the Temporal-Memory ablation experiment.

    Removes the cluster dwell bonus so any hovering behaviour is emergent
    (proves the architecture manages EMA-ADR latency without reward engineering).
    Adds a step penalty to surface the energy cost of stateless aimless motion
    relative to the DQN's frame-stack temporal memory.

    Extended episode metrics logged to info at episode end:
      gini    — Gini coefficient over per-sensor collection ratio (lower = fairer)
      min_cr  — minimum per-sensor collection ratio  ∈ [0, 1]
      max_cr  — maximum per-sensor collection ratio  ∈ [0, 1]
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

    def step(self, action: int):
        phi_s = self._prev_potential

        # Call UAVEnvironment.step directly — bypass RelationalUAVEnv.step so
        # we never invoke _cluster_dwell_bonus.
        _, base_reward, terminated, truncated, info = UAVEnvironment.step(
            self, action
        )

        phi_s_prime = self._compute_potential()
        self._prev_potential = phi_s_prime
        shaping = GAMMA * phi_s_prime - phi_s

        if terminated or truncated:
            ndr, jains, eff = self._compute_episode_metrics()
            gini, min_cr, max_cr = self._compute_buffer_equality()
            EpisodeMetricsStore.record(ndr, jains, eff)
            info.update({
                "ndr":        ndr,
                "jains":      jains,
                "efficiency": eff,
                "gini":       gini,
                "min_cr":     min_cr,
                "max_cr":     max_cr,
            })

        scaled = (base_reward + shaping) * REWARD_SCALE
        step_penalty = -STEP_PENALTY / self.max_steps  # total = -1.0 per episode

        return (
            self._build_relational_obs(),
            scaled + step_penalty,
            terminated,
            truncated,
            info,
        )

    def _compute_buffer_equality(self) -> tuple[float, float, float]:
        """
        Gini coefficient, min, and max collection ratio over all sensors.

        CR_i = total_data_transmitted_i / total_data_generated_i  ∈ [0, 1].
        Low Gini + high min_cr = balanced service across all sensors.
        """
        cr_list = [
            s.total_data_transmitted / max(s.total_data_generated, 1e-9)
            for s in self.sensors
        ]
        if not cr_list:
            return 0.0, 0.0, 0.0
        return _gini(cr_list), float(min(cr_list)), float(max(cr_list))
