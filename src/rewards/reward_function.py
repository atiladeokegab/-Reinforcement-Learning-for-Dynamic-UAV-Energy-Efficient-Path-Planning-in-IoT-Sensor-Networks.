import numpy as np
from typing import List, Dict


class RewardFunction:
    """
    Multi-objective reward function for UAV IoT data collection.

    Optimisation targets: data throughput + sensor coverage fairness.
    Energy is a hard physical constraint (episode ends at 274 Wh); hover
    actions carry an extra power penalty vs movement to reflect rotary-wing
    aerodynamics (hover ~700 W > flight ~500 W).

    R_t = R_coll + R_bonus - C_fair - C_phys - C_term

    Component summary:
      R_coll  = reward_per_byte * bytes * urgency   (urgency-weighted throughput)
      reward_new_sensor     +5000  first visit to each sensor (coverage)
      reward_multi_sensor   +200   per extra simultaneous decodable reception
      reward_urgency_reduction +1000 * urgency reduced (AoI shaping)
      reward_movement       +10    per successful move (anti-hover)
      penalty_step          -0.5   per step (base time cost)
      penalty_hover         -5.0   extra cost for hover vs move (power differential)
      penalty_starvation    up to -250/step based on buffer variance (fairness)
      penalty_unvisited     -15000 per sensor never visited (terminal)
      penalty_boundary      -50    per boundary collision
      penalty_revisit       -2     per empty revisit
    """

    def __init__(
        self,
        # Rewards
        reward_per_byte: float          = 100.0,
        reward_new_sensor: float        = 5000.0,
        reward_multi_sensor: float      = 200.0,
        reward_completion: float        = 100.0,
        reward_urgency_reduction: float = 1000.0,
        reward_movement: float          = 10.0,
        # Penalties
        penalty_revisit: float          = -2.0,
        penalty_boundary: float         = -50.0,
        penalty_collision: float        = -10.0,
        penalty_battery: float          = 0.0,      # hard physical limit — not optimised
        penalty_hover: float            = -5.0,     # extra cost of hover vs move (700W vs 500W)
        penalty_step: float             = -0.5,
        penalty_data_loss: float        = -1.0,
        penalty_starvation: float       = -1000.0,
        penalty_unvisited: float        = -15000.0,
    ):
        self.reward_per_byte          = reward_per_byte
        self.reward_new_sensor        = reward_new_sensor
        self.reward_multi_sensor      = reward_multi_sensor
        self.reward_completion        = reward_completion
        self.reward_urgency_reduction = reward_urgency_reduction
        self.reward_movement          = reward_movement

        self.penalty_revisit    = penalty_revisit
        self.penalty_boundary   = penalty_boundary
        self.penalty_collision  = penalty_collision
        self.penalty_battery    = penalty_battery
        self.penalty_hover      = penalty_hover
        self.penalty_step       = penalty_step
        self.penalty_data_loss  = penalty_data_loss
        self.penalty_starvation = penalty_starvation
        self.penalty_unvisited  = penalty_unvisited

    # ------------------------------------------------------------------ #
    # Fairness helper                                                      #
    # ------------------------------------------------------------------ #

    def calculate_starvation_penalty(self, sensor_buffers: List[float]) -> float:
        """
        Variance-based fairness penalty: -1000 * Var(normalised_buffers).

        Equal buffers → penalty = 0.
        Binary split (half full, half empty) → max penalty = -250/step.
        """
        if not sensor_buffers or len(sensor_buffers) <= 1:
            return 0.0

        max_buffer = max(sensor_buffers)
        if max_buffer == 0:
            return 0.0

        normalized = np.array([b / max_buffer for b in sensor_buffers])
        variance   = float(np.var(normalized))
        return self.penalty_starvation * variance

    # ------------------------------------------------------------------ #
    # Movement reward                                                      #
    # ------------------------------------------------------------------ #

    def calculate_movement_reward(
        self, move_success: bool, battery_used: float
    ) -> float:
        """
        Reward for a movement action.
        Move steps pay only penalty_step (no hover surcharge).
        """
        reward = self.penalty_step
        if move_success:
            reward += self.reward_movement
        else:
            reward += self.penalty_boundary
        reward += self.penalty_battery * battery_used
        return reward

    # ------------------------------------------------------------------ #
    # Collection reward                                                    #
    # ------------------------------------------------------------------ #

    def calculate_collection_reward(
        self,
        bytes_collected: float,
        was_new_sensor: bool,
        was_empty: bool,
        all_sensors_collected: bool,
        battery_used: float,
        num_sensors_collected: int  = 1,
        collision_count: int        = 0,
        data_loss: float            = 0.0,
        urgency_reduced: float      = 0.0,
        sensor_buffers: List[float] = None,
        unvisited_count: int        = 0,
        sensor_urgency: float       = 0.0,
    ) -> float:
        """
        Fairness-coupled reward for a hover/collect action.

        Byte reward = reward_per_byte * bytes * sensor_urgency
          sensor_urgency in [0, 1]: 0 = just collected, 1 = buffer full + data loss.
          A neglected sensor (urgency=1) earns full 100/byte.
          A recently-serviced sensor (urgency=0.1) earns 10/byte.
          This directly couples throughput maximisation with fairness.

        Hover surcharge (penalty_hover) is applied every collection step
        to reflect the 700W hover cost vs 500W flight cost.

        unvisited_count: pass non-zero only on the terminal step
          (self.num_sensors - len(self.sensors_visited)).
        """
        # Base step cost + hover surcharge
        reward = self.penalty_step + self.penalty_hover

        # ── Urgency-weighted data collection ──────────────────────────
        if bytes_collected > 0:
            reward += self.reward_per_byte * bytes_collected * sensor_urgency

            if was_new_sensor:
                reward += self.reward_new_sensor

            if num_sensors_collected > 1:
                reward += self.reward_multi_sensor * (num_sensors_collected - 1)

        # ── Urgency reduction bonus ────────────────────────────────────
        if urgency_reduced > 0:
            reward += self.reward_urgency_reduction * urgency_reduced

        # ── Empty-visit penalty ────────────────────────────────────────
        if was_empty and bytes_collected == 0:
            reward += self.penalty_revisit

        # ── Battery (disabled — physical constraint only) ──────────────
        reward += self.penalty_battery * battery_used

        # ── Collisions ─────────────────────────────────────────────────
        if collision_count > 0:
            reward += self.penalty_collision * collision_count

        # ── Data loss ──────────────────────────────────────────────────
        if data_loss > 0:
            reward += self.penalty_data_loss * data_loss

        # ── Variance-based starvation penalty ─────────────────────────
        if sensor_buffers is not None:
            reward += self.calculate_starvation_penalty(sensor_buffers)

        # ── Terminal bonuses / penalties ───────────────────────────────
        if all_sensors_collected:
            reward += self.reward_completion

        if unvisited_count > 0:
            reward += self.penalty_unvisited * unvisited_count

        return reward

    # ------------------------------------------------------------------ #
    # Diagnostic breakdown                                                 #
    # ------------------------------------------------------------------ #

    def get_reward_breakdown(
        self,
        bytes_collected: float      = 0.0,
        was_new_sensor: bool        = False,
        was_empty: bool             = False,
        all_sensors_collected: bool = False,
        battery_used: float         = 0.0,
        num_sensors_collected: int  = 1,
        collision_count: int        = 0,
        data_loss: float            = 0.0,
        urgency_reduced: float      = 0.0,
        sensor_buffers: List[float] = None,
        unvisited_count: int        = 0,
        sensor_urgency: float       = 0.0,
        is_hover: bool              = True,
    ) -> Dict:
        """Return a per-component reward breakdown for diagnostics."""
        starvation = (
            self.calculate_starvation_penalty(sensor_buffers)
            if sensor_buffers is not None else 0.0
        )

        breakdown = {
            "step_penalty":            self.penalty_step,
            "hover_penalty":           self.penalty_hover if is_hover else 0.0,
            "movement_bonus":          0.0 if is_hover else self.reward_movement,
            "data_reward":             self.reward_per_byte * bytes_collected * sensor_urgency
                                       if bytes_collected > 0 else 0.0,
            "new_sensor_bonus":        self.reward_new_sensor if was_new_sensor else 0.0,
            "multi_sensor_bonus":      self.reward_multi_sensor * (num_sensors_collected - 1)
                                       if num_sensors_collected > 1 else 0.0,
            "urgency_reduction_bonus": self.reward_urgency_reduction * urgency_reduced
                                       if urgency_reduced > 0 else 0.0,
            "empty_penalty":           self.penalty_revisit
                                       if was_empty and bytes_collected == 0 else 0.0,
            "battery_penalty":         self.penalty_battery * battery_used,
            "collision_penalty":       self.penalty_collision * collision_count
                                       if collision_count > 0 else 0.0,
            "data_loss_penalty":       self.penalty_data_loss * data_loss
                                       if data_loss > 0 else 0.0,
            "starvation_penalty":      starvation,
            "completion_bonus":        self.reward_completion if all_sensors_collected else 0.0,
            "unvisited_penalty":       self.penalty_unvisited * unvisited_count
                                       if unvisited_count > 0 else 0.0,
        }

        breakdown["total"] = sum(breakdown.values())
        return breakdown
