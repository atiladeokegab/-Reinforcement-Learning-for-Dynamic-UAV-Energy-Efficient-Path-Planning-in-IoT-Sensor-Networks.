import numpy as np
from typing import List, Dict


class RewardFunction:
    """
    Fairness-constrained reward function for UAV IoT data collection.

    Key fixes vs previous version:
    1. reward_new_sensor: 50 → 5000
       Visiting an unseen sensor must compete with revisiting known sensors.
       A revisit step earns ~300,000 (100 * 3000 bytes). A 50-point new-sensor
       bonus was completely invisible to the optimiser.

    2. penalty_starvation: -50 → -500
       Was -500 originally and was incorrectly reduced. At -50 the maximum
       starvation penalty per step is -12.5 (variance ≤ 0.25), which is
       drowned out by a single good collection step worth thousands.
       Restoring to -500 gives up to -125/step — still not dominant, but
       meaningful over 1400 steps.

    3. penalty_unvisited: NEW — -2000 per unvisited sensor at episode end.
       The agent was earning ~7000 reward by revisiting known sensors for
       the last 900 steps rather than searching for the 5 it missed.
       A terminal penalty of -2000 per missed sensor (-10,000 total for 5
       missed) makes it worth spending battery to find them.

    4. reward_completion: kept at 100 — the new penalty_unvisited makes
       this effectively a per-sensor completion bonus already.
    """

    def __init__(
        self,
        # Rewards
        reward_per_byte: float          = 100.0,
        reward_new_sensor: float        = 5000.0,   # FIX 1: was 50
        reward_multi_sensor: float      = 200.0,
        reward_completion: float        = 100.0,
        reward_urgency_reduction: float = 1000.0,
        # Penalties
        penalty_revisit: float          = -2.0,
        penalty_boundary: float         = -50.0,
        penalty_collision: float        = -10.0,
        penalty_battery: float          = -0.5,
        penalty_step: float             = -0.5,
        penalty_data_loss: float        = -1.0,
        penalty_starvation: float       = -500.0,   # FIX 2: was -50
        penalty_unvisited: float        = -2000.0,  # FIX 3: NEW
    ):
        self.reward_per_byte          = reward_per_byte
        self.reward_new_sensor        = reward_new_sensor
        self.reward_multi_sensor      = reward_multi_sensor
        self.reward_completion        = reward_completion
        self.reward_urgency_reduction = reward_urgency_reduction

        self.penalty_revisit    = penalty_revisit
        self.penalty_boundary   = penalty_boundary
        self.penalty_collision  = penalty_collision
        self.penalty_battery    = penalty_battery
        self.penalty_step       = penalty_step
        self.penalty_data_loss  = penalty_data_loss
        self.penalty_starvation = penalty_starvation
        self.penalty_unvisited  = penalty_unvisited  # FIX 3: NEW

    # ------------------------------------------------------------------ #
    # Fairness helper                                                      #
    # ------------------------------------------------------------------ #

    def calculate_starvation_penalty(self, sensor_buffers: List[float]) -> float:
        """
        Variance-based fairness penalty.

        If all sensors have equal buffer levels → penalty ≈ 0.
        If buffers are unequal → penalty grows with variance, up to
        penalty_starvation * 0.25 per step (variance of a 0/1 binary split).

        Args:
            sensor_buffers: current data buffer level for each sensor

        Returns:
            Negative penalty proportional to buffer-level variance.
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
        """Reward for a movement action."""
        reward = self.penalty_step
        if not move_success:
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
        unvisited_count: int        = 0,    # FIX 4: NEW — sensors not yet visited
    ) -> float:
        """
        Fairness-constrained reward for a data collection action.

        unvisited_count should be passed by the environment as:
            self.num_sensors - len(self.sensors_visited)

        It is applied as a terminal penalty so the agent feels the cost
        of missing sensors only at episode end, not on every step.
        """
        reward = self.penalty_step

        # ── Data collection ────────────────────────────────────────────
        if bytes_collected > 0:
            reward += self.reward_per_byte * bytes_collected

            if was_new_sensor:
                reward += self.reward_new_sensor          # FIX 1

            if num_sensors_collected > 1:
                reward += self.reward_multi_sensor * (num_sensors_collected - 1)

        # ── Urgency reduction bonus ────────────────────────────────────
        if urgency_reduced > 0:
            reward += self.reward_urgency_reduction * urgency_reduced

        # ── Empty-visit penalty ────────────────────────────────────────
        if was_empty and bytes_collected == 0:
            reward += self.penalty_revisit

        # ── Battery & movement ─────────────────────────────────────────
        reward += self.penalty_battery * battery_used

        # ── Collisions ─────────────────────────────────────────────────
        if collision_count > 0:
            reward += self.penalty_collision * collision_count

        # ── Data loss ──────────────────────────────────────────────────
        if data_loss > 0:
            reward += self.penalty_data_loss * data_loss

        # ── Variance-based starvation penalty (FIX 2) ─────────────────
        if sensor_buffers is not None:
            reward += self.calculate_starvation_penalty(sensor_buffers)

        # ── Terminal penalties / bonuses ───────────────────────────────
        if all_sensors_collected:
            reward += self.reward_completion

        # FIX 3 & 4: penalty per sensor the agent never visited
        # Applied every step unvisited_count > 0 would be too noisy;
        # the env should only pass a non-zero value on the terminal step.
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
    ) -> Dict:
        """
        Return a detailed per-component breakdown of the reward.
        Useful for diagnosing which signal dominates during training.
        """
        starvation = (
            self.calculate_starvation_penalty(sensor_buffers)
            if sensor_buffers is not None else 0.0
        )

        breakdown = {
            "step_penalty":           self.penalty_step,
            "data_reward":            self.reward_per_byte * bytes_collected
                                      if bytes_collected > 0 else 0.0,
            "new_sensor_bonus":       self.reward_new_sensor
                                      if was_new_sensor else 0.0,
            "multi_sensor_bonus":     self.reward_multi_sensor * (num_sensors_collected - 1)
                                      if num_sensors_collected > 1 else 0.0,
            "urgency_reduction_bonus":self.reward_urgency_reduction * urgency_reduced
                                      if urgency_reduced > 0 else 0.0,
            "empty_penalty":          self.penalty_revisit
                                      if was_empty and bytes_collected == 0 else 0.0,
            "battery_penalty":        self.penalty_battery * battery_used,
            "collision_penalty":      self.penalty_collision * collision_count
                                      if collision_count > 0 else 0.0,
            "data_loss_penalty":      self.penalty_data_loss * data_loss
                                      if data_loss > 0 else 0.0,
            "starvation_penalty":     starvation,
            "completion_bonus":       self.reward_completion
                                      if all_sensors_collected else 0.0,
            "unvisited_penalty":      self.penalty_unvisited * unvisited_count
                                      if unvisited_count > 0 else 0.0,  # FIX 3
        }

        breakdown["total"] = sum(breakdown.values())
        return breakdown