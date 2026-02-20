import numpy as np
from typing import List, Dict


class RewardFunction:
    """
    Fixed reward function balancing throughput with fairness.

    Key Changes from Original:
    1. Step penalty reduced from -50 to -0.5 (prevents "stay still" trap)
    2. Data loss penalty reduced from -5000 to -1 (makes learning possible)
    3. Reward per byte increased from 55 to 100 (encourages collection)
    4. Battery penalty reduced from -5 to -0.5 (enables exploration)
    5. Added variance-based starvation penalty (implements fairness properly)
    """

    def __init__(
        self,
        # Rewards
        reward_per_byte: float = 100.0,  # INCREASED from 55
        reward_new_sensor: float = 50.0,
        reward_multi_sensor: float = 200.0,
        reward_completion: float = 100.0,
        reward_urgency_reduction: float = 1000.0,
        # Penalties
        penalty_revisit: float = -2.0,
        penalty_boundary: float = -50.0,
        penalty_collision: float = -10.0,
        penalty_battery: float = -0.5,  # REDUCED from -5
        penalty_step: float = -0.5,  # REDUCED from -50
        penalty_data_loss: float = -1.0,  # REDUCED from -5000
        # penalty_starvation: float = -500.0,
        penalty_starvation: float = -50.0,
    ):  # NEW: Variance-based
        """
        Initialize fairness-constrained reward function.

        Critical Parameters:
        - penalty_step: Should be small (-0.5 to -2.0) to avoid "stay still" trap
        - penalty_data_loss: Should be reduced (-1 to -10) to make learning possible
        - reward_per_byte: Should be high (100+) to make exploration rewarding
        - penalty_battery: Should be low (-0.5 to -1) to enable movement
        - penalty_starvation: Variance-based penalty for fairness
        """
        self.reward_per_byte = reward_per_byte
        self.reward_new_sensor = reward_new_sensor
        self.reward_multi_sensor = reward_multi_sensor
        self.reward_completion = reward_completion
        self.reward_urgency_reduction = reward_urgency_reduction

        self.penalty_revisit = penalty_revisit
        self.penalty_boundary = penalty_boundary
        self.penalty_collision = penalty_collision
        self.penalty_battery = penalty_battery
        self.penalty_step = penalty_step
        self.penalty_data_loss = penalty_data_loss
        self.penalty_starvation = penalty_starvation  # NEW

    def calculate_starvation_penalty(self, sensor_buffers: List[float]) -> float:
        """
        Calculate fairness penalty based on buffer variance.

        Core Idea:
        - If all sensors have equal buffers → penalty ≈ 0
        - If buffers are unequal → penalty increases with variance
        - This encourages fair distribution without destroying learning

        Args:
            sensor_buffers: List of current buffer levels for each sensor

        Returns:
            Penalty based on variance (0 if all equal, higher if unequal)
        """
        if not sensor_buffers or len(sensor_buffers) <= 1:
            return 0.0

        # Normalize to 0-1 range
        max_buffer = max(sensor_buffers)
        if max_buffer == 0:  # All buffers empty
            return 0.0

        normalized = np.array([b / max_buffer for b in sensor_buffers])

        # Variance: 0 if all equal, up to 0.25 for maximum spread
        variance = np.var(normalized)

        # Apply penalty proportional to variance
        penalty = self.penalty_starvation * variance

        return penalty

    def calculate_movement_reward(
        self, move_success: bool, battery_used: float
    ) -> float:
        """Calculate reward for movement action."""
        reward = self.penalty_step

        if not move_success:
            reward += self.penalty_boundary

        reward += self.penalty_battery * battery_used

        return reward

    def calculate_collection_reward(
        self,
        bytes_collected: float,
        was_new_sensor: bool,
        was_empty: bool,
        all_sensors_collected: bool,
        battery_used: float,
        num_sensors_collected: int = 1,
        collision_count: int = 0,
        data_loss: float = 0.0,
        urgency_reduced: float = 0.0,
        sensor_buffers: List[float] = None,
    ) -> float:
        """
        Calculate fairness-constrained reward for data collection.

        FIXED VERSION: Balances throughput with fairness learning.
        """
        reward = self.penalty_step

        # Base data collection reward (INCREASED to incentivize movement)
        if bytes_collected > 0:
            reward += self.reward_per_byte * bytes_collected

            if was_new_sensor:
                reward += self.reward_new_sensor

            if num_sensors_collected > 1:
                multi_sensor_bonus = self.reward_multi_sensor * (
                    num_sensors_collected - 1
                )
                reward += multi_sensor_bonus

        # Urgency reduction bonus
        if urgency_reduced > 0:
            reward += self.reward_urgency_reduction * urgency_reduced

        # Penalty for attempting empty collection
        if was_empty and bytes_collected == 0:
            reward += self.penalty_revisit

        # Battery penalty (REDUCED)
        reward += self.penalty_battery * battery_used

        # Collision penalty
        if collision_count > 0:
            reward += self.penalty_collision * collision_count

        # Data loss penalty (GREATLY REDUCED - makes learning possible)
        if data_loss > 0:
            reward += self.penalty_data_loss * data_loss

        # VARIANCE-BASED STARVATION PENALTY (replaces flat data loss)
        if sensor_buffers is not None:
            starvation_penalty = self.calculate_starvation_penalty(sensor_buffers)
            reward += starvation_penalty

        # Mission completion bonus
        if all_sensors_collected:
            reward += self.reward_completion

        return reward

    def get_reward_breakdown(
        self,
        bytes_collected: float = 0.0,
        was_new_sensor: bool = False,
        was_empty: bool = False,
        all_sensors_collected: bool = False,
        battery_used: float = 0.0,
        num_sensors_collected: int = 1,
        collision_count: int = 0,
        data_loss: float = 0.0,
        urgency_reduced: float = 0.0,
        sensor_buffers: List[float] = None,
    ) -> Dict:
        """Get detailed breakdown of reward components for analysis."""
        starvation_penalty = 0.0
        if sensor_buffers is not None:
            starvation_penalty = self.calculate_starvation_penalty(sensor_buffers)

        breakdown = {
            "step_penalty": self.penalty_step,
            "data_reward": (
                self.reward_per_byte * bytes_collected if bytes_collected > 0 else 0.0
            ),
            "new_sensor_bonus": self.reward_new_sensor if was_new_sensor else 0.0,
            "multi_sensor_bonus": (
                self.reward_multi_sensor * (num_sensors_collected - 1)
                if num_sensors_collected > 1
                else 0.0
            ),
            "urgency_reduction_bonus": (
                self.reward_urgency_reduction * urgency_reduced
                if urgency_reduced > 0
                else 0.0
            ),
            "empty_penalty": (
                self.penalty_revisit if was_empty and bytes_collected == 0 else 0.0
            ),
            "battery_penalty": self.penalty_battery * battery_used,
            "collision_penalty": (
                self.penalty_collision * collision_count if collision_count > 0 else 0.0
            ),
            "data_loss_penalty": (
                self.penalty_data_loss * data_loss if data_loss > 0 else 0.0
            ),
            "starvation_penalty": starvation_penalty,  # NEW
            "completion_bonus": (
                self.reward_completion if all_sensors_collected else 0.0
            ),
        }

        breakdown["total"] = sum(breakdown.values())

        return breakdown
