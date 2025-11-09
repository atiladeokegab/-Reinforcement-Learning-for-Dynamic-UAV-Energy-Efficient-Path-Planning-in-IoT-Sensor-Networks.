"""
Fairness-Constrained Reward Function for UAV Data Collection

Enforces network health through massive penalties for data loss.
Balances throughput optimization with fairness/safety constraints.

Author: ATILADE GABRIEL OKE
Date: November 2025
"""


class RewardFunction:
    """
    Reward function with fairness constraints.

    Key Features:
    - Massive penalty for data loss (-500.0+)
    - Bonus for reducing urgency
    - Maintains throughput incentives
    - Ensures no sensor is neglected
    """

    def __init__(self,
                 reward_per_byte: float = 0.1,
                 reward_new_sensor: float = 10.0,
                 reward_multi_sensor: float = 5.0,
                 reward_completion: float = 50.0,
                 reward_urgency_reduction: float = 20.0,  # NEW
                 penalty_revisit: float = -2.0,
                 penalty_boundary: float = -5.0,
                 penalty_collision: float = -1.0,
                 penalty_battery: float = -0.1,
                 penalty_step: float = -0.05,
                 penalty_data_loss: float = -500.0):
        """
        Initialize fairness-constrained reward function.

        Args:
            reward_per_byte: Reward per byte collected
            reward_new_sensor: Bonus for first collection from sensor
            reward_multi_sensor: Bonus per additional sensor in multi-collection
            reward_completion: Bonus for completing mission
            reward_urgency_reduction: Bonus for reducing high-urgency sensor buffer
            penalty_revisit: Penalty for collecting from empty sensor
            penalty_boundary: Penalty for hitting boundary
            penalty_collision: Penalty per SF collision
            penalty_battery: Penalty per Wh of battery used
            penalty_step: Penalty per step taken
            penalty_data_loss: MASSIVE penalty per byte lost (fairness constraint)
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
        self.penalty_data_loss = penalty_data_loss  # ✅ CRITICAL

    def calculate_movement_reward(self,
                                  move_success: bool,
                                  battery_used: float) -> float:
        """Calculate reward for movement action."""
        reward = self.penalty_step

        if not move_success:
            reward += self.penalty_boundary

        reward += self.penalty_battery * battery_used

        return reward

    def calculate_collection_reward(self,
                                    bytes_collected: float,
                                    was_new_sensor: bool,
                                    was_empty: bool,
                                    all_sensors_collected: bool,
                                    battery_used: float,
                                    num_sensors_collected: int = 1,
                                    collision_count: int = 0,
                                    data_loss: float = 0.0,
                                    urgency_reduced: float = 0.0) -> float:  # NEW
        """
        Calculate fairness-constrained reward for data collection.

        Args:
            bytes_collected: Amount of data collected (bytes)
            was_new_sensor: True if first time collecting from any sensor
            was_empty: True if attempted to collect from empty sensor
            all_sensors_collected: True if all sensors now empty
            battery_used: Amount of battery consumed (Wh)
            num_sensors_collected: Number of sensors collected from simultaneously
            collision_count: Number of SF collisions detected
            data_loss: Total bytes lost due to buffer overflow
            urgency_reduced: Sum of urgency reduction across all sensors

        Returns:
            Total reward for the collection action
        """
        reward = self.penalty_step

        # Base data collection reward
        if bytes_collected > 0:
            reward += self.reward_per_byte * bytes_collected

            if was_new_sensor:
                reward += self.reward_new_sensor

            # Multi-sensor bonus
            if num_sensors_collected > 1:
                multi_sensor_bonus = self.reward_multi_sensor * (num_sensors_collected - 1)
                reward += multi_sensor_bonus

        # ✅ NEW: Urgency reduction bonus
        # Reward for collecting from high-urgency sensors
        if urgency_reduced > 0:
            reward += self.reward_urgency_reduction * urgency_reduced

        # Penalty for attempting empty collection
        if was_empty and bytes_collected == 0:
            reward += self.penalty_revisit

        # Battery penalty
        reward += self.penalty_battery * battery_used

        # Collision penalty
        if collision_count > 0:
            reward += self.penalty_collision * collision_count

        # ✅ MASSIVE DATA LOSS PENALTY (Fairness Constraint)
        # This is the key: make data loss so expensive that the agent
        # MUST prioritize high-urgency sensors regardless of SF
        if data_loss > 0:
            reward += self.penalty_data_loss * data_loss

        # Mission completion bonus
        if all_sensors_collected:
            reward += self.reward_completion

        return reward

    def get_reward_breakdown(self,
                            bytes_collected: float = 0.0,
                            was_new_sensor: bool = False,
                            was_empty: bool = False,
                            all_sensors_collected: bool = False,
                            battery_used: float = 0.0,
                            num_sensors_collected: int = 1,
                            collision_count: int = 0,
                            data_loss: float = 0.0,
                            urgency_reduced: float = 0.0) -> dict:
        """
        Get detailed breakdown of reward components for analysis.

        Returns:
            Dictionary with individual reward components
        """
        breakdown = {
            'step_penalty': self.penalty_step,
            'data_reward': self.reward_per_byte * bytes_collected if bytes_collected > 0 else 0.0,
            'new_sensor_bonus': self.reward_new_sensor if was_new_sensor else 0.0,
            'multi_sensor_bonus': self.reward_multi_sensor * (num_sensors_collected - 1) if num_sensors_collected > 1 else 0.0,
            'urgency_reduction_bonus': self.reward_urgency_reduction * urgency_reduced if urgency_reduced > 0 else 0.0,
            'empty_penalty': self.penalty_revisit if was_empty and bytes_collected == 0 else 0.0,
            'battery_penalty': self.penalty_battery * battery_used,
            'collision_penalty': self.penalty_collision * collision_count if collision_count > 0 else 0.0,
            'data_loss_penalty': self.penalty_data_loss * data_loss if data_loss > 0 else 0.0,  # ✅ CRITICAL
            'completion_bonus': self.reward_completion if all_sensors_collected else 0.0,
        }

        breakdown['total'] = sum(breakdown.values())

        return breakdown