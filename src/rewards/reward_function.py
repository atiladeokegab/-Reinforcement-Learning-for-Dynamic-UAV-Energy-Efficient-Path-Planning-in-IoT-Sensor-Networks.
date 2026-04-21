import numpy as np
from typing import List, Dict

class RewardFunction:
    """ Refactored Multi-objective reward function for UAV IoT data collection. """

    def __init__(
            self,
            # Rewards
            reward_per_byte: float = 100.0,
            reward_new_sensor: float = 5000.0,
            reward_completion: float = 100.0,
            reward_urgency_reduction: float = 1000.0,
            reward_movement: float = 10.0,
            # Penalties
            penalty_revisit: float = -2.0,
            penalty_boundary: float = -50.0,
            penalty_collision: float = -10.0,
            penalty_battery: float = 0.0,
            penalty_hover: float = -5.0,
            penalty_step: float = -0.5,
            penalty_data_loss: float = -1.0,
            penalty_starvation: float = -1000.0,
            penalty_unvisited: float = -5000.0,
            penalty_starved: float = -1000.0,
            starvation_cr_threshold: float = 0.20,
    ):
        self.reward_per_byte = reward_per_byte
        self.reward_new_sensor = reward_new_sensor
        self.reward_completion = reward_completion
        self.reward_urgency_reduction = reward_urgency_reduction
        self.reward_movement = reward_movement

        self.penalty_revisit = penalty_revisit
        self.penalty_boundary = penalty_boundary
        self.penalty_collision = penalty_collision
        self.penalty_battery = penalty_battery
        self.penalty_hover = penalty_hover
        self.penalty_step = penalty_step
        self.penalty_data_loss = penalty_data_loss
        self.penalty_starvation = penalty_starvation
        self.penalty_unvisited = penalty_unvisited
        self.penalty_starved = penalty_starved
        self.starvation_cr_threshold = starvation_cr_threshold

    def calculate_starvation_penalty(self, sensor_buffers: List[float]) -> float:
        """Variance-based fairness penalty to keep buffers balanced across the fleet."""
        if not sensor_buffers or len(sensor_buffers) <= 1:
            return 0.0

        max_buffer = max(sensor_buffers)
        if max_buffer == 0:
            return 0.0

        normalized = np.array([b / max_buffer for b in sensor_buffers])
        variance = float(np.var(normalized))
        return self.penalty_starvation * variance

    def calculate_terminal_starvation_penalty(self, sensors) -> float:
        """Penalty for sensors visited but left with low collection ratios."""
        penalty = 0.0
        for s in sensors:
            if s.total_data_generated > 0:
                cr = s.total_data_transmitted / s.total_data_generated
                if cr < self.starvation_cr_threshold:
                    penalty += self.penalty_starved
        return penalty

    def calculate_movement_reward(
            self, move_success: bool, battery_used: float
    ) -> float:
        """Reward logic for the UAV moving to a new coordinate."""
        reward = self.penalty_step
        if move_success:
            reward += self.reward_movement
        else:
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
            collision_count: int = 0,
            data_loss: float = 0.0,
            urgency_reduced: float = 0.0,
            sensor_buffers: List[float] = None,
            unvisited_count: int = 0,
            sensor_urgency: float = 0.0,
    ) -> float:
        """Reward logic for hovering and collecting data from a sensor."""
        # Step cost + hover surcharge (rotary wing aerodynamics penalty)
        reward = self.penalty_step + self.penalty_hover

        # Data collection reward weighted by urgency/buffer state
        if bytes_collected > 0:
            reward += self.reward_per_byte * bytes_collected * sensor_urgency
            if was_new_sensor:
                reward += self.reward_new_sensor

        if urgency_reduced > 0:
            reward += self.reward_urgency_reduction * urgency_reduced

        if was_empty and bytes_collected == 0:
            reward += self.penalty_revisit

        reward += self.penalty_battery * battery_used

        if collision_count > 0:
            reward += self.penalty_collision * collision_count

        if data_loss > 0:
            reward += self.penalty_data_loss * data_loss

        if sensor_buffers is not None:
            reward += self.calculate_starvation_penalty(sensor_buffers)

        if all_sensors_collected:
            reward += self.reward_completion

        if unvisited_count > 0:
            reward += self.penalty_unvisited * unvisited_count

        return reward

    def get_reward_breakdown(
            self,
            bytes_collected: float = 0.0,
            was_new_sensor: bool = False,
            was_empty: bool = False,
            all_sensors_collected: bool = False,
            battery_used: float = 0.0,
            collision_count: int = 0,
            data_loss: float = 0.0,
            urgency_reduced: float = 0.0,
            sensor_buffers: List[float] = None,
            unvisited_count: int = 0,
            sensor_urgency: float = 0.0,
            is_hover: bool = True,
    ) -> Dict:
        """Detailed breakdown for diagnostic logging."""
        starvation = (
            self.calculate_starvation_penalty(sensor_buffers)
            if sensor_buffers is not None else 0.0
        )

        breakdown = {
            "step_penalty": self.penalty_step,
            "hover_penalty": self.penalty_hover if is_hover else 0.0,
            "movement_bonus": 0.0 if is_hover else self.reward_movement,
            "data_reward": self.reward_per_byte * bytes_collected * sensor_urgency
            if bytes_collected > 0 else 0.0,
            "new_sensor_bonus": self.reward_new_sensor if was_new_sensor else 0.0,
            "urgency_reduction_bonus": self.reward_urgency_reduction * urgency_reduced
            if urgency_reduced > 0 else 0.0,
            "empty_penalty": self.penalty_revisit
            if was_empty and bytes_collected == 0 else 0.0,
            "battery_penalty": self.penalty_battery * battery_used,
            "collision_penalty": self.penalty_collision * collision_count
            if collision_count > 0 else 0.0,
            "data_loss_penalty": self.penalty_data_loss * data_loss
            if data_loss > 0 else 0.0,
            "starvation_penalty": starvation,
            "completion_bonus": self.reward_completion if all_sensors_collected else 0.0,
            "unvisited_penalty": self.penalty_unvisited * unvisited_count
            if unvisited_count > 0 else 0.0,
        }

        breakdown["total"] = sum(breakdown.values())
        return breakdown