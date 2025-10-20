"""
Reward Function for UAV Data Collection

Calculates rewards for different UAV actions including movement,
data collection, and mission completion.

Author: ATILADE GABRIEL OKE
Date: October 2025
Project: Reinforcement Learning for Dynamic UAV Energy-Efficient Path Planning
         in IoT Sensor Networks
"""


class RewardFunction:
    """
    Reward function for UAV data collection task.

    Rewards:
        +0.1 per byte collected
        +10.0 for collecting from new sensor
        +50.0 for completing mission (all sensors empty)
        -2.0 for attempting to collect from empty sensor
        -5.0 for boundary collision
        -0.1 per Wh of battery used
        -0.05 per step (encourages efficiency)
    """

    def __init__(self,
                 reward_per_byte: float = 0.1,
                 reward_new_sensor: float = 10.0,
                 reward_completion: float = 50.0,
                 penalty_revisit: float = -2.0,
                 penalty_boundary: float = -5.0,
                 penalty_battery: float = -0.1,
                 penalty_step: float = -0.05):
        """
        Initialize reward function with configurable parameters.

        Args:
            reward_per_byte: Reward per byte of data collected
            reward_new_sensor: Bonus for first collection from a sensor
            reward_completion: Bonus for completing mission
            penalty_revisit: Penalty for collecting from empty sensor
            penalty_boundary: Penalty for hitting boundary
            penalty_battery: Penalty per Wh of battery used
            penalty_step: Penalty per step taken
        """
        self.reward_per_byte = reward_per_byte
        self.reward_new_sensor = reward_new_sensor
        self.reward_completion = reward_completion
        self.penalty_revisit = penalty_revisit
        self.penalty_boundary = penalty_boundary
        self.penalty_battery = penalty_battery
        self.penalty_step = penalty_step

    def calculate_movement_reward(self,
                                  move_success: bool,
                                  battery_used: float) -> float:
        """
        Calculate reward for movement action.

        Args:
            move_success: Whether move was successful (not boundary collision)
            battery_used: Amount of battery consumed (Wh)

        Returns:
            Total reward for the movement
        """
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
                                    battery_used: float) -> float:
        """
        Calculate reward for data collection action.

        Args:
            bytes_collected: Amount of data collected (bytes)
            was_new_sensor: True if this was first time collecting from this sensor
            was_empty: True if attempted to collect from empty sensor
            all_sensors_collected: True if all sensors now have empty buffers
            battery_used: Amount of battery consumed (Wh)

        Returns:
            Total reward for the collection action
        """
        reward = self.penalty_step

        # Reward for collecting data
        if bytes_collected > 0:
            reward += self.reward_per_byte * bytes_collected

            # Bonus for new sensor
            if was_new_sensor:
                reward += self.reward_new_sensor

        # Penalty for attempting empty collection
        if was_empty:
            reward += self.penalty_revisit

        # Battery penalty
        reward += self.penalty_battery * battery_used

        # Mission completion bonus
        if all_sensors_collected:
            reward += self.reward_completion

        return reward


# Testing
if __name__ == "__main__":
    print("=" * 70)
    print("Testing Reward Function")
    print("=" * 70)
    print()

    rf = RewardFunction()

    # Test 1: Successful movement
    print("Test 1: Successful Movement")
    reward = rf.calculate_movement_reward(move_success=True, battery_used=0.167)
    print(f"  Reward: {reward:.3f}")
    print()

    # Test 2: Boundary collision
    print("Test 2: Boundary Collision")
    reward = rf.calculate_movement_reward(move_success=False, battery_used=0.083)
    print(f"  Reward: {reward:.3f}")
    print()

    # Test 3: Collect from new sensor
    print("Test 3: First Collection from Sensor")
    reward = rf.calculate_collection_reward(
        bytes_collected=100.0,
        was_new_sensor=True,
        was_empty=False,
        all_sensors_collected=False,
        battery_used=0.556
    )
    print(f"  Reward: {reward:.3f}")
    print()

    # Test 4: Empty sensor revisit
    print("Test 4: Attempt to Collect from Empty Sensor")
    reward = rf.calculate_collection_reward(
        bytes_collected=0.0,
        was_new_sensor=False,
        was_empty=True,
        all_sensors_collected=False,
        battery_used=0.111
    )
    print(f"  Reward: {reward:.3f}")
    print()

    # Test 5: Mission completion
    print("Test 5: Mission Completion")
    reward = rf.calculate_collection_reward(
        bytes_collected=50.0,
        was_new_sensor=False,
        was_empty=False,
        all_sensors_collected=True,
        battery_used=0.556
    )
    print(f"  Reward: {reward:.3f}")
    print()

    print("=" * 70)
    print("✓ Reward function test complete!")
    print("=" * 70)