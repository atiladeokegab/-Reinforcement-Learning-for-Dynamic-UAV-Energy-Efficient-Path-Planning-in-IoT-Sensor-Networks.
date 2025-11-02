"""
Reward Function for UAV Data Collection

Calculates rewards for different UAV actions including movement,
data collection, and mission completion with multi-sensor collection bonuses.

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
        +5.0 per additional sensor in multi-sensor collection
        +50.0 for completing mission (all sensors empty)
        -2.0 for attempting to collect from empty sensor
        -5.0 for boundary collision
        -1.0 per SF collision detected
        -0.1 per Wh of battery used
        -0.05 per step (encourages efficiency)
    """

    def __init__(self,
                 reward_per_byte: float = 0.1,
                 reward_new_sensor: float = 10.0,
                 reward_multi_sensor: float = 5.0,  # NEW
                 reward_completion: float = 50.0,
                 penalty_revisit: float = -2.0,
                 penalty_boundary: float = -5.0,
                 penalty_collision: float = -1.0,  # NEW
                 penalty_battery: float = -0.1,
                 penalty_step: float = -0.05,
                 penalty_data_loss: float = -0.05):  # NEW
        """
        Initialize reward function with configurable parameters.

        Args:
            reward_per_byte: Reward per byte of data collected
            reward_new_sensor: Bonus for first collection from a sensor
            reward_multi_sensor: Bonus per additional sensor in simultaneous collection
            reward_completion: Bonus for completing mission
            penalty_revisit: Penalty for collecting from empty sensor
            penalty_boundary: Penalty for hitting boundary
            penalty_collision: Penalty per SF collision (encourages better positioning)
            penalty_battery: Penalty per Wh of battery used
            penalty_step: Penalty per step taken
            penalty_data_loss: Penalty per byte of data lost due to buffer overflow
        """
        self.reward_per_byte = reward_per_byte
        self.reward_new_sensor = reward_new_sensor
        self.reward_multi_sensor = reward_multi_sensor
        self.reward_completion = reward_completion
        self.penalty_revisit = penalty_revisit
        self.penalty_boundary = penalty_boundary
        self.penalty_collision = penalty_collision
        self.penalty_battery = penalty_battery
        self.penalty_step = penalty_step
        self.penalty_data_loss = penalty_data_loss

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
                                    battery_used: float,
                                    num_sensors_collected: int = 1,  # NEW
                                    collision_count: int = 0,  # NEW
                                    data_loss: float = 0.0) -> float:  # NEW
        """
        Calculate reward for data collection action.

        Args:
            bytes_collected: Amount of data collected (bytes)
            was_new_sensor: True if this was first time collecting from any sensor
            was_empty: True if attempted to collect from empty sensor
            all_sensors_collected: True if all sensors now have empty buffers
            battery_used: Amount of battery consumed (Wh)
            num_sensors_collected: Number of sensors collected from simultaneously
            collision_count: Number of SF collisions detected
            data_loss: Total bytes lost due to buffer overflow

        Returns:
            Total reward for the collection action
        """
        reward = self.penalty_step

        # Reward for collecting data
        if bytes_collected > 0:
            # Base data collection reward
            reward += self.reward_per_byte * bytes_collected

            # Bonus for new sensor discovery
            if was_new_sensor:
                reward += self.reward_new_sensor

            # NEW: Multi-sensor collection bonus
            # Encourages agent to position for collecting from multiple sensors
            if num_sensors_collected > 1:
                multi_sensor_bonus = self.reward_multi_sensor * (num_sensors_collected - 1)
                reward += multi_sensor_bonus

        # Penalty for attempting empty collection
        if was_empty and bytes_collected == 0:
            reward += self.penalty_revisit

        # Battery penalty
        reward += self.penalty_battery * battery_used

        # NEW: Collision penalty
        # Encourages agent to position to minimize SF collisions
        if collision_count > 0:
            reward += self.penalty_collision * collision_count

        # NEW: Data loss penalty
        # Penalizes letting sensor buffers overflow
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
                            data_loss: float = 0.0) -> dict:
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
            'empty_penalty': self.penalty_revisit if was_empty and bytes_collected == 0 else 0.0,
            'battery_penalty': self.penalty_battery * battery_used,
            'collision_penalty': self.penalty_collision * collision_count if collision_count > 0 else 0.0,
            'data_loss_penalty': self.penalty_data_loss * data_loss if data_loss > 0 else 0.0,
            'completion_bonus': self.reward_completion if all_sensors_collected else 0.0,
        }

        breakdown['total'] = sum(breakdown.values())

        return breakdown


# Example usage and testing
if __name__ == "__main__":
    print("=" * 70)
    print("Testing Reward Function")
    print("=" * 70)
    print()

    reward_fn = RewardFunction()

    # Test 1: Single sensor collection
    print("Test 1: Single sensor collection (100 bytes, new sensor)")
    reward = reward_fn.calculate_collection_reward(
        bytes_collected=100.0,
        was_new_sensor=True,
        was_empty=False,
        all_sensors_collected=False,
        battery_used=0.5,
        num_sensors_collected=1,
        collision_count=0
    )
    breakdown = reward_fn.get_reward_breakdown(
        bytes_collected=100.0,
        was_new_sensor=True,
        battery_used=0.5,
        num_sensors_collected=1
    )
    print(f"Total Reward: {reward:.2f}")
    print("Breakdown:")
    for key, value in breakdown.items():
        print(f"  {key}: {value:.2f}")
    print()

    # Test 2: Multi-sensor collection (3 sensors)
    print("Test 2: Multi-sensor collection (300 bytes from 3 sensors)")
    reward = reward_fn.calculate_collection_reward(
        bytes_collected=300.0,
        was_new_sensor=True,
        was_empty=False,
        all_sensors_collected=False,
        battery_used=1.0,
        num_sensors_collected=3,  # Collecting from 3 sensors simultaneously!
        collision_count=0
    )
    breakdown = reward_fn.get_reward_breakdown(
        bytes_collected=300.0,
        was_new_sensor=True,
        battery_used=1.0,
        num_sensors_collected=3
    )
    print(f"Total Reward: {reward:.2f}")
    print("Breakdown:")
    for key, value in breakdown.items():
        print(f"  {key}: {value:.2f}")
    print()

    # Test 3: Collection with collisions
    print("Test 3: Collection with 2 SF collisions")
    reward = reward_fn.calculate_collection_reward(
        bytes_collected=200.0,
        was_new_sensor=False,
        was_empty=False,
        all_sensors_collected=False,
        battery_used=0.8,
        num_sensors_collected=2,
        collision_count=2  # 2 collisions detected
    )
    breakdown = reward_fn.get_reward_breakdown(
        bytes_collected=200.0,
        battery_used=0.8,
        num_sensors_collected=2,
        collision_count=2
    )
    print(f"Total Reward: {reward:.2f}")
    print("Breakdown:")
    for key, value in breakdown.items():
        print(f"  {key}: {value:.2f}")
    print()

    # Test 4: Data loss scenario
    print("Test 4: Collection with 50 bytes data loss")
    reward = reward_fn.calculate_collection_reward(
        bytes_collected=150.0,
        was_new_sensor=False,
        was_empty=False,
        all_sensors_collected=False,
        battery_used=0.6,
        num_sensors_collected=1,
        collision_count=0,
        data_loss=50.0  # 50 bytes lost to overflow
    )
    breakdown = reward_fn.get_reward_breakdown(
        bytes_collected=150.0,
        battery_used=0.6,
        num_sensors_collected=1,
        data_loss=50.0
    )
    print(f"Total Reward: {reward:.2f}")
    print("Breakdown:")
    for key, value in breakdown.items():
        print(f"  {key}: {value:.2f}")
    print()

    # Test 5: Mission completion
    print("Test 5: Mission completion")
    reward = reward_fn.calculate_collection_reward(
        bytes_collected=100.0,
        was_new_sensor=False,
        was_empty=False,
        all_sensors_collected=True,  # Mission complete!
        battery_used=0.5,
        num_sensors_collected=1,
        collision_count=0
    )
    breakdown = reward_fn.get_reward_breakdown(
        bytes_collected=100.0,
        all_sensors_collected=True,
        battery_used=0.5,
        num_sensors_collected=1
    )
    print(f"Total Reward: {reward:.2f}")
    print("Breakdown:")
    for key, value in breakdown.items():
        print(f"  {key}: {value:.2f}")
    print()

    # Test 6: Movement to boundary
    print("Test 6: Boundary collision during movement")
    reward = reward_fn.calculate_movement_reward(
        move_success=False,
        battery_used=0.2
    )
    print(f"Total Reward: {reward:.2f}")
    print()

    print("=" * 70)
    print("Key Reward Insights:")
    print("=" * 70)
    print("✓ Multi-sensor bonus encourages strategic positioning")
    print("✓ Collision penalty discourages poor positioning")
    print("✓ Data loss penalty encourages timely collection")
    print("✓ Battery penalty encourages energy efficiency")
    print("=" * 70)