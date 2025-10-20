"""
UAV (Unmanned Aerial Vehicle) Implementation

This module contains the UAV class for the Q-Learning path planning simulation.
The UAV is a mobile agent that navigates a 2D grid to collect data from IoT sensors
via LoRa communication.

Key Assumptions:
    - The UAV hovers at a constant height of 100m above ground level
    - Constant speed assumed for all movements (simplified physics model) (10 m/s)
    - 2D grid approximation (x, y coordinates, fixed altitude)
    - Discrete time steps for movement and actions
    - Battery depletion is deterministic (no stochastic failures - continuous time problem)
    - Constant Altitude of 100m above ground level
    - No environmental effects (wind, temperature)

Key Features:
    - Position tracking in 2D space
    - Battery management with configurable drain rates
    - Movement in four cardinal directions (UP, DOWN, LEFT, RIGHT)
    - Hovering capability (for data collection at fixed position)
    - Boundary collision detection and prevention
    - Episode reset functionality for training
    - UAV cannot collect data from IoT sensors while in motion (must hover)

Battery Consumption Model:
    The UAV consumes battery energy when:
    - Moving between positions (higher drain rate)
    - Hovering at a location for data collection (lower drain rate)
    - Attempting invalid moves / boundary collisions (partial drain)

State Variables:
    - Position: (x, y) coordinates in grid space
    - Battery: Remaining energy level (0 to max_battery)
    - Velocity: Implicitly constant (movement speed parameter)

Constraints:
    - Battery level must remain > 0 (mission failure if depleted)
    - Position must stay within grid boundaries
    - Cannot move and collect data simultaneously

Example UAV:
    - TB60 Intelligent
    - Capacity: 5935mAh, 52.8V, 274W
    - Typical Range: 5 m/s - 15 m/s
    - Max speed 10m/s
    - Flight power 600 W - 800 W (depending on speed)
    - Hovering at a location for data collection (lower drain rate)
    - Hover power 400 W
    - Weight - 6.3 kg


Author: Atilade Oke
Date: January 2025
Project: Reinforcement Learning for Dynamic UAV Energy-Efficient Path Planning
         in IoT Sensor Networks
"""
from typing import Tuple,List
import numpy as np


class UAV:
    """
    Represents a UAV with position tracking and battery management.

    The UAV navigates a 2D grid world to collect data from IoT sensors.
    It has limited battery capacity and consumes energy during movement
    and hovering operations.

    Attributes:
        position (np.ndarray): Current (x, y) position in grid
        battery (float): Current battery level in Wh (0 to max_battery)
        max_battery (float): Maximum battery capacity in Wh
        speed (float): Movement speed in grid units per second
        power_move (float): Power consumption during movement (Watts)
        power_hover (float): Power consumption while hovering (Watts)
        power_collision (float): Power consumption on boundary collision (Watts)
        start_position (np.ndarray): Initial position for episode reset

    Example:
        >>> uav = UAV(start_position=(5.0, 5.0), max_battery=274.0)
        >>> success = uav.move('UP', grid_size=(10, 10))
        >>> print(f"Position: {uav.position}, Battery: {uav.battery:.2f} Wh")
        Position: [5. 6.], Battery: 273.83 Wh
    """

    def __init__(self,
                 start_position: Tuple[float, float] = (5.0, 5.0),
                 max_battery: float = 274.0,  # Wh (TB60 capacity)
                 speed: float = 10.0,  # m/s (cruising speed)
                 power_move: float = 600.0,  # W (cruise power)
                 power_hover: float = 400.0,  # W (hover power)
                 altitude: float = 100.0):  # m (constant altitude)
        """
        Initialize UAV with position and battery parameters.

        Args:
            start_position: Initial (x, y) position in grid
            max_battery: Maximum battery capacity in Wh (default: 274 Wh for TB60)
            speed: Movement speed in m/s (default: 10 m/s cruising)
            power_move: Power consumption during movement in Watts (default: 600W)
            power_hover: Power consumption while hovering in Watts (default: 400W)
            altitude: Operating altitude in meters (default: 100m, constant)

        Example:
            >>> uav = UAV(start_position=(0.0, 0.0), max_battery=274.0)
            >>> uav.battery
            274.0
        """
        self.start_position = np.array(start_position, dtype=np.float32)
        self.position = self.start_position.copy()

        self.max_battery = max_battery
        self.battery = max_battery

        self.speed = speed
        self.altitude = altitude

        # Power consumption (Watts)
        self.power_move = power_move
        self.power_hover = power_hover
        self.power_collision = power_move * 0.5  # Half power on failed move

    def move(self, direction: str, grid_size: Tuple[int, int], time_step: float = 1.0) -> bool:
        """
        Move UAV in specified direction.

        Attempts to move the UAV one grid unit in the given direction.
        Movement fails if it would go outside grid boundaries.
        Battery is consumed for both successful and failed moves.

        Args:
            direction: One of 'UP', 'DOWN', 'LEFT', 'RIGHT'
            grid_size: (width, height) of the grid for boundary checking
            time_step: Duration of movement in seconds (default: 1.0)

        Returns:
            True if move was successful, False if blocked by boundary

        Raises:
            ValueError: If direction is not one of the valid options

        Example:
            >>> uav = UAV(start_position=(5.0, 5.0))
            >>> uav.move('UP', grid_size=(10, 10))
            True
            >>> uav.position
            array([5., 6.], dtype=float32)
        """
        # Calculate new position based on direction
        new_position = self.position.copy()

        if direction == 'UP':
            new_position[1] += 1.0
        elif direction == 'DOWN':
            new_position[1] -= 1.0
        elif direction == 'LEFT':
            new_position[0] -= 1.0
        elif direction == 'RIGHT':
            new_position[0] += 1.0
        else:
            raise ValueError(f"Invalid direction: {direction}. Must be UP, DOWN, LEFT, or RIGHT")

        # Check boundaries
        if (0 <= new_position[0] < grid_size[0] and
                0 <= new_position[1] < grid_size[1]):
            # Valid move - update position and consume full movement power
            self.position = new_position
            energy_consumed = (self.power_move * time_step) / 3600  # Convert W*s to Wh
            self.battery -= energy_consumed
            return True
        else:
            # Boundary collision - don't move but consume partial power
            energy_consumed = (self.power_collision * time_step) / 3600
            self.battery -= energy_consumed
            return False

    def hover(self, duration: float = 1.0) -> None:
        """
        Hover in place for data collection.

        UAV remains stationary while hovering, typically used for
        collecting data from nearby sensors. Consumes hover power.

        Args:
            duration: Hovering duration in seconds (default: 1.0)

        Example:
            >>> uav = UAV(max_battery=274.0)
            >>> battery_before = uav.battery
            >>> uav.hover(duration=5.0)
            >>> battery_used = battery_before - uav.battery
            >>> print(f"Energy used: {battery_used:.3f} Wh")
            Energy used: 0.556 Wh
        """
        energy_consumed = (self.power_hover * duration) / 3600  # Convert W*s to Wh
        self.battery -= energy_consumed

    def is_alive(self) -> bool:
        """
        Check if UAV has sufficient battery to continue operation.

        Returns:
            True if battery > 0, False if depleted

        Example:
            >>> uav = UAV(max_battery=1.0)
            >>> uav.battery = 0.5
            >>> uav.is_alive()
            True
            >>> uav.battery = 0.0
            >>> uav.is_alive()
            False
        """
        return self.battery > 0

    def get_battery_percentage(self) -> float:
        """
        Get current battery level as percentage.

        Returns:
            Battery level as percentage (0-100)

        Example:
            >>> uav = UAV(max_battery=100.0)
            >>> uav.battery = 50.0
            >>> uav.get_battery_percentage()
            50.0
        """
        return (self.battery / self.max_battery) * 100

    def reset(self) -> None:
        """
        Reset UAV to initial state for new episode.

        Resets position to start position and battery to maximum capacity.
        Used at the beginning of each training episode.

        Example:
            >>> uav = UAV(start_position=(5.0, 5.0), max_battery=100.0)
            >>> uav.move('UP', grid_size=(10, 10))
            >>> uav.hover(5.0)
            >>> uav.reset()
            >>> print(f"Position: {uav.position}, Battery: {uav.battery}")
            Position: [5. 5.], Battery: 100.0
        """
        self.position = self.start_position.copy()
        self.battery = self.max_battery


    @property
    def battery_drain_hover(self) -> float:
        """Calculate hover battery drain per second in Wh."""
        return self.power_hover / 3600  # Convert W to Wh/s

    def __repr__(self) -> str:
        """String representation of UAV state."""
        return (f"UAV(position={tuple(self.position)}, "
                f"battery={self.battery:.2f}Wh/{self.max_battery}Wh, "
                f"charge={self.get_battery_percentage():.1f}%)")