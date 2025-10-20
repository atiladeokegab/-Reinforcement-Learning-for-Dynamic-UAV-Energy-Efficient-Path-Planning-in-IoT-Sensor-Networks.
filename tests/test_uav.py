"""
Unit Tests for UAV Implementation

Tests all UAV functionality including:
- Initialization and configuration
- Movement in all directions
- Boundary collision detection
- Battery consumption (move, hover, collision)
- Episode reset
- Status checking

Author: Atilade Oke
Date: October 2025
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from src.environment.uav import UAV


class TestUAVInitialization:
    """Test UAV initialization and parameter validation."""

    def test_default_initialization(self):
        """Test UAV initializes with default parameters."""
        uav = UAV()

        assert np.array_equal(uav.position, np.array([5.0, 5.0], dtype=np.float32))
        assert uav.battery == 274.0
        assert uav.max_battery == 274.0
        assert uav.speed == 10.0
        assert uav.power_move == 600.0
        assert uav.power_hover == 400.0

    def test_custom_initialization(self):
        """Test UAV initializes with custom parameters."""
        uav = UAV(
            start_position=(10.0, 20.0),
            max_battery=100.0,
            speed=15.0,
            power_move=700.0,
            power_hover=500.0
        )

        assert np.array_equal(uav.position, np.array([10.0, 20.0], dtype=np.float32))
        assert uav.battery == 100.0
        assert uav.max_battery == 100.0
        assert uav.speed == 15.0
        assert uav.power_move == 700.0
        assert uav.power_hover == 500.0

    def test_position_is_numpy_array(self):
        """Test position is stored as numpy array."""
        uav = UAV(start_position=(5, 5))

        assert isinstance(uav.position, np.ndarray)
        assert uav.position.dtype == np.float32

    def test_start_position_preserved(self):
        """Test start position is stored separately from current position."""
        uav = UAV(start_position=(3.0, 7.0))

        assert np.array_equal(uav.start_position, np.array([3.0, 7.0]))
        assert not np.shares_memory(uav.position, uav.start_position)


class TestUAVMovement:
    """Test UAV movement in all directions."""

    def test_move_up(self):
        """Test UAV moves up correctly."""
        uav = UAV(start_position=(5.0, 5.0))

        success = uav.move('UP', grid_size=(10, 10))

        assert success == True
        assert uav.position[0] == 5.0  # x unchanged
        assert uav.position[1] == 6.0  # y increased

    def test_move_down(self):
        """Test UAV moves down correctly."""
        uav = UAV(start_position=(5.0, 5.0))

        success = uav.move('DOWN', grid_size=(10, 10))

        assert success == True
        assert uav.position[0] == 5.0  # x unchanged
        assert uav.position[1] == 4.0  # y decreased

    def test_move_left(self):
        """Test UAV moves left correctly."""
        uav = UAV(start_position=(5.0, 5.0))

        success = uav.move('LEFT', grid_size=(10, 10))

        assert success == True
        assert uav.position[0] == 4.0  # x decreased
        assert uav.position[1] == 5.0  # y unchanged

    def test_move_right(self):
        """Test UAV moves right correctly."""
        uav = UAV(start_position=(5.0, 5.0))

        success = uav.move('RIGHT', grid_size=(10, 10))

        assert success == True
        assert uav.position[0] == 6.0  # x increased
        assert uav.position[1] == 5.0  # y unchanged

    def test_sequential_movements(self):
        """Test multiple sequential movements."""
        uav = UAV(start_position=(5.0, 5.0))
        grid_size = (10, 10)

        uav.move('UP', grid_size)
        uav.move('RIGHT', grid_size)
        uav.move('RIGHT', grid_size)
        uav.move('DOWN', grid_size)

        assert uav.position[0] == 7.0
        assert uav.position[1] == 5.0

    def test_invalid_direction_raises_error(self):
        """Test invalid direction raises ValueError."""
        uav = UAV()

        with pytest.raises(ValueError, match="Invalid direction"):
            uav.move('DIAGONAL', grid_size=(10, 10))


class TestBoundaryDetection:
    """Test UAV boundary collision detection."""

    def test_upper_boundary_collision(self):
        """Test UAV cannot move beyond upper boundary."""
        uav = UAV(start_position=(5.0, 9.0))  # Near top

        success = uav.move('UP', grid_size=(10, 10))

        assert success == False
        assert uav.position[1] == 9.0  # Position unchanged

    def test_lower_boundary_collision(self):
        """Test UAV cannot move beyond lower boundary."""
        uav = UAV(start_position=(5.0, 0.0))  # At bottom

        success = uav.move('DOWN', grid_size=(10, 10))

        assert success == False
        assert uav.position[1] == 0.0  # Position unchanged

    def test_left_boundary_collision(self):
        """Test UAV cannot move beyond left boundary."""
        uav = UAV(start_position=(0.0, 5.0))  # At left edge

        success = uav.move('LEFT', grid_size=(10, 10))

        assert success == False
        assert uav.position[0] == 0.0  # Position unchanged

    def test_right_boundary_collision(self):
        """Test UAV cannot move beyond right boundary."""
        uav = UAV(start_position=(9.0, 5.0))  # Near right edge

        success = uav.move('RIGHT', grid_size=(10, 10))

        assert success == False
        assert uav.position[0] == 9.0  # Position unchanged

    def test_corner_boundary(self):
        """Test UAV behavior at corners."""
        uav = UAV(start_position=(0.0, 0.0))  # Bottom-left corner

        success_left = uav.move('LEFT', grid_size=(10, 10))
        success_down = uav.move('DOWN', grid_size=(10, 10))

        assert success_left == False
        assert success_down == False
        assert np.array_equal(uav.position, np.array([0.0, 0.0]))


class TestBatteryConsumption:
    """Test battery consumption for different actions."""

    def test_battery_drains_on_move(self):
        """Test battery decreases after movement."""
        uav = UAV(max_battery=274.0)
        initial_battery = uav.battery

        uav.move('UP', grid_size=(10, 10), time_step=1.0)

        assert uav.battery < initial_battery

    def test_move_energy_calculation(self):
        """Test correct energy consumption for movement."""
        uav = UAV(max_battery=274.0, power_move=600.0)
        initial_battery = uav.battery

        uav.move('UP', grid_size=(10, 10), time_step=1.0)

        # Energy = (600W * 1s) / 3600 = 0.1667 Wh
        expected_energy = (600.0 * 1.0) / 3600
        actual_energy = initial_battery - uav.battery

        assert abs(actual_energy - expected_energy) < 0.001

    def test_hover_energy_calculation(self):
        """Test correct energy consumption for hovering."""
        uav = UAV(max_battery=274.0, power_hover=400.0)
        initial_battery = uav.battery

        uav.hover(duration=5.0)

        # Energy = (400W * 5s) / 3600 = 0.5556 Wh
        expected_energy = (400.0 * 5.0) / 3600
        actual_energy = initial_battery - uav.battery

        assert abs(actual_energy - expected_energy) < 0.001

    def test_collision_partial_energy(self):
        """Test boundary collision consumes partial energy."""
        uav = UAV(start_position=(9.0, 5.0), max_battery=274.0, power_move=600.0)
        initial_battery = uav.battery

        uav.move('RIGHT', grid_size=(10, 10), time_step=1.0)  # Collision

        # Collision energy = (power_move * 0.5 * time) / 3600
        expected_energy = (600.0 * 0.5 * 1.0) / 3600
        actual_energy = initial_battery - uav.battery

        assert abs(actual_energy - expected_energy) < 0.001

    def test_collision_uses_less_energy_than_move(self):
        """Test collision uses less energy than successful move."""
        uav1 = UAV(start_position=(5.0, 5.0), max_battery=274.0)
        uav2 = UAV(start_position=(9.0, 5.0), max_battery=274.0)

        uav1.move('UP', grid_size=(10, 10))  # Successful move
        uav2.move('RIGHT', grid_size=(10, 10))  # Collision

        energy_move = 274.0 - uav1.battery
        energy_collision = 274.0 - uav2.battery

        assert energy_collision < energy_move
        assert abs(energy_collision - energy_move * 0.5) < 0.001

    def test_multiple_moves_accumulate_energy(self):
        """Test energy consumption accumulates over multiple moves."""
        uav = UAV(max_battery=274.0)

        for _ in range(5):
            uav.move('UP', grid_size=(100, 100))

        energy_used = 274.0 - uav.battery
        assert energy_used > 0.5  # Should have used significant energy


class TestHovering:
    """Test UAV hovering functionality."""

    def test_hover_drains_battery(self):
        """Test hovering consumes battery."""
        uav = UAV(max_battery=274.0)
        initial_battery = uav.battery

        uav.hover(duration=1.0)

        assert uav.battery < initial_battery

    def test_hover_duration_affects_energy(self):
        """Test longer hover duration uses more energy."""
        uav1 = UAV(max_battery=274.0)
        uav2 = UAV(max_battery=274.0)

        uav1.hover(duration=1.0)
        uav2.hover(duration=5.0)

        energy1 = 274.0 - uav1.battery
        energy2 = 274.0 - uav2.battery

        assert energy2 > energy1
        assert abs(energy2 - 5 * energy1) < 0.001  # Should be 5x

    def test_hover_does_not_move_position(self):
        """Test hovering doesn't change UAV position."""
        uav = UAV(start_position=(5.0, 5.0))
        initial_position = uav.position.copy()

        uav.hover(duration=10.0)

        assert np.array_equal(uav.position, initial_position)


class TestBatteryStatus:
    """Test battery status checking."""

    def test_is_alive_with_battery(self):
        """Test is_alive returns True when battery > 0."""
        uav = UAV(max_battery=100.0)
        uav.battery = 50.0

        assert uav.is_alive() == True

    def test_is_alive_with_zero_battery(self):
        """Test is_alive returns False when battery = 0."""
        uav = UAV(max_battery=100.0)
        uav.battery = 0.0

        assert uav.is_alive() == False

    def test_is_alive_with_negative_battery(self):
        """Test is_alive returns False when battery < 0."""
        uav = UAV(max_battery=100.0)
        uav.battery = -5.0

        assert uav.is_alive() == False

    def test_is_alive_with_minimal_battery(self):
        """Test is_alive returns True with tiny positive battery."""
        uav = UAV(max_battery=100.0)
        uav.battery = 0.001

        assert uav.is_alive() == True

    def test_get_battery_percentage_full(self):
        """Test battery percentage calculation when full."""
        uav = UAV(max_battery=100.0)

        assert uav.get_battery_percentage() == 100.0

    def test_get_battery_percentage_half(self):
        """Test battery percentage calculation when half."""
        uav = UAV(max_battery=100.0)
        uav.battery = 50.0

        assert uav.get_battery_percentage() == 50.0

    def test_get_battery_percentage_empty(self):
        """Test battery percentage calculation when empty."""
        uav = UAV(max_battery=100.0)
        uav.battery = 0.0

        assert uav.get_battery_percentage() == 0.0


class TestEpisodeReset:
    """Test episode reset functionality."""

    def test_reset_restores_position(self):
        """Test reset restores UAV to start position."""
        uav = UAV(start_position=(5.0, 5.0))

        # Move away from start
        uav.move('UP', grid_size=(10, 10))
        uav.move('RIGHT', grid_size=(10, 10))

        # Reset
        uav.reset()

        assert np.array_equal(uav.position, np.array([5.0, 5.0]))

    def test_reset_restores_battery(self):
        """Test reset restores battery to maximum."""
        uav = UAV(max_battery=274.0)

        # Drain battery
        for _ in range(10):
            uav.move('UP', grid_size=(100, 100))

        # Reset
        uav.reset()

        assert uav.battery == 274.0

    def test_reset_multiple_times(self):
        """Test reset works consistently across multiple episodes."""
        uav = UAV(start_position=(3.0, 7.0), max_battery=100.0)

        for _ in range(3):
            # Move and use battery
            uav.move('UP', grid_size=(10, 10))
            uav.hover(5.0)

            # Reset
            uav.reset()

            # Verify
            assert np.array_equal(uav.position, np.array([3.0, 7.0]))
            assert uav.battery == 100.0

    def test_reset_after_battery_depletion(self):
        """Test reset works even after battery is depleted."""
        uav = UAV(max_battery=1.0)  # Small battery

        # Deplete battery
        uav.battery = 0.0
        assert uav.is_alive() == False

        # Reset
        uav.reset()

        assert uav.is_alive() == True
        assert uav.battery == 1.0


class TestUAVRepresentation:
    """Test string representation."""

    def test_repr_contains_key_info(self):
        """Test __repr__ contains position, battery, and percentage."""
        uav = UAV(start_position=(5.0, 5.0), max_battery=274.0)
        uav.battery = 137.0  # 50%

        repr_str = repr(uav)

        assert "UAV" in repr_str
        assert "5.0" in repr_str
        assert "137.00" in repr_str or "137" in repr_str
        assert "274" in repr_str
        assert "50" in repr_str  # Percentage

    def test_repr_is_string(self):
        """Test __repr__ returns a string."""
        uav = UAV()

        repr_str = repr(uav)

        assert isinstance(repr_str, str)
        assert len(repr_str) > 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_move_with_custom_timestep(self):
        """Test movement with non-default timestep."""
        uav = UAV(max_battery=274.0, power_move=600.0)
        initial_battery = uav.battery

        uav.move('UP', grid_size=(10, 10), time_step=2.0)

        # Energy should be 2x normal
        expected_energy = (600.0 * 2.0) / 3600
        actual_energy = initial_battery - uav.battery

        assert abs(actual_energy - expected_energy) < 0.001

    def test_hover_with_zero_duration(self):
        """Test hovering with zero duration."""
        uav = UAV(max_battery=274.0)
        initial_battery = uav.battery

        uav.hover(duration=0.0)

        assert uav.battery == initial_battery  # No energy consumed

    def test_fractional_positions(self):
        """Test UAV handles fractional start positions."""
        uav = UAV(start_position=(5.5, 7.3))

        assert uav.position[0] == 5.5
        assert uav.position[1] == 7.3

    def test_very_small_battery(self):
        """Test UAV with very small battery capacity."""
        uav = UAV(max_battery=0.1)

        assert uav.battery == 0.1
        assert uav.is_alive() == True

    def test_large_grid_movement(self):
        """Test UAV in large grid."""
        uav = UAV(start_position=(50.0, 50.0))

        success = uav.move('UP', grid_size=(100, 100))

        assert success == True
        assert uav.position[1] == 51.0


# Pytest fixtures
@pytest.fixture
def uav_default():
    """Create a UAV with default parameters."""
    return UAV()


@pytest.fixture
def uav_custom():
    """Create a UAV with custom parameters."""
    return UAV(
        start_position=(10.0, 10.0),
        max_battery=100.0,
        speed=15.0
    )

# Run tests with: pytest tests/test_uav.py -v