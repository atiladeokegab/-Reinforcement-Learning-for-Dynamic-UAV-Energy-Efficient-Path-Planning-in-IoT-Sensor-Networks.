"""
Unit Tests for IoT Sensor Implementation

Tests all functionality including:
- Initialization and validation
- Data generation
- LoRa communication (RSSI, range)
- Data collection
- Buffer management

Author: Atilade Okeat
Date: January 2025
"""

import pytest
import numpy as np
import sys
from pathlib import Path

import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Now this works!
from src.environment.iot_sensors import IoTSensor

class TestIoTSensorInitialization:
    """Test sensor initialization and parameter validation."""

    def test_basic_initialization(self):
        """Test sensor initializes with default parameters."""
        sensor = IoTSensor(sensor_id=1, position=(5.0, 5.0))

        assert sensor.sensor_id == 1
        assert np.array_equal(sensor.position, np.array([5.0, 5.0], dtype=np.float32))
        assert sensor.data_buffer == 0.0
        assert sensor.data_collected == False
        assert sensor.spreading_factor == 7

    def test_custom_parameters(self):
        """Test sensor initializes with custom parameters."""
        sensor = IoTSensor(
            sensor_id=42,
            position=(10.5, 20.3),
            data_generation_rate=0.5,
            max_buffer_size=5000.0,
            spreading_factor=12
        )

        assert sensor.sensor_id == 42
        assert sensor.data_generation_rate == 0.5
        assert sensor.max_buffer_size == 5000.0
        assert sensor.spreading_factor == 12

    def test_position_conversion(self):
        """Test position is converted to numpy array."""
        sensor = IoTSensor(sensor_id=1, position=(3, 4))

        assert isinstance(sensor.position, np.ndarray)
        assert sensor.position.dtype == np.float32

    def test_data_rate_lookup(self):
        """Test LoRa data rate is correctly looked up from spreading factor."""
        sensor_sf7 = IoTSensor(1, (0, 0), spreading_factor=7)
        sensor_sf12 = IoTSensor(2, (0, 0), spreading_factor=12)

        assert sensor_sf7.data_rate == IoTSensor.LORA_DATA_RATES[7]
        assert sensor_sf12.data_rate == IoTSensor.LORA_DATA_RATES[12]
        assert sensor_sf7.data_rate > sensor_sf12.data_rate  # SF7 faster than SF12


class TestDataGeneration:
    """Test data generation and buffer management."""

    def test_data_generation_single_step(self):
        """Test data is generated in a single step."""
        sensor = IoTSensor(1, (0, 0), data_generation_rate=1.0)

        sensor.step(time_step=1.0)

        assert sensor.data_buffer == 1.0
        assert sensor.total_data_generated == 1.0

    def test_data_generation_multiple_steps(self):
        """Test data accumulates over multiple steps."""
        sensor = IoTSensor(1, (0, 0), data_generation_rate=0.5)

        sensor.step(time_step=1.0)
        sensor.step(time_step=1.0)
        sensor.step(time_step=1.0)

        assert sensor.data_buffer == 1.5
        assert sensor.total_data_generated == 1.5

    def test_data_generation_custom_timestep(self):
        """Test data generation with custom time step."""
        sensor = IoTSensor(1, (0, 0), data_generation_rate=2.0)

        sensor.step(time_step=5.0)

        assert sensor.data_buffer == 10.0
        assert sensor.total_data_generated == 10.0

    def test_buffer_overflow_prevention(self):
        """Test buffer doesn't exceed maximum capacity."""
        sensor = IoTSensor(1, (0, 0), data_generation_rate=100.0, max_buffer_size=50.0)

        sensor.step(time_step=1.0)  # Would generate 100, but capped at 50

        assert sensor.data_buffer == 50.0
        assert sensor.total_data_generated == 100.0  # Total tracks what was generated

    def test_buffer_remains_at_max_when_full(self):
        """Test buffer stays at max when continuously generating."""
        sensor = IoTSensor(1, (0, 0), data_generation_rate=10.0, max_buffer_size=20.0)

        sensor.step(time_step=1.0)  # Buffer: 10
        sensor.step(time_step=1.0)  # Buffer: 20 (max)
        sensor.step(time_step=1.0)  # Buffer: 20 (stays at max)

        assert sensor.data_buffer == 20.0
        assert sensor.total_data_generated == 30.0


class TestRSSICalculation:
    """Test RSSI (signal strength) calculations."""

    def test_rssi_at_reference_distance(self):
        """Test RSSI equals reference RSSI at reference distance."""
        sensor = IoTSensor(
            1, (0, 0),
            rssi_reference=-60.0,
            reference_distance=1.0,
            path_loss_exponent=2.0
        )

        rssi = sensor.calculate_rssi((1.0, 0.0))  # Distance = 1.0

        assert abs(rssi - (-60.0)) < 0.01  # Should be approximately -60 dBm

    def test_rssi_decreases_with_distance(self):
        """Test RSSI decreases as distance increases."""
        sensor = IoTSensor(1, (0, 0), rssi_reference=-60.0, path_loss_exponent=2.0)

        rssi_1m = sensor.calculate_rssi((1.0, 0.0))
        rssi_10m = sensor.calculate_rssi((10.0, 0.0))
        rssi_100m = sensor.calculate_rssi((100.0, 0.0))

        assert rssi_1m > rssi_10m > rssi_100m

    def test_rssi_path_loss_exponent_2(self):
        """Test RSSI follows d^-2 path loss (20 dB per decade)."""
        sensor = IoTSensor(1, (0, 0), rssi_reference=-60.0, path_loss_exponent=2.0)

        rssi_1m = sensor.calculate_rssi((1.0, 0.0))
        rssi_10m = sensor.calculate_rssi((10.0, 0.0))

        # For n=2, going 10x distance = 20 dB loss
        expected_loss = 20.0
        actual_loss = rssi_1m - rssi_10m

        assert abs(actual_loss - expected_loss) < 0.1

    def test_rssi_symmetric_distance(self):
        """Test RSSI is same for equal distances in different directions."""
        sensor = IoTSensor(1, (5.0, 5.0))

        rssi_north = sensor.calculate_rssi((5.0, 8.0))  # 3 units north
        rssi_east = sensor.calculate_rssi((8.0, 5.0))  # 3 units east
        rssi_diagonal = sensor.calculate_rssi((7.12, 7.12))  # ~3 units diagonal

        assert abs(rssi_north - rssi_east) < 0.01
        assert abs(rssi_north - rssi_diagonal) < 0.1

    def test_rssi_very_close_distance_handling(self):
        """Test RSSI doesn't break with very small distances (avoid log(0))."""
        sensor = IoTSensor(1, (5.0, 5.0))

        # Should not raise exception or return inf/nan
        rssi = sensor.calculate_rssi((5.0, 5.0))  # Distance ≈ 0

        assert not np.isnan(rssi)
        assert not np.isinf(rssi)
        assert rssi > -200  # Reasonable value


class TestCommunicationRange:
    """Test communication range checking."""

    def test_in_range_at_close_distance(self):
        """Test UAV is in range when close to sensor."""
        sensor = IoTSensor(1, (5.0, 5.0), rssi_threshold=-120.0)

        assert sensor.is_in_range((5.0, 5.0)) == True  # Same position
        assert sensor.is_in_range((6.0, 5.0)) == True  # 1 unit away

    def test_out_of_range_at_far_distance(self):
        """Test UAV is out of range when far from sensor."""
        sensor = IoTSensor(1, (5.0, 5.0), rssi_threshold=-120.0)

        # Very far away
        assert sensor.is_in_range((5000.0, 5000.0)) == False

    def test_range_boundary_threshold(self):
        """Test range boundary is determined by RSSI threshold."""
        sensor = IoTSensor(
            1, (0, 0),
            rssi_reference=-60.0,
            rssi_threshold=-100.0,
            path_loss_exponent=2.0
        )

        # Find approximate range where RSSI = -100 dBm
        # RSSI = -60 - 20*log10(d)
        # -100 = -60 - 20*log10(d)
        # -40 = -20*log10(d)
        # 2 = log10(d)
        # d = 100

        assert sensor.is_in_range((50.0, 0.0)) == True  # Within range
        assert sensor.is_in_range((150.0, 0.0)) == False  # Outside range

    def test_stricter_threshold_reduces_range(self):
        """Test higher RSSI threshold reduces communication range."""
        sensor_lenient = IoTSensor(1, (0, 0), rssi_threshold=-120.0)
        sensor_strict = IoTSensor(1, (0, 0), rssi_threshold=-80.0)

        far_position = (50.0, 0.0)

        # Lenient threshold allows communication, strict does not
        assert sensor_lenient.is_in_range(far_position) == True
        assert sensor_strict.is_in_range(far_position) == False


class TestDataCollection:
    """Test data collection functionality."""

    def test_collect_data_when_in_range(self):
        """Test data collection succeeds when UAV is in range."""
        sensor = IoTSensor(1, (5.0, 5.0))
        sensor.data_buffer = 100.0

        bytes_collected, success = sensor.collect_data((5.0, 5.0), collection_duration=1.0)

        assert success == True
        assert bytes_collected > 0

    def test_collect_data_when_out_of_range(self):
        """Test data collection fails when UAV is out of range."""
        sensor = IoTSensor(1, (5.0, 5.0), rssi_threshold=-80.0)  # Strict threshold
        sensor.data_buffer = 100.0

        bytes_collected, success = sensor.collect_data((5000.0, 5000.0), collection_duration=1.0)

        assert success == False
        assert bytes_collected == 0.0
        assert sensor.data_buffer == 100.0  # Buffer unchanged

    def test_collect_data_when_buffer_empty(self):
        """Test data collection fails when buffer is empty."""
        sensor = IoTSensor(1, (5.0, 5.0))
        sensor.data_buffer = 0.0

        bytes_collected, success = sensor.collect_data((5.0, 5.0), collection_duration=1.0)

        assert success == False
        assert bytes_collected == 0.0

    def test_collect_data_limited_by_data_rate(self):
        """Test collection is limited by LoRa data rate."""
        sensor = IoTSensor(1, (5.0, 5.0), spreading_factor=7)  # ~684 bytes/sec
        sensor.data_buffer = 10000.0  # Lots of data

        bytes_collected, success = sensor.collect_data((5.0, 5.0), collection_duration=1.0)

        assert success == True
        # Should collect approximately data_rate * duration
        expected = sensor.data_rate * 1.0
        assert abs(bytes_collected - expected) < 1.0

    def test_collect_data_limited_by_buffer(self):
        """Test collection is limited by available buffer data."""
        sensor = IoTSensor(1, (5.0, 5.0), spreading_factor=7)
        sensor.data_buffer = 50.0  # Less than data_rate * duration

        bytes_collected, success = sensor.collect_data((5.0, 5.0), collection_duration=1.0)

        assert success == True
        assert bytes_collected == 50.0  # Collected all available
        assert sensor.data_buffer == 0.0

    def test_collect_data_removes_from_buffer(self):
        """Test collected data is removed from buffer."""
        sensor = IoTSensor(1, (5.0, 5.0), spreading_factor=7)
        sensor.data_buffer = 1000.0
        initial_buffer = sensor.data_buffer

        bytes_collected, success = sensor.collect_data((5.0, 5.0), collection_duration=1.0)

        assert success == True
        assert sensor.data_buffer == initial_buffer - bytes_collected

    def test_collect_data_marks_as_collected_when_empty(self):
        """Test sensor is marked as collected when buffer becomes empty."""
        sensor = IoTSensor(1, (5.0, 5.0))
        sensor.data_buffer = 50.0

        bytes_collected, success = sensor.collect_data((5.0, 5.0), collection_duration=1.0)

        assert sensor.data_buffer == 0.0
        assert sensor.data_collected == True

    def test_collect_data_longer_duration(self):
        """Test longer collection duration allows more data transfer."""
        sensor1 = IoTSensor(1, (5.0, 5.0), spreading_factor=7)
        sensor1.data_buffer = 10000.0

        sensor2 = IoTSensor(2, (5.0, 5.0), spreading_factor=7)
        sensor2.data_buffer = 10000.0

        bytes_1sec, _ = sensor1.collect_data((5.0, 5.0), collection_duration=1.0)
        bytes_5sec, _ = sensor2.collect_data((5.0, 5.0), collection_duration=5.0)

        assert bytes_5sec > bytes_1sec
        assert abs(bytes_5sec - 5 * bytes_1sec) < 1.0  # Should be ~5x


class TestTransmissionTime:
    """Test transmission time calculations."""

    def test_transmission_time_calculation(self):
        """Test transmission time is correctly calculated."""
        sensor = IoTSensor(1, (0, 0), spreading_factor=7)  # ~684 bytes/sec

        time = sensor.calculate_transmission_time(684.0)  # Should take ~1 sec

        assert abs(time - 1.0) < 0.01

    def test_transmission_time_zero_data(self):
        """Test transmission time is zero for zero data."""
        sensor = IoTSensor(1, (0, 0))

        time = sensor.calculate_transmission_time(0.0)

        assert time == 0.0

    def test_transmission_time_proportional(self):
        """Test transmission time is proportional to data amount."""
        sensor = IoTSensor(1, (0, 0), spreading_factor=7)

        time_100 = sensor.calculate_transmission_time(100.0)
        time_200 = sensor.calculate_transmission_time(200.0)

        assert abs(time_200 - 2 * time_100) < 0.001


class TestBufferStatus:
    """Test buffer status reporting."""

    def test_buffer_status_empty(self):
        """Test buffer status when empty."""
        sensor = IoTSensor(1, (0, 0))

        status = sensor.get_buffer_status()

        assert status['buffer_bytes'] == 0.0
        assert status['buffer_percent'] == 0.0
        assert status['is_empty'] == True
        assert status['is_full'] == False

    def test_buffer_status_partial(self):
        """Test buffer status when partially filled."""
        sensor = IoTSensor(1, (0, 0), max_buffer_size=1000.0)
        sensor.data_buffer = 250.0

        status = sensor.get_buffer_status()

        assert status['buffer_bytes'] == 250.0
        assert status['buffer_percent'] == 25.0
        assert status['is_empty'] == False
        assert status['is_full'] == False

    def test_buffer_status_full(self):
        """Test buffer status when full."""
        sensor = IoTSensor(1, (0, 0), max_buffer_size=1000.0)
        sensor.data_buffer = 1000.0

        status = sensor.get_buffer_status()

        assert status['buffer_bytes'] == 1000.0
        assert status['buffer_percent'] == 100.0
        assert status['is_full'] == True

    def test_buffer_utilization_property(self):
        """Test buffer_utilization property."""
        sensor = IoTSensor(1, (0, 0), max_buffer_size=200.0)
        sensor.data_buffer = 50.0

        assert sensor.buffer_utilization == 25.0

    def test_has_data_property(self):
        """Test has_data property."""
        sensor = IoTSensor(1, (0, 0))

        assert sensor.has_data == False

        sensor.data_buffer = 10.0
        assert sensor.has_data == True


class TestSensorReset:
    """Test sensor reset functionality."""

    def test_reset_clears_buffer(self):
        """Test reset clears data buffer."""
        sensor = IoTSensor(1, (0, 0))
        sensor.data_buffer = 500.0

        sensor.reset()

        assert sensor.data_buffer == 0.0

    def test_reset_clears_collected_flag(self):
        """Test reset clears collection status."""
        sensor = IoTSensor(1, (0, 0))
        sensor.data_collected = True

        sensor.reset()

        assert sensor.data_collected == False

    def test_reset_preserves_total_generated(self):
        """Test reset does not clear total data generated statistic."""
        sensor = IoTSensor(1, (0, 0), data_generation_rate=1.0)

        sensor.step(time_step=10.0)
        total_before_reset = sensor.total_data_generated

        sensor.reset()

        assert sensor.total_data_generated == total_before_reset

    def test_reset_allows_reuse(self):
        """Test sensor can be reused after reset."""
        sensor = IoTSensor(1, (0, 0), data_generation_rate=1.0)

        # First episode
        sensor.step(time_step=5.0)
        sensor.data_buffer = 100.0

        # Reset for new episode
        sensor.reset()

        # Should work normally
        sensor.step(time_step=1.0)
        assert sensor.data_buffer == 1.0


class TestSensorRepresentation:
    """Test string representation."""

    def test_repr_contains_key_info(self):
        """Test __repr__ contains key sensor information."""
        sensor = IoTSensor(1, (5.0, 5.0), spreading_factor=7)
        sensor.data_buffer = 123.4

        repr_str = repr(sensor)

        assert "IoTSensor" in repr_str
        assert "id=1" in repr_str
        assert "5.0" in repr_str
        assert "123.4" in repr_str or "123" in repr_str
        assert "SF=7" in repr_str


class TestSpreadingFactorVariations:
    """Test behavior with different spreading factors."""

    def test_sf7_fastest_data_rate(self):
        """Test SF7 has highest data rate."""
        sensor = IoTSensor(1, (0, 0), spreading_factor=7)

        assert sensor.data_rate == IoTSensor.LORA_DATA_RATES[7]
        assert sensor.data_rate > 600  # Should be ~684 bytes/sec

    def test_sf12_slowest_data_rate(self):
        """Test SF12 has lowest data rate."""
        sensor = IoTSensor(1, (0, 0), spreading_factor=12)

        assert sensor.data_rate == IoTSensor.LORA_DATA_RATES[12]
        assert sensor.data_rate < 50  # Should be ~31 bytes/sec

    def test_all_spreading_factors_valid(self):
        """Test all spreading factors 7-12 initialize correctly."""
        for sf in range(7, 13):
            sensor = IoTSensor(1, (0, 0), spreading_factor=sf)
            assert sensor.spreading_factor == sf
            assert sensor.data_rate > 0


# Pytest fixtures for common test setups
@pytest.fixture
def basic_sensor():
    """Create a basic sensor for testing."""
    return IoTSensor(sensor_id=1, position=(5.0, 5.0))


@pytest.fixture
def sensor_with_data():
    """Create a sensor with data in buffer."""
    sensor = IoTSensor(sensor_id=1, position=(5.0, 5.0))
    sensor.data_buffer = 100.0
    return sensor

# Run tests with: pytest tests/test_iot_sensors.py -v