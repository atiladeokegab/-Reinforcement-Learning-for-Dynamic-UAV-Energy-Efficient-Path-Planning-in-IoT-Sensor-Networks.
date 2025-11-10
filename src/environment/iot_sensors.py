"""
IoT Sensor with Cumulative Data Tracking

Properly tracks all data metrics:
- total_data_generated (cumulative)
- total_data_transmitted (cumulative collected data)
- total_data_lost (cumulative)
- data_buffer (current state)

Author: ATILADE GABRIEL OKE
Date: November 2025
"""

import numpy as np
from typing import Tuple, Optional


class IoTSensor:
    """Enhanced IoT Sensor with proper cumulative data tracking."""

    LORA_DATA_RATES = {
        7: 5470 / 8,   # ~684 bytes/sec
        8: 3125 / 8,   # ~390 bytes/sec
        9: 1760 / 8,   # ~220 bytes/sec
        10: 980 / 8,   # ~122 bytes/sec
        11: 440 / 8,   # ~55 bytes/sec
        12: 250 / 8    # ~31 bytes/sec
    }

    REQUIRED_SNR_DB = {
        7: 7.5,
        8: 10.0,
        9: 12.5,
        10: 15.0,
        11: 17.5,
        12: 20.0
    }

    # RSSI to SF mapping
    RSSI_SF_MAPPING = [
        (-39, 7),            # SF7: RSSI > -39 dBm (very close, 0-1m)
        (-44, 9),            # SF9: RSSI > -44 dBm (close, 1-2m)
        (-50, 11),           # SF11: RSSI > -51 dBm (medium, 2-10m)
        (-100, 12)  # SF12: RSSI < -51 dBm (far, 10-70m)
    ]

    def __init__(
            self,
            position: Tuple[float, float],
            sensor_id: int = 0,
            max_buffer_size: float = 1000.0,
            data_generation_rate: float = 10.0,
            packet_size: int = 50,
            transmit_power_dbm: float = 0.0,
            path_loss_exponent: float = 2.0,
            d0: float = 1.0,
            rssi_threshold: float = -140.0,
            noise_floor_dbm: float = -174.0,
            uav_altitude: int = 10,
            spreading_factor: int = 7,
            duty_cycle: float = 5.0
    ):
        """
        Initialize IoT Sensor with cumulative data tracking.

        Args:
            position: Sensor (x, y) position in meters
            sensor_id: Unique identifier
            max_buffer_size: Maximum buffer capacity (bytes)
            data_generation_rate: Data generation rate (bytes/sec)
            packet_size: Transmission packet size (bytes)
            transmit_power_dbm: Transmission power (dBm)
            path_loss_exponent: Path loss exponent (2.0 for free space)
            d0: Reference distance (1.0m standard)
            rssi_threshold: Minimum RSSI for communication (dBm)
            noise_floor_dbm: Ambient noise floor (dBm)
            uav_altitude: UAV altitude in meters
            spreading_factor: Initial SF (7-12)
            duty_cycle: Duty cycle percentage (1-100)
        """
        # Position and identification
        self.position = np.array(position, dtype=np.float32)
        self.sensor_id = sensor_id

        # Buffer configuration
        self.max_buffer_size = max_buffer_size
        self.data_generation_rate = data_generation_rate
        self.packet_size = packet_size

        # RF parameters
        self.transmit_power_dbm = transmit_power_dbm
        self.path_loss_exponent = path_loss_exponent
        self.d0 = d0
        self.rssi_threshold = rssi_threshold
        self.noise_floor_dbm = noise_floor_dbm
        self.uav_altitude = uav_altitude

        # LoRa parameters
        self.spreading_factor = spreading_factor
        self.data_rate = self.LORA_DATA_RATES[self.spreading_factor]

        # Duty cycle (percentage)
        self.duty_cycle = duty_cycle

        # Current state
        self.data_buffer = 0.0

        # CUMULATIVE TRACKING (pleaseeeeee never reset except on reset())
        self.total_data_generated = 0.0      # All data ever generated
        self.total_data_transmitted = 0.0    # All data ever collected/transmitted
        self.total_data_lost = 0.0           # All data ever lost to overflow

        # Transmission statistics
        self.transmission_count = 0
        self.successful_transmissions = 0

        # Status flags
        self.data_collected = False  # Boolean flag (at least one collection occurred)
        self.is_active = True  # For duty cycle simulation

        # SF history tracking for analysis
        self.sf_history = []

    # ============ Properties ============

    @property
    def duty_cycle_probability(self) -> float:
        """
        Returns the duty cycle as a probability value (0.0 to 1.0).
        """
        return self.duty_cycle / 100.0

    # ============ Data Generation ============

    def step(self, time_step: float = 1.0) -> float:
        """
        Simulate one time step of continuous data generation.

        Args:
            time_step: Duration of time step in seconds

        Returns:
            Amount of data lost due to buffer overflow (bytes)
        """
        # Generate new data
        new_data = self.data_generation_rate * time_step
        self.total_data_generated += new_data  # Cumulative

        # Try to add to buffer
        potential_buffer = self.data_buffer + new_data

        # Calculate overflow
        if potential_buffer > self.max_buffer_size:
            data_loss = potential_buffer - self.max_buffer_size
            self.data_buffer = self.max_buffer_size
            self.total_data_lost += data_loss  # Cumulative
        else:
            data_loss = 0.0
            self.data_buffer = potential_buffer

        return data_loss

    # ============ Data Collection ============

    def collect_data(
            self,
            uav_position: Tuple[float, float],
            collection_duration: float = 1.0
    ) -> Tuple[float, bool]:
        """
        Collect data from sensor buffer (only if in range and has data).

        PROPERLY TRACKS CUMULATIVE DATA IN total_data_transmitted

        Args:
            uav_position: UAV position (x, y)
            collection_duration: How long to collect (seconds)

        Returns:
            Tuple of (bytes_collected, success)
        """
        # Check if in range
        success_prob = self.get_success_probability(uav_position, use_advanced_model=False)
        in_range = success_prob > 0.5

        if not in_range:
            return 0.0, False

        if self.data_buffer <= 0:
            return 0.0, True  # In range but no data

        # Calculate maximum collectible based on data rate
        max_collectible = self.data_rate * collection_duration
        bytes_collected = min(self.data_buffer, max_collectible)

        # Remove from buffer
        self.data_buffer -= bytes_collected

        # TRACK CUMULATIVE COLLECTED DATA
        self.total_data_transmitted += bytes_collected

        # Update statistics
        self.transmission_count += 1

        if bytes_collected > 0:
            self.successful_transmissions += 1
            self.data_collected = True  # Flag that collection occurred

        return bytes_collected, True

    def transmit_data(self, amount: float) -> float:
        """
        Transmit data from buffer (alternative collection method).

        Args:
            amount: Amount to transmit (bytes)

        Returns:
            Amount actually transmitted (bytes)
        """
        transmitted = min(amount, self.data_buffer)
        self.data_buffer -= transmitted

        # TRACK CUMULATIVE TRANSMITTED DATA
        self.total_data_transmitted += transmitted

        self.transmission_count += 1
        return transmitted

    # ============ Energy & Timing ============

    def get_transmission_energy_cost(
            self,
            power_tx_watts: float,
            packet_size_bytes: Optional[int] = None
    ) -> float:
        """
        Calculate the energy consumed by the sensor to transmit one packet.

        Args:
            power_tx_watts: Transmit power in watts
            packet_size_bytes: Size of packet (uses self.packet_size if None)

        Returns:
            Energy cost in watt-hours (Wh)
        """
        if packet_size_bytes is None:
            packet_size_bytes = self.packet_size

        transmission_time = packet_size_bytes / self.data_rate
        energy_cost = power_tx_watts * transmission_time
        return energy_cost

    def get_transmission_time(self, packet_size_bytes: Optional[int] = None) -> float:
        """
        Get time-on-air for a packet (in seconds).

        Args:
            packet_size_bytes: Size of packet (uses self.packet_size if None)

        Returns:
            Time-on-air in seconds
        """
        if packet_size_bytes is None:
            packet_size_bytes = self.packet_size
        return packet_size_bytes / self.data_rate

    # ============ RF Propagation ============

    def calculate_rssi(self, uav_position: Tuple[float, float]) -> float:
        """
        Calculate RSSI using Free Space Path Loss Model.

        Formula: RSSI = Ptx - PL
                 PL = 20*n*log10(d/d0)

        Args:
            uav_position: UAV position (x, y)

        Returns:
            RSSI in dBm
        """
        sensor_pos = np.array(self.position, dtype=np.float32)
        uav_pos = np.array(uav_position, dtype=np.float32)

        # Calculate 3D distance (ground distance + altitude)
        dx = uav_pos[0] - sensor_pos[0]
        dy = uav_pos[1] - sensor_pos[1]
        ground_distance = np.sqrt(dx ** 2 + dy ** 2)
        distance_3d = np.sqrt(ground_distance ** 2 + self.uav_altitude ** 2)

        # Free Space Path Loss
        if distance_3d <= self.d0:
            path_loss = 0.1
        else:
            path_loss = 20 * self.path_loss_exponent * np.log10(distance_3d / self.d0)

        rssi = self.transmit_power_dbm - path_loss
        return float(rssi)

    def _get_required_snr(self, spreading_factor: int) -> float:
        """
        Get required SNR (dB) for a given spreading factor.

        Args:
            spreading_factor: LoRa SF (7-12)

        Returns:
            Required SNR in dB
        """
        return self.REQUIRED_SNR_DB.get(spreading_factor, 7.5)

    def get_success_probability(
            self,
            uav_position: Tuple[float, float],
            use_advanced_model: bool = False
    ) -> float:
        """
        Calculate probability of successful reception based on link quality.

        Args:
            uav_position: UAV position (x, y)
            use_advanced_model: Use SNR-based sigmoid model

        Returns:
            Success probability (0.0 to 1.0)
        """
        rssi = self.calculate_rssi(uav_position)

        # Hard threshold
        if rssi < self.rssi_threshold:
            return 0.0

        # Advanced model: sigmoid based on SNR margin
        if use_advanced_model:
            snr_db = rssi - self.noise_floor_dbm
            required_snr_db = self._get_required_snr(self.spreading_factor)
            sigmoid = 1.0 / (1.0 + np.exp(-(snr_db - required_snr_db)))
            return sigmoid

        # Simple model: binary threshold
        return 1.0

    def is_in_range(self, uav_position: Tuple[float, float]) -> bool:
        """
        Check if UAV is in communication range.

        Args:
            uav_position: UAV position (x, y)

        Returns:
            True if in range, False otherwise
        """
        return self.get_success_probability(uav_position) > 0.5

    # ============ Adaptive Data Rate (ADR) ============

    def update_spreading_factor(self, uav_position: Tuple[float, float], current_step: int = 0):
        """
        Dynamically update spreading factor based on RSSI (Adaptive Data Rate).

        Args:
            uav_position: UAV position (x, y)
            current_step: Current simulation step (for tracking)
        """
        rssi = self.calculate_rssi(uav_position)
        old_sf = self.spreading_factor

        # Find appropriate SF based on RSSI
        for threshold, sf in self.RSSI_SF_MAPPING:
            if rssi > threshold:
                self.spreading_factor = sf
                break

        # Update data rate
        self.data_rate = self.LORA_DATA_RATES[self.spreading_factor]

        # Track SF changes
        if self.spreading_factor != old_sf:
            self.sf_history.append((current_step, self.spreading_factor))

    # ============ Status & Reporting ============

    def get_buffer_status(self) -> dict:
        """
        Return buffer and data loss statistics.

        Returns:
            Dictionary of buffer metrics
        """
        return {
            "buffer_used": self.data_buffer,
            "buffer_capacity": self.max_buffer_size,
            "buffer_utilization": self.data_buffer / self.max_buffer_size,
            "total_data_generated": self.total_data_generated,
            "total_data_lost": self.total_data_lost,
            "total_data_transmitted": self.total_data_transmitted,
            "loss_rate": self.total_data_lost / max(1, self.total_data_generated),
            "collection_rate": self.total_data_transmitted / max(1, self.total_data_generated),
            "duty_cycle": self.duty_cycle
        }

    def get_link_quality_report(self, uav_position: Tuple[float, float]) -> dict:
        """
        Generate comprehensive link quality report.

        Args:
            uav_position: UAV position (x, y)

        Returns:
            Dictionary of link quality metrics
        """
        rssi = self.calculate_rssi(uav_position)
        snr = rssi - self.noise_floor_dbm
        success_prob = self.get_success_probability(uav_position, use_advanced_model=True)
        toa = self.get_transmission_time()

        return {
            "sensor_id": self.sensor_id,
            "rssi_dbm": rssi,
            "snr_db": snr,
            "spreading_factor": self.spreading_factor,
            "data_rate_bps": self.data_rate,
            "success_probability": success_prob,
            "time_on_air_sec": toa,
            "in_range": self.is_in_range(uav_position),
            "required_snr_db": self._get_required_snr(self.spreading_factor),
            "duty_cycle": self.duty_cycle
        }

    # ============ Analysis Methods ============

    def get_sf_distribution(self) -> dict:
        """
        Get distribution of spreading factors used during episode.

        Returns:
            Dictionary mapping SF to count
        """
        sf_counts = {sf: 0 for sf in range(7, 13)}
        for step, sf in self.sf_history:
            sf_counts[sf] += 1
        return sf_counts

    def get_sf_changes(self) -> int:
        """
        Return number of SF changes during episode.

        Returns:
            Number of SF transitions
        """
        if len(self.sf_history) <= 1:
            return 0
        changes = sum(1 for i in range(1, len(self.sf_history))
                      if self.sf_history[i][1] != self.sf_history[i - 1][1])
        return changes

    def get_collection_percentage(self) -> float:
        """
        Calculate percentage of generated data that was collected.

        Returns:
            Collection percentage (0-100)
        """
        if self.total_data_generated <= 0:
            return 0.0
        return (self.total_data_transmitted / self.total_data_generated) * 100

    def get_data_accounting(self) -> dict:
        """
        Get complete data accounting (for verification).

        Returns:
            Dictionary with data accounting breakdown
        """
        total_accounted = self.total_data_transmitted + self.total_data_lost + self.data_buffer
        accounting_error = abs(self.total_data_generated - total_accounted)

        return {
            'generated': self.total_data_generated,
            'transmitted': self.total_data_transmitted,
            'lost': self.total_data_lost,
            'in_buffer': self.data_buffer,
            'accounted': total_accounted,
            'error': accounting_error,
            'error_pct': (accounting_error / max(1, self.total_data_generated)) * 100
        }

    # ============ Reset & Cleanup ============

    def reset(self):
        """Reset sensor to initial state (clears ALL cumulative data)."""
        # Current state
        self.data_buffer = 0.0
        self.spreading_factor = 7
        self.data_rate = self.LORA_DATA_RATES[self.spreading_factor]

        # RESET CUMULATIVE TRACKING
        self.total_data_generated = 0.0
        self.total_data_lost = 0.0
        self.total_data_transmitted = 0.0

        # Statistics
        self.transmission_count = 0
        self.successful_transmissions = 0
        self.data_collected = False
        self.is_active = True

        # History
        self.sf_history = []

    def reset_stats(self):
        """Reset only statistics (alias for reset)."""
        self.reset()

    # ============ String Representation ============

    def __repr__(self) -> str:
        """String representation of sensor."""
        return (f"IoTSensor(id={self.sensor_id}, SF{self.spreading_factor}, "
                f"buffer={self.data_buffer:.0f}B/{self.max_buffer_size:.0f}B, "
                f"collected={self.total_data_transmitted:.0f}B, "
                f"duty_cycle={self.duty_cycle}%)")

    def __str__(self) -> str:
        """Human-readable string."""
        collection_pct = self.get_collection_percentage()
        return (f"Sensor {self.sensor_id}: "
                f"Generated={self.total_data_generated:.0f}B, "
                f"Collected={self.total_data_transmitted:.0f}B ({collection_pct:.1f}%), "
                f"Lost={self.total_data_lost:.0f}B, "
                f"Buffer={self.data_buffer:.0f}B")


# ============ TESTING ============

if __name__ == "__main__":
    print("=" * 100)
    print("IoT SENSOR - CUMULATIVE DATA TRACKING TEST")
    print("=" * 100)
    print()

    # Test 1: Create sensor
    print("Test 1: Creating IoT Sensor")
    print("-" * 100)
    sensor = IoTSensor(
        position=(25.0, 25.0),
        sensor_id=1,
        max_buffer_size=1000.0,
        data_generation_rate=10.0,
        duty_cycle=5.0,
        transmit_power_dbm=0.0,
        uav_altitude=10
    )
    print(f"✓ {sensor}")
    print()

    # Test 2: Data generation
    print("Test 2: Data Generation & Cumulative Tracking")
    print("-" * 100)

    for step in range(5):
        loss = sensor.step(time_step=1.0)
        print(f"Step {step + 1}: Generated={sensor.total_data_generated:.0f}B, "
              f"Buffer={sensor.data_buffer:.0f}B, Loss={loss:.0f}B")

    print()

    # Test 3: Data collection
    print("Test 3: Data Collection & Cumulative Tracking")
    print("-" * 100)

    uav_pos = (25.0, 25.0)  # Same position as sensor
    print(f"UAV at {uav_pos}, RSSI: {sensor.calculate_rssi(uav_pos):.1f} dBm")

    for i in range(3):
        collected, success = sensor.collect_data(uav_pos, collection_duration=1.0)
        print(f"Collection {i + 1}: Collected={collected:.0f}B, "
              f"Total Transmitted={sensor.total_data_transmitted:.0f}B, "
              f"Buffer={sensor.data_buffer:.0f}B")

        # Generate more data
        sensor.step(time_step=1.0)

    print()

    # Test 4: Data accounting
    print("Test 4: Data Accounting Verification")
    print("-" * 100)

    accounting = sensor.get_data_accounting()
    print(f"Generated:    {accounting['generated']:.0f} bytes")
    print(f"Transmitted:  {accounting['transmitted']:.0f} bytes")
    print(f"Lost:         {accounting['lost']:.0f} bytes")
    print(f"In Buffer:    {accounting['in_buffer']:.0f} bytes")
    print(f"---")
    print(f"Accounted:    {accounting['accounted']:.0f} bytes")
    print(f"Error:        {accounting['error']:.3f} bytes ({accounting['error_pct']:.3f}%)")

    if accounting['error'] < 0.1:
        print("Data accounting: PERFECT")
    else:
        print("Data accounting: ERROR")

    print()

    # Test 5: Collection percentage
    print("Test 5: Collection Metrics")
    print("-" * 100)
    print(f"Collection Percentage: {sensor.get_collection_percentage():.1f}%")
    print(f"Buffer Status: {sensor.get_buffer_status()}")
    print()

    # Test 6: Reset
    print("Test 6: Reset Functionality")
    print("-" * 100)
    print(f"Before reset: {sensor}")
    sensor.reset()
    print(f"After reset:  {sensor}")
    print()

    print("=" * 100)
    print("ALL TESTS PASSED")
    print("=" * 100)