import numpy as np
from typing import Tuple, Optional


class IoTSensor:
    """Enhanced IoT Sensor with LoRa duty cycling, energy awareness, and realistic communication model."""

    LORA_DATA_RATES = {
        7: 5470 / 8,  # ~684 bytes/sec
        8: 3125 / 8,  # ~390 bytes/sec
        9: 1760 / 8,  # ~220 bytes/sec
        10: 980 / 8,  # ~122 bytes/sec
        11: 440 / 8,  # ~55 bytes/sec
        12: 250 / 8  # ~31 bytes/sec
    }

    REQUIRED_SNR_DB = {
        7: 7.5,
        8: 10.0,
        9: 12.5,
        10: 15.0,
        11: 17.5,
        12: 20.0
    }

    RSSI_SF_MAPPING = [
        (-100, 7),
        (-110, 9),
        (-120, 11),
        (float('-inf'), 12)
    ]

    def __init__(
            self,
            position: Tuple[float, float],
            sensor_id: int = 0,
            max_buffer_size: float = 1000.0,
            data_generation_rate: float = 10.0,
            packet_size: int = 50,
            transmit_power_dbm: float = 14.0,
            path_loss_exponent: float = 2.0,
            d0: float = 1.0,
            rssi_threshold: float = -120.0,
            noise_floor_dbm: float = -130.0,
            uav_altitude: int = 100,
            spreading_factor: int = 7,
            duty_cycle: float = 1.0  # NEW: Duty cycle percentage (1-100%)
    ):
        """
        Initialize IoT Sensor with duty cycling.

        Args:
            position: Sensor (x, y) position
            sensor_id: Unique identifier for the sensor
            max_buffer_size: Maximum buffer capacity (bytes)
            data_generation_rate: Data generation rate (bytes/sec) WHEN ACTIVE
            packet_size: Size of each transmission packet (bytes)
            transmit_power_dbm: Transmission power (dBm)
            path_loss_exponent: Path loss exponent (n)
            d0: Reference distance (meters)
            rssi_threshold: Minimum RSSI for communication (dBm)
            noise_floor_dbm: Ambient noise floor (dBm)
            uav_altitude: UAV altitude above ground (meters)
            spreading_factor: Initial LoRa spreading factor (7-12)
            duty_cycle: Duty cycle percentage (1-100%)
                1.0 = 1% (EU regulations: 1 sec on, 99 sec off)
                5.0 = 5% (5 sec on, 95 sec off)
                100.0 = always on (unrealistic)
        """
        self.position = np.array(position, dtype=np.float32)
        self.sensor_id = sensor_id
        self.max_buffer_size = max_buffer_size
        self.data_generation_rate = data_generation_rate
        self.packet_size = packet_size
        self.transmit_power_dbm = transmit_power_dbm
        self.path_loss_exponent = path_loss_exponent
        self.d0 = d0
        self.rssi_threshold = rssi_threshold
        self.noise_floor_dbm = noise_floor_dbm
        self.uav_altitude = uav_altitude

        # State variables
        self.data_buffer = 0.0
        self.spreading_factor = spreading_factor
        self.data_rate = self.LORA_DATA_RATES[self.spreading_factor]

        # ============ NEW: Duty Cycle ============
        self.duty_cycle = duty_cycle  # Percentage (1-100)
        self.cycle_length = int(100 / duty_cycle)  # Total cycle steps
        self.active_duration = int(self.cycle_length * (duty_cycle / 100))  # Steps active
        self.current_cycle_step = 0  # Current step in cycle (0 to cycle_length-1)
        self.is_active = False  # Is sensor currently generating data?

        # Tracking metrics
        self.total_data_generated = 0.0
        self.total_data_lost = 0.0
        self.total_data_transmitted = 0.0
        self.transmission_count = 0
        self.successful_transmissions = 0
        self.data_collected = False

    # ============ NEW: Duty Cycle Methods ============

    def update_duty_cycle(self):
        """Update duty cycle state (call once per step)."""
        # Determine if active or sleeping
        self.is_active = (self.current_cycle_step < self.active_duration)

        # Advance cycle counter
        self.current_cycle_step += 1
        if self.current_cycle_step >= self.cycle_length:
            self.current_cycle_step = 0  # Reset to beginning of cycle

    def get_duty_cycle_info(self) -> dict:
        """Get current duty cycle status."""
        return {
            "is_active": self.is_active,
            "current_cycle_step": self.current_cycle_step,
            "cycle_length": self.cycle_length,
            "active_duration": self.active_duration,
            "duty_cycle_percent": self.duty_cycle,
            "steps_until_active": (
                self.active_duration - self.current_cycle_step
                if self.current_cycle_step >= self.active_duration
                else 0
            )
        }

    # ============ Modified: Data Generation with Duty Cycle ============

    def step(self, time_step: float = 1.0) -> float:
        """
        Simulate one time step of data generation and buffer management.
        NOW: Only generates data during ACTIVE periods (duty cycle aware)

        Returns:
            data_loss: Amount of data lost due to buffer overflow.
        """
        # Update duty cycle state first
        self.update_duty_cycle()

        # Calculate potential new data ONLY if active
        if self.is_active:
            new_data = self.data_generation_rate * time_step
        else:
            new_data = 0.0  # No data generation during sleep

        self.total_data_generated += new_data

        # Calculate overflow
        potential_buffer = self.data_buffer + new_data
        data_loss = max(0.0, potential_buffer - self.max_buffer_size)

        # Update buffer
        self.data_buffer = min(potential_buffer, self.max_buffer_size)

        # Track total data lost
        self.total_data_lost += data_loss

        return data_loss

    # ============ Original Methods (unchanged) ============

    def get_transmission_energy_cost(
            self,
            power_tx_watts: float,
            packet_size_bytes: Optional[int] = None
    ) -> float:
        """Calculate the energy consumed by the sensor to transmit one packet."""
        if packet_size_bytes is None:
            packet_size_bytes = self.packet_size

        transmission_time = packet_size_bytes / self.data_rate
        energy_cost = power_tx_watts * transmission_time
        return energy_cost

    def get_transmission_time(self, packet_size_bytes: Optional[int] = None) -> float:
        """Get time-on-air for a packet (in seconds)."""
        if packet_size_bytes is None:
            packet_size_bytes = self.packet_size
        return packet_size_bytes / self.data_rate

    def transmit_data(self, amount: float) -> float:
        """Transmit data from buffer."""
        transmitted = min(amount, self.data_buffer)
        self.data_buffer -= transmitted
        self.total_data_transmitted += transmitted
        self.transmission_count += 1
        return transmitted

    def collect_data(self, uav_position: Tuple[float, float], collection_duration: float = 1.0) -> Tuple[float, bool]:
        """Collect data from sensor buffer (only if in range and has data)."""
        success_prob = self.get_success_probability(uav_position, use_advanced_model=False)
        in_range = success_prob > 0.5

        if not in_range:
            return 0.0, False

        # NEW: Can only collect if sensor has data (was active at some point)
        if self.data_buffer <= 0:
            return 0.0, True  # In range but nothing to collect

        max_collectible = self.data_rate * collection_duration
        bytes_collected = min(self.data_buffer, max_collectible)

        self.data_buffer -= bytes_collected
        self.total_data_transmitted += bytes_collected
        self.transmission_count += 1

        if bytes_collected > 0:
            self.successful_transmissions += 1
            self.data_collected = True

        return bytes_collected, True

    def get_buffer_status(self) -> dict:
        """Return buffer and data loss statistics."""
        return {
            "buffer_used": self.data_buffer,
            "buffer_capacity": self.max_buffer_size,
            "buffer_utilization": self.data_buffer / self.max_buffer_size,
            "total_data_generated": self.total_data_generated,
            "total_data_lost": self.total_data_lost,
            "total_data_transmitted": self.total_data_transmitted,
            "loss_rate": self.total_data_lost / max(1, self.total_data_generated),
            "is_active": self.is_active,  # NEW
            "duty_cycle": self.duty_cycle  # NEW
        }

    def calculate_rssi(self, uav_position: Tuple[float, float]) -> float:
        """Two-Ray Ground Reflection Model"""
        uav_altitude = self.uav_altitude
        sensor_pos = np.array(self.position, dtype=np.float32)
        uav_pos = np.array(uav_position, dtype=np.float32)

        dx = uav_pos[0] - sensor_pos[0]
        dy = uav_pos[1] - sensor_pos[1]
        ground_distance = np.sqrt(dx ** 2 + dy ** 2)
        distance_3d = np.sqrt(ground_distance ** 2 + uav_altitude ** 2)

        h_sensor = 1.0
        h_uav = uav_altitude
        f = 868e6
        c = 3e8
        wavelength = c / f
        critical_distance = (4 * h_sensor * h_uav) / wavelength

        if distance_3d < critical_distance:
            path_loss = 10 * self.path_loss_exponent * np.log10(distance_3d / self.d0)
        else:
            path_loss = 40 * np.log10(distance_3d) - 10 * np.log10(h_sensor ** 2 * h_uav ** 2)

        rssi = self.transmit_power_dbm - path_loss
        return float(rssi)

    def _get_required_snr(self, spreading_factor: int) -> float:
        """Get required SNR (dB) for a given spreading factor."""
        return self.REQUIRED_SNR_DB.get(spreading_factor, 7.5)

    def get_success_probability(
            self,
            uav_position: Tuple[float, float],
            use_advanced_model: bool = False
    ) -> float:
        """Calculate probability of successful reception based on link quality."""
        rssi = self.calculate_rssi(uav_position)

        if rssi < self.rssi_threshold:
            return 0.0

        if use_advanced_model:
            snr_db = rssi - self.noise_floor_dbm
            required_snr_db = self._get_required_snr(self.spreading_factor)
            sigmoid = 1.0 / (1.0 + np.exp(-(snr_db - required_snr_db)))
            return sigmoid

        return 1.0

    def is_in_range(self, uav_position: Tuple[float, float]) -> bool:
        """Check if UAV is in communication range."""
        return self.get_success_probability(uav_position) > 0.5

    def update_spreading_factor(self, uav_position: Tuple[float, float]):
        """Dynamically update spreading factor based on RSSI."""
        rssi = self.calculate_rssi(uav_position)

        for threshold, sf in self.RSSI_SF_MAPPING:
            if rssi > threshold:
                self.spreading_factor = sf
                break

        self.data_rate = self.LORA_DATA_RATES[self.spreading_factor]

    def get_link_quality_report(self, uav_position: Tuple[float, float]) -> dict:
        """Generate comprehensive link quality report."""
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
            "is_active": self.is_active,  # NEW
            "duty_cycle": self.duty_cycle  # NEW
        }

    def reset(self):
        """Reset sensor to initial state."""
        self.data_buffer = 0.0
        self.spreading_factor = 7
        self.data_rate = self.LORA_DATA_RATES[self.spreading_factor]

        self.total_data_generated = 0.0
        self.total_data_lost = 0.0
        self.total_data_transmitted = 0.0
        self.transmission_count = 0
        self.successful_transmissions = 0
        self.data_collected = False

        # Reset duty cycle to beginning
        self.current_cycle_step = 0
        self.is_active = False

    def reset_stats(self):
        """Reset all tracking statistics."""
        self.total_data_generated = 0.0
        self.total_data_lost = 0.0
        self.total_data_transmitted = 0.0
        self.transmission_count = 0
        self.successful_transmissions = 0
        self.data_buffer = 0.0

    def __repr__(self) -> str:
        """String representation of sensor."""
        status = "🟢 ACTIVE" if self.is_active else "🔴 SLEEP"
        return f"IoTSensor(id={self.sensor_id}, {status}, duty_cycle={self.duty_cycle}%, buffer={self.data_buffer:.0f}B)"


# ============ EXAMPLE USAGE ============

if __name__ == "__main__":
    print("=" * 80)
    print("Testing IoT Sensor with Duty Cycling")
    print("=" * 80)
    print()

    # Create sensors with different duty cycles
    sensor_1percent = IoTSensor(
        position=(0, 0),
        sensor_id=1,
        duty_cycle=1.0,  # 1% duty cycle (EU regulations)
        data_generation_rate=10.0,
    )

    sensor_10percent = IoTSensor(
        position=(10, 10),
        sensor_id=2,
        duty_cycle=10.0,  # 10% duty cycle
        data_generation_rate=10.0,
    )

    print("Simulating 120 steps with different duty cycles:")
    print()
    print("Step | Sensor 1% | Data1 | Sensor 10% | Data10")
    print("-" * 55)

    for step in range(120):
        sensor_1percent.step(time_step=1.0)
        sensor_10percent.step(time_step=1.0)

        if step % 10 == 0 or step < 20:
            status_1 = "🟢" if sensor_1percent.is_active else "🔴"
            status_10 = "🟢" if sensor_10percent.is_active else "🔴"
            print(
                f"{step:3d}  | {status_1} {sensor_1percent.current_cycle_step:3d}/{sensor_1percent.cycle_length:3d} "
                f"| {sensor_1percent.data_buffer:5.0f} | "
                f"{status_10} {sensor_10percent.current_cycle_step:3d}/{sensor_10percent.cycle_length:3d} "
                f"| {sensor_10percent.data_buffer:5.0f}"
            )

    print()
    print("=" * 80)
    print("KEY OBSERVATIONS:")
    print("=" * 80)
    print(f"Sensor 1% (EU regulation): Active for {sensor_1percent.active_duration} out of {sensor_1percent.cycle_length} steps")
    print(f"Sensor 10%: Active for {sensor_10percent.active_duration} out of {sensor_10percent.cycle_length} steps")
    print()
    print(f"Sensor 1% total data generated: {sensor_1percent.total_data_generated:.0f} bytes")
    print(f"Sensor 10% total data generated: {sensor_10percent.total_data_generated:.0f} bytes")
    print()
    print("Notice: 10% duty cycle generates ~10x more data")
    print("UAV must arrive during ACTIVE windows to collect data!")