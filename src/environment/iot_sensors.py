"""
IoT Sensor with EMA-based ADR and Gaussian Shadowing

UPDATED: Added Gaussian Random Variable for Shadowing
Formula: RSSI = RSS(d) + X_sigma
Where X_sigma ~ N(0, 4.0)

This stochasticity simulates the effect of trees, buildings, and
orientation changes, validating the need for the EMA smoothing algorithm.

Author: ATILADE GABRIEL OKE
Date: November 2025
"""

import numpy as np
from typing import Tuple, Optional


class IoTSensor:
    """Enhanced IoT Sensor with EMA-based Adaptive Data Rate and Stochastic Shadowing."""

    LORA_DATA_RATES = {
        7: 5470 / 8,
        8: 3125 / 8,
        9: 1760 / 8,
        10: 980 / 8,
        11: 440 / 8,
        12: 250 / 8
    }

    REQUIRED_SNR_DB = {
        7: 7.5,
        8: 10.0,
        9: 12.5,
        10: 15.0,
        11: 17.5,
        12: 20.0
    }

    # RSSI to SF mapping (Capped SF12 range)
    RSSI_SF_MAPPING = [
        (-55, 7),
        (-70, 9),
        (-90, 11),
        (-110, 12)
    ]

    def __init__(
            self,
            position: Tuple[float, float],
            sensor_id: int = 0,
            max_buffer_size: float = 1000.0,
            data_generation_rate: float = 10.0,
            packet_size: int = 50,
            transmit_power_dbm: float = 14.0,  # Updated to realistic 14dBm
            path_loss_exponent: float = 3.8,   # Urban Path Loss
            d0: float = 1.0,
            rssi_threshold: float = -120.0,
            noise_floor_dbm: float = -105.0,
            uav_altitude: int = 100,           # Fixed at 100m
            spreading_factor: int = 12,
            duty_cycle: float = 10.0,
            adr_lambda: float = 0.1,
            use_ema_adr: bool = True,
            shadowing_std_db: float = 4.0      # Shadowing Standard Deviation
    ):
        """
        Initialize IoT Sensor with Stochastic Shadowing.

        Args:
            shadowing_std_db: Standard deviation of Gaussian shadowing (dB).
                              User specified "disturbance of 4 dB".
                              We model this as X ~ N(0, 4.0).
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
        self.spreading_factor = spreading_factor
        self.duty_cycle = duty_cycle
        self.data_rate = self.LORA_DATA_RATES[self.spreading_factor]

        # EMA ADR
        self.use_ema_adr = use_ema_adr
        self.adr_lambda = adr_lambda
        self.avg_rssi = None
        self.current_rssi = None

        # Shadowing
        self.shadowing_std_db = shadowing_std_db

        # State tracking
        self.data_buffer = 0.0
        self.total_data_generated = 0.0
        self.total_data_transmitted = 0.0
        self.total_data_lost = 0.0
        self.transmission_count = 0
        self.successful_transmissions = 0
        self.data_collected = False

        # History
        self.sf_history = []
        self.rssi_history = []
        self.avg_rssi_history = []

    # [Properties and Step methods remain unchanged...]
    @property
    def duty_cycle_probability(self) -> float:
        return self.duty_cycle / 100.0

    @property
    def adr_latency_steps(self) -> int:
        if self.adr_lambda <= 0: return float('inf')
        return int(np.ceil(-1.0 / np.log(1 - self.adr_lambda)))

    def step(self, time_step: float = 1.0) -> float:
        new_data = self.data_generation_rate * time_step
        self.total_data_generated += new_data
        potential_buffer = self.data_buffer + new_data
        if potential_buffer > self.max_buffer_size:
            data_loss = potential_buffer - self.max_buffer_size
            self.data_buffer = self.max_buffer_size
            self.total_data_lost += data_loss
        else:
            data_loss = 0.0
            self.data_buffer = potential_buffer
        return data_loss

    def collect_data(self, uav_position: Tuple[float, float], collection_duration: float = 1.0) -> Tuple[float, bool]:
        # Use simple model for binary range check
        success_prob = self.get_success_probability(uav_position, use_advanced_model=False)
        if success_prob <= 0.5 or self.data_buffer <= 0:
            return 0.0, success_prob > 0.5

        max_collectible = self.data_rate * collection_duration
        bytes_collected = min(self.data_buffer, max_collectible)
        self.data_buffer -= bytes_collected
        self.total_data_transmitted += bytes_collected
        self.transmission_count += 1
        if bytes_collected > 0:
            self.successful_transmissions += 1
            self.data_collected = True
        return bytes_collected, True

    def calculate_rssi(self, uav_position: Tuple[float, float]) -> float:
        """
        Calculate RSSI with Gaussian Shadowing.

        Formula: RSSI = Ptx - PL(d) + X_sigma

        Where:
            Ptx = Transmit Power (14 dBm)
            PL(d) = 20 * n * log10(d)  (Log-Distance Path Loss)
            X_sigma ~ N(0, 4.0)        (Log-Normal Shadowing)

        Returns:
            RSSI in dBm (including random noise)
        """
        sensor_pos = np.array(self.position, dtype=np.float32)
        uav_pos = np.array(uav_position, dtype=np.float32)

        dx = (uav_pos[0] - sensor_pos[0]) * 10
        dy = (uav_pos[1] - sensor_pos[1]) * 10
        ground_distance = np.sqrt(dx ** 2 + dy ** 2)
        distance_3d = np.sqrt(ground_distance ** 2 + self.uav_altitude ** 2)

        # Thesis Equation 1 & 8 Implementation
        ht = 0.5  # Sensor height (meters)
        hr = self.uav_altitude  # UAV height (meters)
        d = distance_3d

        # 1. Calculate Breakpoint Distance
        # (4 * pi * ht * hr) / lambda, assuming 868MHz (lambda approx 0.345m)
        d_break = (4 * np.pi * ht * hr) / 0.345

        # Two-Ray Ground Reflection
        if d < d_break:
            # standard free space path loss (d^2)
            path_loss = (20 * (np.log10(d))) + (20 * (np.log10(868))) - 28
        else:
            # Two-Ray Ground Reflection (d^4) - The Thesis Claim
            # L = 40log(d) - 20log(ht) - 20log(hr)
            path_loss = (40 * np.log10(d)) - (20 * np.log10(ht)) - (20 * np.log10(hr))

        # Base RSSI (Deterministic)
        rssi_deterministic = self.transmit_power_dbm - path_loss

        # Add Gaussian Shadowing (Stochastic)
        # "Disturbance of 4 dB, zero mean"
        shadowing = np.random.normal(0, self.shadowing_std_db)

        # Final RSSI
        rssi_total = rssi_deterministic + shadowing

        return float(rssi_total)

    def _get_required_snr(self, spreading_factor: int) -> float:
        return self.REQUIRED_SNR_DB.get(spreading_factor, 7.5)

    def get_success_probability(self, uav_position: Tuple[float, float], use_advanced_model: bool = False) -> float:
        rssi = self.calculate_rssi(uav_position)
        if rssi < self.rssi_threshold:
            return 0.0
        if use_advanced_model:
            snr_db = rssi - self.noise_floor_dbm
            required_snr_db = self._get_required_snr(self.spreading_factor)
            return 1.0 / (1.0 + np.exp(-(snr_db - required_snr_db)))
        return 1.0

    def is_in_range(self, uav_position: Tuple[float, float]) -> bool:
        rssi = self.calculate_rssi(uav_position)

        # 2. Check against the hardware sensitivity floor
        # This matches the definition of 'Coverage' in your Thesis Section 4.1.2
        return rssi >= self.rssi_threshold

    # ============ ADR WITH NOISE FILTERING ============

    def update_spreading_factor(self, uav_position: Tuple[float, float], current_step: int = 0):
        """
        Update SF using noisy RSSI and EMA smoothing.

        The Gaussian noise added in calculate_rssi() simulates real-world
        signal fluctuation. The EMA filter smooths this noise to prevent
        erratic SF switching (ping-pong effect).
        """
        # This call now includes random shadowing noise!
        current_rssi = self.calculate_rssi(uav_position)
        self.current_rssi = current_rssi
        self.rssi_history.append((current_step, current_rssi))

        # EMA Smoothing
        if self.avg_rssi is None:
            self.avg_rssi = current_rssi
        elif self.use_ema_adr:
            self.avg_rssi = (self.adr_lambda * current_rssi) + ((1 - self.adr_lambda) * self.avg_rssi)
        else:
            self.avg_rssi = current_rssi

        self.avg_rssi_history.append((current_step, self.avg_rssi))

        # Select SF based on SMOOTHED RSSI
        old_sf = self.spreading_factor
        for threshold, sf in self.RSSI_SF_MAPPING:
            if self.avg_rssi > threshold:
                self.spreading_factor = sf
                break

        self.data_rate = self.LORA_DATA_RATES[self.spreading_factor]
        if self.spreading_factor != old_sf:
            self.sf_history.append((current_step, self.spreading_factor))

    # [Remaining reporting methods unchanged...]
    def get_buffer_status(self) -> dict:
        return {
            "buffer_used": self.data_buffer,
            "buffer_capacity": self.max_buffer_size,
            "loss_rate": self.total_data_lost / max(1, self.total_data_generated)
        }

    def get_sf_distribution(self) -> dict:
        sf_counts = {sf: 0 for sf in range(7, 13)}
        for _, sf in self.sf_history: sf_counts[sf] += 1
        return sf_counts

    def get_sf_histogram(self) -> str:
        dist = self.get_sf_distribution()
        max_count = max(dist.values()) if max(dist.values()) > 0 else 1
        hist = "SF Distribution:\n"
        for sf in range(7, 13):
            bar = "█" * int((dist[sf] / max_count) * 40)
            hist += f"  SF{sf:2d}: {bar:<40} ({dist[sf]:3d})\n"
        return hist

    def get_sf_changes(self) -> int:
        if len(self.sf_history) <= 1: return 0
        return sum(1 for i in range(1, len(self.sf_history)) if self.sf_history[i][1] != self.sf_history[i-1][1])

    def get_adr_performance(self) -> dict:
        if not self.rssi_history: return {}
        rssi_vals = [r for _, r in self.rssi_history]
        avg_vals = [a for _, a in self.avg_rssi_history]
        return {
            'rssi_std': np.std(rssi_vals),
            'avg_std': np.std(avg_vals),
            'smoothing_gain': np.std(rssi_vals) - np.std(avg_vals),
            'sf_changes': self.get_sf_changes()
        }

    def reset(self):
        self.data_buffer = 0.0
        self.spreading_factor = 12
        self.data_rate = self.LORA_DATA_RATES[12]
        self.avg_rssi = None
        self.current_rssi = None
        self.total_data_generated = 0.0
        self.total_data_transmitted = 0.0
        self.total_data_lost = 0.0
        self.sf_history = []
        self.rssi_history = []
        self.avg_rssi_history = []

    def get_transmission_time(self, packet_size_bytes: int = 50) -> float:
        return packet_size_bytes / self.data_rate

    def get_transmission_energy_cost(self, power_tx: float) -> float:
        return power_tx * (self.packet_size / self.data_rate)


# ============ TESTING ============

if __name__ == "__main__":
    print("=" * 100)
    print("IoT SENSOR WITH EMA-BASED ADR (EQUATION 18)")
    print("=" * 100)
    print()

    # Test 1: Create sensor with EMA
    print("Test 1: Creating IoT Sensor with EMA-based ADR")
    print("-" * 100)
    sensor = IoTSensor(
        position=(25.0, 25.0),
        sensor_id=1,
        max_buffer_size=1000.0,
        data_generation_rate=10.0,
        duty_cycle=10.0,
        transmit_power_dbm=0.0,
        uav_altitude=100,
        adr_lambda=0.1,
        use_ema_adr=True
    )

    print(f"  ADR Lambda: {sensor.adr_lambda}")
    print(f"  ADR Latency (time constant): ~{sensor.adr_latency_steps} steps")
    print()

    # Test 2: Simulate approaching UAV with EMA lag
    print("Test 2: Simulating Drone Approach with EMA ADR Lag")
    print("-" * 100)
    print(f"{'Step':<6} {'Position':<15} {'Current RSSI':<15} {'Avg RSSI':<15} {'SF':<5}")
    print("-" * 100)

    uav_start = (50.0, 25.0)  # Far away
    for step in range(15):
        # Move UAV closer
        progress = step / 14.0
        uav_x = 50.0 - (25.0 * progress)  # Move from x=50 to x=25
        uav_pos = (uav_x, 25.0)

        # Update ADR
        sensor.update_spreading_factor(uav_pos, current_step=step)

        # Generate data
        sensor.step(time_step=1.0)

        print(f"{step:<6} ({uav_x:>5.1f}, 25.0) {sensor.current_rssi:>14.1f} dBm {sensor.avg_rssi:>14.1f} dBm {sensor.spreading_factor:<5}")

    print()

    # Test 3: SF Distribution
    print("Test 3: SF Distribution (Shows ADR Lag Effect)")
    print("-" * 100)
    print(sensor.get_sf_histogram())
    print()

    # Test 4: ADR Performance
    print("Test 4: ADR Performance Metrics")
    print("-" * 100)
    adr_perf = sensor.get_adr_performance()
    for key, value in adr_perf.items():
        print(f"  {key}: {value:.2f}")
    print()

    # Test 5: Comparison (instantaneous vs EMA)
    print("Test 5: Instantaneous vs EMA Comparison")
    print("-" * 100)

    sensor_inst = IoTSensor(
        position=(25.0, 25.0),
        sensor_id=2,
        adr_lambda=0.1,
        use_ema_adr=False  # Instantaneous
    )

    sensor_ema = IoTSensor(
        position=(25.0, 25.0),
        sensor_id=3,
        adr_lambda=0.1,
        use_ema_adr=True  # EMA
    )

    print(f"{'Step':<6} {'Instantaneous SF':<20} {'EMA SF':<20}")
    print("-" * 100)

    for step in range(10):
        progress = step / 9.0
        uav_x = 50.0 - (25.0 * progress)
        uav_pos = (uav_x, 25.0)

        sensor_inst.update_spreading_factor(uav_pos, current_step=step)
        sensor_ema.update_spreading_factor(uav_pos, current_step=step)

        print(f"{step:<6} SF{sensor_inst.spreading_factor:<18} SF{sensor_ema.spreading_factor:<18}")

    print()

    print("=" * 100)
    print("KEY INSIGHT: EMA Creates ADR Lag")
    print("=" * 100)
    print(f"Instantaneous: Switches immediately when distance threshold crossed")
    print(f"EMA (λ=0.1):  Takes ~{sensor_ema.adr_latency_steps} steps to reach 63% convergence")
    print(f"→ Forces DQN to learn HOVERING strategies (not just passing by)")
    print(f"→ Frame stacking becomes valuable for predicting future RSSI trends")
    print("=" * 100)