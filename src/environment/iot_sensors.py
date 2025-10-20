"""
IoT Sensor Node with Data Generation and LoRa Transmission Model

Models realistic sensor behavior including:
- Data generation over time
- Data buffer management
- LoRa communication with d^-2 path loss
- Transmission time calculation

Key equations implementation:
        >>> Path loss: PL(d) = 10*n*log10(d/d_0)
        >>> path_loss = 10 * self.path_loss_exponent * np.log10(distance / self.reference_distance)
        >>> rssi = self.rssi_reference - path_loss

Author: ATILADE GABRIEL OKE
Date: October 2025
Project: Reinforcement Learning for Dynamic UAV Energy-Efficient Path Planning
         in IoT Sensor Networks
"""

from typing import Tuple, Optional, List
import numpy as np
import logging

from triton.experimental.gluon.language.nvidia.hopper.mbarrier import expect


class IoTSensor:
    """
    IoT sensor node with realistic data generation and LoRa communication.

    The sensor generates data at regular intervals and stores it in a buffer.
    Data transmission via LoRa depends on signal strength (d^-2 path loss model)
    and available data rate.

    Attributes:
        sensor_id (int): Unique sensor identifier
        position (Tuple[float, float]): (x, y) coordinates
        data_buffer (float): Amount of data waiting to be collected (bytes)
        max_buffer_size (float): Maximum buffer capacity (bytes)
        data_generation_rate (float): Data generated per time step (bytes/step)
        packet_size (int): Size of each LoRa packet (bytes)

        # LoRa communication parameters
        rssi_reference (float): Reference signal strength (dBm)
        path_loss_exponent (float): Path loss exponent (2.0 for d^-2)
        rssi_threshold (float): Minimum RSSI for communication (dBm)
        spreading_factor (int): LoRa spreading factor (7-12)
        data_rate (float): LoRa data rate (bytes/second)
    """

    # LoRa data rates based on spreading factor (bytes per second)
    LORA_DATA_RATES = {
        7: 5470 / 8,   # ~684 bytes/sec       (SF7, highest speed)
        8: 3125 / 8,   # ~390 bytes/sec   ↓
        9: 1760 / 8,   # ~220 bytes/sec   ↓
        10: 980 / 8,   # ~122 bytes/sec   ↓
        11: 440 / 8,   # ~55 bytes/sec    ↓
        12: 250 / 8    # ~31 bytes/sec    ↓   (SF12, maximum range)
    }

    def __init__(self,
                 sensor_id: int,
                 position: Tuple[float, float],
                 # Data generation parameters
                 data_generation_rate: float = 0.073,  # bytes per time step
                 max_buffer_size: float = 10000.0,  # bytes
                 packet_size: int = 51,  # bytes per LoRa packet
                 # LoRa communication parameters
                 rssi_reference: float = -60.0,  # dBm at 1m
                 path_loss_exponent: float = 2.0,  # d^-2 for free space
                 reference_distance: float = 1,  # meters
                 rssi_threshold: float = -120.0,  # dBm
                 spreading_factor: int = 7):  # SF7-SF12
        """
        Initialize IoT sensor with data generation and LoRa parameters.

        Args:
            sensor_id: Unique identifier
            position: (x, y) coordinates in grid
            data_generation_rate: Data generated per time step (bytes/step)
                                 Default: 0.073 bytes/step ≈ 22 bytes/5min
            max_buffer_size: Maximum data buffer capacity (bytes)
            packet_size: LoRa packet payload size (bytes)
            rssi_reference: Signal strength at reference distance (dBm)
            path_loss_exponent: n in d^-n model (2.0 for free space)
            reference_distance: Reference distance d_0
            rssi_threshold: Minimum RSSI for communication (dBm)
            spreading_factor: LoRa SF (7=fast/short, 12=slow/long)

        Example:
            >>> # Environmental sensor: 22 bytes every 5 minutes
            >>> sensor = IoTSensor(
            ...     sensor_id=1,
            ...     position=(5.0, 5.0),
            ...     data_generation_rate=22.0 / 300,  # 22 bytes per 300 steps
            ...     spreading_factor=7
            ... )
        """
        self.sensor_id = sensor_id
        self.position = np.array(position, dtype=np.float32)

        # Data generation
        self.data_generation_rate = data_generation_rate
        self.max_buffer_size = max_buffer_size
        self.data_buffer = 0.0  # Current data in buffer (bytes)
        self.total_data_generated = 0.0  # Total data generated (for stats)
        self.data_collected = False  # Whether UAV has visited

        # LoRa transmission
        self.packet_size = packet_size
        self.spreading_factor = spreading_factor
        self.data_rate = self.LORA_DATA_RATES[spreading_factor]  # bytes/sec

        # LoRa signal propagation
        self.rssi_reference = rssi_reference
        self.path_loss_exponent = path_loss_exponent
        self.reference_distance = reference_distance
        self.rssi_threshold = rssi_threshold

    def step(self, time_step: float = 1.0) -> None:
        """
        Simulate one time step of sensor operation.

        Generates new data and adds to buffer (if not full).

        Args:
            time_step: Duration of this step (default: 1 time unit)

        Example:
            >>> sensor = IoTSensor(1, (5.0, 5.0), data_generation_rate=0.1)
            >>> sensor.step()
            >>> print(f"Buffer: {sensor.data_buffer:.2f} bytes")
            Buffer: 0.10 bytes
        """
        # Generate new data
        new_data = self.data_generation_rate * time_step
        self.total_data_generated += new_data

        # Add to buffer (respect max capacity)
        self.data_buffer = min(self.data_buffer + new_data, self.max_buffer_size)

        # If buffer is full, data is lost (overflow)
        if self.data_buffer >= self.max_buffer_size:
            # In real systems, this would be data loss
            self.max_buffer_size  # TODO : find out actual data loss rate


    def calculate_rssi(self, uav_position: Tuple[float, float]) -> float:
        """
        Calculate received signal strength (RSSI) at UAV position.

        Uses path loss model: RSSI(d) = RSSI_0 - 10*n*log10(d/d_0)
        where n is the path loss exponent (2 for d^-2 free space propagation).

        Args:
            uav_position: (x, y) coordinates of UAV

        Returns:
            Received signal strength in dBm
        """
        uav_pos = np.array(uav_position, dtype=np.float32)
        distance = np.linalg.norm(self.position - uav_pos)

        # Avoid log(0) for very close distances
        if distance < 0.01:
            distance = 0.01

        # Path loss: PL(d) = 10*n*log10(d/d_0)
        path_loss = 10 * self.path_loss_exponent * np.log10(distance / self.reference_distance)
        rssi = self.rssi_reference - path_loss

        return rssi

    def is_in_range(self, uav_position: Tuple[float, float]) -> bool:
        """
        Check if UAV is within LoRa communication range.

        Communication is possible when RSSI is above threshold.
        Uses d^-2 path loss model.

        Args:
            uav_position: (x, y) coordinates of UAV

        Returns:
            True if communication is possible, False otherwise
        """
        rssi = self.calculate_rssi(uav_position)
        return rssi >= self.rssi_threshold

    def calculate_transmission_time(self, data_amount: float) -> float:
        """
        Calculate time required to transmit data via LoRa.

        Time = data_amount / data_rate

        Args:
            data_amount: Amount of data to transmit (bytes)

        Returns:
            Transmission time (seconds or time steps)

        Example:
            >>> sensor = IoTSensor(1, (5.0, 5.0), spreading_factor=7)
            >>> time = sensor.calculate_transmission_time(100)  # 100 bytes
            >>> print(f"Transmission time: {time:.3f} seconds")
            Transmission time: 0.146 seconds
        """
        if data_amount <= 0:
            return 0.0
        return data_amount / self.data_rate

    def collect_data(self,
                     uav_position: Tuple[float, float],
                     collection_duration: float = 1.0) -> Tuple[float, bool]:
        """
        Attempt to collect data from sensor via LoRa.

        Collection succeeds only if:
        1. UAV is in range (RSSI >= threshold)
        2. There is data in the buffer

        Amount collected depends on:
        - Available data in buffer
        - LoRa data rate
        - Collection duration

        Args:
            uav_position: (x, y) coordinates of UAV
            collection_duration: Time spent collecting (time steps/seconds)

        Returns:
            Tuple of (bytes_collected, success)
            - bytes_collected: Amount of data transferred (bytes)
            - success: True if any data was collected

        Example:
            >>> sensor = IoTSensor(1, (5.0, 5.0))
            >>> sensor.data_buffer = 100.0  # 100 bytes waiting
            >>> bytes_collected, success = sensor.collect_data((5.0, 5.0), duration=1.0)
            >>> print(f"Collected {bytes_collected:.2f} bytes")
        """
        # Check if in range
        if not self.is_in_range(uav_position):
            return 0.0, False

        # Check if buffer has data
        if self.data_buffer <= 0:
            return 0.0, False

        # Calculate how much can be transmitted in the given duration
        max_transferable = self.data_rate * collection_duration

        # Transfer the minimum of: available data, or max transferable
        bytes_collected = min(self.data_buffer, max_transferable)

        # Remove from buffer
        self.data_buffer -= bytes_collected

        # Mark as collected if buffer is now empty
        if self.data_buffer <= 0:
            self.data_collected = True

        return bytes_collected, True

    def get_buffer_status(self) -> dict:
        """
        Get current buffer status information.

        Returns:
            Dictionary with buffer statistics
        """
        return {
            'buffer_bytes': self.data_buffer,
            'buffer_percent': (self.data_buffer / self.max_buffer_size) * 100,
            'total_generated': self.total_data_generated,
            'is_full': self.data_buffer >= self.max_buffer_size,
            'is_empty': self.data_buffer <= 0
        }

    def reset(self) -> None:
        """
        Reset sensor state for new episode.

        Clears buffer and resets collection status.
        Note: Does NOT reset total_data_generated (for statistics).
        """
        self.data_buffer = 0.0
        self.data_collected = False

    @property
    def buffer_utilization(self) -> float:
        """Get buffer utilization as percentage (0-100)."""
        return (self.data_buffer / self.max_buffer_size) * 100

    @property
    def has_data(self) -> bool:
        """Check if sensor has data waiting to be collected."""
        return self.data_buffer > 0

    def __repr__(self) -> str:
        """String representation of the sensor."""
        return (f"IoTSensor(id={self.sensor_id}, "
                f"position={Tuple(self.position)}, "
                f"buffer={self.data_buffer:.1f}B, "
                f"SF={self.spreading_factor}, "
                f"data_rate={self.data_rate:.1f}B/s)")

