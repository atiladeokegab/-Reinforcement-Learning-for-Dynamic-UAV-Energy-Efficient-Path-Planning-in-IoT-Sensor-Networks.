"""
Gymnasium-Compatible UAV Environment for Q-Learning

Environment for training UAV to collect data from IoT sensors using Q-Learning
with realistic LoRa communication and energy constraints.

State Space:
    - UAV position (x, y)
    - Battery level
    - Sensor buffer states (for each sensor)

Action Space:
    0: UP
    1: DOWN
    2: LEFT
    3: RIGHT
    4: COLLECT (hover and collect data from nearby sensors)

Episode Termination:
    - Success: All sensors have empty buffers (data collected)
    - Failure: Battery depleted (battery <= 0)
    - Timeout: Maximum steps reached

Author: ATILADE GABRIEL OKE
Date: October 2025
Project: Reinforcement Learning for Dynamic UAV Energy-Efficient Path Planning
         in IoT Sensor Networks
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch, Circle, Wedge
import sys
from pathlib import Path
import time
from enum import IntEnum

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Absolute imports
from environment.iot_sensors import IoTSensor
from environment.uav import UAV
from rewards.reward_function import RewardFunction


class SensorState(IntEnum):
    """Visual states for sensors based on buffer level"""
    EMPTY = 0           # 0% buffer
    LOW = 1             # 1-33% buffer
    MEDIUM = 2          # 34-66% buffer
    HIGH = 3            # 67-99% buffer
    FULL = 4            # 100% buffer
    COLLECTING = 5      # UAV actively collecting
    COLLECTED = 6       # Recently emptied


def get_sensor_visual_state(sensor,
                            uav_position,
                            current_action=None) -> SensorState:
    """
    Determine visual state of sensor based on buffer level and UAV interaction.

    Args:
        sensor: IoTSensor object
        uav_position: Current UAV position
        current_action: Current action (0-4, where 4=COLLECT)

    Returns:
        SensorState enum value
    """
    # Use sensor's own range calculation (RSSI-based)
    is_in_range = sensor.is_in_range(uav_position)

    # Determine state based on buffer percentage
    buffer_pct = (sensor.data_buffer / sensor.max_buffer_size) * 100

    # Priority 1: COLLECTING - UAV actively collecting (action=4, in range, has data)
    if current_action == 4 and is_in_range and sensor.data_buffer > 0:
        return SensorState.COLLECTING

    # Priority 2: COLLECTED - Recently emptied (empty buffer, in range, was collected)
    if sensor.data_buffer <= 10 and sensor.data_collected and is_in_range:
        return SensorState.COLLECTED

    # Priority 3: States based on buffer level
    if buffer_pct == 0:
        return SensorState.EMPTY
    elif buffer_pct <= 33:
        return SensorState.LOW
    elif buffer_pct <= 66:
        return SensorState.MEDIUM
    elif buffer_pct < 100:
        return SensorState.HIGH
    else:
        return SensorState.FULL


def render_sensor_enhanced(ax, sensor, current_step, uav_position,
                          current_action=None):
    """
    Render a single sensor with enhanced visual states.

    Args:
        ax: Matplotlib axis
        sensor: IoTSensor object
        current_step: Current simulation step (for animations)
        uav_position: UAV position tuple (x, y)
        current_action: Current action (0-4)
    """
    x, y = sensor.position

    # Get sensor state (now uses sensor's own range calculation)
    state = get_sensor_visual_state(sensor, uav_position, current_action)

    # Define visual properties for each state
    if state == SensorState.EMPTY:
        color = 'lightblue'
        marker_size = 80
        alpha = 0.5
        marker = 'o'
    elif state == SensorState.LOW:
        color = 'yellow'
        marker_size = 120
        alpha = 0.7
        marker = 'o'
    elif state == SensorState.MEDIUM:
        color = 'yellow'
        marker_size = 160
        alpha = 0.8
        marker = 'o'
    elif state == SensorState.HIGH:
        color = 'yellow'
        marker_size = 200
        alpha = 0.9
        marker = 'o'
    elif state == SensorState.FULL:
        color = 'blue'
        # Pulsing effect for full sensors
        pulse = 0.5 + 0.5 * np.sin(current_step * 0.3)
        marker_size = 250 * (0.9 + 0.2 * pulse)
        alpha = 0.7 + 0.2 * pulse
        marker = 'o'
    elif state == SensorState.COLLECTING:
        color = 'purple'
        marker_size = 300
        alpha = 1.0
        marker = '*'
    elif state == SensorState.COLLECTED:
        color = 'green'
        marker_size = 100
        alpha = 0.5
        marker = 'o'
    else:
        color = 'gray'
        marker_size = 80
        alpha = 0.5
        marker = 'o'

    # Draw sensor
    ax.scatter(x, y,
               c=color,
               marker=marker,
               s=marker_size,
               alpha=alpha,
               edgecolors='black',
               linewidths=2,
               zorder=5)

    # Add buffer percentage text
    buffer_pct = (sensor.data_buffer / sensor.max_buffer_size) * 100
    text_color = 'white' if state in [SensorState.FULL, SensorState.COLLECTING] else 'black'

    ax.text(x, y - 0.4,
            f'{int(buffer_pct)}%',
            ha='center',
            va='top',
            fontsize=7,
            fontweight='bold',
            color=text_color,
            bbox=dict(boxstyle='round,pad=0.2',
                      facecolor='white',
                      edgecolor='black',
                      alpha=0.7))

    # Add sensor ID
    ax.text(x, y + 0.5,
            f'S{sensor.sensor_id}',
            ha='center',
            va='bottom',
            fontsize=7,
            fontweight='bold')

    # Show collection progress animation for COLLECTING state
    if state == SensorState.COLLECTING:
        # Draw a small circle around sensor to show it's being collected
        collection_circle = Circle((x, y), 0.5,
                                   facecolor='none',
                                   edgecolor='purple',
                                   linewidth=3,
                                   linestyle='--',
                                   alpha=0.6,
                                   zorder=6)
        ax.add_patch(collection_circle)


def render_uav_enhanced(ax, uav):
    """
    Render the UAV with battery indicator.

    Args:
        ax: Matplotlib axis
        uav: UAV object
    """
    uav_x, uav_y = uav.position

    # Draw UAV body
    uav_marker = patches.FancyBboxPatch(
        (uav_x - 0.25, uav_y - 0.25), 0.5, 0.5,
        boxstyle="round,pad=0.05",
        edgecolor='red',
        facecolor='orange',
        linewidth=2.5,
        zorder=10
    )
    ax.add_patch(uav_marker)

    # UAV symbol
    ax.text(uav_x, uav_y, '✈',
            ha='center', va='center',
            fontweight='bold', fontsize=14,
            color='white',
            zorder=11)

    # Battery indicator below UAV
    battery_pct = uav.get_battery_percentage()
    if battery_pct > 50:
        battery_color = 'green'
    elif battery_pct > 25:
        battery_color = 'orange'
    else:
        battery_color = 'red'

    ax.text(uav_x, uav_y - 0.8,
            f'⚡{battery_pct:.0f}%',
            ha='center',
            va='top',
            fontsize=8,
            fontweight='bold',
            color=battery_color,
            bbox=dict(boxstyle='round,pad=0.3',
                      facecolor='white',
                      edgecolor=battery_color,
                      linewidth=2,
                      alpha=0.9),
            zorder=11)


class UAVEnvironment(gym.Env):
    """
    Custom Gymnasium Environment for UAV IoT Data Collection.

    The UAV must navigate a grid to collect data from IoT sensors.
    Sensors continuously generate data that must be collected before
    buffers overflow. LoRa communication uses d^-2 path loss model
    with probabilistic duty cycle constraints.

    Observation Space:
        Box: [uav_x, uav_y, battery, sensor1_buffer, ..., sensorN_buffer]

    Action Space:
        Discrete(5): [UP, DOWN, LEFT, RIGHT, COLLECT]

    Reward Structure:
        +10.0: Successful data collection from sensor
        +50.0: All sensors collected (mission complete)
        -2.0: Attempted collection from empty sensor
        -5.0: Boundary collision
        -0.1: Battery drain penalty
        -0.05: Step penalty (encourages efficiency)
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self,
                 grid_size: Tuple[int, int] = (10, 10),
                 sensor_positions: Optional[List[Tuple[float, float]]] = None,
                 num_sensors: int = 20,
                 # Sensor parameters
                 data_generation_rate: float = 22.0 / 10,
                 max_buffer_size: float = 1000.0,
                 lora_spreading_factor: int = 7,
                 path_loss_exponent: float = 2.0,
                 rssi_threshold: float = -90.0,
                 sensor_duty_cycle: float = 10.0,
                 # UAV parameters
                 uav_start_position: Optional[Tuple[float, float]] = None,
                 max_battery: float = 274.0,
                 collection_duration: float = 1.0,
                 # Episode parameters
                 max_steps: int = 300,
                 render_mode: Optional[str] = None):
        """
        Initialize UAV environment.

        Args:
            grid_size: (width, height) of grid world
            sensor_positions: List of (x, y) for sensors (or None for random)
            num_sensors: Number of sensors (if positions not specified)
            data_generation_rate: Sensor data generation (bytes/step)
            max_buffer_size: Maximum sensor buffer capacity (bytes)
            lora_spreading_factor: LoRa SF (7-12)
            path_loss_exponent: n in d^-n model (2.0 for free space)
            rssi_threshold: Minimum RSSI for communication (dBm)
            sensor_duty_cycle: Sensor duty cycle percentage (1-100)
                              1.0 = EU regulation (1%)
                              10.0 = 10% (recommended for testing)
                              100.0 = always active
            uav_start_position: UAV starting position (or None for center)
            max_battery: UAV battery capacity (Wh)
            collection_duration: Time UAV hovers to collect data (seconds)
            max_steps: Maximum steps per episode
            render_mode: 'human' or 'rgb_array'
        """
        super().__init__()

        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.collection_duration = collection_duration
        self.last_successful_collections = []
        self.last_action = None

        # Create sensors
        if sensor_positions is None:
            self.sensor_positions = self._generate_uniform_sensor_positions(num_sensors)
        else:
            self.sensor_positions = sensor_positions

        self.num_sensors = len(self.sensor_positions)

        # Initialize sensors
        self.sensors: List[IoTSensor] = []
        for i, pos in enumerate(self.sensor_positions):
            sensor = IoTSensor(
                sensor_id=i,
                position=pos,
                data_generation_rate=data_generation_rate,
                max_buffer_size=max_buffer_size,
                spreading_factor=lora_spreading_factor,
                path_loss_exponent=path_loss_exponent,
                rssi_threshold=rssi_threshold,
                duty_cycle=sensor_duty_cycle
            )
            self.sensors.append(sensor)

        # Initialize UAV
        if uav_start_position is None:
            uav_start_position = (0,0)

        self.uav = UAV(
            start_position=uav_start_position,
            max_battery=max_battery
        )

        # Initialize reward function
        self.reward_fn = RewardFunction()

        # Action space: UP, DOWN, LEFT, RIGHT, COLLECT
        self.action_space = spaces.Discrete(5)

        # Observation space: [x, y, battery, sensor1_buffer, ..., sensorN_buffer]
        obs_low = np.array([0, 0, 0] + [0] * self.num_sensors, dtype=np.float32)
        obs_high = np.array(
            [grid_size[0], grid_size[1], max_battery] +
            [max_buffer_size] * self.num_sensors,
            dtype=np.float32
        )
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        # Episode tracking
        self.current_step = 0
        self.total_reward = 0.0
        self.total_data_collected = 0.0
        self.sensors_visited = set()

        # Rendering
        self.fig = None
        self.ax = None

    def _generate_uniform_sensor_positions(self, num_sensors: int) -> List[Tuple[float, float]]:
        """
        Generate uniformly distributed sensor positions across the grid.

        Args:
            num_sensors: Number of sensors to place.

        Returns:
            List of (x, y) positions.
        """
        x_min, y_min = 0.0, 0.0
        x_max, y_max = float(self.grid_size[0]), float(self.grid_size[1])

        low_bounds = [x_min, y_min]
        high_bounds = [x_max, y_max]

        coordinates = np.random.uniform(low=low_bounds, high=high_bounds, size=(num_sensors, 2))
        positions = [(float(x), float(y)) for x, y in coordinates]

        return positions

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed for reproducibility
            options: Additional options (unused)

        Returns:
            observation: Initial observation
            info: Additional information dictionary
        """
        super().reset(seed=seed)

        # Reset UAV
        self.uav.reset()

        # Reset all sensors (no duty cycle state to randomize - it's probabilistic now)
        for sensor in self.sensors:
            sensor.reset()

        # Reset episode tracking
        self.current_step = 0
        self.total_reward = 0.0
        self.total_data_collected = 0.0
        self.sensors_visited = set()
        self.last_action = None

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in environment.

        Args:
            action: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT, 4=COLLECT

        Returns:
            observation: Current observation
            reward: Reward for this step
            terminated: True if episode ended naturally (success/failure)
            truncated: True if episode was cut short (timeout)
            info: Additional information dictionary
        """
        self.current_step += 1
        self.last_action = action

        # IMPORTANT: All sensors generate data each step
        for sensor in self.sensors:
            sensor.step(time_step=1.0)

        # Execute action
        battery_before = self.uav.battery

        if action in [0, 1, 2, 3]:  # Movement actions
            reward = self._execute_move_action(action)
        elif action == 4:  # COLLECT action
            reward = self._execute_collect_action()
        else:
            raise ValueError(f"Invalid action: {action}")

        battery_used = battery_before - self.uav.battery

        # Check termination conditions
        terminated = False
        truncated = False

        # Failure: Battery depleted
        if not self.uav.is_alive():
            truncated = True

        # Timeout: Max steps reached
        if self.current_step >= self.max_steps:
            truncated = True

        self.total_reward += reward

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _execute_move_action(self, action: int) -> float:
        """Execute movement action and return reward."""
        direction_map = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}
        direction = direction_map[action]

        battery_before = self.uav.battery
        move_success = self.uav.move(direction, self.grid_size, time_step=5)
        battery_used = battery_before - self.uav.battery

        # Clear last successful collections (not collecting anymore)
        self.last_successful_collections = []

        reward = self.reward_fn.calculate_movement_reward(
            move_success=move_success,
            battery_used=battery_used
        )

        return reward

    def _execute_collect_action(self) -> float:
        """
        Execute data collection action with probabilistic LoRa transmission
        and SF-based concurrency handling.
        """
        # UAV hovers while collecting
        self.uav.hover(duration=self.collection_duration)
        battery_used = self.uav.battery_drain_hover * self.collection_duration

        # Dictionary to track successful collections by Spreading Factor
        successful_sf_slots = {}

        total_bytes_collected = 0.0
        new_sensors_collected = []
        attempted_empty = False
        collision_count = 0

        # STEP 1: Probabilistic transmission attempt for all sensors
        for sensor in self.sensors:
            # Skip if buffer is empty
            if sensor.data_buffer <= 0:
                attempted_empty = True
                continue

            # Calculate distance to UAV
            distance = np.linalg.norm(sensor.position - self.uav.position)

            # Update Spreading Factor based on distance (ADR)
            sensor.update_spreading_factor(self.uav.position)

            # Calculate P_overall = P_link * P_cycle
            P_link = sensor.get_success_probability(self.uav.position, use_advanced_model=True)
            P_cycle = sensor.duty_cycle_probability
            P_overall = P_link * P_cycle

            # Probabilistic transmission attempt
            if np.random.rand() < P_overall:
                current_sf = sensor.spreading_factor

                # STEP 2: Handle SF-based concurrency (Capture Effect)
                if current_sf not in successful_sf_slots:
                    # First sensor to transmit on this SF - wins the slot
                    successful_sf_slots[current_sf] = {
                        'sensor_id': sensor.sensor_id,
                        'distance': distance,
                        'sensor': sensor,
                        'data_to_collect': min(sensor.data_buffer, sensor.packet_size)
                    }
                else:
                    # Collision detected! Multiple sensors on same SF
                    collision_count += 1

                    # Capture effect: Closer sensor wins (stronger RSSI)
                    existing_distance = successful_sf_slots[current_sf]['distance']

                    if distance < existing_distance:
                        # Current sensor is closer - it wins, replace existing
                        successful_sf_slots[current_sf] = {
                            'sensor_id': sensor.sensor_id,
                            'distance': distance,
                            'sensor': sensor,
                            'data_to_collect': min(sensor.data_buffer, sensor.packet_size)
                        }

        # Store successful collections for visualization
        self.last_successful_collections = []

        # STEP 3: Process successful transmissions
        for sf, slot_info in successful_sf_slots.items():
            winning_sensor = slot_info['sensor']

            # Collect data from winning sensor
            bytes_collected, success = winning_sensor.collect_data(
                uav_position=tuple(self.uav.position),
                collection_duration=self.collection_duration
            )

            if success and bytes_collected > 0:
                total_bytes_collected += bytes_collected
                self.total_data_collected += bytes_collected

                # Track if this is first time collecting from this sensor
                if winning_sensor.sensor_id not in self.sensors_visited:
                    new_sensors_collected.append(winning_sensor.sensor_id)
                    self.sensors_visited.add(winning_sensor.sensor_id)

                # Add to visualization list
                self.last_successful_collections.append((winning_sensor, sf))

        # Check if all sensors now have empty buffers
        all_sensors_collected = all(sensor.data_buffer <= 0 for sensor in self.sensors)
        total_data_loss = sum(sensor.total_data_lost for sensor in self.sensors)

        # STEP 4: Calculate reward
        reward = self.reward_fn.calculate_collection_reward(
            bytes_collected=total_bytes_collected,
            was_new_sensor=len(new_sensors_collected) > 0,
            was_empty=attempted_empty,
            all_sensors_collected=all_sensors_collected,
            battery_used=battery_used,
            num_sensors_collected=len(successful_sf_slots),
            collision_count=collision_count,
            data_loss=total_data_loss
        )

        return reward

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        obs = np.concatenate([
            self.uav.position,
            [self.uav.battery],
            [sensor.data_buffer for sensor in self.sensors],
        ]).astype(np.float32)

        return obs

    def _get_info(self) -> dict:
        """Get additional information."""
        return {
            'uav_position': self.uav.position.copy(),
            'battery': self.uav.battery,
            'battery_percent': self.uav.get_battery_percentage(),
            'sensors_collected': len(self.sensors_visited),
            'total_sensors': self.num_sensors,
            'current_step': self.current_step,
            'total_reward': self.total_reward,
            'total_data_collected': self.total_data_collected,
            'coverage_percentage': (len(self.sensors_visited) / self.num_sensors) * 100,
            'is_alive': self.uav.is_alive()
        }

    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return None

        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            if self.fig is None:
                plt.ion()  # Enable interactive mode
                self.fig, self.ax = plt.subplots(figsize=(12, 10))

            self._render_frame()
            plt.pause(0.01)  # Small pause to update display

            return None  # Important: return None for 'human' mode

    def _render_frame(self):
        """Enhanced render frame method for UAVEnvironment."""
        if self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(14, 10))

        self.ax.clear()

        # Draw grid lines
        for i in range(self.grid_size[0] + 1):
            self.ax.axhline(i, color='gray', linewidth=0.5, alpha=0.3)
        for i in range(self.grid_size[1] + 1):
            self.ax.axvline(i, color='gray', linewidth=0.5, alpha=0.3)

        # Calculate which sensors are in communication range (based on RSSI)
        sensors_in_range = []

        for sensor in self.sensors:
            in_range = sensor.is_in_range(self.uav.position)
            if in_range:
                sensors_in_range.append(sensor)

        # Get ACTUAL collecting sensors from last successful collections
        collecting_sensors = []
        collecting_sensors_sf = {}

        if self.last_action == 4 and hasattr(self, 'last_successful_collections'):
            for sensor, sf in self.last_successful_collections:
                collecting_sensors.append(sensor)
                collecting_sensors_sf[sensor.sensor_id] = sf

        # Draw connection lines for sensors in range (but not actively collecting)
        for sensor in sensors_in_range:
            if sensor not in collecting_sensors:
                self.ax.plot([sensor.position[0], self.uav.position[0]],
                             [sensor.position[1], self.uav.position[1]],
                             color='lightblue',
                             linewidth=1,
                             linestyle=':',
                             alpha=0.3,
                             zorder=2)

        # Draw all sensors with enhanced visuals
        for sensor in self.sensors:
            is_collecting = sensor in collecting_sensors
            render_sensor_enhanced(self.ax, sensor, self.current_step,
                                   self.uav.position,
                                   current_action=self.last_action if is_collecting else None)

        # Draw active collection lines (ONLY for actual winners)
        for sensor in collecting_sensors:
            self.ax.plot([sensor.position[0], self.uav.position[0]],
                         [sensor.position[1], self.uav.position[1]],
                         color='purple',
                         linewidth=2.5,
                         linestyle='--',
                         alpha=0.7,
                         zorder=8)

            mid_x = (sensor.position[0] + self.uav.position[0]) / 2
            mid_y = (sensor.position[1] + self.uav.position[1]) / 2
            sf = collecting_sensors_sf.get(sensor.sensor_id, sensor.spreading_factor)

            self.ax.text(mid_x, mid_y, f'SF{sf}',
                         fontsize=8,
                         color='white',
                         fontweight='bold',
                         ha='center',
                         va='center',
                         bbox=dict(boxstyle='round,pad=0.3',
                                   facecolor='purple',
                                   edgecolor='white',
                                   linewidth=1.5,
                                   alpha=0.9),
                         zorder=9)

        # Draw UAV
        render_uav_enhanced(self.ax, self.uav)

        # Set axis properties
        self.ax.set_xlim(0, self.grid_size[0])
        self.ax.set_ylim(0, self.grid_size[1])
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('X Coordinate', fontsize=12)
        self.ax.set_ylabel('Y Coordinate', fontsize=12)
        self.ax.axhline(0, color='black', linewidth=1.5, alpha=0.7, zorder=1)
        self.ax.axvline(0, color='black', linewidth=1.5, alpha=0.7, zorder=1)

        # Enhanced title with more info
        title = (f'Step: {self.current_step}/{self.max_steps} | '
                 f'Battery: {self.uav.battery:.1f}Wh ({self.uav.get_battery_percentage():.0f}%) | '
                 f'Collected: {len(self.sensors_visited)}/{self.num_sensors} | '
                 f'Reward: {self.total_reward:.1f}')
        self.ax.set_title(title, fontsize=11, fontweight='bold', pad=10)

        # Add statistics panel
        avg_buffer = np.mean([s.data_buffer for s in self.sensors])

        stats_text = (
            f'Data Collected: {self.total_data_collected:.1f} bytes\n'
            f'Coverage: {(len(self.sensors_visited) / self.num_sensors) * 100:.0f}%\n'
            f'Battery Used: {self.uav.max_battery - self.uav.battery:.1f}Wh\n'
            f'Avg Buffer: {avg_buffer:.1f} bytes'
        )

        if len(collecting_sensors) > 0:
            stats_text += f'\n\n📡 Collecting: {len(collecting_sensors)} sensor(s)'
            if len(collecting_sensors) > 1:
                stats_text += '\n🔗 Multi-sensor collection!'
            sf_list = sorted(set(collecting_sensors_sf.values()))
            stats_text += f'\n📶 SFs: {sf_list}'

        if len(sensors_in_range) > 0:
            stats_text += f'\n📡 In Range: {len(sensors_in_range)} sensor(s)'

        self.ax.text(0.02, 0.98, stats_text,
                     transform=self.ax.transAxes,
                     fontsize=9,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round',
                               facecolor='wheat',
                               alpha=0.8))

        # Updated Legend
        legend_elements = [
            Patch(facecolor='blue', alpha=0.9, edgecolor='black',
                  label='Full Buffer (100%)'),
            Patch(facecolor='yellow', alpha=0.7, edgecolor='black',
                  label='Partial Buffer (1-99%)'),
            Patch(facecolor='green', alpha=0.5, edgecolor='black',
                  label='Collected (empty)'),
            Patch(facecolor='purple', alpha=1.0, edgecolor='black',
                  label='Currently Collecting'),
            Patch(facecolor='lightblue', alpha=0.5, edgecolor='black',
                  label='Empty Buffer'),
            Patch(facecolor='orange', edgecolor='red', linewidth=2,
                  label='UAV Position'),
            plt.Line2D([0], [0], color='purple', linewidth=2.5, linestyle='--',
                       label='Active Collection Link'),
            plt.Line2D([0], [0], color='lightblue', linewidth=1, linestyle=':',
                       label='In Communication Range'),
        ]

        # Place legend OUTSIDE on the right
        self.ax.legend(handles=legend_elements,
                       loc='center left',
                       bbox_to_anchor=(1.02, 0.5),
                       fontsize=8,
                       framealpha=0.9,
                       title='Legend',
                       title_fontsize=9,
                       borderaxespad=0)

        # Make room for the legend
        plt.tight_layout()
        plt.subplots_adjust(right=0.82)

        if self.render_mode == "rgb_array":
            self.fig.canvas.draw()
            image = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return image

    def close(self):
        """Close the environment."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


# Testing
if __name__ == "__main__":

    print("=" * 70)
    print("Testing UAV Environment (Probabilistic Duty Cycle)")
    print("=" * 70)
    print()

    # Create environment WITH RENDERING
    env = UAVEnvironment(
        grid_size=(50, 50),
        num_sensors=20,
        max_steps=500,
        sensor_duty_cycle=10.0,
        render_mode='human'
    )

    print(f"✓ Environment created")
    print(f"  Action Space: {env.action_space}")
    print(f"  Observation Space: {env.observation_space}")
    print(f"  Grid Size: {env.grid_size}")
    print(f"  Number of Sensors: {env.num_sensors}")
    print()

    # Reset environment
    obs, info = env.reset(seed=42)

    print(f"✓ Environment reset")
    print(f"  Initial observation shape: {obs.shape}")
    print(f"  UAV position: {info['uav_position']}")
    print(f"  Battery: {info['battery']:.2f} Wh ({info['battery_percent']:.1f}%)")
    print()

    # Run random episode
    print("   Running random episode...")
    print("   Watch the window - UAV will move randomly!")
    print()

    action_names = ['UP', 'DOWN', 'LEFT', 'RIGHT', 'COLLECT']

    try:
        for step in range(10000):
            # Sample random action
            action = env.action_space.sample()

            # Take step
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            # Print every 20 steps or at end
            if step % 20 == 0 or terminated or truncated:
                print(f"Step {step + 1:3d}: {action_names[action]:7s} | "
                      f"Pos: ({info['uav_position'][0]:.1f}, {info['uav_position'][1]:.1f}) | "
                      f"Battery: {info['battery']:6.1f}Wh | "
                      f"Collected: {info['sensors_collected']:2d}/{info['total_sensors']:2d} | "
                      f"Reward: {reward:+7.2f}")

            # Check if done
            if terminated:
                print("\n✓ Mission complete! All sensors collected.")
                env.render()
                time.sleep(5)
                break
            elif truncated:
                if not info['is_alive']:
                    print("\n✗ Battery depleted!")
                else:
                    print("\n✗ Timeout reached.")
                env.render()
                time.sleep(5)
                break

    except KeyboardInterrupt:
        print("\n\n⏸️ Stopped by user (Ctrl+C)")

    # Summary
    print()
    print("=" * 70)
    print("Episode Summary:")
    print("=" * 70)
    print(f"  Total Steps: {info['current_step']}")
    print(f"  Total Reward: {info['total_reward']:.2f}")
    print(f"  Coverage: {info['coverage_percentage']:.1f}%")
    print(f"  Data Collected: {info['total_data_collected']:.2f} bytes")
    print(f"  Battery Used: {274.0 - info['battery']:.2f} Wh")
    print(f"  Battery Remaining: {info['battery']:.2f} Wh ({info['battery_percent']:.1f}%)")
    print("=" * 70)

    # Keep window open at the end
    print("\n✓ Test complete! Close the matplotlib window to exit.")
    plt.ioff()
    plt.show()