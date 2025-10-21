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
                            communication_range=0.2,
                            current_action=None,
                            steps_since_collection=10) -> SensorState:
    """
    Determine visual state of sensor based on buffer level and UAV interaction.

    Args:
        sensor: IoTSensor object
        uav_position: Current UAV position
        communication_range: Distance for UAV to collect data
        current_action: Current action (0-4, where 4=COLLECT)
        steps_since_collection: How long ago sensor was collected

    Returns:
        SensorState enum value
    """
    # Check if UAV is currently collecting from this sensor
    distance = np.linalg.norm(sensor.position - np.array(uav_position))
    is_in_range = distance <= communication_range

    # Determine state based on buffer percentage
    buffer_pct = (sensor.data_buffer / sensor.max_buffer_size) * 100

    # Priority 1: COLLECTING - UAV actively collecting (action=4, in range, has data)
    if current_action == 4 and is_in_range:
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
                          communication_range=0.2, current_action=None):
    """
    Render a single sensor with enhanced visual states.

    Args:
        ax: Matplotlib axis
        sensor: IoTSensor object
        current_step: Current simulation step (for animations)
        uav_position: UAV position tuple (x, y)
        communication_range: Distance for communication
        current_action: Current action (0-4)
    """
    x, y = sensor.position

    # Get sensor state
    state = get_sensor_visual_state(sensor, uav_position, communication_range, current_action)

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

    # Add sensor ID above
    ax.text(x, y + 0.4,
            f'S{sensor.sensor_id}',
            ha='center',
            va='bottom',
            fontsize=6,
            color='gray')

    # Show collection progress animation for COLLECTING state
    if state == SensorState.COLLECTING:
        # Draw a small circle around sensor to show it's being collected
        collection_circle = Circle((x, y), 0.4,
                                   facecolor='none',
                                   edgecolor='purple',
                                   linewidth=3,
                                   linestyle='--',
                                   alpha=0.6,
                                   zorder=4)
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
    buffers overflow. LoRa communication uses d^-2 path loss model.

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
                 data_generation_rate: float = 22.0/10,
                 max_buffer_size: float = 1000.0,
                 lora_spreading_factor: int = 7,
                 path_loss_exponent: float = 2.0,
                 rssi_threshold: float = -90.0,
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
                rssi_threshold=rssi_threshold
            )
            self.sensors.append(sensor)

        # Initialize UAV
        if uav_start_position is None:
            uav_start_position = (grid_size[0] / 2, grid_size[1] / 2)

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
            [grid_size[0]-1, grid_size[1]-1, max_battery] +
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

        Creates a near-uniform grid layout with optional small jitter
        to avoid perfectly aligned sensors.

        Args:
            num_sensors: Number of sensors to place

        Returns:
            List of (x, y) positions
        """
        positions = []

        # Calculate grid layout for uniform distribution
        rows = int(np.sqrt(num_sensors))
        cols = int(np.ceil(num_sensors / rows))

        # Add some margin from edges
        margin = 1.0
        x_spacing = (self.grid_size[0] - 2 * margin) / (cols - 1) if cols > 1 else 0
        y_spacing = (self.grid_size[1] - 2 * margin) / (rows - 1) if rows > 1 else 0

        sensor_count = 0
        for row in range(rows):
            for col in range(cols):
                if sensor_count >= num_sensors:
                    break

                x = margin + col * x_spacing
                y = margin + row * y_spacing

                # Add small random jitter (optional - remove for perfect grid)
                jitter = 0.2
                x += np.random.uniform(-jitter, jitter)
                y += np.random.uniform(-jitter, jitter)

                # Ensure within bounds
                x = np.clip(x, 0.5, self.grid_size[0] - 0.5)
                y = np.clip(y, 0.5, self.grid_size[1] - 0.5)

                positions.append((float(x), float(y)))
                sensor_count += 1

            if sensor_count >= num_sensors:
                break

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

        # Reset all sensors
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

        # Success: All sensors have empty buffers
        if all(sensor.data_buffer <= 0 for sensor in self.sensors):
            terminated = True

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
        move_success = self.uav.move(direction, self.grid_size)
        battery_used = battery_before - self.uav.battery

        reward = self.reward_fn.calculate_movement_reward(
            move_success=move_success,
            battery_used=battery_used
        )

        return reward

    def _execute_collect_action(self) -> float:
        """Execute data collection action and return reward."""
        # UAV hovers while collecting
        self.uav.hover(duration=self.collection_duration)
        battery_used = self.uav.battery_drain_hover * self.collection_duration

        total_bytes_collected = 0.0
        new_sensors_collected = []
        attempted_empty = False

        # Try to collect from all sensors in range
        for sensor in self.sensors:
            bytes_collected, success = sensor.collect_data(
                uav_position=tuple(self.uav.position),
                collection_duration=self.collection_duration
            )

            if success and bytes_collected > 0:
                total_bytes_collected += bytes_collected
                self.total_data_collected += bytes_collected

                # Track if this is first time collecting from this sensor
                if sensor.sensor_id not in self.sensors_visited:
                    new_sensors_collected.append(sensor.sensor_id)
                    self.sensors_visited.add(sensor.sensor_id)
            elif success and bytes_collected == 0:
                # In range but no data (empty buffer)
                attempted_empty = True

        # Check if all sensors now have empty buffers
        all_sensors_collected = all(sensor.data_buffer <= 0 for sensor in self.sensors)

        # Calculate reward
        reward = self.reward_fn.calculate_collection_reward(
            bytes_collected=total_bytes_collected,
            was_new_sensor=len(new_sensors_collected) > 0,
            was_empty=attempted_empty,
            all_sensors_collected=all_sensors_collected,
            battery_used=battery_used
        )

        return reward

    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        obs = np.concatenate([
            self.uav.position,
            [self.uav.battery],
            [sensor.data_buffer for sensor in self.sensors]
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
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            if self.fig is None:
                plt.ion()  # Enable interactive mode
                self.fig, self.ax = plt.subplots(figsize=(10, 10))

            self.ax.clear()
            self._render_frame()
            plt.pause(0.05)  # Update display

    def _render_frame(self):
        """Enhanced render frame method for UAVEnvironment."""
        if self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(12, 10))

        self.ax.clear()

        # Draw grid
        for i in range(self.grid_size[0] + 1):
            self.ax.axhline(i, color='gray', linewidth=0.5, alpha=0.3)
            self.ax.axvline(i, color='gray', linewidth=0.5, alpha=0.3)

        # Draw communication range
        comm_range = 2.0  # Adjust based on your RSSI settings
        range_circle = Circle(self.uav.position, comm_range,
                             facecolor='none', edgecolor='cyan',
                             linewidth=2, linestyle='--', alpha=0.4)
        self.ax.add_patch(range_circle)

        # Draw all sensors with enhanced visuals
        for sensor in self.sensors:
            render_sensor_enhanced(self.ax, sensor, self.current_step,
                                   self.uav.position, communication_range=2.0,
                                   current_action=self.last_action)

        # Draw UAV
        render_uav_enhanced(self.ax, self.uav)

        # Set axis properties
        self.ax.set_xlim(-0.5, self.grid_size[0] - 0.5)
        self.ax.set_ylim(-0.5, self.grid_size[1] - 0.5)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('X Coordinate', fontsize=12)
        self.ax.set_ylabel('Y Coordinate', fontsize=12)

        # Enhanced title with more info
        title = (f'Step: {self.current_step}/{self.max_steps} | '
                 f'Battery: {self.uav.battery:.1f}Wh ({self.uav.get_battery_percentage():.0f}%) | '
                 f'Collected: {len(self.sensors_visited)}/{self.num_sensors} | '
                 f'Reward: {self.total_reward:.1f}')
        self.ax.set_title(title, fontsize=11, fontweight='bold', pad=10)

        # Add statistics panel
        stats_text = (
            f'Data Collected: {self.total_data_collected:.1f} bytes\n'
            f'Coverage: {(len(self.sensors_visited) / self.num_sensors) * 100:.0f}%\n'
            f'Battery Used: {self.uav.max_battery - self.uav.battery:.1f}Wh'
        )
        self.ax.text(0.02, 0.98, stats_text,
                     transform=self.ax.transAxes,
                     fontsize=9,
                     verticalalignment='top',
                     bbox=dict(boxstyle='round',
                               facecolor='wheat',
                               alpha=0.8))

        # Legend
        legend_elements = [
            Patch(facecolor='blue', alpha=0.9, edgecolor='black',
                  label='Sensor: Full Buffer (100%)'),
            Patch(facecolor='yellow', alpha=0.7, edgecolor='black',
                  label='Sensor: Partial (1-99%)'),
            Patch(facecolor='green', alpha=0.5, edgecolor='black',
                  label='Sensor: Collected (empty)'),
            Patch(facecolor='purple', alpha=1.0, edgecolor='black',
                  label='Sensor: Collecting'),
            Patch(facecolor='lightblue', alpha=0.5, edgecolor='black',
                  label='Sensor: Empty'),
            Patch(facecolor='orange', edgecolor='red', linewidth=2,
                  label='UAV')
        ]
        self.ax.legend(handles=legend_elements,
                       loc='upper right',
                       fontsize=8,
                       framealpha=0.9)

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
    print("Testing UAV Environment")
    print("=" * 70)
    print()

    # Create environment WITH RENDERING
    env = UAVEnvironment(
        grid_size=(20, 20),
        num_sensors=20,
        max_steps=500,
        rssi_threshold=-80.0,
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
        for step in range(500):
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