"""
Gymnasium-Compatible UAV Environment with Fairness Constraints and Battery Constraints

Environment for training UAV to collect data from IoT sensors using RL
with fairness constraints to prevent sensor neglect.

NEW FEATURES:
- Urgency metrics in observation space
- Fairness-constrained reward function
- Data loss tracking per step (GLOBAL for all actions)
- Urgency reduction tracking
- Observation space bounds match actual observation values

CRITICAL FIX:
- Data loss penalty applies to ALL actions (movement AND collection)
- Prevents agent from learning to ignore buffer overflows while moving

Author: ATILADE GABRIEL OKE
Date: November 2025
Project: Energy-Efficient UAV Path Planning in IOT Networks: A Deep Reinforcement Learning Aided Approach
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
    EMPTY = 0
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    FULL = 4
    COLLECTING = 5
    COLLECTED = 6


def get_sensor_visual_state(sensor, uav_position, current_action=None) -> SensorState:
    """Determine visual state of sensor based on buffer level and UAV interaction."""
    is_in_range = sensor.is_in_range(uav_position)
    buffer_pct = (sensor.data_buffer / sensor.max_buffer_size) * 100

    if current_action == 4 and is_in_range and sensor.data_buffer > 0:
        return SensorState.COLLECTING

    if sensor.data_buffer <= 10 and sensor.data_collected and is_in_range:
        return SensorState.COLLECTED

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


def render_sensor_enhanced(ax, sensor, current_step, uav_position, current_action=None, urgency=0.0, is_visited=True):
    """Render a single sensor with enhanced visual states and urgency indicator."""
    x, y = sensor.position
    state = get_sensor_visual_state(sensor, uav_position, current_action)

    if state == SensorState.EMPTY:
        color, marker_size, alpha, marker = 'lightblue', 80, 0.5, 'o'
    elif state == SensorState.LOW:
        color, marker_size, alpha, marker = 'yellow', 120, 0.7, 'o'
    elif state == SensorState.MEDIUM:
        color, marker_size, alpha, marker = 'yellow', 160, 0.8, 'o'
    elif state == SensorState.HIGH:
        color, marker_size, alpha, marker = 'yellow', 200, 0.9, 'o'
    elif state == SensorState.FULL:
        pulse = 0.5 + 0.5 * np.sin(current_step * 0.3)
        color, marker_size, alpha, marker = 'blue', 250 * (0.9 + 0.2 * pulse), 0.7 + 0.2 * pulse, 'o'
    elif state == SensorState.COLLECTING:
        color, marker_size, alpha, marker = 'purple', 300, 1.0, '*'
    elif state == SensorState.COLLECTED:
        color, marker_size, alpha, marker = 'green', 100, 0.5, 'o'
    else:
        color, marker_size, alpha, marker = 'gray', 80, 0.5, 'o'

    ax.scatter(x, y, c=color, marker=marker, s=marker_size, alpha=alpha, edgecolors='black', linewidths=2, zorder=5)
    if sensor.data_buffer > 0:
        # Create a Green Ring
        # Radius 0.8 ensures it circles the sensor dot nicely on a 50x50 grid
        ready_ring = Circle((x, y), 0.8, fill=False, edgecolor='lime',
                            linewidth=1.5, linestyle='-', alpha=0.8, zorder=4)
        ax.add_patch(ready_ring)

    buffer_pct = (sensor.data_buffer / sensor.max_buffer_size) * 100
    text_color = 'white' if state in [SensorState.FULL, SensorState.COLLECTING] else 'black'

    ax.text(x, y - 0.4, f'{int(buffer_pct)}%', ha='center', va='top', fontsize=7, fontweight='bold',
            color=text_color, bbox=dict(boxstyle='round,pad=0.2', facecolor='white', edgecolor='black', alpha=0.7))

    ax.text(x, y + 0.5, f'S{sensor.sensor_id}', ha='center', va='bottom', fontsize=7, fontweight='bold')

    if urgency > 0.8:
        urgency_symbol = '🔴'
    elif urgency > 0.5:
        urgency_symbol = '🟠'
    elif urgency > 0.2:
        urgency_symbol = '🟡'
    else:
        urgency_symbol = '🟢'

    ax.text(x + 0.5, y + 0.5, urgency_symbol, fontsize=8, ha='center', va='center', zorder=12)

    if urgency > 0.3:
        ax.text(x, y - 0.8, f'U:{urgency:.2f}', fontsize=6, ha='center', va='top',
                color='red' if urgency > 0.8 else 'orange', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))

    if is_visited:
        # Draw a bold green checkmark at Top-Left (x - 0.6)
        # to distinguish it from the urgency symbol at Top-Right
        ax.text(x - 0.6, y + 0.5, '✓', fontsize=12, fontweight='bold',
                color='darkgreen', ha='center', va='center', zorder=20,
                bbox=dict(boxstyle='circle,pad=0.1', facecolor='white', edgecolor='green', alpha=0.8))

    if state == SensorState.COLLECTING:
        collection_circle = Circle((x, y), 0.5, facecolor='none', edgecolor='purple', linewidth=3, linestyle='--', alpha=0.6, zorder=6)
        ax.add_patch(collection_circle)


def render_uav_enhanced(ax, uav):
    """Render the UAV with battery indicator."""
    uav_x, uav_y = uav.position

    uav_marker = patches.FancyBboxPatch(
        (uav_x - 0.25, uav_y - 0.25), 0.5, 0.5,
        boxstyle="round,pad=0.05", edgecolor='red', facecolor='orange', linewidth=2.5, zorder=10
    )
    ax.add_patch(uav_marker)

    ax.text(uav_x, uav_y, '✈', ha='center', va='center', fontweight='bold', fontsize=14, color='white', zorder=11)

    battery_pct = uav.get_battery_percentage()
    battery_color = 'green' if battery_pct > 50 else ('orange' if battery_pct > 25 else 'red')

    ax.text(uav_x, uav_y - 0.8, f'⚡{battery_pct:.0f}%', ha='center', va='top', fontsize=8, fontweight='bold',
            color=battery_color, bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=battery_color, linewidth=2, alpha=0.9), zorder=11)


class UAVEnvironment(gym.Env):
    """
    Custom Gymnasium Environment for UAV IoT Data Collection with Fairness Constraints.

    Observation Space:
        Box: [uav_x, uav_y, battery, sensor1_buffer, sensor1_urgency, ..., sensorN_buffer, sensorN_urgency]

    Action Space:
        Discrete(5): [UP, DOWN, LEFT, RIGHT, COLLECT]

    Reward Structure (Fairness-Constrained):
        +0.1 per byte: Data collection
        +10.0: New sensor collected
        +20.0 per unit: Urgency reduction
        -500.0 per byte: Data loss (MASSIVE PENALTY - applies to ALL actions)
        -2.0: Attempted collection from empty sensor
        -5.0: Boundary collision
        -0.1 per Wh: Battery drain
        -0.05: Step penalty
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}

    def __init__(self,
                 grid_size: Tuple[int, int] = (10, 10),
                 sensor_positions: Optional[List[Tuple[float, float]]] = None,
                 num_sensors: int = 20,
                 data_generation_rate: float = 22.0 / 10,
                 max_buffer_size: float = 1000.0,
                 lora_spreading_factor: int = 7,
                 path_loss_exponent: float = 2.0,
                 rssi_threshold: float = -90.0,
                 sensor_duty_cycle: float = 10.0,
                 uav_start_position: Optional[Tuple[float, float]] = None,
                 max_battery: float = 274.0,
                 collection_duration: float = 1.0,
                 max_steps: int = 2100,
                 render_mode: Optional[str] = None,
                 penalty_data_loss: float = -500.0,
                 reward_urgency_reduction: float = 20.0):

        """Initialize UAV environment with fairness constraints."""

        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.collection_duration = collection_duration
        self.last_successful_collections = []
        self.last_action = None

        if sensor_positions is None:
            self.sensor_positions = self._generate_uniform_sensor_positions(num_sensors)
        else:
            self.sensor_positions = sensor_positions

        self.num_sensors = len(self.sensor_positions)

        self.sensors: List[IoTSensor] = []
        for i, pos in enumerate(self.sensor_positions):
            sensor = IoTSensor(
                sensor_id=i, position=pos, data_generation_rate=data_generation_rate,
                max_buffer_size=max_buffer_size, spreading_factor=lora_spreading_factor,
                path_loss_exponent=path_loss_exponent, rssi_threshold=rssi_threshold,
                duty_cycle=sensor_duty_cycle
            )
            self.sensors.append(sensor)

        if uav_start_position is None:
            uav_start_position = (0, 0)

        self.uav = UAV(start_position=uav_start_position, max_battery=max_battery)
        self.reward_fn = RewardFunction(penalty_data_loss=penalty_data_loss, reward_urgency_reduction=reward_urgency_reduction)

        self.action_space = spaces.Discrete(5)

        obs_low = np.array([0, 0, 0] + [0, 0] * self.num_sensors, dtype=np.float32)
        obs_high = np.array([grid_size[0], grid_size[1], max_battery] + [max_buffer_size, 1.0] * self.num_sensors, dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.current_step = 0
        self.total_reward = 0.0
        self.total_data_collected = 0.0
        self.sensors_visited = set()
        self.previous_data_loss = 0.0

        self.fig = None
        self.ax = None

    def _generate_uniform_sensor_positions(self, num_sensors: int) -> List[Tuple[float, float]]:
        """Generate uniformly distributed sensor positions across the grid."""
        x_min, y_min = 0.0, 0.0
        x_max, y_max = float(self.grid_size[0]), float(self.grid_size[1])
        coordinates = np.random.uniform(low=[x_min, y_min], high=[x_max, y_max], size=(num_sensors, 2))
        return [(float(x), float(y)) for x, y in coordinates]

    def _calculate_urgency(self, sensor: IoTSensor) -> float:
        """Calculate urgency metric for a sensor."""
        buffer_utilization = sensor.data_buffer / sensor.max_buffer_size
        data_loss_rate = (sensor.total_data_lost / sensor.total_data_generated) if sensor.total_data_generated > 0 else 0.0
        urgency = buffer_utilization * (1.0 + data_loss_rate * 10.0)
        return np.clip(urgency, 0.0, 1.0)

    def _get_sensor_urgencies(self) -> np.ndarray:
        """Calculate urgency for all sensors."""
        return np.array([self._calculate_urgency(sensor) for sensor in self.sensors])

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)
        self.uav.reset()
        for sensor in self.sensors:
            sensor.reset()
        self.current_step = 0
        self.total_reward = 0.0
        self.total_data_collected = 0.0
        self.sensors_visited = set()
        self.last_action = None
        self.previous_data_loss = 0.0
        return self._get_observation(), self._get_info()

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one step in environment.

        Time synchronization fix: Step duration is now dynamic based on action type.
        - Move actions (0-3): 1.0 second
        - Collect action (4): self.collection_duration seconds

        This ensures sensors age realistically during collection, preventing the agent
        from "cheating" by collecting for extended periods while time is artificially frozen.

        Args:
            action: Integer in [0, 1, 2, 3, 4] representing UAV action
                    0: Move North, 1: Move South, 2: Move East, 3: Move West, 4: Collect

        Returns:
            Tuple of (observation, reward, terminated, truncated, info)
        """
        self.current_step += 1
        self.last_action = action

        # ===== CRITICAL FIX: CALCULATE DYNAMIC STEP DURATION =====
        # If Moving (0-3): Takes 1.0 second (fixed grid step time)
        # If Collecting (4): Takes self.collection_duration seconds (realistic)
        if action == 4:
            step_duration = self.collection_duration
        else:
            step_duration = 1.0

        # ===== CRITICAL FIX: SYNC SENSORS TO REAL TIME =====
        # Sensors now age exactly as much time as the UAV action takes
        # This prevents the agent from exploiting artificial time freezing during collection
        for sensor in self.sensors:
            sensor.step(time_step=step_duration)

        # STEP 2: CALCULATE DATA LOSS (Global for this step)
        # ===== CRITICAL: This applies to BOTH Move and Collect actions =====
        current_data_loss = sum(sensor.total_data_lost for sensor in self.sensors)
        step_data_loss = current_data_loss - self.previous_data_loss
        self.previous_data_loss = current_data_loss

        # STEP 3: EXECUTE ACTION (with data loss passed as parameter)
        battery_before = self.uav.battery

        if action in [0, 1, 2, 3]:  # Movement actions
            reward = self._execute_move_action(action, step_data_loss)
        elif action == 4:  # COLLECT action
            reward = self._execute_collect_action(step_data_loss)
        else:
            raise ValueError(f"Invalid action: {action}")

        battery_used = battery_before - self.uav.battery

        # STEP 4: CHECK TERMINATION CONDITIONS
        terminated = False
        truncated = False

        if not self.uav.is_alive():
            truncated = True

        if self.current_step >= self.max_steps:
            truncated = True

        # STEP 5: ACCUMULATE REWARD AND PREPARE OUTPUT
        self.total_reward += reward
        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _execute_move_action(self, action: int, step_data_loss: float) -> float:
        """
        Execute movement action and return reward.

        ===== CRITICAL FIX =====
        Data loss penalty applies to movement actions too.
        This prevents agent from learning to ignore overflows while moving.
        """
        direction_map = {0: 'UP', 1: 'DOWN', 2: 'LEFT', 3: 'RIGHT'}
        direction = direction_map[action]

        battery_before = self.uav.battery
        move_success = self.uav.move(direction, self.grid_size, time_step=1)
        battery_used = battery_before - self.uav.battery

        self.last_successful_collections = []

        # Call reward function WITHOUT data_loss parameter
        reward = self.reward_fn.calculate_movement_reward(
            move_success=move_success,
            battery_used=battery_used
        )

        # ===== APPLY DATA LOSS PENALTY SEPARATELY =====
        data_loss_penalty = self.reward_fn.penalty_data_loss * step_data_loss
        reward += data_loss_penalty

        return reward

    def _get_sensor_urgencies(self) -> np.ndarray:
        """
        Calculate urgency metric for each sensor (Age of Information).

        Used to track fairness: How much are we reducing the staleness
        of the network by collecting from certain sensors?

        Returns:
            Array of urgency values (one per sensor)
        """
        urgencies = np.zeros(len(self.sensors))
        for i, sensor in enumerate(self.sensors):
            # Simple AoI: how old is the data in the buffer?
            # Approximation: buffer_level / generation_rate
            if sensor.data_generation_rate > 0:
                urgencies[i] = sensor.data_buffer / sensor.data_generation_rate
            else:
                urgencies[i] = 0.0
        return urgencies


    def _execute_collect_action(self, step_data_loss: float) -> float:
        """
        Execute data collection action with Capture Effect collision handling.

        ===== IMPLEMENTS EQUATION 11 FROM DISSERTATION =====
        Capture Effect: Gateway successfully decodes signal from sensor i if:
            P_r,i / (sum(P_r,j for j in co-channel) + N0) >= tau_cap (6 dB)

        Key improvements:
        1. RSSI-based collision resolution (physics)
        2. Destructive interference when neither sensor dominates
        3. Fairness tracking via urgency reduction
        4. Pre-calculated data loss passed as parameter (consistency)

        Args:
            step_data_loss: Data loss from this step (pre-calculated in step())
                           Passed to maintain consistency across action types

        Returns:
            float: Reward value calculated by reward function
        """
        # ===== PHASE 0: TRACK URGENCY BEFORE COLLECTION =====
        urgencies_before = self._get_sensor_urgencies()

        # ===== PHASE 1: UAV HOVER/COLLECTION SETUP =====
        self.uav.hover(duration=self.collection_duration)
        battery_used = self.uav.battery_drain_hover * self.collection_duration

        # ===== PHASE 2: PROBABILISTIC TRANSMISSION ATTEMPT =====
        # All sensors decide whether to transmit based on duty cycle and link quality
        transmission_attempts = {}  # SF -> list of sensors attempting to transmit

        for sensor in self.sensors:
            if sensor.data_buffer <= 0:
                continue

            # Update ADR based on current RSSI (Equation 18)
            sensor.update_spreading_factor(tuple(self.uav.position), current_step=self.current_step)

            # Calculate transmission probability
            P_link = sensor.get_success_probability(tuple(self.uav.position), use_advanced_model=True)
            P_cycle = sensor.duty_cycle_probability
            P_overall = P_link * P_cycle

            # Probabilistic transmission attempt
            if np.random.rand() < P_overall:
                current_sf = sensor.spreading_factor

                if current_sf not in transmission_attempts:
                    transmission_attempts[current_sf] = []

                transmission_attempts[current_sf].append(sensor)

        # ===== PHASE 3: COLLISION RESOLUTION VIA CAPTURE EFFECT =====
        # For each SF, resolve collisions using RSSI power comparison
        successful_sf_slots = {}  # SF -> winning sensor
        collision_count = 0

        for current_sf, attempting_sensors in transmission_attempts.items():

            if len(attempting_sensors) == 1:
                # No collision: Single sensor on this SF
                sensor = attempting_sensors[0]
                successful_sf_slots[current_sf] = sensor

            else:
                # ===== COLLISION DETECTED: Multiple sensors on same SF =====
                collision_count += len(attempting_sensors) - 1

                # Sort by RSSI (strongest first)
                sorted_by_rssi = sorted(
                    attempting_sensors,
                    key=lambda s: s.current_rssi,
                    reverse=True
                )

                strongest_sensor = sorted_by_rssi[0]
                second_strongest_sensor = sorted_by_rssi[1]

                rssi_strongest = strongest_sensor.current_rssi
                rssi_second = second_strongest_sensor.current_rssi

                # ===== CAPTURE EFFECT CRITERION (Equation 11) =====
                # Threshold is typically 6 dB for LoRa
                capture_threshold_db = 6.0

                if rssi_strongest > (rssi_second + capture_threshold_db):
                    # Strongest sensor is >6 dB above interference
                    # Gateway successfully decodes strongest signal
                    successful_sf_slots[current_sf] = strongest_sensor

                else:
                    # Destructive interference: Neither signal dominates
                    # Both packets corrupted, none are received
                    # Don't add this SF to successful_sf_slots
                    pass

        # ===== PHASE 4: COLLECT DATA FROM SUCCESSFUL TRANSMISSIONS =====
        total_bytes_collected = 0.0
        new_sensors_collected = []
        attempted_empty = False

        self.last_successful_collections = []

        for sf, winning_sensor in successful_sf_slots.items():
            # Attempt to collect from the winning sensor
            bytes_collected, success = winning_sensor.collect_data(
                uav_position=tuple(self.uav.position),
                collection_duration=self.collection_duration
            )

            if success and bytes_collected > 0:
                total_bytes_collected += bytes_collected
                self.total_data_collected += bytes_collected

                # Track new sensor visitation
                if winning_sensor.sensor_id not in self.sensors_visited:
                    new_sensors_collected.append(winning_sensor.sensor_id)
                    self.sensors_visited.add(winning_sensor.sensor_id)

                self.last_successful_collections.append((winning_sensor, sf))

        # Check if any sensor has empty buffer (attempted collection from empty)
        attempted_empty = any(s.data_buffer <= 0 for s in self.sensors)

        # ===== PHASE 5: FAIRNESS METRICS =====
        # Track urgency reduction (how much we helped stale sensors)
        urgencies_after = self._get_sensor_urgencies()
        urgency_reduced = np.sum(np.maximum(0, urgencies_before - urgencies_after))

        # Check if mission complete (all sensors have empty buffers)
        all_sensors_collected = all(sensor.data_buffer <= 0 for sensor in self.sensors)

        # ===== PHASE 6: CALCULATE REWARD =====
        # ===== CRITICAL: Use pre-calculated data loss (passed as parameter) =====
        reward = self.reward_fn.calculate_collection_reward(
            bytes_collected=total_bytes_collected,
            was_new_sensor=len(new_sensors_collected) > 0,
            was_empty=attempted_empty,
            all_sensors_collected=all_sensors_collected,
            battery_used=battery_used,
            num_sensors_collected=len(successful_sf_slots),
            collision_count=collision_count,
            data_loss=step_data_loss,  # ===== USE PASSED VALUE (Consistency) =====
            urgency_reduced=urgency_reduced
        )

        return reward

    def _get_observation(self) -> np.ndarray:
        """Get current observation with urgency metrics."""
        obs_list = [self.uav.position[0], self.uav.position[1], self.uav.battery]
        for sensor in self.sensors:
            urgency = self._calculate_urgency(sensor)
            obs_list.extend([sensor.data_buffer, urgency])
        return np.array(obs_list, dtype=np.float32)

    def _get_info(self) -> dict:
        """Get additional information including urgency stats."""
        urgencies = self._get_sensor_urgencies()
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
            'is_alive': self.uav.is_alive(),
            'max_urgency': np.max(urgencies),
            'avg_urgency': np.mean(urgencies),
            'high_urgency_sensors': np.sum(urgencies > 0.8),
        }

    def render(self):
        """Render the environment."""
        if self.render_mode is None:
            return None

        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            if self.fig is None:
                plt.ion()
                self.fig, self.ax = plt.subplots(figsize=(12, 10))

            self._render_frame()
            plt.pause(0.01)
            return None

    def _render_frame(self):
        """Enhanced render frame with urgency indicators."""
        if self.ax is None:
            self.fig, self.ax = plt.subplots(figsize=(14, 10))

        self.ax.clear()

        for i in range(self.grid_size[0] + 1):
            self.ax.axhline(i, color='gray', linewidth=0.5, alpha=0.3)
        for i in range(self.grid_size[1] + 1):
            self.ax.axvline(i, color='gray', linewidth=0.5, alpha=0.3)

        urgencies = self._get_sensor_urgencies()

        sensors_in_range = [sensor for sensor in self.sensors if sensor.is_in_range(self.uav.position)]

        collecting_sensors = []
        collecting_sensors_sf = {}

        if self.last_action == 4 and hasattr(self, 'last_successful_collections'):
            for sensor, sf in self.last_successful_collections:
                collecting_sensors.append(sensor)
                collecting_sensors_sf[sensor.sensor_id] = sf

        for sensor in sensors_in_range:
            if sensor not in collecting_sensors:
                self.ax.plot([sensor.position[0], self.uav.position[0]],
                             [sensor.position[1], self.uav.position[1]],
                             color='lightblue', linewidth=1, linestyle=':', alpha=0.3, zorder=2)

        for i, sensor in enumerate(self.sensors):
            is_collecting = sensor in collecting_sensors
            has_been_visited = sensor.sensor_id in self.sensors_visited
            render_sensor_enhanced(self.ax, sensor, self.current_step, self.uav.position,
                                   current_action=self.last_action if is_collecting else None,
                                   urgency=urgencies[i], is_visited=has_been_visited)

        for sensor in collecting_sensors:
            self.ax.plot([sensor.position[0], self.uav.position[0]],
                         [sensor.position[1], self.uav.position[1]],
                         color='purple', linewidth=2.5, linestyle='--', alpha=0.7, zorder=8)

            mid_x = (sensor.position[0] + self.uav.position[0]) / 2
            mid_y = (sensor.position[1] + self.uav.position[1]) / 2
            sf = collecting_sensors_sf.get(sensor.sensor_id, sensor.spreading_factor)

            self.ax.text(mid_x, mid_y, f'SF{sf}', fontsize=8, color='white', fontweight='bold',
                         ha='center', va='center', bbox=dict(boxstyle='round,pad=0.3',
                         facecolor='purple', edgecolor='white', linewidth=1.5, alpha=0.9), zorder=9)

        render_uav_enhanced(self.ax, self.uav)

        self.ax.set_xlim(0, self.grid_size[0])
        self.ax.set_ylim(0, self.grid_size[1])
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('X Coordinate', fontsize=12)
        self.ax.set_ylabel('Y Coordinate', fontsize=12)
        self.ax.axhline(0, color='black', linewidth=1.5, alpha=0.7, zorder=1)
        self.ax.axvline(0, color='black', linewidth=1.5, alpha=0.7, zorder=1)

        title = (f'Step: {self.current_step}/{self.max_steps} | '
                 f'Battery: {self.uav.battery:.1f}Wh ({self.uav.get_battery_percentage():.0f}%) | '
                 f'Collected: {len(self.sensors_visited)}/{self.num_sensors} | '
                 f'Reward: {self.total_reward:.1f}')
        self.ax.set_title(title, fontsize=11, fontweight='bold', pad=10)

        avg_buffer = np.mean([s.data_buffer for s in self.sensors])
        max_urgency = np.max(urgencies)
        avg_urgency = np.mean(urgencies)
        high_urgency_count = np.sum(urgencies > 0.8)

        stats_text = (
            f'Data Collected: {self.total_data_collected:.1f} bytes\n'
            f'Coverage: {(len(self.sensors_visited) / self.num_sensors) * 100:.0f}%\n'
            f'Battery Used: {self.uav.max_battery - self.uav.battery:.1f}Wh\n'
            f'Avg Buffer: {avg_buffer:.1f} bytes\n'
            f'\nURGENCY METRICS:\n'
            f'Max Urgency: {max_urgency:.2f}\n'
            f'Avg Urgency: {avg_urgency:.2f}\n'
            f'High Urgency (>0.8): {high_urgency_count}'
        )

        if len(collecting_sensors) > 0:
            stats_text += f'\n\nCollecting: {len(collecting_sensors)} sensor(s)'
            if len(collecting_sensors) > 1:
                stats_text += '\nMulti-sensor collection!'
            sf_list = sorted(set(collecting_sensors_sf.values()))
            stats_text += f'\nSFs: {sf_list}'

        if len(sensors_in_range) > 0:
            stats_text += f'\nIn Range: {len(sensors_in_range)} sensor(s)'

        self.ax.text(0.02, 0.98, stats_text, transform=self.ax.transAxes, fontsize=9,
                     verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        legend_elements = [
            Patch(facecolor='blue', alpha=0.9, edgecolor='black', label='Full Buffer (100%)'),
            Patch(facecolor='yellow', alpha=0.7, edgecolor='black', label='Partial Buffer (1-99%)'),
            Patch(facecolor='green', alpha=0.5, edgecolor='black', label='Collected (empty)'),
            Patch(facecolor='purple', alpha=1.0, edgecolor='black', label='Currently Collecting'),
            Patch(facecolor='lightblue', alpha=0.5, edgecolor='black', label='Empty Buffer'),
            Patch(facecolor='orange', edgecolor='red', linewidth=2, label='UAV Position'),
            # Add this line:
            plt.Line2D([0], [0], marker='$✓$', color='w', markeredgecolor='green', markersize=10,
                       label='Visited Sensor'),
            plt.Line2D([0], [0], color='purple', linewidth=2.5, linestyle='--', label='Active Collection Link'),
            plt.Line2D([0], [0], color='lightblue', linewidth=1, linestyle=':', label='In Communication Range'),
            plt.Line2D([0], [0], marker='o', color='w', label='Urgency: 🔴>0.8 🟠>0.5 🟡>0.2 🟢≤0.2', markersize=0, linestyle='None'),
        ]

        self.ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.02, 0.5),
                       fontsize=8, framealpha=0.9, title='Legend', title_fontsize=9, borderaxespad=0)

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
    print("Testing UAV Environment with FAIRNESS CONSTRAINTS")
    print("=" * 70)
    print()

    # Create environment WITH FAIRNESS
    env = UAVEnvironment(
        grid_size=(100, 100),
        uav_start_position=(50, 50),
        num_sensors=20,
        max_steps=2100,
        sensor_duty_cycle=10.0,
        penalty_data_loss=-500.0,
        reward_urgency_reduction=20.0,
        render_mode='human'
    )

    # Reset environment
    obs, info = env.reset(seed=42)

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
                      f"Urgency: Max={info['max_urgency']:.2f} Avg={info['avg_urgency']:.2f} High={info['high_urgency_sensors']} | "
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
        print("\n\n⏸Stopped by user (Ctrl+C)")

    # Summary
    print()
    print("=" * 70)
    print("Episode Summary (with Fairness Metrics):")
    print("=" * 70)
    print(f"  Total Steps: {info['current_step']}")
    print(f"  Total Reward: {info['total_reward']:.2f}")
    print(f"  Coverage: {info['coverage_percentage']:.1f}%")
    print(f"  Data Collected: {info['total_data_collected']:.2f} bytes")
    print(f"  Battery Used: {274.0 - info['battery']:.2f} Wh")
    print(f"  Final Max Urgency: {info['max_urgency']:.3f}")
    print(f"  Final Avg Urgency: {info['avg_urgency']:.3f}")
    print(f"  High Urgency Sensors: {info['high_urgency_sensors']}")
    print("=" * 70)

    # Keep window open at the end
    print("\n✓ Test complete! Close the matplotlib window to exit.")
    plt.ioff()
    plt.show()