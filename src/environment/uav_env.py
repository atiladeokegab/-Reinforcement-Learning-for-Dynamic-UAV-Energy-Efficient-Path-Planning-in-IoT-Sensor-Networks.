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
from typing import List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Patch, Circle
import sys
from pathlib import Path
import time
import random
from enum import IntEnum

_HERE = Path(__file__).resolve().parent  # …/src/environment
_SRC  = _HERE.parent                     # …/src

for _p in [str(_HERE), str(_SRC)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

from iot_sensors import IoTSensor
from uav import UAV
from rewards.reward_function import RewardFunction


# ─────────────────────────────────────────────────────────────────────────────
# Visual helpers
# ─────────────────────────────────────────────────────────────────────────────

class SensorState(IntEnum):
    """Visual states for sensors based on buffer level"""
    EMPTY      = 0
    LOW        = 1
    MEDIUM     = 2
    HIGH       = 3
    FULL       = 4
    COLLECTING = 5
    COLLECTED  = 6


def get_sensor_visual_state(sensor, uav_position, current_action=None) -> SensorState:
    """Determine visual state of sensor based on buffer level and UAV interaction."""
    is_in_range = sensor.is_in_range(uav_position)
    buffer_pct  = (sensor.data_buffer / sensor.max_buffer_size) * 100

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


def render_sensor_enhanced(
    ax,
    sensor,
    current_step,
    uav_position,
    current_action=None,
    urgency=0.0,
    is_visited=True,
):
    """Render a single sensor with enhanced visual states and urgency indicator."""
    x, y  = sensor.position
    state = get_sensor_visual_state(sensor, uav_position, current_action)

    if state == SensorState.EMPTY:
        color, marker_size, alpha, marker = "lightblue", 80,  0.5, "o"
    elif state == SensorState.LOW:
        color, marker_size, alpha, marker = "yellow",   120, 0.7, "o"
    elif state == SensorState.MEDIUM:
        color, marker_size, alpha, marker = "yellow",   160, 0.8, "o"
    elif state == SensorState.HIGH:
        color, marker_size, alpha, marker = "yellow",   200, 0.9, "o"
    elif state == SensorState.FULL:
        pulse = 0.5 + 0.5 * np.sin(current_step * 0.3)
        color = "blue"
        marker_size = 250 * (0.9 + 0.2 * pulse)
        alpha  = 0.7 + 0.2 * pulse
        marker = "o"
    elif state == SensorState.COLLECTING:
        color, marker_size, alpha, marker = "purple", 300, 1.0, "*"
    elif state == SensorState.COLLECTED:
        color, marker_size, alpha, marker = "green",  100, 0.5, "o"
    else:
        color, marker_size, alpha, marker = "gray",   80,  0.5, "o"

    ax.scatter(x, y, c=color, marker=marker, s=marker_size, alpha=alpha,
               edgecolors="black", linewidths=2, zorder=5)

    if sensor.data_buffer > 0:
        ready_ring = Circle((x, y), 0.8, fill=False, edgecolor="lime",
                            linewidth=1.5, linestyle="-", alpha=0.8, zorder=4)
        ax.add_patch(ready_ring)

    buffer_pct = (sensor.data_buffer / sensor.max_buffer_size) * 100
    text_color = ("white"
                  if state in [SensorState.FULL, SensorState.COLLECTING]
                  else "black")

    ax.text(x, y - 0.4, f"{int(buffer_pct)}%",
            ha="center", va="top", fontsize=7, fontweight="bold", color=text_color,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      edgecolor="black", alpha=0.7))

    ax.text(x, y + 0.5, f"S{sensor.sensor_id}",
            ha="center", va="bottom", fontsize=7, fontweight="bold")

    if urgency > 0.8:
        urgency_symbol = "🔴"
    elif urgency > 0.5:
        urgency_symbol = "🟠"
    elif urgency > 0.2:
        urgency_symbol = "🟡"
    else:
        urgency_symbol = "🟢"

    ax.text(x + 0.5, y + 0.5, urgency_symbol,
            fontsize=8, ha="center", va="center", zorder=12)

    if urgency > 0.3:
        ax.text(x, y - 0.8, f"U:{urgency:.2f}",
                fontsize=6, ha="center", va="top",
                color="red" if urgency > 0.8 else "orange",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    if is_visited:
        ax.text(x - 0.6, y + 0.5, "✓",
                fontsize=12, fontweight="bold", color="darkgreen",
                ha="center", va="center", zorder=20,
                bbox=dict(boxstyle="circle,pad=0.1", facecolor="white",
                          edgecolor="green", alpha=0.8))

    if state == SensorState.COLLECTING:
        collection_circle = Circle((x, y), 0.5, facecolor="none",
                                   edgecolor="purple", linewidth=3,
                                   linestyle="--", alpha=0.6, zorder=6)
        ax.add_patch(collection_circle)


def render_uav_enhanced(ax, uav):
    """Render the UAV with battery indicator."""
    uav_x, uav_y = uav.position

    uav_marker = patches.FancyBboxPatch(
        (uav_x - 0.25, uav_y - 0.25), 0.5, 0.5,
        boxstyle="round,pad=0.05", edgecolor="red", facecolor="orange",
        linewidth=2.5, zorder=10)
    ax.add_patch(uav_marker)

    ax.text(uav_x, uav_y, "✈",
            ha="center", va="center", fontweight="bold", fontsize=14,
            color="white", zorder=11)

    battery_pct   = uav.get_battery_percentage()
    battery_color = ("green" if battery_pct > 50
                     else ("orange" if battery_pct > 25 else "red"))

    ax.text(uav_x, uav_y - 0.8, f"⚡{battery_pct:.0f}%",
            ha="center", va="top", fontsize=8, fontweight="bold",
            color=battery_color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=battery_color, linewidth=2, alpha=0.9),
            zorder=11)


# ─────────────────────────────────────────────────────────────────────────────
# Main environment
# ─────────────────────────────────────────────────────────────────────────────

class UAVEnvironment(gym.Env):
    """
    Custom Gymnasium Environment for UAV IoT Data Collection with Fairness Constraints.

    Observation Space:
        Box: [uav_x, uav_y, battery,
              sensor1_buffer, sensor1_urgency, sensor1_link_quality, ...,
              sensorN_buffer, sensorN_urgency, sensorN_link_quality]
        (+ 2 relative-position features per sensor when include_sensor_positions=True)

    Action Space:
        Discrete(5): [UP, DOWN, LEFT, RIGHT, COLLECT]

    Reward Structure (Fairness-Constrained):
        +0.1 per byte  : Data collection
        +10.0          : New sensor collected
        +20.0 per unit : Urgency reduction
        -500.0 per byte: Data loss (MASSIVE PENALTY - applies to ALL actions)
        -2.0           : Attempted collection from empty sensor
        -5.0           : Boundary collision
        -0.1 per Wh    : Battery drain
        -0.05          : Step penalty
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        grid_size: Tuple[int, int] = (10, 10),
        sensor_positions: Optional[List[Tuple[float, float]]] = None,
        num_sensors: int = 20,
        data_generation_rate: float = 22.0 / 10,
        max_buffer_size: float = 1000.0,
        lora_spreading_factor: int = 7,
        path_loss_exponent: float = 2.0,
        rssi_threshold: float = -85.0,
        sensor_duty_cycle: float = 10.0,
        uav_start_position: Optional[Tuple[float, float]] = None,
        max_battery: float = 274.0,
        collection_duration: float = 1.0,
        max_steps: int = 2100,
        render_mode: Optional[str] = None,
        penalty_data_loss: float = -1.0,
        reward_urgency_reduction: float = 20.0,
        penalty_battery: float = -0.5,
        reward_movement: float = 10.0,
        include_sensor_positions: bool = False,
    ):
        """Initialize UAV environment with fairness constraints."""
        super().__init__()

        self.grid_size   = grid_size
        self.max_steps   = max_steps
        self.render_mode = render_mode
        self.collection_duration = collection_duration

        # ── FIX 2: initialise ALL instance variables here, BEFORE reset() ──
        # reset() (called at the bottom of __init__) references these fields,
        # so they must exist first.
        self.last_successful_collections = []
        self.last_action                 = None
        self.previous_data_loss          = 0.0
        self.capture_effect_triggers     = 0   # ← was missing; caused AttributeError
        self.last_step_bytes_collected   = 0.0
        self.total_reward                = 0.0
        self.total_data_collected        = 0.0
        self.sensors_visited: set        = set()
        self.current_step                = 0

        # ── Sensor setup ──
        if sensor_positions is None:
            self.sensor_positions = self._generate_uniform_sensor_positions(num_sensors)
        else:
            self.sensor_positions = sensor_positions

        self.num_sensors = len(self.sensor_positions)

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
                duty_cycle=sensor_duty_cycle,
            )
            self.sensors.append(sensor)

        # ── UAV & reward function ──
        if uav_start_position is None:
            uav_start_position = (0, 0)

        self.uav = UAV(
            start_position=uav_start_position,
            max_battery=max_battery,
        )
        self.reward_fn = RewardFunction(
            penalty_data_loss=penalty_data_loss,
            reward_urgency_reduction=reward_urgency_reduction,
            penalty_battery=penalty_battery,
            reward_movement=reward_movement,
        )

        # ── Action / observation spaces ──
        self.action_space = spaces.Discrete(5)
        self.include_sensor_positions = include_sensor_positions
        self._features_per_sensor     = 5 if include_sensor_positions else 3

        obs_dim = 3 + self._features_per_sensor * self.num_sensors
        obs_low  = np.full(obs_dim, -1.0, dtype=np.float32)
        obs_high = np.ones(obs_dim,       dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high,
                                            dtype=np.float32)

        # ── Rendering handles ──
        self.fig = None
        self.ax  = None

        # NOTE: we intentionally do NOT call self.reset() here.
        # Gymnasium's convention is for the caller to call reset() before
        # the first step().  Calling it in __init__ is harmless but it used
        # to hide bugs; keeping it explicit makes the lifecycle clearer.

    # ─────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────────────────────

    def _generate_uniform_sensor_positions(
        self, num_sensors: int
    ) -> List[Tuple[float, float]]:
        """Generate uniformly distributed sensor positions across the grid."""
        x_max, y_max = float(self.grid_size[0]), float(self.grid_size[1])
        coordinates  = np.random.uniform(
            low=[0.0, 0.0], high=[x_max, y_max], size=(num_sensors, 2)
        )
        return [(float(x), float(y)) for x, y in coordinates]

    def _calculate_urgency(self, sensor: IoTSensor) -> float:
        """Calculate urgency metric for a sensor (buffer utilisation + loss rate)."""
        buffer_utilization = sensor.data_buffer / sensor.max_buffer_size
        data_loss_rate     = (
            sensor.total_data_lost / sensor.total_data_generated
            if sensor.total_data_generated > 0 else 0.0
        )
        urgency = buffer_utilization * (1.0 + data_loss_rate * 10.0)
        return float(np.clip(urgency, 0.0, 1.0))

    def _get_sensor_urgencies(self) -> np.ndarray:
        """Calculate urgency for all sensors (AoI approximation)."""
        urgencies = np.zeros(len(self.sensors), dtype=np.float32)
        for i, sensor in enumerate(self.sensors):
            if sensor.data_generation_rate > 0:
                urgencies[i] = sensor.data_buffer / sensor.data_generation_rate
            else:
                urgencies[i] = 0.0
        return urgencies

    # ─────────────────────────────────────────────────────────────────────
    # Gymnasium API
    # ─────────────────────────────────────────────────────────────────────

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[np.ndarray, dict]:
        """Reset environment to initial state."""
        super().reset(seed=seed)

        self.uav.reset()
        for sensor in self.sensors:
            sensor.reset(initial_buffer_fill=self.np_random.uniform(0.20, 0.60))

        # Reset all episode-level counters
        self.current_step              = 0
        self.total_reward              = 0.0
        self.total_data_collected      = 0.0
        self.sensors_visited           = set()
        self.last_action               = None
        self.previous_data_loss        = 0.0
        self.capture_effect_triggers   = 0
        self.last_step_bytes_collected = 0.0
        self.last_successful_collections = []

        return self._get_observation(), self._get_info()

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one environment step.

        Step duration is dynamic:
          - Move  (0-3): 1.0 s
          - Collect (4): self.collection_duration s
        """
        self.current_step += 1
        self.last_action  = action

        # ── Dynamic step duration ──────────────────────────────────────
        step_duration = self.collection_duration if action == 4 else 1.0

        # ── Age all sensors by the real elapsed time ───────────────────
        for sensor in self.sensors:
            sensor.step(time_step=step_duration)

        # ── Global data-loss delta for this step ──────────────────────
        current_data_loss = sum(s.total_data_lost for s in self.sensors)
        step_data_loss    = current_data_loss - self.previous_data_loss
        self.previous_data_loss = current_data_loss

        # ── Execute chosen action ──────────────────────────────────────
        if action in [0, 1, 2, 3]:
            self.last_step_bytes_collected = 0.0
            reward = self._execute_move_action(action, step_data_loss)
        elif action == 4:
            reward = self._execute_collect_action(step_data_loss)
        else:
            raise ValueError(f"Invalid action: {action}")

        # ── Termination checks ────────────────────────────────────────
        terminated = False
        truncated  = False

        if not self.uav.is_alive():
            truncated = True

        if self.current_step >= self.max_steps:
            truncated = True

        # Terminal penalties
        if truncated:
            unvisited = self.num_sensors - len(self.sensors_visited)
            if unvisited > 0:
                reward += self.reward_fn.penalty_unvisited * unvisited
            reward += self.reward_fn.calculate_terminal_starvation_penalty(self.sensors)

        self.total_reward += reward
        return self._get_observation(), reward, terminated, truncated, self._get_info()

    # ─────────────────────────────────────────────────────────────────────
    # Action helpers
    # ─────────────────────────────────────────────────────────────────────

    def _execute_move_action(self, action: int, step_data_loss: float) -> float:
        """Execute movement action; data-loss penalty applied here too."""
        direction_map = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
        direction     = direction_map[action]

        battery_before = self.uav.battery
        move_success   = self.uav.move(direction, self.grid_size, time_step=1)
        battery_used   = battery_before - self.uav.battery

        self.last_successful_collections = []

        reward = self.reward_fn.calculate_movement_reward(
            move_success=move_success,
            battery_used=battery_used,
        )

        # Data-loss penalty (same formula as collect path)
        reward += self.reward_fn.penalty_data_loss * step_data_loss

        return reward

    def _execute_collect_action(self, step_data_loss: float) -> float:
        """
        Execute data collection with Capture Effect collision handling.

        Implements Equation 11 (Capture Effect):
            P_r,i / (Σ P_r,j + N0) >= τ_cap  (6 dB threshold)
        """
        # ── Phase 0: urgency snapshot before collection ────────────────
        urgencies_before = self._get_sensor_urgencies()

        # ── Phase 1: UAV hover ─────────────────────────────────────────
        self.uav.hover(duration=self.collection_duration)
        battery_used = self.uav.battery_drain_hover * self.collection_duration

        # ── Phase 2: probabilistic transmission attempts ───────────────
        transmission_attempts: dict = {}   # SF -> [sensors]

        for sensor in self.sensors:
            if sensor.data_buffer <= 0:
                continue

            sensor.update_spreading_factor(
                tuple(self.uav.position), current_step=self.current_step
            )

            p_link    = sensor.get_success_probability(
                tuple(self.uav.position), use_advanced_model=True
            )
            p_cycle   = sensor.duty_cycle_probability
            p_overall = p_link * p_cycle

            if p_overall > random.random():
                sf = sensor.spreading_factor
                transmission_attempts.setdefault(sf, []).append(sensor)

        # ── Phase 3: collision resolution (Capture Effect) ────────────
        successful_sf_slots: dict = {}   # SF -> winning sensor
        collision_count = 0

        for sf, attempting in transmission_attempts.items():
            if len(attempting) == 1:
                successful_sf_slots[sf] = attempting[0]
            else:
                collision_count += len(attempting) - 1

                sorted_sensors = sorted(
                    attempting, key=lambda s: s.current_rssi, reverse=True
                )
                strongest = sorted_sensors[0]
                second    = sorted_sensors[1]

                if strongest.current_rssi > (second.current_rssi + 6.0):
                    successful_sf_slots[sf] = strongest
                    self.capture_effect_triggers += 1
                # else: destructive interference – no winner

        # ── Phase 4: collect data from winning sensors ─────────────────
        total_bytes_collected = 0.0
        new_sensors_collected: List[int] = []

        self.last_successful_collections = []

        for sf, winning_sensor in successful_sf_slots.items():
            bytes_collected, success = winning_sensor.collect_data(
                uav_position=tuple(self.uav.position),
                collection_duration=self.collection_duration,
            )

            if success and bytes_collected > 0:
                total_bytes_collected += bytes_collected
                self.total_data_collected += bytes_collected

                if winning_sensor.sensor_id not in self.sensors_visited:
                    new_sensors_collected.append(winning_sensor.sensor_id)
                    self.sensors_visited.add(winning_sensor.sensor_id)

                self.last_successful_collections.append((winning_sensor, sf))

        attempted_empty = any(s.data_buffer <= 0 for s in self.sensors)

        # ── Phase 5: fairness metrics ──────────────────────────────────
        urgencies_after  = self._get_sensor_urgencies()
        urgency_reduced  = float(
            np.sum(np.maximum(0, urgencies_before - urgencies_after))
        )

        all_sensors_collected = all(s.data_buffer <= 0 for s in self.sensors)

        # ── Phase 6: reward ────────────────────────────────────────────
        self.last_step_bytes_collected = total_bytes_collected
        current_buffers = [float(s.data_buffer) for s in self.sensors]

        if successful_sf_slots:
            # Use _calculate_urgency (clipped [0,1]) not _get_sensor_urgencies
            # (which returns AoI in time-units and would inflate the byte reward ~250x)
            mean_urgency = float(
                np.mean([self._calculate_urgency(s) for s in successful_sf_slots.values()])
            )
        else:
            mean_urgency = 0.0

        reward = self.reward_fn.calculate_collection_reward(
            bytes_collected=total_bytes_collected,
            was_new_sensor=len(new_sensors_collected) > 0,
            was_empty=attempted_empty,
            all_sensors_collected=all_sensors_collected,
            battery_used=battery_used,
            collision_count=collision_count,
            data_loss=step_data_loss,
            urgency_reduced=urgency_reduced,
            sensor_buffers=current_buffers,
            sensor_urgency=mean_urgency,
        )

        return reward

    # ─────────────────────────────────────────────────────────────────────
    # Observation / info
    # ─────────────────────────────────────────────────────────────────────

    def _get_observation(self) -> np.ndarray:
        """Return normalised observation vector."""
        W, H   = float(self.grid_size[0]), float(self.grid_size[1])

        # ── FIX 3: extract scalars explicitly to avoid 0-d array indexing ──
        # uav.position is np.ndarray shape (2,); index safely with [0] / [1].
        uav_x = float(self.uav.position[0])
        uav_y = float(self.uav.position[1])

        obs_list = [uav_x / W, uav_y / H, self.uav.battery / self.uav.max_battery]

        sf_quality = {7: 1.0, 8: 0.8, 9: 0.6, 10: 0.4, 11: 0.2, 12: 0.1}

        # ── FIX 4: pass position as a plain Python tuple of floats ──────────
        # Some code paths call np.array(uav_position) inside IoTSensor;
        # passing a tuple of Python floats avoids accidental 0-d arrays.
        uav_pos_tuple = (uav_x, uav_y)

        for sensor in self.sensors:
            urgency = self._calculate_urgency(sensor)

            sensor.update_spreading_factor(uav_pos_tuple)

            link_quality = (
                sf_quality.get(sensor.spreading_factor, 0.1)
                if sensor.is_in_range(uav_pos_tuple)
                else 0.0
            )

            obs_list.extend([
                sensor.data_buffer / sensor.max_buffer_size,
                urgency,
                link_quality,
            ])

            if self.include_sensor_positions:
                s_x = float(sensor.position[0])
                s_y = float(sensor.position[1])
                obs_list.append((s_x - uav_x) / W)
                obs_list.append((s_y - uav_y) / H)

        return np.array(obs_list, dtype=np.float32)

    def _get_info(self) -> dict:
        """Return info dict including urgency statistics."""
        urgencies = self._get_sensor_urgencies()
        return {
            "uav_position":            self.uav.position.copy(),
            "battery":                 self.uav.battery,
            "battery_percent":         self.uav.get_battery_percentage(),
            "sensors_collected":       len(self.sensors_visited),
            "current_step":            self.current_step,
            "total_reward":            self.total_reward,
            "total_data_collected":    self.total_data_collected,
            "coverage_percentage":     (len(self.sensors_visited) / self.num_sensors) * 100,
            "is_alive":                self.uav.is_alive(),
            "max_urgency":             float(np.max(urgencies)),
            "avg_urgency":             float(np.mean(urgencies)),
            "high_urgency_sensors":    int(np.sum(urgencies > 0.8)),
            "capture_effect_triggers": self.capture_effect_triggers,
            "last_step_bytes_collected": self.last_step_bytes_collected,
            "sensor_collection_ratios": [
                s.total_data_transmitted / max(s.total_data_generated, 1e-6)
                for s in self.sensors
            ],
        }

    # ─────────────────────────────────────────────────────────────────────
    # Rendering
    # ─────────────────────────────────────────────────────────────────────

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
            self.ax.axhline(i, color="gray", linewidth=0.5, alpha=0.3)
        for i in range(self.grid_size[1] + 1):
            self.ax.axvline(i, color="gray", linewidth=0.5, alpha=0.3)

        urgencies = self._get_sensor_urgencies()

        sensors_in_range = [
            s for s in self.sensors if s.is_in_range(self.uav.position)
        ]

        collecting_sensors    = []
        collecting_sensors_sf = {}

        if self.last_action == 4 and self.last_successful_collections:
            for sensor, sf in self.last_successful_collections:
                collecting_sensors.append(sensor)
                collecting_sensors_sf[sensor.sensor_id] = sf

        for sensor in sensors_in_range:
            if sensor not in collecting_sensors:
                self.ax.plot(
                    [sensor.position[0], self.uav.position[0]],
                    [sensor.position[1], self.uav.position[1]],
                    color="lightblue", linewidth=1, linestyle=":", alpha=0.3, zorder=2,
                )

        for i, sensor in enumerate(self.sensors):
            is_collecting  = sensor in collecting_sensors
            has_been_visited = sensor.sensor_id in self.sensors_visited
            render_sensor_enhanced(
                self.ax, sensor, self.current_step, self.uav.position,
                current_action=self.last_action if is_collecting else None,
                urgency=urgencies[i],
                is_visited=has_been_visited,
            )

        for sensor in collecting_sensors:
            self.ax.plot(
                [sensor.position[0], self.uav.position[0]],
                [sensor.position[1], self.uav.position[1]],
                color="purple", linewidth=2.5, linestyle="--", alpha=0.7, zorder=8,
            )
            mid_x = (sensor.position[0] + self.uav.position[0]) / 2
            mid_y = (sensor.position[1] + self.uav.position[1]) / 2
            sf    = collecting_sensors_sf.get(sensor.sensor_id, sensor.spreading_factor)
            self.ax.text(
                mid_x, mid_y, f"SF{sf}",
                fontsize=8, color="white", fontweight="bold",
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="purple",
                          edgecolor="white", linewidth=1.5, alpha=0.9),
                zorder=9,
            )

        render_uav_enhanced(self.ax, self.uav)

        self.ax.set_xlim(0, self.grid_size[0])
        self.ax.set_ylim(0, self.grid_size[1])
        self.ax.set_aspect("equal")
        self.ax.set_xlabel("X Coordinate", fontsize=12)
        self.ax.set_ylabel("Y Coordinate", fontsize=12)
        self.ax.axhline(0, color="black", linewidth=1.5, alpha=0.7, zorder=1)
        self.ax.axvline(0, color="black", linewidth=1.5, alpha=0.7, zorder=1)

        title = (
            f"Step: {self.current_step}/{self.max_steps} | "
            f"Battery: {self.uav.battery:.1f}Wh "
            f"({self.uav.get_battery_percentage():.0f}%) | "
            f"Collected: {len(self.sensors_visited)}/{self.num_sensors} | "
            f"Reward: {self.total_reward:.1f}"
        )
        self.ax.set_title(title, fontsize=11, fontweight="bold", pad=10)

        avg_buffer        = np.mean([s.data_buffer for s in self.sensors])
        max_urgency       = np.max(urgencies)
        avg_urgency       = np.mean(urgencies)
        high_urgency_count = int(np.sum(urgencies > 0.8))

        stats_text = (
            f"Data Collected: {self.total_data_collected:.1f} bytes\n"
            f"Coverage: {(len(self.sensors_visited) / self.num_sensors) * 100:.0f}%\n"
            f"Battery Used: {self.uav.max_battery - self.uav.battery:.1f}Wh\n"
            f"Avg Buffer: {avg_buffer:.1f} bytes\n"
            f"\nURGENCY METRICS:\n"
            f"Max Urgency: {max_urgency:.2f}\n"
            f"Avg Urgency: {avg_urgency:.2f}\n"
            f"High Urgency (>0.8): {high_urgency_count}"
        )

        if collecting_sensors:
            stats_text += f"\n\nCollecting: {len(collecting_sensors)} sensor(s)"
            if len(collecting_sensors) > 1:
                stats_text += "\nMulti-sensor collection!"
            sf_list = sorted(set(collecting_sensors_sf.values()))
            stats_text += f"\nSFs: {sf_list}"

        if sensors_in_range:
            stats_text += f"\nIn Range: {len(sensors_in_range)} sensor(s)"

        self.ax.text(
            0.02, 0.98, stats_text,
            transform=self.ax.transAxes, fontsize=9, verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )

        legend_elements = [
            Patch(facecolor="blue",     alpha=0.9, edgecolor="black",
                  label="Full Buffer (100%)"),
            Patch(facecolor="yellow",   alpha=0.7, edgecolor="black",
                  label="Partial Buffer (1-99%)"),
            Patch(facecolor="green",    alpha=0.5, edgecolor="black",
                  label="Collected (empty)"),
            Patch(facecolor="purple",   alpha=1.0, edgecolor="black",
                  label="Currently Collecting"),
            Patch(facecolor="lightblue",alpha=0.5, edgecolor="black",
                  label="Empty Buffer"),
            Patch(facecolor="orange",   edgecolor="red", linewidth=2,
                  label="UAV Position"),
            plt.Line2D([0], [0], marker="$✓$", color="w",
                       markeredgecolor="green", markersize=10,
                       label="Visited Sensor"),
            plt.Line2D([0], [0], color="purple", linewidth=2.5, linestyle="--",
                       label="Active Collection Link"),
            plt.Line2D([0], [0], color="lightblue", linewidth=1, linestyle=":",
                       label="In Communication Range"),
            plt.Line2D([0], [0], marker="o", color="w", markersize=0,
                       linestyle="None",
                       label="Urgency: 🔴>0.8 🟠>0.5 🟡>0.2 🟢≤0.2"),
        ]

        self.ax.legend(
            handles=legend_elements,
            loc="center left", bbox_to_anchor=(1.02, 0.5),
            fontsize=8, framealpha=0.9,
            title="Legend", title_fontsize=9, borderaxespad=0,
        )

        plt.tight_layout()
        plt.subplots_adjust(right=0.82)

        if self.render_mode == "rgb_array":
            self.fig.canvas.draw()
            image = np.frombuffer(
                self.fig.canvas.tostring_rgb(), dtype="uint8"
            )
            image = image.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return image

    def close(self):
        """Close the environment."""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax  = None


# ─────────────────────────────────────────────────────────────────────────────
# Quick smoke-test (run directly with: uv run uav_env.py)
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 70)
    print("Testing UAV Environment with FAIRNESS CONSTRAINTS")
    print("=" * 70)

    env = UAVEnvironment(
        grid_size=(100, 100),
        uav_start_position=(0, 0),
        num_sensors=20,
        max_steps=2100,
        sensor_duty_cycle=10.0,
        penalty_data_loss=-1.0,
        reward_urgency_reduction=20.0,
        render_mode="human",
    )

    obs, info = env.reset(seed=42)
    action_names = ["UP", "DOWN", "LEFT", "RIGHT", "COLLECT"]

    try:
        for step in range(10_000):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()

            if step % 20 == 0 or terminated or truncated:
                print(
                    f"Step {step + 1:3d}: {action_names[action]:7s} | "
                    f"Pos: ({info['uav_position'][0]:.1f}, "
                    f"{info['uav_position'][1]:.1f}) | "
                    f"Battery: {info['battery']:6.1f}Wh | "
                    f"Urgency: Max={info['max_urgency']:.2f} "
                    f"Avg={info['avg_urgency']:.2f} "
                    f"High={info['high_urgency_sensors']} | "
                    f"Reward: {reward:+7.2f}"
                )

            if terminated:
                print("\n✓ Mission complete! All sensors collected.")
                env.render()
                time.sleep(5)
                break
            elif truncated:
                msg = "Battery depleted!" if not info["is_alive"] else "Timeout reached."
                print(f"\n✗ {msg}")
                env.render()
                time.sleep(5)
                break

    except KeyboardInterrupt:
        print("\n\n⏸ Stopped by user (Ctrl+C)")

    print()
    print("=" * 70)
    print("Episode Summary:")
    print("=" * 70)
    print(f"  Total Steps:          {info['current_step']}")
    print(f"  Total Reward:         {info['total_reward']:.2f}")
    print(f"  Coverage:             {info['coverage_percentage']:.1f}%")
    print(f"  Data Collected:       {info['total_data_collected']:.2f} bytes")
    print(f"  Battery Used:         {274.0 - info['battery']:.2f} Wh")
    print(f"  Final Max Urgency:    {info['max_urgency']:.3f}")
    print(f"  Final Avg Urgency:    {info['avg_urgency']:.3f}")
    print(f"  High Urgency Sensors: {info['high_urgency_sensors']}")
    print("=" * 70)
    print("\n✓ Test complete! Close the matplotlib window to exit.")
    plt.ioff()
    plt.show()