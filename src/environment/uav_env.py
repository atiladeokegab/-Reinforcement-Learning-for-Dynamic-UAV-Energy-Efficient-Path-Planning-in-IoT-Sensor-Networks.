"""Gymnasium-compatible UAV environment for IoT data collection with fairness and battery constraints.

Author: ATILADE GABRIEL OKE
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

# src/ and src/environment/ are pre-inserted by dqn.py before this module is
# imported.  Recompute here only as a fallback for direct script execution.
if not any("environment" in p for p in sys.path):
    _HERE = Path(__file__).resolve().parent
    _SRC  = _HERE.parent
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
    grid_size=None,
):
    """Render a single sensor with a soft glow/halo treatment that scales with grid."""
    x, y  = sensor.position
    state = get_sensor_visual_state(sensor, uav_position, current_action)

    # Scale all sensor geometry with grid extent so they stay visible on big maps.
    if grid_size is not None:
        span = float(max(grid_size[0], grid_size[1]))
    else:
        xlim = ax.get_xlim()
        span = float(max(abs(xlim[1] - xlim[0]), 1.0))
    r_core  = max(span * 0.008, 0.35)           # core dot radius (data units)
    r_halo  = r_core * 2.6                       # outer halo radius
    label_fs = min(max(int(span * 0.012), 7), 13)

    # Per-state palette — tuned for strong contrast on white background.
    if state == SensorState.EMPTY:
        core_c, halo_c, core_a = "#334155", "#64748b", 0.85   # slate
    elif state == SensorState.LOW:
        core_c, halo_c, core_a = "#fb923c", "#c2410c", 0.9    # light orange
    elif state == SensorState.MEDIUM:
        core_c, halo_c, core_a = "#ea580c", "#9a3412", 0.95   # deep orange
    elif state == SensorState.HIGH:
        core_c, halo_c, core_a = "#dc2626", "#7f1d1d", 1.0    # strong red
    elif state == SensorState.FULL:
        pulse = 0.5 + 0.5 * np.sin(current_step * 0.3)
        core_c, halo_c = "#1d4ed8", "#1e3a8a"                  # deep blue
        core_a = 0.9 + 0.1 * pulse
        r_core *= 1.0 + 0.15 * pulse
        r_halo *= 1.0 + 0.25 * pulse
    elif state == SensorState.COLLECTING:
        core_c, halo_c, core_a = "#7e22ce", "#4c1d95", 1.0    # rich purple
        r_core *= 1.25
        r_halo *= 1.4
    elif state == SensorState.COLLECTED:
        core_c, halo_c, core_a = "#15803d", "#14532d", 0.95   # deep green
    else:
        core_c, halo_c, core_a = "#6b7280", "#374151", 0.8    # gray

    # Outer halo — soft glow
    ax.add_patch(Circle((x, y), r_halo, facecolor=halo_c,
                        edgecolor="none", alpha=0.22, zorder=3))
    # Mid ring
    ax.add_patch(Circle((x, y), r_halo * 0.65, facecolor=halo_c,
                        edgecolor="none", alpha=0.32, zorder=4))
    # Core
    ax.add_patch(Circle((x, y), r_core, facecolor=core_c,
                        edgecolor="white", linewidth=1.2,
                        alpha=core_a, zorder=5))

    # Ready/data-available ring
    if sensor.data_buffer > 0 and state not in (SensorState.COLLECTING,):
        ax.add_patch(Circle((x, y), r_core * 1.5, fill=False,
                            edgecolor="#7cf27c", linewidth=1.2,
                            linestyle="-", alpha=0.7, zorder=6))

    # Collecting highlight — pulsing dashed ring
    if state == SensorState.COLLECTING:
        ax.add_patch(Circle((x, y), r_halo * 1.1, facecolor="none",
                            edgecolor="#a64bff", linewidth=2.0,
                            linestyle="--", alpha=0.75, zorder=7))

    # Sensor ID label
    ax.text(x, y + r_halo * 1.15, f"S{sensor.sensor_id}",
            ha="center", va="bottom", fontsize=label_fs, fontweight="bold",
            color="#222", zorder=11,
            bbox=dict(boxstyle="round,pad=0.18", facecolor="white",
                      edgecolor="none", alpha=0.75))

    # Visited tick — small corner badge
    if is_visited:
        ax.text(x - r_halo * 0.95, y + r_halo * 0.95, "✓",
                fontsize=label_fs, fontweight="bold", color="#1b8a3a",
                ha="center", va="center", zorder=12,
                bbox=dict(boxstyle="circle,pad=0.1", facecolor="white",
                          edgecolor="#1b8a3a", linewidth=1.0, alpha=0.9))

    # High-urgency flag
    if urgency > 0.5:
        flag_c = "#e63946" if urgency > 0.8 else "#ff8c42"
        ax.add_patch(Circle((x + r_halo * 0.9, y - r_halo * 0.9),
                            r_core * 0.55, facecolor=flag_c,
                            edgecolor="white", linewidth=1.0,
                            alpha=0.95, zorder=12))


def render_uav_enhanced(ax, uav, grid_size=None):
    """Render the UAV as a stylized quadcopter (body + 4 rotors) that scales with grid."""
    uav_x, uav_y = float(uav.position[0]), float(uav.position[1])

    if grid_size is not None:
        span = float(max(grid_size[0], grid_size[1]))
    else:
        xlim = ax.get_xlim()
        span = float(max(abs(xlim[1] - xlim[0]), 1.0))

    # Overall UAV scale in data units.
    R       = max(span * 0.022, 0.8)     # rotor-centre distance from UAV centre
    rotor_r = R * 0.55                    # rotor disc radius
    body_r  = R * 0.55                    # central body radius
    battery_fs = min(max(int(span * 0.015), 8), 18)
    offset  = R * 2.6                     # battery label offset below

    # Four rotor arms (dark struts) — X-config quadcopter
    arm_offsets = [(+R, +R), (-R, +R), (-R, -R), (+R, -R)]
    for dx, dy in arm_offsets:
        ax.plot([uav_x, uav_x + dx], [uav_y, uav_y + dy],
                color="#2a2a2a", linewidth=3.5, solid_capstyle="round",
                zorder=9)

    # Rotor discs — spin blur (soft outer ring + hub)
    for dx, dy in arm_offsets:
        cx, cy = uav_x + dx, uav_y + dy
        ax.add_patch(Circle((cx, cy), rotor_r * 1.15,
                            facecolor="#4aa8ff", edgecolor="none",
                            alpha=0.25, zorder=9))
        ax.add_patch(Circle((cx, cy), rotor_r,
                            facecolor="#111111", edgecolor="#4aa8ff",
                            linewidth=1.5, alpha=0.85, zorder=10))
        ax.add_patch(Circle((cx, cy), rotor_r * 0.25,
                            facecolor="#dddddd", edgecolor="#111111",
                            linewidth=0.8, zorder=11))

    # Central body — glossy orange disc with red rim
    ax.add_patch(Circle((uav_x, uav_y), body_r * 1.25,
                        facecolor="#ff6a00", edgecolor="none",
                        alpha=0.25, zorder=10))
    ax.add_patch(Circle((uav_x, uav_y), body_r,
                        facecolor="#ff8a1a", edgecolor="#b22222",
                        linewidth=2.0, zorder=11))
    # Camera / gimbal dot
    ax.add_patch(Circle((uav_x, uav_y), body_r * 0.32,
                        facecolor="#1a1a1a", edgecolor="#dddddd",
                        linewidth=1.0, zorder=12))

    # Forward-heading indicator (small nose arrow pointing +x)
    nose = R * 1.25
    ax.annotate("",
                xy=(uav_x + nose, uav_y),
                xytext=(uav_x + body_r * 0.8, uav_y),
                arrowprops=dict(arrowstyle="-|>", color="#b22222",
                                lw=2.0, mutation_scale=12),
                zorder=12)

    battery_pct   = uav.get_battery_percentage()
    battery_color = ("#2ecc71" if battery_pct > 50
                     else ("#ff8c42" if battery_pct > 25 else "#e63946"))

    ax.text(uav_x, uav_y - offset, f"⚡ {battery_pct:.0f}%",
            ha="center", va="top", fontsize=battery_fs, fontweight="bold",
            color=battery_color,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                      edgecolor=battery_color, linewidth=1.8, alpha=0.92),
            zorder=12)


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
        +100 × urgency per byte : Data collection (urgency-weighted)
        +5000                   : New sensor first visit
        +1000 × reduction       : Urgency reduction
        -2                      : Revisit empty sensor
        -50                     : Boundary hit
        -1000 × variance        : Starvation fairness penalty
        -1000 per sensor        : Terminal starvation (CR < 20%)
        -5000 per sensor        : Unvisited sensors at episode end
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
        reward_urgency_reduction: float = 1000.0,
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

        # All instance variables must be defined before reset() is called.
        self.last_successful_collections = []
        self.last_action                 = None
        self.previous_data_loss          = 0.0
        self.capture_effect_triggers     = 0   # ← was missing; caused AttributeError
        self.last_step_bytes_collected   = 0.0
        self.total_reward                = 0.0
        self.total_data_collected        = 0.0
        self.sensors_visited: set        = set()
        self.current_step                = 0
        self.boundary_hits               = 0   # failed moves (attempted to leave grid)
        self.edge_steps                  = 0   # steps spent on boundary cells

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
        self.boundary_hits             = 0
        self.edge_steps                = 0
        self.last_step_bytes_collected = 0.0
        self.last_successful_collections = []
        self._path_history = [(float(self.uav.position[0]),
                               float(self.uav.position[1]))]

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

        # Track whether UAV is currently parked on a boundary cell
        W, H = float(self.grid_size[0]), float(self.grid_size[1])
        ux, uy = float(self.uav.position[0]), float(self.uav.position[1])
        eps = 1e-6
        if ux <= eps or uy <= eps or ux >= W - 1 - eps or uy >= H - 1 - eps:
            self.edge_steps += 1

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

        if not move_success:
            self.boundary_hits += 1

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

        uav_x = float(self.uav.position[0])
        uav_y = float(self.uav.position[1])

        obs_list = [uav_x / W, uav_y / H, self.uav.battery / self.uav.max_battery]

        sf_quality = {7: 1.0, 8: 0.8, 9: 0.6, 10: 0.4, 11: 0.2, 12: 0.1}

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
            "boundary_hits":           self.boundary_hits,
            "edge_steps":              self.edge_steps,
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
        # Clean white background — no gridlines.
        self.ax.set_facecolor("white")

        # Append current position to path history and draw the trail.
        pos = (float(self.uav.position[0]), float(self.uav.position[1]))
        if not hasattr(self, "_path_history") or self._path_history is None:
            self._path_history = [pos]
        elif self._path_history[-1] != pos:
            self._path_history.append(pos)

        if len(self._path_history) >= 2:
            px = [p[0] for p in self._path_history]
            py = [p[1] for p in self._path_history]
            # White halo underneath for contrast on any background
            self.ax.plot(px, py, color="white", linewidth=4.0,
                         alpha=0.35, zorder=6, solid_capstyle="round")
            self.ax.plot(px, py, color="deepskyblue", linewidth=2.2,
                         alpha=0.85, zorder=7, solid_capstyle="round")

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
                grid_size=self.grid_size,
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

        render_uav_enhanced(self.ax, self.uav, grid_size=self.grid_size)

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