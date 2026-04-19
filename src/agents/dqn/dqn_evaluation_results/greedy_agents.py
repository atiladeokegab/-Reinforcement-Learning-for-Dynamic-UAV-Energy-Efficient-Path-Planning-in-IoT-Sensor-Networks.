"""
Greedy baseline agents for UAV data collection evaluation.

Two baselines used across all evaluation scripts:
  1. NearestSensorGreedy  — distance-based (SF-agnostic)
  2. MaxThroughputGreedyV2 — SF-aware with multi-objective scoring

Author: ATILADE GABRIEL OKE
"""

import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

import numpy as np
from typing import List, Optional, Dict

from environment.uav_env import UAVEnvironment


# ==================== BASE CLASS ====================

class GreedyAgent:
    """Base class for greedy algorithms."""

    ACTION_UP      = 0
    ACTION_DOWN    = 1
    ACTION_LEFT    = 2
    ACTION_RIGHT   = 3
    ACTION_COLLECT = 4

    def __init__(self, env: UAVEnvironment):
        self.env = env

    def select_action(self, observation: np.ndarray) -> int:
        raise NotImplementedError

    def _move_toward(self, target_pos: np.ndarray) -> int:
        """Move one step toward target_pos using cardinal directions."""
        uav_pos = self.env.uav.position
        dx = target_pos[0] - uav_pos[0]
        dy = target_pos[1] - uav_pos[1]

        if abs(dx) <= 0.5 and abs(dy) <= 0.5:
            return self.ACTION_COLLECT

        if abs(dx) > abs(dy):
            new_x = uav_pos[0] + (1 if dx > 0 else -1)
            new_y = uav_pos[1]
        else:
            new_x = uav_pos[0]
            new_y = uav_pos[1] + (1 if dy > 0 else -1)

        if (new_x < 0 or new_x >= self.env.grid_size[0] or
                new_y < 0 or new_y >= self.env.grid_size[1]):
            return self.ACTION_COLLECT

        move_dx = new_x - uav_pos[0]
        move_dy = new_y - uav_pos[1]

        if move_dx > 0:   return self.ACTION_RIGHT
        if move_dx < 0:   return self.ACTION_LEFT
        if move_dy > 0:   return self.ACTION_UP
        if move_dy < 0:   return self.ACTION_DOWN
        return self.ACTION_COLLECT


# ==================== NEAREST SENSOR GREEDY ====================

class NearestSensorGreedy(GreedyAgent):
    """
    Distance-based baseline: always move toward the nearest sensor
    that has data in its buffer. Collects immediately if any in-range
    sensor has data.
    """

    def select_action(self, observation: np.ndarray) -> int:
        uav_pos = self.env.uav.position

        # Collect if any in-range sensor has data
        for sensor in self.env.sensors:
            if sensor.data_buffer > 0 and sensor.is_in_range(tuple(uav_pos)):
                return self.ACTION_COLLECT

        # Otherwise navigate to nearest sensor with data
        target = self._nearest_with_data()
        if target is None:
            return self.ACTION_COLLECT
        return self._move_toward(target.position)

    def _nearest_with_data(self):
        uav_pos = self.env.uav.position
        candidates = [s for s in self.env.sensors if s.data_buffer > 0]
        if not candidates:
            return None
        return min(candidates,
                   key=lambda s: np.linalg.norm(s.position - uav_pos))


# ==================== SF-AWARE GREEDY (V2) ====================

class MaxThroughputGreedyV2(GreedyAgent):
    """
    SF-aware baseline with multi-objective scoring.

    Phase 1 — if any in-range sensors have data and acceptable SF:
               collect from the best one (highest SF priority,
               largest buffer, closest).
    Phase 2 — navigate to the globally best sensor using a scored
               objective that balances SF priority, buffer fullness,
               and travel distance.  Out-of-range sensors are included
               as navigation targets; the distance penalty naturally
               deprioritises far sensors without excluding them.
    """

    WEIGHT_SF_PRIORITY    = 5.0
    WEIGHT_BUFFER         = 10.0
    WEIGHT_DISTANCE       = 5.0
    WEIGHT_DUTY_CYCLE     = 2.0

    def __init__(self, env: UAVEnvironment):
        super().__init__(env)
        self.target_sensor = None

    def select_action(self, observation: np.ndarray) -> int:
        uav_pos      = self.env.uav.position
        battery_pct  = self.env.uav.battery / 274.0
        steps_left   = (self.env.max_steps - self.env.current_step
                        if hasattr(self.env, "max_steps") else float("inf"))
        sf_threshold = self._adaptive_sf_threshold(battery_pct, steps_left)

        # Phase 1: collect from best in-range sensor with acceptable SF
        immediate = [
            s for s in self.env.sensors
            if s.data_buffer > 0
            and s.is_in_range(tuple(uav_pos))
            and s.spreading_factor <= sf_threshold
        ]
        if immediate:
            self.target_sensor = max(
                immediate,
                key=lambda s: (
                    self._sf_priority(s.spreading_factor),
                    s.data_buffer,
                    -np.linalg.norm(s.position - uav_pos),
                ),
            )
            return self.ACTION_COLLECT

        # Phase 2: navigate to globally best target
        self.target_sensor = self._best_global_target(uav_pos, battery_pct, steps_left)
        if self.target_sensor is None:
            return self.ACTION_COLLECT
        return self._move_toward(self.target_sensor.position)

    def _best_global_target(self, uav_pos, battery_pct, steps_left):
        """Score all sensors with data and return the best navigation target."""
        if battery_pct < 0.1 or steps_left < 50:
            sf_w, dist_w = 1.0, 1.0
        elif battery_pct < 0.3 or steps_left < 150:
            sf_w, dist_w = 2.0, 1.0
        else:
            sf_w, dist_w = 5.0, 1.0

        best_score  = -np.inf
        best_target = None

        for sensor in self.env.sensors:
            if sensor.data_buffer <= 0:
                continue

            distance         = np.linalg.norm(sensor.position - uav_pos)
            sf_score         = self._sf_priority(sensor.spreading_factor) * self.WEIGHT_SF_PRIORITY * sf_w
            buffer_score     = (sensor.data_buffer / sensor.max_buffer_size) * self.WEIGHT_BUFFER
            duty_score       = sensor.duty_cycle_probability * self.WEIGHT_DUTY_CYCLE
            distance_penalty = (distance / self.env.grid_size[0]) * self.WEIGHT_DISTANCE * dist_w

            score = sf_score + buffer_score + duty_score - distance_penalty
            if score > best_score:
                best_score  = score
                best_target = sensor

        return best_target

    def _sf_priority(self, sf: int) -> float:
        """Higher value = better SF (SF7 → 6, SF12 → 1)."""
        return max(0, 13 - sf)

    def _adaptive_sf_threshold(self, battery_pct: float, steps_left) -> int:
        steps_ratio = min(1.0, steps_left / self.env.max_steps
                          if hasattr(self.env, "max_steps") else 1.0)
        if battery_pct > 0.5 and steps_ratio > 0.5:
            return 9
        elif battery_pct > 0.2 and steps_ratio > 0.2:
            return 10
        else:
            return 12


# ==================== QUICK SMOKE TEST ====================

if __name__ == "__main__":
    env = UAVEnvironment(grid_size=(500, 500), num_sensors=20, max_steps=2100)
    obs, _ = env.reset(seed=0)

    for name, AgentClass in [("NearestSensorGreedy", NearestSensorGreedy),
                              ("MaxThroughputGreedyV2", MaxThroughputGreedyV2)]:
        env.reset(seed=0)
        agent = AgentClass(env)
        done  = False
        total_reward = 0.0
        while not done:
            action = agent.select_action(obs)
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            done = term or trunc
        ndr = len(env.sensors_visited) / env.num_sensors * 100
        print(f"{name}: reward={total_reward:.0f}  NDR={ndr:.1f}%")

    env.close()
