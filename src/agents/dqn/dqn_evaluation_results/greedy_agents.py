"""
Greedy baseline agents for UAV data collection evaluation.

Agents in this module:
  1. NearestSensorGreedy    — distance-based (SF-agnostic)
  2. MaxThroughputGreedyV2  — SF-aware with multi-objective scoring
  3. TSPOracleAgent         — near-optimal TSP tour (NN + 2-opt), upper-bound benchmark

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


# ==================== TSP ORACLE ====================

def _tsp_nearest_neighbour(positions: np.ndarray, start_idx: int = 0) -> List[int]:
    """
    Nearest-neighbour construction heuristic.
    Returns a list of indices into `positions`, starting and ending at `start_idx`.
    """
    n = len(positions)
    unvisited = set(range(n))
    tour = [start_idx]
    unvisited.remove(start_idx)
    while unvisited:
        last = tour[-1]
        nearest = min(unvisited,
                      key=lambda j: np.linalg.norm(positions[last] - positions[j]))
        tour.append(nearest)
        unvisited.remove(nearest)
    tour.append(start_idx)  # return to origin
    return tour


def _tour_length(positions: np.ndarray, tour: List[int]) -> float:
    return sum(
        np.linalg.norm(positions[tour[i]] - positions[tour[i + 1]])
        for i in range(len(tour) - 1)
    )


def _two_opt_improve(positions: np.ndarray, tour: List[int]) -> List[int]:
    """
    2-opt local search.  Iterates until no improving swap is found.
    The first and last elements of `tour` are the depot (start/end) and are kept fixed.
    """
    best = tour[:]
    best_len = _tour_length(positions, best)
    improved = True
    inner = best[1:-1]  # don't touch depot endpoints
    n = len(inner)
    while improved:
        improved = False
        for i in range(n - 1):
            for j in range(i + 1, n):
                new_inner = inner[:i] + inner[i:j + 1][::-1] + inner[j + 1:]
                candidate = [best[0]] + new_inner + [best[-1]]
                cand_len = _tour_length(positions, candidate)
                if cand_len < best_len - 1e-9:
                    best = candidate
                    best_len = cand_len
                    inner = best[1:-1]
                    improved = True
                    break
            if improved:
                break
    return best


class TSPOracleAgent(GreedyAgent):
    """
    TSP Oracle — near-optimal distance upper-bound benchmark.

    At the start of each episode the agent:
      1. Builds a node list: [UAV start] + [all sensor positions]
      2. Solves the TSP with nearest-neighbour + 2-opt local search
         (polynomial, finds near-optimal tours for N ≤ ~100 sensors)
      3. Stores the ordered sensor visit list as its policy

    During the episode it follows that fixed tour, navigating to each sensor
    in order and collecting data (action COLLECT) at each waypoint.

    Key metrics exposed after an episode:
      tsp_tour_length_units : Euclidean tour length in grid units (theoretical min)
      actual_path_length    : Euclidean distance of actual steps taken (always ≥ TSP)
      path_efficiency_pct   : tsp_tour_length / actual * 100  (100% = perfect)

    The TSP Oracle ignores dynamic factors (battery, SF, buffer urgency) by design.
    Its value is as a distance-optimal *lower bound* on path length, letting you
    measure how much extra distance the DQN (or greedy agents) fly and why.
    """

    def __init__(self, env: UAVEnvironment):
        super().__init__(env)
        self._tour_sensor_order: List[int] = []   # indices into env.sensors
        self._current_waypoint: int = 0
        self.tsp_tour_length_units: float = 0.0
        self.actual_path_length: float = 0.0
        self._prev_uav_pos: Optional[np.ndarray] = None
        self._plan_computed: bool = False

    # ── Public interface ─────────────────────────────────────────────────────

    def reset(self):
        """Call after env.reset() to rebuild the TSP tour for the new layout."""
        self._plan_computed = False
        self.actual_path_length = 0.0
        self._prev_uav_pos = None

    @property
    def path_efficiency_pct(self) -> float:
        if self.actual_path_length < 1e-6:
            return 100.0
        return min(100.0, self.tsp_tour_length_units / self.actual_path_length * 100.0)

    # ── Core logic ───────────────────────────────────────────────────────────

    def _ensure_plan(self):
        """Lazily compute TSP tour on first call (after env has been reset)."""
        if self._plan_computed:
            return

        uav_start = self.env.uav.position.copy()
        sensor_positions = np.array([s.position for s in self.env.sensors])

        # Node 0 = UAV start;  nodes 1..N = sensors
        all_positions = np.vstack([uav_start[np.newaxis, :], sensor_positions])

        nn_tour = _tsp_nearest_neighbour(all_positions, start_idx=0)
        opt_tour = _two_opt_improve(all_positions, nn_tour)

        self.tsp_tour_length_units = _tour_length(all_positions, opt_tour)

        # Convert tour node indices (which include depot=0) to sensor indices (0-based)
        self._tour_sensor_order = [idx - 1 for idx in opt_tour if idx > 0]
        self._current_waypoint = 0
        self._prev_uav_pos = uav_start.copy()
        self._plan_computed = True

        print(
            f"  [TSPOracle] Tour planned: {len(self._tour_sensor_order)} sensors, "
            f"tour length = {self.tsp_tour_length_units:.1f} grid units"
        )

    def select_action(self, observation: np.ndarray) -> int:
        self._ensure_plan()

        uav_pos = self.env.uav.position
        if self._prev_uav_pos is not None:
            self.actual_path_length += float(
                np.linalg.norm(uav_pos - self._prev_uav_pos)
            )
        self._prev_uav_pos = uav_pos.copy()

        if self._current_waypoint >= len(self._tour_sensor_order):
            return self.ACTION_COLLECT  # all sensors visited — hover/collect

        target_sensor_idx = self._tour_sensor_order[self._current_waypoint]
        target_sensor = self.env.sensors[target_sensor_idx]
        target_pos = target_sensor.position

        dx = target_pos[0] - uav_pos[0]
        dy = target_pos[1] - uav_pos[1]
        at_waypoint = abs(dx) <= 0.5 and abs(dy) <= 0.5

        if at_waypoint:
            # Collect while there is data; advance waypoint once drained or on first arrival
            if target_sensor.data_buffer > 0:
                return self.ACTION_COLLECT
            # Buffer empty — move on
            self._current_waypoint += 1
            if self._current_waypoint >= len(self._tour_sensor_order):
                return self.ACTION_COLLECT
            target_sensor_idx = self._tour_sensor_order[self._current_waypoint]
            target_sensor = self.env.sensors[target_sensor_idx]

        return self._move_toward(target_sensor.position)


# ==================== QUICK SMOKE TEST ====================

if __name__ == "__main__":
    env = UAVEnvironment(grid_size=(500, 500), num_sensors=20, max_steps=2100)
    obs, _ = env.reset(seed=0)

    for name, AgentClass in [
        ("NearestSensorGreedy",   NearestSensorGreedy),
        ("MaxThroughputGreedyV2", MaxThroughputGreedyV2),
        ("TSPOracleAgent",        TSPOracleAgent),
    ]:
        obs, _ = env.reset(seed=0)
        agent = AgentClass(env)
        if hasattr(agent, "reset"):
            agent.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = agent.select_action(obs)
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            done = term or trunc
        ndr = len(env.sensors_visited) / env.num_sensors * 100
        extra = ""
        if isinstance(agent, TSPOracleAgent):
            extra = (
                f"  TSP_tour={agent.tsp_tour_length_units:.0f}u  "
                f"actual={agent.actual_path_length:.0f}u  "
                f"efficiency={agent.path_efficiency_pct:.1f}%"
            )
        print(f"{name}: reward={total_reward:.0f}  NDR={ndr:.1f}%{extra}")

    env.close()
