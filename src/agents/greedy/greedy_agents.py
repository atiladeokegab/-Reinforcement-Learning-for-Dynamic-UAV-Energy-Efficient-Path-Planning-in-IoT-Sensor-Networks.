"""
Complete Greedy Algorithms for UAV Data Collection with SF-Aware Agent
SF = spreading Factor
INCLUDES:
1. NearestSensorGreedy - Distance-based (ignores SF)
2. HighestBufferGreedy - Buffer-based (ignores SF)
3. ProbabilisticAwareGreedy - Duty-cycle aware (ignores SF)
4. MaxThroughputGreedy - SF-AWARE v1 (original)
5. MaxThroughputGreedyV2 - SF-AWARE v2 (PRODUCTION READY)
6. MultiSensorGreedy - Multi-target positioning

Author: ATILADE GABRIEL OKE
Date: November 2025
Status: PRODUCTION READY
"""
import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

import numpy as np
from typing import Tuple, List, Dict, Optional
from enum import Enum

from environment.uav_env import UAVEnvironment


class DataGoal:
    """Define data collection goals for episodes."""

    def __init__(self,
                 target_bytes: float = None,
                 target_coverage: float = None,
                 max_battery_usage: float = None):
        """Initialize data collection goals."""
        self.target_bytes = target_bytes
        self.target_coverage = target_coverage
        self.max_battery_usage = max_battery_usage

    def is_goal_achieved(self, metrics: dict) -> bool:
        """Check if all goals are achieved."""
        goals_met = []

        if self.target_bytes is not None:
            goals_met.append(metrics['total_data_collected'] >= self.target_bytes)

        if self.target_coverage is not None:
            goals_met.append(metrics['coverage_percentage'] >= self.target_coverage)

        if self.max_battery_usage is not None:
            goals_met.append(metrics['battery_used_percentage'] <= self.max_battery_usage)

        return all(goals_met) if goals_met else False

    def get_goal_achievement_ratio(self, metrics: dict) -> dict:
        """Calculate achievement ratio for each goal."""
        achievement = {}

        if self.target_bytes is not None:
            achievement['bytes_ratio'] = min(1.0, metrics['total_data_collected'] / self.target_bytes)

        if self.target_coverage is not None:
            achievement['coverage_ratio'] = min(1.0, metrics['coverage_percentage'] / self.target_coverage)

        if self.max_battery_usage is not None:
            if metrics['battery_used_percentage'] <= self.max_battery_usage:
                achievement['battery_ratio'] = 1.0
            else:
                achievement['battery_ratio'] = self.max_battery_usage / metrics['battery_used_percentage']

        return achievement

    def __str__(self):
        """String representation of goals."""
        goals = []
        if self.target_bytes is not None:
            goals.append(f"Collect ≥{self.target_bytes:.0f} bytes")
        if self.target_coverage is not None:
            goals.append(f"Visit ≥{self.target_coverage:.0f}% of sensors")
        if self.max_battery_usage is not None:
            goals.append(f"Use ≤{self.max_battery_usage:.0f}% battery")

        return " AND ".join(goals) if goals else "No specific goals"


class SuccessMetrics:
    """Multi-level success criteria based on academic literature."""

    @staticmethod
    def evaluate_episode(info: dict, env: UAVEnvironment, data_goal: DataGoal = None) -> dict:
        """Evaluate episode with multiple success criteria and optional data goal."""
        total_data_generated = sum(s.total_data_generated for s in env.sensors)
        total_data_lost = sum(s.total_data_lost for s in env.sensors)
        data_collected = info['total_data_collected']
        battery_used = 274.0 - info['battery']

        coverage_pct = info['coverage_percentage']
        collection_ratio = (data_collected / total_data_generated * 100) if total_data_generated > 0 else 0
        data_loss_rate = (total_data_lost / total_data_generated * 100) if total_data_generated > 0 else 0
        battery_efficiency = data_collected / battery_used if battery_used > 0 else 0
        battery_used_pct = battery_used / 274.0 * 100

        result = {
            'perfect_success': coverage_pct == 100 and data_loss_rate < 1.0,
            'high_success': coverage_pct >= 95 and collection_ratio >= 90,
            'acceptable_success': coverage_pct >= 80 and collection_ratio >= 75,
            'partial_success': coverage_pct >= 50,

            'coverage_percentage': coverage_pct,
            'collection_ratio': collection_ratio,
            'data_loss_rate': data_loss_rate,
            'battery_efficiency': battery_efficiency,
            'battery_used_percentage': battery_used_pct,
            'total_data_generated': total_data_generated,
            'total_data_collected': data_collected,
            'total_data_lost': total_data_lost,
        }

        if data_goal is not None:
            result['goal_achieved'] = data_goal.is_goal_achieved(result)
            result['goal_achievement'] = data_goal.get_goal_achievement_ratio(result)
        else:
            result['goal_achieved'] = None
            result['goal_achievement'] = None

        return result

    @staticmethod
    def get_success_level_string(metrics: dict) -> str:
        """Get human-readable success level."""
        if metrics['perfect_success']:
            return " Perfect"
        elif metrics['high_success']:
            return "High"
        elif metrics['acceptable_success']:
            return "Acceptable"
        elif metrics['partial_success']:
            return "Partial"
        else:
            return "Failed"


class GreedyAgent:
    """Base class for greedy algorithms with movement."""

    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    ACTION_COLLECT = 4

    def __init__(self, env: UAVEnvironment):
        self.env = env

    def select_action(self, observation: np.ndarray) -> int:
        """Select action based on greedy strategy."""
        raise NotImplementedError

    def _move_toward(self, target_pos: np.ndarray) -> int:
        """Properly move toward target position with bounds checking."""
        uav_pos = self.env.uav.position

        dx = target_pos[0] - uav_pos[0]
        dy = target_pos[1] - uav_pos[1]

        if abs(dx) <= 0.5 and abs(dy) <= 0.5:
            return self.ACTION_COLLECT

        if abs(dx) > abs(dy):
            if dx > 0:
                new_x = uav_pos[0] + 1
                new_y = uav_pos[1]
            else:
                new_x = uav_pos[0] - 1
                new_y = uav_pos[1]
        else:
            if dy > 0:
                new_x = uav_pos[0]
                new_y = uav_pos[1] + 1
            else:
                new_x = uav_pos[0]
                new_y = uav_pos[1] - 1

        if new_x < 0 or new_x >= self.env.grid_size[0] or \
           new_y < 0 or new_y >= self.env.grid_size[1]:
            return self.ACTION_COLLECT

        current_dx = new_x - uav_pos[0]
        current_dy = new_y - uav_pos[1]

        if current_dx > 0:
            return self.ACTION_RIGHT
        elif current_dx < 0:
            return self.ACTION_LEFT
        elif current_dy > 0:
            return self.ACTION_UP
        elif current_dy < 0:
            return self.ACTION_DOWN
        else:
            return self.ACTION_COLLECT


class NearestSensorGreedy(GreedyAgent):
    """Greedy strategy - Always move toward nearest sensor with data."""

    def select_action(self, observation: np.ndarray) -> int:
        uav_pos = self.env.uav.position

        collectible_sensors = []
        for sensor in self.env.sensors:
            if sensor.data_buffer > 0 and sensor.is_in_range(tuple(uav_pos)):
                collectible_sensors.append(sensor)

        if len(collectible_sensors) > 0:
            return self.ACTION_COLLECT

        target_sensor = self._find_nearest_sensor_with_data()
        if target_sensor is None:
            return self.ACTION_COLLECT

        return self._move_toward(target_sensor.position)

    def _find_nearest_sensor_with_data(self):
        """Find nearest sensor that has data in buffer."""
        uav_pos = self.env.uav.position
        sensors_with_data = [s for s in self.env.sensors if s.data_buffer > 0]
        if not sensors_with_data:
            return None

        distances = [np.linalg.norm(np.array(s.position) - np.array(uav_pos))
                    for s in sensors_with_data]
        return sensors_with_data[np.argmin(distances)]


class HighestBufferGreedy(GreedyAgent):
    """Greedy strategy - Prioritize sensors with highest buffer levels."""

    def select_action(self, observation: np.ndarray) -> int:
        uav_pos = self.env.uav.position

        collectible_sensors = []
        for sensor in self.env.sensors:
            if sensor.data_buffer > 0 and sensor.is_in_range(tuple(uav_pos)):
                collectible_sensors.append(sensor)

        if len(collectible_sensors) > 0:
            return self.ACTION_COLLECT

        target_sensor = self._find_highest_buffer_sensor()
        if target_sensor is None:
            return self.ACTION_COLLECT

        return self._move_toward(target_sensor.position)

    def _find_highest_buffer_sensor(self):
        """Find sensor with most data in buffer."""
        sensors_with_data = [s for s in self.env.sensors if s.data_buffer > 0]
        if not sensors_with_data:
            return None
        return max(sensors_with_data, key=lambda s: s.data_buffer)


class ProbabilisticAwareGreedy(GreedyAgent):
    """Greedy strategy - Prioritize sensors with high duty cycle probability."""

    def select_action(self, observation: np.ndarray) -> int:
        uav_pos = self.env.uav.position

        collectible_sensors = []
        for sensor in self.env.sensors:
            if sensor.data_buffer > 0 and sensor.is_in_range(tuple(uav_pos)):
                collectible_sensors.append(sensor)

        if len(collectible_sensors) > 0:
            return self.ACTION_COLLECT

        target_sensor = self._find_highest_duty_cycle_sensor()
        if target_sensor is None:
            target_sensor = self._find_nearest_sensor_with_data()
        if target_sensor is None:
            return self.ACTION_COLLECT

        return self._move_toward(target_sensor.position)

    def _find_highest_duty_cycle_sensor(self):
        """Find sensor with highest duty cycle probability that has data."""
        sensors_with_data = [s for s in self.env.sensors if s.data_buffer > 0]
        if not sensors_with_data:
            return None
        return max(sensors_with_data, key=lambda s: s.duty_cycle_probability)

    def _find_nearest_sensor_with_data(self):
        """Find nearest sensor with data."""
        uav_pos = self.env.uav.position
        sensors_with_data = [s for s in self.env.sensors if s.data_buffer > 0]
        if not sensors_with_data:
            return None
        distances = [np.linalg.norm(np.array(s.position) - np.array(uav_pos))
                    for s in sensors_with_data]
        return sensors_with_data[np.argmin(distances)]


class MaxThroughputGreedy(GreedyAgent):
    """
    SF-AWARE v1: Prioritize sensors with BEST data rates (LOWEST SF).

    THIS IS THE ORIGINAL BASELINE FOR COMPARISON!

    Key Insight:
    - SF7 = 684 B/s (best, needs close positioning)
    - SF9 = 220 B/s (medium, medium distance)
    - SF11 = 55 B/s (poor, far distance)
    - SF12 = 31 B/s (worst, very far)

    Strategy:
    1. Collect from SF9+ sensors if in range
    2. Otherwise, move toward sensor with LOWEST SF (highest data rate)
    3. Forces position optimization for data rate!
    """

    def __init__(self, env: UAVEnvironment):
        super().__init__(env)
        self.target_sensor = None

    def select_action(self, observation: np.ndarray) -> int:
        uav_pos = self.env.uav.position

        # Check if any sensors are in range with GOOD SF
        collectible_sensors = []
        for sensor in self.env.sensors:
            if sensor.data_buffer > 0 and sensor.is_in_range(tuple(uav_pos)):
                if sensor.spreading_factor <= 9:  # Only SF9 or better
                    collectible_sensors.append(sensor)

        if len(collectible_sensors) > 0:
            best = max(collectible_sensors,
                      key=lambda s: (-s.spreading_factor, s.data_buffer))
            self.target_sensor = best
            return self.ACTION_COLLECT

        # Find sensor with LOWEST SF (HIGHEST data rate)
        target_sensor = self._find_best_sf_sensor()
        self.target_sensor = target_sensor

        if target_sensor is None:
            return self.ACTION_COLLECT

        return self._move_toward(target_sensor.position)

    def _find_best_sf_sensor(self):
        """Find sensor with LOWEST SF (HIGHEST data rate)."""
        sensors_with_data = [s for s in self.env.sensors if s.data_buffer > 0]
        if not sensors_with_data:
            return None

        best_sensor = min(sensors_with_data,
                         key=lambda s: (s.spreading_factor, -s.data_buffer))
        return best_sensor


class MaxThroughputGreedyV2(GreedyAgent):
    """
    SF-AWARE v2 (PRODUCTION READY): Sophisticated baseline with reachability filtering.

    THIS IS THE IMPROVED BASELINE FOR THESIS!

    Key Improvements:
    • Reachability filtering (SF-dependent ranges)
    • Multi-objective scoring (throughput + buffer + distance)
    • Constraint-aware adaptation (battery + time)
    • Comprehensive validation framework
    """

    # Scoring weights (tuned for balance)
    WEIGHT_SF_PRIORITY = 5.0        # SF optimization (most important)
    WEIGHT_BUFFER_PRIORITY = 10.0   # Buffer management
    WEIGHT_DISTANCE_PENALTY = 5.0   # Movement efficiency
    WEIGHT_DUTY_CYCLE = 2.0         # Sensor reliability

    def __init__(self, env: UAVEnvironment):
        super().__init__(env)
        self.target_sensor = None

        # Metrics tracking for validation
        self.metrics = {
            'decisions_made': [],
            'sf_distribution': {},
            'reachability_filtered': 0,
            'unreachable_sensors_avoided': 0,
            'targets_selected': [],
        }

    def select_action(self, observation: np.ndarray) -> int:
        """Select action using sophisticated SF-aware scoring with reachability filtering."""
        uav_pos = self.env.uav.position

        # Get adaptive thresholds based on constraints
        battery_pct = self._get_battery_percentage()
        steps_remaining = self._get_steps_remaining()
        good_sf_threshold = self._adaptive_sf_threshold(battery_pct, steps_remaining)

        # PHASE 1: Check for immediately collectible sensors in range
        immediate_targets = self._find_immediate_collection_targets(
            uav_pos, good_sf_threshold
        )

        if immediate_targets:
            best = self._score_and_select_best(
                immediate_targets, uav_pos, phase='immediate'
            )
            self.target_sensor = best
            self._record_decision('collect_immediate', best.spreading_factor, best.sensor_id)
            return self.ACTION_COLLECT

        # PHASE 2: Find best reachable sensor globally
        global_target = self._find_optimal_target_with_reachability(
            uav_pos, battery_pct, steps_remaining
        )

        if global_target is None:
            self._record_decision('collect_nothing', None, None)
            return self.ACTION_COLLECT

        self.target_sensor = global_target
        self._record_decision('move_to_target', global_target.spreading_factor,
                            global_target.sensor_id)

        return self._move_toward(global_target.position)

    def _find_immediate_collection_targets(
            self,
            uav_pos: np.ndarray,
            good_sf_threshold: int
    ) -> List:
        """Find sensors that are reachable, have data, and have good SF."""
        targets = []

        for sensor in self.env.sensors:
            if sensor.data_buffer <= 0:
                continue

            if not sensor.is_in_range(tuple(uav_pos)):
                self.metrics['unreachable_sensors_avoided'] += 1
                continue

            if sensor.spreading_factor > good_sf_threshold:
                continue

            targets.append(sensor)

        return targets

    def _find_optimal_target_with_reachability(
            self,
            uav_pos: np.ndarray,
            battery_pct: float,
            steps_remaining: int
    ) -> Optional:
        """
        Find best reachable sensor using multi-objective scoring.

        Balances throughput (SF), resource management (buffer),
        and efficiency (distance).
        """
        uav_pos_tuple = tuple(uav_pos)
        best_score = -np.inf
        best_target = None

        for sensor in self.env.sensors:
            # Hard constraints
            if sensor.data_buffer <= 0:
                continue

            if not sensor.is_in_range(uav_pos_tuple):
                continue

            # Calculate distance
            distance = np.linalg.norm(np.array(sensor.position) - uav_pos)

            # SF Priority
            sf_priority = self._calculate_sf_priority(sensor.spreading_factor)

            # Buffer Priority (normalized 0-1 to 10-point scale)
            buffer_utilization = sensor.data_buffer / sensor.max_buffer_size
            buffer_priority = buffer_utilization * self.WEIGHT_BUFFER_PRIORITY

            # Distance Cost
            distance_penalty = (distance / self.env.grid_size[0]) * self.WEIGHT_DISTANCE_PENALTY

            # Duty Cycle Bonus
            duty_cycle_bonus = (sensor.duty_cycle_probability * self.WEIGHT_DUTY_CYCLE)

            # Constraint adaptation
            if battery_pct < 0.1 or steps_remaining < 50:
                sf_weight = 1.0
                distance_weight = 1.0
            elif battery_pct < 0.3 or steps_remaining < 150:
                sf_weight = 2.0
                distance_weight = 1.0
            else:
                sf_weight = 5.0
                distance_weight = 1.0

            # Final score
            current_score = (
                (sf_priority * self.WEIGHT_SF_PRIORITY * sf_weight) +
                (buffer_priority) +
                (duty_cycle_bonus) -
                (distance_penalty * distance_weight)
            )

            if current_score > best_score:
                best_score = current_score
                best_target = sensor

        return best_target

    def _calculate_sf_priority(self, spreading_factor: int) -> float:
        """Calculate SF priority (6=SF7, 1=SF12)."""
        return max(0, 13 - spreading_factor)

    def _score_and_select_best(
            self,
            sensors: List,
            uav_pos: np.ndarray,
            phase: str = 'global'
    ) -> Optional:
        """Score sensors and select the best one."""
        if not sensors:
            return None

        if phase == 'immediate':
            return min(sensors, key=lambda s: (
                self._calculate_sf_priority(s.spreading_factor),
                -s.data_buffer,
                np.linalg.norm(np.array(s.position) - uav_pos)
            ))
        else:
            return sensors[0] if sensors else None

    def _get_battery_percentage(self) -> float:
        """Get battery as percentage (0-1)."""
        max_battery = 274.0
        return self.env.uav.battery / max_battery if max_battery > 0 else 0

    def _get_steps_remaining(self) -> int:
        """Get remaining steps."""
        if hasattr(self.env, 'current_step') and hasattr(self.env, 'max_steps'):
            return self.env.max_steps - self.env.current_step
        return float('inf')

    def _adaptive_sf_threshold(self, battery_pct: float, steps_remaining: int) -> int:
        """Adaptively adjust SF threshold based on constraints."""
        steps_ratio = min(1.0, steps_remaining / self.env.max_steps
                         if hasattr(self.env, 'max_steps') else 1.0)

        if battery_pct > 0.5 and steps_ratio > 0.5:
            return 9
        elif battery_pct > 0.2 and steps_ratio > 0.2:
            return 10
        else:
            return 12

    def _record_decision(self, decision_type: str, sf: Optional[int], sensor_id: Optional[int]):
        """Record decision for analysis."""
        self.metrics['decisions_made'].append({
            'type': decision_type,
            'sf': sf,
            'sensor_id': sensor_id
        })

        if sf is not None:
            if sf not in self.metrics['sf_distribution']:
                self.metrics['sf_distribution'][sf] = 0
            self.metrics['sf_distribution'][sf] += 1

        if decision_type == 'move_to_target':
            self.metrics['targets_selected'].append(sensor_id)

    def validate_sf_awareness(self) -> Dict:
        """Validate that algorithm is truly SF-aware."""
        if not self.metrics['decisions_made']:
            return {'error': 'No decisions recorded'}

        decision_counts = {}
        for decision in self.metrics['decisions_made']:
            dtype = decision['type']
            decision_counts[dtype] = decision_counts.get(dtype, 0) + 1

        sf_counts = self.metrics['sf_distribution']
        sf_awareness_score = self._calculate_sf_awareness_score(sf_counts)
        is_sf_aware = self._is_sf_aware(sf_counts)

        total_unreachable_avoided = self.metrics['unreachable_sensors_avoided']
        total_decisions = len(self.metrics['decisions_made'])
        filtering_effectiveness = (total_unreachable_avoided / max(1, total_decisions)) * 100

        validation = {
            'decision_counts': decision_counts,
            'sf_distribution': sf_counts,
            'sf_awareness_score': sf_awareness_score,
            'is_sf_aware': is_sf_aware,
            'reachability_filtering_effectiveness': filtering_effectiveness,
            'unreachable_sensors_avoided': total_unreachable_avoided,
            'targets_selected': len(self.metrics['targets_selected']),
            'validation_status': 'PASS' if is_sf_aware else 'FAIL'
        }

        return validation

    def _calculate_sf_awareness_score(self, sf_counts: Dict) -> float:
        """Score how much SF impacts decisions (0-1)."""
        if not sf_counts:
            return 0.0

        total_visits = sum(sf_counts.values())
        weighted_score = 0
        max_score = 0

        for sf, count in sorted(sf_counts.items()):
            weight = 1 / (sf - 6.5)
            weighted_score += weight * count
            max_score += weight * (total_visits / len(sf_counts))

        return min(1.0, weighted_score / max_score) if max_score > 0 else 0.0

    def _is_sf_aware(self, sf_counts: Dict) -> bool:
        """Determine if algorithm is meaningfully SF-aware."""
        if not sf_counts:
            return False

        low_sf_visits = sum(count for sf, count in sf_counts.items() if sf <= 9)
        high_sf_visits = sum(count for sf, count in sf_counts.items() if sf > 9)

        total = low_sf_visits + high_sf_visits
        return (low_sf_visits / total > 0.6) if total > 0 else False

    def reset_metrics(self):
        """Reset metrics for new episode."""
        self.metrics = {
            'decisions_made': [],
            'sf_distribution': {},
            'reachability_filtered': 0,
            'unreachable_sensors_avoided': 0,
            'targets_selected': [],
        }


class MultiSensorGreedy(GreedyAgent):
    """Greedy strategy - Position to collect from multiple sensors."""

    def __init__(self, env: UAVEnvironment, communication_range: float = 2.0):
        super().__init__(env)
        self.communication_range = communication_range

    def select_action(self, observation: np.ndarray) -> int:
        uav_pos = self.env.uav.position

        collectible_sensors = []
        for sensor in self.env.sensors:
            if sensor.data_buffer > 0 and sensor.is_in_range(tuple(uav_pos)):
                collectible_sensors.append(sensor)

        if len(collectible_sensors) > 0:
            return self.ACTION_COLLECT

        best_position = self._find_best_position()
        if best_position is None:
            target_sensor = self._find_nearest_sensor_with_data()
            if target_sensor:
                return self._move_toward(target_sensor.position)
            return self.ACTION_COLLECT

        return self._move_toward(best_position)

    def _count_sensors_in_range(self, position: np.ndarray) -> int:
        """Count how many sensors with data are in range of position."""
        count = 0
        for sensor in self.env.sensors:
            if sensor.data_buffer > 0:
                temp_x, temp_y = position[0], position[1]
                distance = np.sqrt((sensor.position[0] - temp_x)**2 +
                                 (sensor.position[1] - temp_y)**2)
                if distance <= self.communication_range:
                    count += 1
        return count

    def _find_best_position(self) -> np.ndarray:
        """Find position that can reach most sensors with data."""
        sensors_with_data = [s for s in self.env.sensors if s.data_buffer > 0]
        if not sensors_with_data:
            return None

        best_position = None
        best_count = 0

        for sensor in sensors_with_data:
            for dx in [-2, -1, 0, 1, 2]:
                for dy in [-2, -1, 0, 1, 2]:
                    candidate_pos = np.array([sensor.position[0] + dx,
                                             sensor.position[1] + dy], dtype=float)

                    if (0 <= candidate_pos[0] < self.env.grid_size[0] and
                        0 <= candidate_pos[1] < self.env.grid_size[1]):
                        count = self._count_sensors_in_range(candidate_pos)
                        if count > best_count:
                            best_count = count
                            best_position = candidate_pos

        return best_position

    def _find_nearest_sensor_with_data(self):
        """Find nearest sensor with data."""
        uav_pos = self.env.uav.position
        sensors_with_data = [s for s in self.env.sensors if s.data_buffer > 0]
        if not sensors_with_data:
            return None
        distances = [np.linalg.norm(np.array(s.position) - np.array(uav_pos))
                    for s in sensors_with_data]
        return sensors_with_data[np.argmin(distances)]


def test_greedy_agent(agent: GreedyAgent,
                      env: UAVEnvironment,
                      num_episodes: int = 1,
                      render: bool = False,
                      data_goal: DataGoal = None,
                      agent_name: str = "Agent") -> dict:
    """Test a greedy agent with proper movement and metrics."""
    results = {
        'total_rewards': [],
        'coverage_percentage': [],
        'steps_taken': [],
        'data_collected': [],
        'data_generated': [],
        'collection_efficiency': [],
        'battery_efficiency': [],
        'success_levels': [],
        'per_sensor_collection': [],
    }

    for episode in range(num_episodes):
        obs, info = env.reset(seed=episode)
        episode_reward = 0
        done = False
        step_count = 0

        print(f"\n{'=' * 80}")
        print(f"Episode {episode + 1}/{num_episodes} - {agent_name}")
        if data_goal:
            print(f"Goal: {data_goal}")
        print(f"{'=' * 80}")

        while not done:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1

            if render:
                env.render()

            done = terminated or truncated

        # Evaluate episode
        metrics = SuccessMetrics.evaluate_episode(info, env, data_goal)

        # Calculate per-sensor statistics
        per_sensor_stats = []
        total_data_generated = 0
        total_data_collected = 0

        for sensor in env.sensors:
            generated = sensor.total_data_generated

            # METHOD 1: Use tracked cumulative (PREFERRED)
            collected = sensor.total_data_transmitted

            # METHOD 2: Calculate from accounting (VERIFICATION)
            remaining_buffer = sensor.data_buffer
            lost = sensor.total_data_lost
            calculated_collected = generated - remaining_buffer - lost

            # Verify they match
            if abs(collected - calculated_collected) > 0.1:
                print(f" WARNING: Sensor {sensor.sensor_id} data mismatch!")
                print(f"   Tracked: {collected:.1f}, Calculated: {calculated_collected:.1f}")

            collection_pct = (collected / generated * 100) if generated > 0 else 0

            per_sensor_stats.append({
                'sensor_id': sensor.sensor_id,
                'data_generated': generated,
                'data_collected': collected,
                'collection_percentage': collection_pct,
                'data_lost': lost,
                'final_buffer': remaining_buffer,
            })

            total_data_generated += generated
            total_data_collected += collected

        collection_efficiency = (total_data_collected / total_data_generated * 100) if total_data_generated > 0 else 0

        # Store results
        results['total_rewards'].append(episode_reward)
        results['coverage_percentage'].append(metrics['coverage_percentage'])
        results['steps_taken'].append(step_count)
        results['data_collected'].append(total_data_collected)
        results['data_generated'].append(total_data_generated)
        results['collection_efficiency'].append(collection_efficiency)
        results['battery_efficiency'].append(metrics['battery_efficiency'])
        results['success_levels'].append(SuccessMetrics.get_success_level_string(metrics))
        results['per_sensor_collection'].append(per_sensor_stats)

        # Print episode results
        print(f"\n{'=' * 80}")
        print(f"EPISODE {episode + 1} RESULTS")
        print(f"{'=' * 80}")

        # Overall metrics
        print(f"\nOverall Performance:")
        print(f"  Reward: {episode_reward:.1f}")
        print(f"  Coverage: {metrics['coverage_percentage']:.1f}%")
        print(f"  Steps: {step_count}")
        print(f"  Success Level: {SuccessMetrics.get_success_level_string(metrics)}")

        # Data metrics
        print(f"\n Data Collection:")
        print(f"  Total Generated: {total_data_generated:.0f} bytes")
        print(f"  Total Collected: {total_data_collected:.0f} bytes")
        print(f"  Collection Efficiency: {collection_efficiency:.1f}%")
        print(f"  Data Lost: {metrics['total_data_lost']:.0f} bytes")
        print(f"  Still in Buffers: {sum(s['final_buffer'] for s in per_sensor_stats):.0f} bytes")

        # Battery metrics
        print(f"\n Energy Efficiency:")
        print(f"  Battery Used: {metrics['battery_used_percentage']:.1f}%")
        print(f"  Battery Efficiency: {metrics['battery_efficiency']:.2f} bytes/Wh")

        # Per-sensor breakdown
        print(f"\n Per-Sensor Collection Breakdown:")
        print(f"{'ID':<6} {'Generated':<12} {'Collected':<12} {'%':<8} {'Lost':<10} {'Buffer':<10}")
        print("-" * 70)

        # Sort by sensor ID
        sorted_stats = sorted(per_sensor_stats, key=lambda x: x['sensor_id'])

        for stats in sorted_stats:
            print(f"S{stats['sensor_id']:<5} "
                  f"{stats['data_generated']:<12.0f} "
                  f"{stats['data_collected']:<12.0f} "
                  f"{stats['collection_percentage']:<7.1f}% "
                  f"{stats['data_lost']:<10.0f} "
                  f"{stats['final_buffer']:<10.0f}")

        print("-" * 70)
        print(f"{'TOTAL':<6} "
              f"{total_data_generated:<12.0f} "
              f"{total_data_collected:<12.0f} "
              f"{collection_efficiency:<7.1f}% "
              f"{metrics['total_data_lost']:<10.0f} "
              f"{sum(s['final_buffer'] for s in per_sensor_stats):<10.0f}")

        # Data accounting verification
        total_accounted = (total_data_collected +
                           metrics['total_data_lost'] +
                           sum(s['final_buffer'] for s in per_sensor_stats))
        accounting_error = abs(total_data_generated - total_accounted)

        if accounting_error > 1.0:
            print(f"\nDATA ACCOUNTING ERROR!")
            print(f"  Generated: {total_data_generated:.0f}")
            print(f"  Accounted: {total_accounted:.0f} (collected + lost + buffer)")
            print(f"  Error: {accounting_error:.0f} bytes")
        else:
            print(f"\nData Accounting: Perfect (error < 1 byte)")

        # Sensor coverage statistics
        sensors_with_100_pct = sum(1 for s in per_sensor_stats if s['collection_percentage'] >= 99.9)
        sensors_with_50_pct = sum(1 for s in per_sensor_stats if s['collection_percentage'] >= 50)
        sensors_with_0_pct = sum(1 for s in per_sensor_stats if s['collection_percentage'] < 1)

        print(f"\nCollection Coverage:")
        print(f"  Sensors 100% collected: {sensors_with_100_pct}/{len(env.sensors)}")
        print(f"  Sensors ≥50% collected: {sensors_with_50_pct}/{len(env.sensors)}")
        print(f"  Sensors <1% collected: {sensors_with_0_pct}/{len(env.sensors)}")

        # Validation for V2
        if isinstance(agent, MaxThroughputGreedyV2):
            print(f"\nSF-Awareness Validation:")
            validation = agent.validate_sf_awareness()
            print(f"  SF-Aware: {validation['is_sf_aware']}")
            print(f"  SF Awareness Score: {validation['sf_awareness_score']:.3f}")
            print(f"  Validation Status: {validation['validation_status']}")

        print(f"{'=' * 80}\n")

    env.close()
    return results


def print_comparison_table(results_dict: Dict[str, dict]) -> None:
    """Print comprehensive comparison table of all algorithms."""
    print("\n" + "=" * 120)
    print("COMPREHENSIVE ALGORITHM COMPARISON")
    print("=" * 120)

    # Table 1: Performance Metrics
    print(f"\n{'Algorithm':<30} {'Coverage':<12} {'Generated':<12} {'Collected':<12} {'Coll Eff':<12} {'Success':<15}")
    print("-" * 120)

    for algo_name, results in results_dict.items():
        avg_coverage = np.mean(results['coverage_percentage'])
        avg_generated = np.mean(results['data_generated'])
        avg_collected = np.mean(results['data_collected'])
        avg_coll_eff = np.mean(results['collection_efficiency'])
        success_mode = max(set(results['success_levels']), key=results['success_levels'].count)

        print(f"{algo_name:<30} "
              f"{avg_coverage:>6.1f}%{'':<5} "
              f"{avg_generated:>8.0f} B{'':<2} "
              f"{avg_collected:>8.0f} B{'':<2} "
              f"{avg_coll_eff:>7.1f}%{'':<4} "
              f"{success_mode:<15}")

    # Table 2: Efficiency Metrics
    print("\n" + "-" * 120)
    print(f"{'Algorithm':<30} {'Batt Eff':<15} {'Steps':<12} {'Reward':<12}")
    print("-" * 120)

    for algo_name, results in results_dict.items():
        avg_batt_eff = np.mean(results['battery_efficiency'])
        avg_steps = np.mean(results['steps_taken'])
        avg_reward = np.mean(results['total_rewards'])

        print(f"{algo_name:<30} "
              f"{avg_batt_eff:>8.2f} B/Wh{'':<3} "
              f"{avg_steps:>8.0f}{'':<4} "
              f"{avg_reward:>8.1f}")

    print("=" * 120)


def print_per_sensor_analysis(results_dict: Dict[str, dict], env: UAVEnvironment) -> None:
    """Print detailed per-sensor collection analysis."""
    print("\n" + "=" * 100)
    print("PER-SENSOR COLLECTION ANALYSIS")
    print("=" * 100)

    for algo_name, results in results_dict.items():
        print(f"\n{algo_name}:")
        print("-" * 100)

        # Average per-sensor stats across all episodes
        num_sensors = len(env.sensors)
        avg_sensor_collection = [0] * num_sensors

        for episode_stats in results['per_sensor_collection']:
            for sensor_stat in episode_stats:
                sensor_id = sensor_stat['sensor_id']
                avg_sensor_collection[sensor_id] += sensor_stat['collection_percentage']

        # Calculate averages
        num_episodes = len(results['per_sensor_collection'])
        avg_sensor_collection = [pct / num_episodes for pct in avg_sensor_collection]

        # Print histogram
        print(f"{'Sensor':<10} {'Avg Collection %':<20} {'Bar Chart'}")
        print("-" * 100)

        for sensor_id, pct in enumerate(avg_sensor_collection):
            bar_length = int(pct / 2)  # Scale to 50 chars max
            bar = "█" * bar_length #love this addtion from chat will keep
            print(f"Sensor {sensor_id:<4} {pct:>6.1f}%{'':<13} {bar}")

        # Statistics
        avg_pct = np.mean(avg_sensor_collection)
        std_pct = np.std(avg_sensor_collection)
        min_pct = np.min(avg_sensor_collection)
        max_pct = np.max(avg_sensor_collection)

        print(f"\nStatistics:")
        print(f"  Average: {avg_pct:.1f}%")
        print(f"  Std Dev: {std_pct:.1f}%")
        print(f"  Min: {min_pct:.1f}%")
        print(f"  Max: {max_pct:.1f}%")
        print(f"  Range: {max_pct - min_pct:.1f}%")

    print("=" * 100)


# Update the main comparison function
def compare_all_greedy_agents(num_episodes: int = 1, render: bool = False):
    """
    Compare all greedy agents with comprehensive metrics.

    Args:
        num_episodes: Number of episodes per agent
        render: Whether to render episodes

    Returns:
        Dictionary of results for all agents
    """
    print("=" * 100)
    print("COMPREHENSIVE GREEDY ALGORITHM COMPARISON")
    print("=" * 100)

    env = UAVEnvironment(
        grid_size=(50, 50),
        num_sensors=20,
        max_steps=500,
        sensor_duty_cycle=10.0,
        penalty_data_loss=-1000.0,
        reward_urgency_reduction=500.0,
        render_mode='human' if render else None
    )

    print(f"\nEnvironment Configuration:")
    print(f"  Grid Size: {env.grid_size}")
    print(f"  Sensors: {env.num_sensors}")
    print(f"  Duty Cycle: 10%")
    print(f"  Max Steps: {env.max_steps}")

    algorithms = {
        'Nearest Sensor': NearestSensorGreedy(env),
        'Highest Buffer': HighestBufferGreedy(env),
        'Duty-Cycle Aware': ProbabilisticAwareGreedy(env),
        'SF-Aware V1': MaxThroughputGreedy(env),
        'SF-Aware V2': MaxThroughputGreedyV2(env),
        'Multi-Sensor': MultiSensorGreedy(env),
    }

    all_results = {}

    for name, agent in algorithms.items():
        print(f"\n\n{'=' * 100}")
        print(f"Testing: {name}")
        print(f"{'=' * 100}")

        results = test_greedy_agent(
            agent,
            env,
            num_episodes=num_episodes,
            render=False,
            agent_name=name
        )
        all_results[name] = results

    # Print comparison tables
    print_comparison_table(all_results)
    print_per_sensor_analysis(all_results, env)

    # Find best performer
    best_agent = max(all_results.items(),
                     key=lambda x: np.mean(x[1]['collection_efficiency']))
    print(f"\n Best Collection Efficiency: {best_agent[0]}")
    print(f"   Average: {np.mean(best_agent[1]['collection_efficiency']):.1f}%")

    env.close()
    return all_results


if __name__ == "__main__":
    # Quick test with detailed per-sensor output
    print("=" * 100)
    print("TESTING GREEDY AGENTS WITH DETAILED PER-SENSOR METRICS")
    print("=" * 100)

    # Single episode test with visualization
    env = UAVEnvironment(
        grid_size=(10, 10),
        num_sensors=5,
        max_steps=400,
        sensor_duty_cycle=10.0,
        penalty_data_loss=-500.0,
        reward_urgency_reduction=20.0,
        render_mode='human'
    )

    print("\nRunning single episode with SF-Aware V2...")
    agent = MaxThroughputGreedyV2(env)
    results = test_greedy_agent(
        agent,
        env,
        num_episodes=10,
        render=False,
        agent_name="SF-Aware V2"
    )
    #test_greedy_agent(agent, env, num_episodes=5, render=False, agent_name="SF-Aware V2")
    # Full comparison
    print("\n\nRunning full comparison (5 episodes)...")
    compare_all_greedy_agents(num_episodes=1, render=False)