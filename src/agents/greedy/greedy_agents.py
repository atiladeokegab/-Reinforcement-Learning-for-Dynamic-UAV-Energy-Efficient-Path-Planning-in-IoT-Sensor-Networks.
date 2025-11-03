"""
Complete Greedy Algorithms for UAV Data Collection with SF-Aware Agent

INCLUDES:
1. NearestSensorGreedy - Distance-based (ignores SF)
2. HighestBufferGreedy - Buffer-based (ignores SF)
3. ProbabilisticAwareGreedy - Duty-cycle aware (ignores SF)
4. MaxThroughputGreedy - SF-AWARE (prioritizes low SF!)
5. MultiSensorGreedy - Multi-target positioning

Author: ATILADE GABRIEL OKE
Date: November 2025
"""

import sys
from pathlib import Path

src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

import numpy as np
from typing import Tuple, List, Dict

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
            return "🌟 Perfect"
        elif metrics['high_success']:
            return "✅ High"
        elif metrics['acceptable_success']:
            return "👍 Acceptable"
        elif metrics['partial_success']:
            return "⚠️ Partial"
        else:
            return "❌ Failed"


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
    ✅ SF-AWARE: Prioritize sensors with BEST data rates (LOWEST SF).

    THIS IS THE KEY BASELINE FOR RL COMPARISON!

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
                      data_goal: DataGoal = None) -> dict:
    """Test a greedy agent with proper movement and metrics."""
    results = {
        'total_rewards': [],
        'coverage_percentage': [],
        'steps_taken': [],
        'data_collected': [],
    }

    for episode in range(num_episodes):
        obs, info = env.reset(seed=episode)
        episode_reward = 0
        done = False
        step_count = 0

        print(f"\n{'='*80}")
        print(f"Episode {episode + 1}/{num_episodes}")
        if data_goal:
            print(f"Goal: {data_goal}")
        print(f"{'='*80}")

        while not done:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1

            if render:
                env.render()

            done = terminated or truncated

        results['total_rewards'].append(episode_reward)
        results['coverage_percentage'].append(info['coverage_percentage'])
        results['steps_taken'].append(step_count)
        results['data_collected'].append(info['total_data_collected'])

        print(f"\nEpisode {episode + 1} Results:")
        print(f"  Reward: {episode_reward:.1f}")
        print(f"  Coverage: {info['coverage_percentage']:.1f}%")
        print(f"  Steps: {step_count}")
        print(f"  Data Collected: {info['total_data_collected']:.0f} bytes")

    env.close()
    return results


if __name__ == "__main__":
    print("GREEDY ALGORITHM TESTING - WITH SF-AWARE AGENT")
    print("="*80)

    env = UAVEnvironment(
        grid_size=(50, 50),
        sensor_duty_cycle=10.0,
        max_steps=500,
        uav_start_position=(25, 25),
        render_mode='human'
    )


    print("\n\n🎯 Testing: MaxThroughput Greedy (SF-AWARE)")
    print("="*80)
    agent = MaxThroughputGreedy(env)
    results = test_greedy_agent(agent, env, num_episodes=1, render=True)

    print(f"\n✅ Average Reward: {np.mean(results['total_rewards']):.1f}")
    print(f"✅ Average Coverage: {np.mean(results['coverage_percentage']):.1f}%")
    print(f"✅ Average Steps: {np.mean(results['steps_taken']):.0f}")
    print(f"✅ Average Data: {np.mean(results['data_collected']):.0f} bytes")