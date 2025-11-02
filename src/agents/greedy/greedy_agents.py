"""
Greedy Algorithm for UAV Data Collection with Academic Metrics and Data Goals

Baseline algorithms for comparison with RL agents.
Includes data collection goals for fair comparison across algorithms.

Author: ATILADE GABRIEL OKE
Date: November 2025
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

import numpy as np
from typing import Tuple, List, Dict

from environment.uav_env import UAVEnvironment


class DataGoal:
    """
    Define data collection goals for episodes.

    Goals provide fair comparison by setting explicit targets
    rather than just maximizing collection.
    """

    def __init__(self,
                 target_bytes: float = None,
                 target_coverage: float = None,
                 max_battery_usage: float = None):
        """
        Initialize data collection goals.

        Args:
            target_bytes: Target total bytes to collect (e.g., 5000.0)
            target_coverage: Target percentage of sensors to visit (e.g., 95.0)
            max_battery_usage: Maximum battery percentage to use (e.g., 80.0)
        """
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
            # For battery, being under target is good
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
    """
    Multi-level success criteria based on academic literature.

    Success Levels:
        - Perfect: 100% coverage, <1% data loss
        - High: ≥95% coverage, ≥90% collection ratio (typical paper standard)
        - Acceptable: ≥80% coverage, ≥75% collection ratio
        - Partial: ≥50% coverage
    """

    @staticmethod
    def evaluate_episode(info: dict, env: UAVEnvironment, data_goal: DataGoal = None) -> dict:
        """
        Evaluate episode with multiple success criteria and optional data goal.

        Args:
            info: Episode info dictionary from environment
            env: UAV environment instance
            data_goal: Optional DataGoal instance

        Returns:
            Dictionary with success levels, metrics, and goal achievement
        """
        total_data_generated = sum(s.total_data_generated for s in env.sensors)
        total_data_lost = sum(s.total_data_lost for s in env.sensors)
        data_collected = info['total_data_collected']
        battery_used = 274.0 - info['battery']

        # Calculate metrics
        coverage_pct = info['coverage_percentage']
        collection_ratio = (data_collected / total_data_generated * 100) if total_data_generated > 0 else 0
        data_loss_rate = (total_data_lost / total_data_generated * 100) if total_data_generated > 0 else 0
        battery_efficiency = data_collected / battery_used if battery_used > 0 else 0
        battery_used_pct = battery_used / 274.0 * 100

        result = {
            # Success Levels
            'perfect_success': coverage_pct == 100 and data_loss_rate < 1.0,
            'high_success': coverage_pct >= 95 and collection_ratio >= 90,
            'acceptable_success': coverage_pct >= 80 and collection_ratio >= 75,
            'partial_success': coverage_pct >= 50,

            # Supporting Metrics
            'coverage_percentage': coverage_pct,
            'collection_ratio': collection_ratio,
            'data_loss_rate': data_loss_rate,
            'battery_efficiency': battery_efficiency,
            'battery_used_percentage': battery_used_pct,
            'total_data_generated': total_data_generated,
            'total_data_collected': data_collected,
            'total_data_lost': total_data_lost,
        }

        # Add goal-based evaluation
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
    """Base class for greedy algorithms."""

    def __init__(self, env: UAVEnvironment):
        self.env = env

    def select_action(self, observation: np.ndarray) -> int:
        """Select action based on greedy strategy."""
        raise NotImplementedError


class NearestSensorGreedy(GreedyAgent):
    """Greedy strategy: Always move toward the nearest sensor with data."""

    def __init__(self, env: UAVEnvironment):
        super().__init__(env)

    def select_action(self, observation: np.ndarray) -> int:
        uav_pos = self.env.uav.position
        collectible_sensors = []
        for sensor in self.env.sensors:
            if sensor.data_buffer > 0:
                if sensor.is_in_range(tuple(uav_pos)):
                    collectible_sensors.append(sensor)

        if len(collectible_sensors) > 0:
            return 4

        target_sensor = self._find_nearest_sensor_with_data()
        if target_sensor is None:
            return 4
        return self._move_toward(target_sensor.position)

    def _find_nearest_sensor_with_data(self):
        uav_pos = self.env.uav.position
        sensors_with_data = [s for s in self.env.sensors if s.data_buffer > 0]
        if not sensors_with_data:
            return None
        distances = [np.linalg.norm(s.position - uav_pos) for s in sensors_with_data]
        return sensors_with_data[np.argmin(distances)]

    def _move_toward(self, target_pos: np.ndarray) -> int:
        uav_pos = self.env.uav.position
        dx = target_pos[0] - uav_pos[0]
        dy = target_pos[1] - uav_pos[1]
        if dx == 0 and dy == 0:
            return 4
        if abs(dx) > abs(dy):
            return 3 if dx > 0 else 2
        else:
            return 0 if dy > 0 else 1


class HighestBufferGreedy(GreedyAgent):
    """Greedy strategy: Prioritize sensors with highest buffer levels."""

    def __init__(self, env: UAVEnvironment):
        super().__init__(env)

    def select_action(self, observation: np.ndarray) -> int:
        uav_pos = self.env.uav.position
        collectible_sensors = []
        for sensor in self.env.sensors:
            if sensor.data_buffer > 0:
                if sensor.is_in_range(tuple(uav_pos)):
                    collectible_sensors.append(sensor)

        if len(collectible_sensors) > 0:
            return 4

        target_sensor = self._find_highest_buffer_sensor()
        if target_sensor is None:
            return 4
        return self._move_toward(target_sensor.position)

    def _find_highest_buffer_sensor(self):
        sensors_with_data = [s for s in self.env.sensors if s.data_buffer > 0]
        if not sensors_with_data:
            return None
        return max(sensors_with_data, key=lambda s: s.data_buffer)

    def _move_toward(self, target_pos: np.ndarray) -> int:
        uav_pos = self.env.uav.position
        dx = target_pos[0] - uav_pos[0]
        dy = target_pos[1] - uav_pos[1]
        if dx == 0 and dy == 0:
            return 4
        if abs(dx) > abs(dy):
            return 3 if dx > 0 else 2
        else:
            return 0 if dy > 0 else 1


class ActiveSensorGreedy(GreedyAgent):
    """Greedy strategy: Prioritize active sensors (duty cycle aware)."""

    def __init__(self, env: UAVEnvironment):
        super().__init__(env)

    def select_action(self, observation: np.ndarray) -> int:
        uav_pos = self.env.uav.position
        collectible_sensors = []
        for sensor in self.env.sensors:
            if sensor.data_buffer > 0:
                if sensor.is_in_range(tuple(uav_pos)):
                    collectible_sensors.append(sensor)

        if len(collectible_sensors) > 0:
            return 4

        target_sensor = self._find_nearest_active_sensor()
        if target_sensor is None:
            target_sensor = self._find_nearest_sensor_with_data()
        if target_sensor is None:
            return 4
        return self._move_toward(target_sensor.position)

    def _find_nearest_active_sensor(self):
        uav_pos = self.env.uav.position
        active_sensors = [s for s in self.env.sensors if s.is_active and s.data_buffer > 0]
        if not active_sensors:
            return None
        distances = [np.linalg.norm(s.position - uav_pos) for s in active_sensors]
        return active_sensors[np.argmin(distances)]

    def _find_nearest_sensor_with_data(self):
        uav_pos = self.env.uav.position
        sensors_with_data = [s for s in self.env.sensors if s.data_buffer > 0]
        if not sensors_with_data:
            return None
        distances = [np.linalg.norm(s.position - uav_pos) for s in sensors_with_data]
        return sensors_with_data[np.argmin(distances)]

    def _move_toward(self, target_pos: np.ndarray) -> int:
        uav_pos = self.env.uav.position
        dx = target_pos[0] - uav_pos[0]
        dy = target_pos[1] - uav_pos[1]
        if dx == 0 and dy == 0:
            return 4
        if abs(dx) > abs(dy):
            return 3 if dx > 0 else 2
        else:
            return 0 if dy > 0 else 1


class MultiSensorGreedy(GreedyAgent):
    """Greedy strategy: Position to collect from multiple sensors simultaneously."""

    def __init__(self, env: UAVEnvironment, communication_range: float = 2.0):
        super().__init__(env)
        self.communication_range = communication_range

    def select_action(self, observation: np.ndarray) -> int:
        uav_pos = self.env.uav.position
        collectible_sensors = []
        for sensor in self.env.sensors:
            if sensor.data_buffer > 0:
                if sensor.is_in_range(tuple(uav_pos)):
                    collectible_sensors.append(sensor)

        if len(collectible_sensors) > 0:
            return 4

        best_position = self._find_best_position()
        if best_position is None:
            target_sensor = self._find_nearest_sensor_with_data()
            if target_sensor:
                return self._move_toward(target_sensor.position)
            return 4
        return self._move_toward(best_position)

    def _count_sensors_in_range(self, position: np.ndarray) -> int:
        count = 0
        for sensor in self.env.sensors:
            if sensor.data_buffer > 0:
                if sensor.is_in_range(tuple(position)):
                    count += 1
        return count

    def _find_best_position(self) -> np.ndarray:
        sensors_with_data = [s for s in self.env.sensors if s.data_buffer > 0]
        if not sensors_with_data:
            return None

        best_position = None
        best_count = 0

        for sensor in sensors_with_data:
            for dx in [-2, -1, 0, 1, 2]:
                for dy in [-2, -1, 0, 1, 2]:
                    candidate_pos = sensor.position + np.array([dx, dy], dtype=float)
                    if (0 <= candidate_pos[0] < self.env.grid_size[0] and
                        0 <= candidate_pos[1] < self.env.grid_size[1]):
                        count = self._count_sensors_in_range(candidate_pos)
                        if count > best_count:
                            best_count = count
                            best_position = candidate_pos
        return best_position

    def _find_nearest_sensor_with_data(self):
        uav_pos = self.env.uav.position
        sensors_with_data = [s for s in self.env.sensors if s.data_buffer > 0]
        if not sensors_with_data:
            return None
        distances = [np.linalg.norm(s.position - uav_pos) for s in sensors_with_data]
        return sensors_with_data[np.argmin(distances)]

    def _move_toward(self, target_pos: np.ndarray) -> int:
        uav_pos = self.env.uav.position
        dx = target_pos[0] - uav_pos[0]
        dy = target_pos[1] - uav_pos[1]
        if dx == 0 and dy == 0:
            return 4
        if abs(dx) > abs(dy):
            return 3 if dx > 0 else 2
        else:
            return 0 if dy > 0 else 1


# ==================== TESTING FUNCTIONS ====================

def test_greedy_agent(agent: GreedyAgent,
                      env: UAVEnvironment,
                      num_episodes: int = 10,
                      render: bool = False,
                      data_goal: DataGoal = None) -> dict:
    """
    Test a greedy agent with academic-standard metrics and optional data goal.

    Args:
        agent: Greedy agent to test
        env: UAV environment
        num_episodes: Number of episodes to run
        render: Whether to render the environment
        data_goal: Optional data collection goal

    Returns:
        Dictionary of performance metrics
    """
    results = {
        'total_rewards': [],
        'perfect_success': 0,
        'high_success': 0,
        'acceptable_success': 0,
        'partial_success': 0,
        'goal_achieved': 0 if data_goal else None,
        'coverage_percentage': [],
        'collection_ratio': [],
        'data_loss_rate': [],
        'battery_efficiency': [],
        'battery_used_percentage': [],
        'steps_taken': [],
        'data_generated': [],
        'data_collected': [],
        'data_lost': [],
    }

    if data_goal:
        results['bytes_achievement'] = []
        results['coverage_achievement'] = []
        results['battery_achievement'] = []

    for episode in range(num_episodes):
        obs, info = env.reset(seed=episode)
        episode_reward = 0
        done = False

        print(f"\n{'='*80}")
        print(f"Episode {episode + 1}/{num_episodes}")
        if data_goal:
            print(f"Goal: {data_goal}")
        print(f"{'='*80}")

        while not done:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

            if render:
                env.render()

            done = terminated or truncated

        # Evaluate with academic metrics
        metrics = SuccessMetrics.evaluate_episode(info, env, data_goal)

        # Record results
        results['total_rewards'].append(episode_reward)
        results['perfect_success'] += 1 if metrics['perfect_success'] else 0
        results['high_success'] += 1 if metrics['high_success'] else 0
        results['acceptable_success'] += 1 if metrics['acceptable_success'] else 0
        results['partial_success'] += 1 if metrics['partial_success'] else 0

        if data_goal:
            results['goal_achieved'] += 1 if metrics['goal_achieved'] else 0
            achievement = metrics['goal_achievement']
            if 'bytes_ratio' in achievement:
                results['bytes_achievement'].append(achievement['bytes_ratio'] * 100)
            if 'coverage_ratio' in achievement:
                results['coverage_achievement'].append(achievement['coverage_ratio'] * 100)
            if 'battery_ratio' in achievement:
                results['battery_achievement'].append(achievement['battery_ratio'] * 100)

        results['coverage_percentage'].append(metrics['coverage_percentage'])
        results['collection_ratio'].append(metrics['collection_ratio'])
        results['data_loss_rate'].append(metrics['data_loss_rate'])
        results['battery_efficiency'].append(metrics['battery_efficiency'])
        results['battery_used_percentage'].append(metrics['battery_used_percentage'])
        results['steps_taken'].append(info['current_step'])
        results['data_generated'].append(metrics['total_data_generated'])
        results['data_collected'].append(metrics['total_data_collected'])
        results['data_lost'].append(metrics['total_data_lost'])

        # Print episode results
        success_level = SuccessMetrics.get_success_level_string(metrics)

        print(f"\n Episode {episode + 1} Results:")
        print(f"   Reward: {episode_reward:.1f}")
        print(f"   Success Level: {success_level}")
        if data_goal and metrics['goal_achieved'] is not None:
            goal_status = " ACHIEVED" if metrics['goal_achieved'] else " NOT ACHIEVED"
            print(f"   Goal Status: {goal_status}")
            if metrics['goal_achievement']:
                for goal_type, ratio in metrics['goal_achievement'].items():
                    print(f"     {goal_type}: {ratio*100:.1f}%")
        print(f"   Coverage: {metrics['coverage_percentage']:.1f}%")
        print(f"   Collection Ratio: {metrics['collection_ratio']:.1f}%")
        print(f"   Data Loss Rate: {metrics['data_loss_rate']:.1f}%")
        print(f"   Battery Efficiency: {metrics['battery_efficiency']:.2f} bytes/Wh")
        print(f"   Battery Used: {metrics['battery_used_percentage']:.1f}%")
        print(f"   Steps: {info['current_step']}")
        print(f"   Data: Generated={metrics['total_data_generated']:.0f}, "
              f"Collected={metrics['total_data_collected']:.0f}, "
              f"Lost={metrics['total_data_lost']:.0f}")

    # Calculate success rates
    for key in ['perfect_success', 'high_success', 'acceptable_success', 'partial_success']:
        results[key] = (results[key] / num_episodes) * 100

    if data_goal:
        results['goal_achieved'] = (results['goal_achieved'] / num_episodes) * 100

    return results


def compare_greedy_algorithms(num_episodes: int = 10,
                              render: bool = False,
                              data_goal: DataGoal = None, env: UAVEnvironment = None):
    """
    Compare all greedy algorithms with academic metrics and optional data goal.

    Args:
        num_episodes: Number of episodes per algorithm
        render: Whether to render episodes
        data_goal: Optional data collection goal for fair comparison

    Returns:
        Dictionary of results for all algorithms
    """
    print("=" * 90)
    print("ACADEMIC-STANDARD GREEDY ALGORITHM COMPARISON")
    print("=" * 90)
    print("\nSuccess Level Definitions:")
    print("  Perfect:    100% coverage, <1% data loss")
    print("  High:       ≥95% coverage, ≥90% collection ratio (paper standard)")
    print("  Acceptable: ≥80% coverage, ≥75% collection ratio")
    print("  Partial:    ≥50% coverage")
    print("  Failed:     <50% coverage")

    if data_goal:
        print(f"\n Data Collection Goal:")
        print(f"   {data_goal}")

    print("=" * 90)


    print(f"\nEnvironment Configuration:")
    print(f"  Grid Size: {env.grid_size}")
    print(f"  Sensors: {env.num_sensors}")
    print(f"  Duty Cycle: 10%")
    print(f"  Max Steps: {env.max_steps}")
    print(f"  Action Space: {env.action_space}")

    algorithms = {
        'Nearest Sensor': NearestSensorGreedy(env),
        'Highest Buffer': HighestBufferGreedy(env),
        'Active Sensor (Duty Cycle Aware)': ActiveSensorGreedy(env),
        'Multi-Sensor Positioning': MultiSensorGreedy(env),
    }

    all_results = {}

    for name, agent in algorithms.items():
        print(f"\n\n{'='*90}")
        print(f"Testing: {name}")
        print(f"{'='*90}")

        results = test_greedy_agent(agent, env, num_episodes, render=True, data_goal=data_goal)
        all_results[name] = results

    # Print comprehensive comparison table
    print("\n\n" + "=" * 90)
    print("COMPARISON SUMMARY - Success Rates")
    print("=" * 90)

    if data_goal:
        print(f"{'Algorithm':<40} {'Goal':<10} {'Perfect':<10} {'High':<10} {'Accept':<10}")
    else:
        print(f"{'Algorithm':<40} {'Perfect':<10} {'High':<10} {'Accept':<10} {'Partial':<10}")
    print("-" * 90)

    for name, results in all_results.items():
        if data_goal:
            print(f"{name:<40} "
                  f"{results['goal_achieved']:<10.1f}% "
                  f"{results['perfect_success']:<10.1f}% "
                  f"{results['high_success']:<10.1f}% "
                  f"{results['acceptable_success']:<10.1f}%")
        else:
            print(f"{name:<40} "
                  f"{results['perfect_success']:<10.1f}% "
                  f"{results['high_success']:<10.1f}% "
                  f"{results['acceptable_success']:<10.1f}% "
                  f"{results['partial_success']:<10.1f}%")

    print("\n" + "=" * 90)
    print("COMPARISON SUMMARY - Performance Metrics")
    print("=" * 90)
    print(f"{'Algorithm':<40} {'Avg Reward':<12} {'Avg Cov':<10} {'Coll Ratio':<12} {'Loss Rate':<10}")
    print("-" * 90)

    for name, results in all_results.items():
        avg_reward = np.mean(results['total_rewards'])
        avg_cov = np.mean(results['coverage_percentage'])
        avg_coll = np.mean(results['collection_ratio'])
        avg_loss = np.mean(results['data_loss_rate'])

        print(f"{name:<40} "
              f"{avg_reward:<12.1f} "
              f"{avg_cov:<10.1f}% "
              f"{avg_coll:<12.1f}% "
              f"{avg_loss:<10.1f}%")

    print("\n" + "=" * 90)
    print("COMPARISON SUMMARY - Efficiency Metrics")
    print("=" * 90)
    print(f"{'Algorithm':<40} {'Batt Eff':<15} {'Avg Steps':<12} {'Batt Used':<12}")
    print("-" * 90)

    for name, results in all_results.items():
        avg_batt_eff = np.mean(results['battery_efficiency'])
        avg_steps = np.mean(results['steps_taken'])
        avg_batt_used = np.mean(results['battery_used_percentage'])

        print(f"{name:<40} "
              f"{avg_batt_eff:<15.2f} "
              f"{avg_steps:<12.0f} "
              f"{avg_batt_used:<12.1f}%")

    if data_goal:
        print("\n" + "=" * 90)
        print("COMPARISON SUMMARY - Goal Achievement")
        print("=" * 90)

        if results['bytes_achievement']:
            print(f"{'Algorithm':<40} {'Bytes':<12} {'Coverage':<12} {'Battery':<12}")
            print("-" * 90)

            for name, results in all_results.items():
                bytes_ach = np.mean(results['bytes_achievement']) if results['bytes_achievement'] else 0
                cov_ach = np.mean(results['coverage_achievement']) if results['coverage_achievement'] else 0
                batt_ach = np.mean(results['battery_achievement']) if results['battery_achievement'] else 0

                print(f"{name:<40} "
                      f"{bytes_ach:<12.1f}% "
                      f"{cov_ach:<12.1f}% "
                      f"{batt_ach:<12.1f}%")

    # Identify best algorithms
    print("\n" + "=" * 90)
    print("BEST PERFORMERS")
    print("=" * 90)

    if data_goal:
        best_goal = max(all_results.items(), key=lambda x: x[1]['goal_achieved'])
        print(f"Best Goal Achievement: {best_goal[0]} ({best_goal[1]['goal_achieved']:.1f}%)")

    best_reward = max(all_results.items(), key=lambda x: np.mean(x[1]['total_rewards']))
    best_coverage = max(all_results.items(), key=lambda x: np.mean(x[1]['coverage_percentage']))
    best_efficiency = max(all_results.items(), key=lambda x: np.mean(x[1]['battery_efficiency']))
    best_high_success = max(all_results.items(), key=lambda x: x[1]['high_success'])

    print(f"Highest Reward: {best_reward[0]} ({np.mean(best_reward[1]['total_rewards']):.1f})")
    print(f"Best Coverage: {best_coverage[0]} ({np.mean(best_coverage[1]['coverage_percentage']):.1f}%)")
    print(f"Best Efficiency: {best_efficiency[0]} ({np.mean(best_efficiency[1]['battery_efficiency']):.2f} bytes/Wh)")
    print(f"Highest High-Success Rate: {best_high_success[0]} ({best_high_success[1]['high_success']:.1f}%)")

    print("=" * 90)

    env.close()
    return all_results


if __name__ == "__main__":
    print("GREEDY ALGORITHM TESTING WITH DATA COLLECTION GOALS")
    env = UAVEnvironment(
        grid_size=(50, 50),
        num_sensors=30,
        sensor_duty_cycle=100.0,
        max_steps=500,
        rssi_threshold=-90.0,
        transmit_power_dbm=14.0,
        render_mode='human'
    )

    # Define data collection goal
    # Typical goal: Collect 5000 bytes, visit 90% of sensors, use max 80% battery
    data_goal = DataGoal(
        target_bytes=3500.0,      # Collect at least 5000 bytes
        target_coverage=80.0,     # Visit at least 90% of sensors
        max_battery_usage=80.0    # Use at most 80% of battery
    )


    # Full comparison with goal
    print("\nRunning comprehensive comparison (5 episodes per algorithm)...")
    results = compare_greedy_algorithms(num_episodes=1, render=False, data_goal=data_goal,env=env)
