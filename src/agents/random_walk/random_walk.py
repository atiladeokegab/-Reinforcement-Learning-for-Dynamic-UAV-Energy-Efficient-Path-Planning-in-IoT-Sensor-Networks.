"""
Simple Random Walk Agent for UAV Data Collection with Detailed Metrics

The simplest possible baseline - UAV moves randomly and collects opportunistically.

Author: ATILADE GABRIEL OKE
Date: November 2025
"""

import sys
from pathlib import Path

# Add src to path
src_path = Path(__file__).parent.parent.parent
sys.path.insert(0, str(src_path))

import numpy as np

from environment.uav_env import UAVEnvironment


class RandomWalkAgent:
    """
    Simple Random Walk Agent.

    Strategy:
    - If sensors in range with data → COLLECT
    - Otherwise → Move in random direction

    This is the weakest realistic baseline.
    """

    ACTION_UP = 0
    ACTION_DOWN = 1
    ACTION_LEFT = 2
    ACTION_RIGHT = 3
    ACTION_COLLECT = 4

    def __init__(self, env: UAVEnvironment):
        """
        Initialize Random Walk agent.

        Args:
            env: UAV environment
        """
        self.env = env
        self.movement_actions = [
            self.ACTION_UP,
            self.ACTION_DOWN,
            self.ACTION_LEFT,
            self.ACTION_RIGHT
        ]

    def select_action(self, observation: np.ndarray) -> int:
        """
        Select action: collect if possible, otherwise move randomly.

        Args:
            observation: Current state observation

        Returns:
            Selected action (0-4)
        """
        uav_pos = self.env.uav.position

        # Check if any sensors in range with data
        for sensor in self.env.sensors:
            if sensor.data_buffer > 0 and sensor.is_in_range(tuple(uav_pos)):
                return self.ACTION_COLLECT  # Opportunistic collection

        # Otherwise, random movement
        return np.random.choice(self.movement_actions)


# ==================== TESTING WITH DETAILED METRICS ====================

def test_random_walk_agent(agent: RandomWalkAgent,
                           env: UAVEnvironment,
                           num_episodes: int = 1,
                           render: bool = True) -> dict:
    """
    Test random walk agent with detailed per-sensor monitoring.

    Args:
        agent: RandomWalkAgent instance
        env: UAV environment
        num_episodes: Number of episodes to run
        render: Whether to render visualization

    Returns:
        Dictionary of results
    """
    results = {
        'total_rewards': [],
        'coverage_percentage': [],
        'steps_taken': [],
        'data_collected': [],
        'data_generated': [],
        'collection_efficiency': [],
        'battery_efficiency': [],
        'per_sensor_collection': [],
    }

    for episode in range(num_episodes):
        obs, info = env.reset(seed=episode)
        episode_reward = 0
        done = False
        step_count = 0

        print(f"\n{'='*80}")
        print(f"Episode {episode + 1}/{num_episodes} - Random Walk Agent")
        print(f"{'='*80}")

        while not done:
            action = agent.select_action(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_count += 1

            if render:
                env.render()

            done = terminated or truncated

        # Calculate per-sensor statistics
        per_sensor_stats = []
        total_data_generated = 0
        total_data_collected = 0

        for sensor in env.sensors:
            generated = sensor.total_data_generated
            collected = sensor.total_data_transmitted
            remaining_buffer = sensor.data_buffer
            lost = sensor.total_data_lost

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
        battery_used = 274.0 - info['battery']
        battery_efficiency = total_data_collected / battery_used if battery_used > 0 else 0

        # Store results
        results['total_rewards'].append(episode_reward)
        results['coverage_percentage'].append(info['coverage_percentage'])
        results['steps_taken'].append(step_count)
        results['data_collected'].append(total_data_collected)
        results['data_generated'].append(total_data_generated)
        results['collection_efficiency'].append(collection_efficiency)
        results['battery_efficiency'].append(battery_efficiency)
        results['per_sensor_collection'].append(per_sensor_stats)

        # Print detailed results
        print(f"\n{'='*80}")
        print(f"EPISODE {episode + 1} RESULTS")
        print(f"{'='*80}")

        # Overall metrics
        print(f"\n📊 Overall Performance:")
        print(f"  Reward: {episode_reward:.1f}")
        print(f"  Coverage: {info['coverage_percentage']:.1f}%")
        print(f"  Steps: {step_count}")

        # Data metrics
        print(f"\n📦 Data Collection:")
        print(f"  Total Generated: {total_data_generated:.0f} bytes")
        print(f"  Total Collected: {total_data_collected:.0f} bytes")
        print(f"  Collection Efficiency: {collection_efficiency:.1f}%")
        print(f"  Data Lost: {sum(s['data_lost'] for s in per_sensor_stats):.0f} bytes")
        print(f"  Still in Buffers: {sum(s['final_buffer'] for s in per_sensor_stats):.0f} bytes")

        # Battery metrics
        print(f"\n🔋 Energy Efficiency:")
        print(f"  Battery Used: {battery_used:.1f} Wh ({(battery_used/274.0)*100:.1f}%)")
        print(f"  Battery Efficiency: {battery_efficiency:.2f} bytes/Wh")

        # Per-sensor breakdown
        print(f"\n📡 Per-Sensor Collection Breakdown:")
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
              f"{sum(s['data_lost'] for s in per_sensor_stats):<10.0f} "
              f"{sum(s['final_buffer'] for s in per_sensor_stats):<10.0f}")

        # Data accounting verification
        total_accounted = (total_data_collected +
                          sum(s['data_lost'] for s in per_sensor_stats) +
                          sum(s['final_buffer'] for s in per_sensor_stats))
        accounting_error = abs(total_data_generated - total_accounted)

        if accounting_error > 1.0:
            print(f"\n⚠️ DATA ACCOUNTING ERROR!")
            print(f"  Generated: {total_data_generated:.0f}")
            print(f"  Accounted: {total_accounted:.0f} (collected + lost + buffer)")
            print(f"  Error: {accounting_error:.0f} bytes")
        else:
            print(f"\n✅ Data Accounting: Perfect (error < 1 byte)")

        # Sensor coverage statistics
        sensors_with_100_pct = sum(1 for s in per_sensor_stats if s['collection_percentage'] >= 99.9)
        sensors_with_50_pct = sum(1 for s in per_sensor_stats if s['collection_percentage'] >= 50)
        sensors_with_0_pct = sum(1 for s in per_sensor_stats if s['collection_percentage'] < 1)

        print(f"\n📈 Collection Coverage:")
        print(f"  Sensors 100% collected: {sensors_with_100_pct}/{len(env.sensors)}")
        print(f"  Sensors ≥50% collected: {sensors_with_50_pct}/{len(env.sensors)}")
        print(f"  Sensors <1% collected: {sensors_with_0_pct}/{len(env.sensors)}")

        print(f"{'='*80}\n")

    env.close()
    return results


if __name__ == "__main__":
    print("=" * 80)
    print("RANDOM WALK AGENT - DETAILED TESTING")
    print("=" * 80)

    # Create environment
    env = UAVEnvironment(
        grid_size=(50, 50),
        num_sensors=20,
        sensor_duty_cycle=10.0,
        uav_start_position=(0,0),
        max_steps=500,
        render_mode='human'
    )

    print("\nEnvironment Configuration:")
    print(f"  Grid Size: {env.grid_size}")
    print(f"  Sensors: {env.num_sensors}")
    print(f"  Duty Cycle: {env.sensors[0].duty_cycle}%")
    print(f"  Max Steps: {env.max_steps}")

    # Create agent
    agent = RandomWalkAgent(env)

    print("\nRunning Random Walk Agent with detailed monitoring...")
    print("(UAV will move randomly and collect opportunistically)")

    # Run with detailed metrics
    results = test_random_walk_agent(
        agent,
        env,
        num_episodes=1,
        render=True
    )

    # Summary across episodes
    if len(results['total_rewards']) > 1:
        print("\n" + "=" * 80)
        print("SUMMARY ACROSS ALL EPISODES")
        print("=" * 80)
        print(f"Average Reward: {np.mean(results['total_rewards']):.1f}")
        print(f"Average Coverage: {np.mean(results['coverage_percentage']):.1f}%")
        print(f"Average Collection Efficiency: {np.mean(results['collection_efficiency']):.1f}%")
        print(f"Average Battery Efficiency: {np.mean(results['battery_efficiency']):.2f} bytes/Wh")
        print(f"Average Steps: {np.mean(results['steps_taken']):.0f}")
        print("=" * 80)

    print("\n✅ Testing complete!")