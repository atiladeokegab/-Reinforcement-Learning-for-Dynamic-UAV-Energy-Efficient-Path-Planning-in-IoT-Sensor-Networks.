"""Quick DQN Visualization with Detailed Fairness Analysis"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import time  #  ADD THIS
from stable_baselines3 import DQN
from environment.uav_env import UAVEnvironment

print("=" * 100)
print("VISUALIZING TRAINED DQN AGENT")
print("=" * 100)

# Load model
model_path = "models/dqn_fairness/best_model.zip"
print(f"\n Loading model: {model_path}")
model = DQN.load(model_path)
print(f"✓ Model loaded!")

# Create environment
print(f"\n Creating environment...")
env = UAVEnvironment(
    grid_size=(50, 50),
    uav_start_position=(25,25),
    num_sensors=20,
    max_steps=500,
    sensor_duty_cycle=10.0,
    penalty_data_loss=-500.0,
    reward_urgency_reduction=20.0,
    render_mode='human'  #  Make sure this is 'human'
)
print(f"✓ Environment ready!")

print(f"\nStarting visualization...")
print(f" Close window or press Ctrl+C to stop")
print("=" * 100)

# Run episode
obs, info = env.reset()
done = False
step = 0
total_reward = 0

# ADD VISUALIZATION DELAY
DELAY = 0.05  # 50ms delay between steps (adjust for speed)

while not done:
    # Get action from model
    action, _ = model.predict(obs, deterministic=True)
    action = int(action)

    # Execute action
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    done = terminated or truncated
    step += 1

    # RENDER WITH DELAY
    env.render()
    time.sleep(DELAY)  # ← THIS IS THE KEY LINE!

    # Progress update
    if step % 50 == 0:
        print(f"Step {step}: Coverage={info['coverage_percentage']:.1f}%, "
              f"Battery={info['battery_percent']:.1f}%")

# ========== DETAILED RESULTS ==========
print("\n" + "=" * 100)
print("EPISODE COMPLETE")
print("=" * 100)

# Overall metrics
total_generated = sum(s.total_data_generated for s in env.sensors)
total_collected = sum(s.total_data_transmitted for s in env.sensors)
total_lost = sum(s.total_data_lost for s in env.sensors)
efficiency = (total_collected / total_generated * 100) if total_generated > 0 else 0
loss_rate = (total_lost / total_generated * 100) if total_generated > 0 else 0
battery_used = 274.0 - info['battery']
bytes_per_watt = total_collected / battery_used

print(f"\nOverall Performance:")
print(f"  Total Reward:          {total_reward:.1f}")
print(f"  Steps:                 {step}")
print(f"  Coverage:              {info['coverage_percentage']:.1f}%")
print(f"  Collection Efficiency: {efficiency:.1f}%")
print(f"  Data Loss Rate:        {loss_rate:.1f}%")
print(f"  Battery Used:          {274.0 - info['battery']:.1f} Wh ({100 - info['battery_percent']:.1f}%)")
print(f"Bytes per Watt:          {bytes_per_watt:.1f} bytes/watt")
# ========== PER-SENSOR FAIRNESS ANALYSIS ==========
print("\n" + "=" * 100)
print("PER-SENSOR FAIRNESS ANALYSIS")
print("=" * 100)

print(f"\n{'Sensor':<12} {'Avg Collection %':<20} {'Bar Chart':<60}")
print("-" * 100)

sensor_collections = []
for i, sensor in enumerate(env.sensors):
    if sensor.total_data_generated > 0:
        collection_pct = (sensor.total_data_transmitted / sensor.total_data_generated) * 100
    else:
        collection_pct = 0.0

    sensor_collections.append(collection_pct)

    # Create bar chart (each █ = 2%)
    bar_length = int(collection_pct / 2)
    bar = "█" * bar_length

    print(f"Sensor {i:<5} {collection_pct:>6.1f}%{'':<13} {bar}")

# Statistics
if sensor_collections:
    avg_collection = np.mean(sensor_collections)
    std_collection = np.std(sensor_collections)
    min_collection = np.min(sensor_collections)
    max_collection = np.max(sensor_collections)
    range_collection = max_collection - min_collection

    print("\n" + "-" * 100)
    print("Statistics:")
    print(f"  Average:   {avg_collection:.1f}%")
    print(f"  Std Dev:   {std_collection:.1f}% (lower = fairer)")
    print(f"  Min:       {min_collection:.1f}%")
    print(f"  Max:       {max_collection:.1f}%")
    print(f"  Range:     {range_collection:.1f}%")

    # Fairness interpretation
    print(f"\n Fairness Analysis:")
    if std_collection < 15:
        fairness_level = "EXCELLENT"
        emoji = "less than 15 greater then 25"
    elif std_collection < 25:
        fairness_level = "GOOD"
        emoji = "less than 25 greater then 35"
    elif std_collection < 35:
        fairness_level = "MODERATE"
    else:
        fairness_level = "POOR"

    print(f" Fairness Level: {fairness_level} (σ = {std_collection:.1f}%)")

    # Count severely under-served sensors
    under_served = sum(1 for c in sensor_collections if c < 40)
    well_served = sum(1 for c in sensor_collections if c >= 70)

    print(f" Under-served sensors (<40%): {under_served} / {len(sensor_collections)}")
    print(f"  Well-served sensors (≥70%):  {well_served} / {len(sensor_collections)}")

# ========== DETAILED SENSOR BREAKDOWN ==========
print("\n" + "=" * 100)
print("DETAILED SENSOR BREAKDOWN")
print("=" * 100)

print(f"\n{'Sensor':<8} {'Position':<15} {'Generated':<12} {'Collected':<12} {'Lost':<10} {'Buffer':<10} {'Collection %':<15}")
print("-" * 100)

for i, sensor in enumerate(env.sensors):
    pos_str = f"({sensor.position[0]:.0f}, {sensor.position[1]:.0f})"
    generated = sensor.total_data_generated
    collected = sensor.total_data_transmitted
    lost = sensor.total_data_lost
    buffer = sensor.data_buffer
    collection_pct = (collected / generated * 100) if generated > 0 else 0

    print(f"Sensor {i:<2} {pos_str:<15} {generated:<12.0f} {collected:<12.0f} {lost:<10.0f} {buffer:<10.0f} {collection_pct:<6.1f}%")

# ========== GEOGRAPHIC DISTRIBUTION ==========
print("\n" + "=" * 100)
print("GEOGRAPHIC DISTRIBUTION ANALYSIS")
print("=" * 100)

# Divide grid into quadrants
quadrant_stats = {
    'NW': [],  # Top-left
    'NE': [],  # Top-right
    'SW': [],  # Bottom-left
    'SE': []   # Bottom-right
}

for i, sensor in enumerate(env.sensors):
    x, y = sensor.position
    collection_pct = sensor_collections[i]

    if x < 25 and y < 25:
        quadrant_stats['SW'].append(collection_pct)
    elif x >= 25 and y < 25:
        quadrant_stats['SE'].append(collection_pct)
    elif x < 25 and y >= 25:
        quadrant_stats['NW'].append(collection_pct)
    else:
        quadrant_stats['NE'].append(collection_pct)

print(f"\n{'Quadrant':<12} {'Sensors':<10} {'Avg Collection %':<20} {'Std Dev':<12}")
print("-" * 60)

for quad, collections in quadrant_stats.items():
    if collections:
        avg = np.mean(collections)
        std = np.std(collections)
        print(f"{quad:<12} {len(collections):<10} {avg:<20.1f} {std:<12.1f}")

# ========== RECOMMENDATIONS ==========
print("\n" + "=" * 100)
print("RECOMMENDATIONS FOR IMPROVEMENT")
print("=" * 100)

if std_collection > 25:
    print("\n High fairness standard deviation detected!")
    print("   Recommendations:")
    print("   • Increase penalty_data_loss (currently -500)")
    print("   • Increase reward_urgency_reduction (currently 20)")
    print("   • Train for more episodes")

    # Identify problem sensors
    problem_sensors = [i for i, c in enumerate(sensor_collections) if c < 40]
    if problem_sensors:
        print(f"\n   Problem sensors (< 40% collection): {problem_sensors}")
        print("   These sensors may be:")
        print("   • Located far from other sensors")
        print("   • In corners/edges of the grid")
        print("   • Consider adjusting sensor placement")
else:
    print("\n Good fairness achieved!")
    print(f"   Standard deviation of {std_collection:.1f}% indicates balanced service")

if efficiency < 65:
    print("\nTo improve efficiency:")
    print(f"   • Current efficiency: {efficiency:.1f}%")
    print("   • Consider increasing max_steps (currently 500)")
    print("   • Or reduce grid_size for denser sensor network")

print("\n" + "=" * 100)
print("Analysis Complete!")
print("=" * 100)

# ADD PAUSE BEFORE CLOSING
print("\nPress Enter to close visualization...")
input()

env.close()