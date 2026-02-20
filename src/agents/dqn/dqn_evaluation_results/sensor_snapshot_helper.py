"""
Sensor Snapshot Helper
Utility functions for saving sensor-level data during evaluation.
Drop this into your evaluation script to enable the analysis tools.

Author: ATILADE GABRIEL OKE
Modified: February 2026
"""

import json
import time
from pathlib import Path


def save_sensor_snapshot(env, filepath, agent_name=None):
    """
    Save detailed sensor-level data from environment to JSON file.

    Args:
        env: UAVEnvironment instance (or AnalysisUAVEnv)
        filepath: Path object or string for output JSON file
        agent_name: Optional agent identifier for metadata

    Returns:
        Dictionary containing the snapshot data
    """
    snapshot = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "agent_name": agent_name,
        "environment_config": {
            "grid_size": env.grid_size,
            "num_sensors": env.num_sensors,
            "max_steps": env.max_steps,
            "current_step": env.current_step,
        },
        "uav_state": {
            "position": list(env.uav.position),
            "battery": float(env.uav.battery),
            "battery_percent": float(env.uav.get_battery_percentage()),
        },
        "mission_stats": {
            "total_data_collected": float(env.total_data_collected),
            "sensors_visited": (
                len(env.sensors_visited) if hasattr(env, "sensors_visited") else 0
            ),
            "coverage_percentage": (
                (len(env.sensors_visited) / env.num_sensors * 100)
                if hasattr(env, "sensors_visited")
                else 0
            ),
        },
        "sensor_data": [],
    }

    # Extract per-sensor statistics
    for sensor in env.sensors:
        sensor_info = {
            "sensor_id": sensor.sensor_id,
            "position": list(sensor.position),
            "total_data_generated": float(sensor.total_data_generated),
            "total_data_transmitted": float(sensor.total_data_transmitted),
            "total_data_lost": float(sensor.total_data_lost),
            "data_buffer": float(sensor.data_buffer),
        }

        # Calculate derived metrics
        if sensor.total_data_generated > 0:
            sensor_info["collection_rate"] = float(
                sensor.total_data_transmitted / sensor.total_data_generated * 100
            )
            sensor_info["loss_rate"] = float(
                sensor.total_data_lost / sensor.total_data_generated * 100
            )
        else:
            sensor_info["collection_rate"] = 0.0
            sensor_info["loss_rate"] = 0.0

        # Add visit tracking if available
        if hasattr(sensor, "visit_count"):
            sensor_info["visit_count"] = int(sensor.visit_count)

        # Add last collection time if available
        if hasattr(sensor, "last_collection_step"):
            sensor_info["last_collection_step"] = int(sensor.last_collection_step)
            sensor_info["steps_since_collection"] = int(
                env.current_step - sensor.last_collection_step
            )

        snapshot["sensor_data"].append(sensor_info)

    # Save to file
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(snapshot, f, indent=2)

    print(f"✓ Sensor snapshot saved: {filepath}")
    print(f"  • {len(snapshot['sensor_data'])} sensors")
    print(
        f"  • {snapshot['mission_stats']['total_data_collected']:.0f} bytes collected"
    )
    print(f"  • {snapshot['mission_stats']['coverage_percentage']:.1f}% coverage")

    return snapshot


def save_analysis_wrapper_snapshot(env_wrapper, filepath, agent_name=None):
    """
    Save snapshot from AnalysisUAVEnv wrapper that preserves last episode data.

    This function specifically handles the AnalysisUAVEnv wrapper which captures
    the final state before reset() wipes it.

    Args:
        env_wrapper: AnalysisUAVEnv instance with last_episode_sensor_data
        filepath: Path object or string for output JSON file
        agent_name: Optional agent identifier for metadata
    """
    if not hasattr(env_wrapper, "last_episode_sensor_data"):
        print("⚠ WARNING: Environment doesn't have last_episode_sensor_data attribute")
        print("  Using current state instead (may be post-reset)")
        return save_sensor_snapshot(env_wrapper, filepath, agent_name)

    if env_wrapper.last_episode_sensor_data is None:
        print("⚠ WARNING: No episode data captured yet")
        return None

    snapshot = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "agent_name": agent_name,
        "environment_config": {
            "grid_size": env_wrapper.grid_size,
            "num_sensors": env_wrapper.num_sensors,
            "max_steps": env_wrapper.max_steps,
        },
        "uav_state": (
            env_wrapper.last_episode_info if env_wrapper.last_episode_info else {}
        ),
        "mission_stats": {
            "total_data_collected": sum(
                s["total_data_transmitted"]
                for s in env_wrapper.last_episode_sensor_data
            ),
            "total_data_lost": sum(
                s["total_data_lost"] for s in env_wrapper.last_episode_sensor_data
            ),
            "sensors_visited": len(
                [
                    s
                    for s in env_wrapper.last_episode_sensor_data
                    if s["total_data_transmitted"] > 0
                ]
            ),
        },
        "sensor_data": [],
    }

    # Process captured sensor data
    for sensor_data in env_wrapper.last_episode_sensor_data:
        generated = sensor_data["total_data_generated"]
        transmitted = sensor_data["total_data_transmitted"]
        lost = sensor_data["total_data_lost"]

        sensor_info = {
            "sensor_id": sensor_data["sensor_id"],
            "position": list(sensor_data["position"]),
            "total_data_generated": generated,
            "total_data_transmitted": transmitted,
            "total_data_lost": lost,
            "data_buffer": sensor_data["data_buffer"],
            "collection_rate": (
                (transmitted / generated * 100) if generated > 0 else 0.0
            ),
            "loss_rate": (lost / generated * 100) if generated > 0 else 0.0,
        }

        snapshot["sensor_data"].append(sensor_info)

    snapshot["mission_stats"]["coverage_percentage"] = (
        snapshot["mission_stats"]["sensors_visited"]
        / snapshot["environment_config"]["num_sensors"]
        * 100
    )

    # Save to file
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        json.dump(snapshot, f, indent=2)

    print(f"✓ Analysis snapshot saved: {filepath}")
    print(f"  • {len(snapshot['sensor_data'])} sensors")
    print(
        f"  • {snapshot['mission_stats']['total_data_collected']:.0f} bytes collected"
    )
    print(f"  • {snapshot['mission_stats']['coverage_percentage']:.1f}% coverage")

    return snapshot


def quick_fairness_check(snapshot):
    """
    Quick console output of fairness metrics from a snapshot.

    Args:
        snapshot: Dictionary returned by save_sensor_snapshot()

    Returns:
        Dictionary of fairness metrics
    """
    import numpy as np

    collection_rates = [s["collection_rate"] for s in snapshot["sensor_data"]]

    # Jain's Fairness Index
    n = len(collection_rates)
    jains_index = (
        (sum(collection_rates) ** 2) / (n * sum(x**2 for x in collection_rates))
        if n > 0
        else 0
    )

    # Coefficient of Variation
    mean_rate = np.mean(collection_rates)
    std_rate = np.std(collection_rates)
    cv = std_rate / mean_rate if mean_rate > 0 else float("inf")

    # Min/Max ratio
    min_rate = min(collection_rates) if collection_rates else 0
    max_rate = max(collection_rates) if collection_rates else 0
    min_max_ratio = min_rate / max_rate if max_rate > 0 else 0

    # Starvation count
    starved = sum(1 for rate in collection_rates if rate < 20.0)

    metrics = {
        "jains_fairness_index": jains_index,
        "coefficient_of_variation": cv,
        "min_max_ratio": min_max_ratio,
        "min_collection_rate": min_rate,
        "max_collection_rate": max_rate,
        "mean_collection_rate": mean_rate,
        "starved_sensor_count": starved,
    }

    print("\n" + "=" * 60)
    print(f"Quick Fairness Check: {snapshot.get('agent_name', 'Unknown Agent')}")
    print("=" * 60)
    print(f"Jain's Fairness Index: {jains_index:.4f} (1.0 = perfect fairness)")
    print(f"Coefficient of Variation: {cv:.4f} (lower = more consistent)")
    print(f"Min/Max Ratio: {min_max_ratio:.4f} (1.0 = perfect equality)")
    print(f"Collection Rate Range: {min_rate:.1f}% - {max_rate:.1f}%")
    print(f"Starved Sensors (<20%): {starved} / {n}")
    print("=" * 60)

    return metrics
