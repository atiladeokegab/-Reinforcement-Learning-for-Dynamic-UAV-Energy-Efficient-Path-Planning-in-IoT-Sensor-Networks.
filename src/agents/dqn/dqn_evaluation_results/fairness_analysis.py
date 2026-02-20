"""
Fairness Analysis - Sensor-by-Sensor Data Collection
Generates bar chart showing which sensors each agent favors/neglects.
Addresses the "POOR" fairness level by identifying starvation patterns.

Author: ATILADE GABRIEL OKE
Modified: February 2026
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# ==================== PATHS ====================
script_dir = Path(__file__).resolve().parent
src_dir = script_dir.parent.parent.parent  # Adjust based on your structure
results_dir = script_dir / "baseline_results"

print(f"Results directory: {results_dir}")


# ==================== LOAD DATA ====================


def load_agent_results(agent_name):
    """Load CSV results for a specific agent."""
    csv_path = results_dir / f"{agent_name}_results.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    else:
        print(f"⚠ WARNING: Could not find {csv_path}")
        return None


def extract_sensor_data_from_env_snapshot(env_snapshot_path):
    """
    Extract per-sensor data collection statistics.
    This function assumes you save sensor-level data during evaluation.

    Expected format: JSON with sensor_data list containing:
    - sensor_id
    - total_data_generated
    - total_data_transmitted
    - total_data_lost
    """
    import json

    if not env_snapshot_path.exists():
        print(f"⚠ WARNING: Snapshot file not found at {env_snapshot_path}")
        return None

    with open(env_snapshot_path, "r") as f:
        data = json.load(f)

    sensor_stats = []
    for sensor in data.get("sensor_data", []):
        sensor_id = sensor["sensor_id"]
        generated = sensor["total_data_generated"]
        transmitted = sensor["total_data_transmitted"]
        lost = sensor["total_data_lost"]

        collection_rate = (transmitted / generated * 100) if generated > 0 else 0

        sensor_stats.append(
            {
                "sensor_id": sensor_id,
                "data_generated": generated,
                "data_collected": transmitted,
                "data_lost": lost,
                "collection_rate": collection_rate,
            }
        )

    return pd.DataFrame(sensor_stats)


def calculate_fairness_metrics(sensor_df):
    """
    Calculate fairness metrics:
    - Jain's Fairness Index
    - Coefficient of Variation (CV)
    - Min/Max ratio
    """
    collection_rates = sensor_df["collection_rate"].values

    # Jain's Fairness Index: (sum(x_i))^2 / (n * sum(x_i^2))
    n = len(collection_rates)
    jains_index = (
        (np.sum(collection_rates) ** 2) / (n * np.sum(collection_rates**2))
        if n > 0
        else 0
    )

    # Coefficient of Variation: std / mean
    cv = (
        np.std(collection_rates) / np.mean(collection_rates)
        if np.mean(collection_rates) > 0
        else np.inf
    )

    # Min/Max ratio
    min_rate = np.min(collection_rates) if len(collection_rates) > 0 else 0
    max_rate = np.max(collection_rates) if len(collection_rates) > 0 else 0
    min_max_ratio = min_rate / max_rate if max_rate > 0 else 0

    return {
        "jains_fairness_index": jains_index,
        "coefficient_of_variation": cv,
        "min_max_ratio": min_max_ratio,
        "min_collection_rate": min_rate,
        "max_collection_rate": max_rate,
        "mean_collection_rate": np.mean(collection_rates),
        "std_collection_rate": np.std(collection_rates),
    }


# ==================== PLOTTING ====================


def plot_fairness_comparison(dqn_sensors, greedy_smart_sensors, greedy_nearest_sensors):
    """
    Generate comprehensive fairness comparison bar chart.
    Shows side-by-side collection rates for each sensor across all agents.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(16, 7))

    # Prepare data
    sensor_ids = dqn_sensors["sensor_id"].values
    n_sensors = len(sensor_ids)
    x = np.arange(n_sensors)
    width = 0.25

    # Extract collection rates
    dqn_rates = dqn_sensors["collection_rate"].values
    smart_rates = greedy_smart_sensors["collection_rate"].values
    nearest_rates = greedy_nearest_sensors["collection_rate"].values

    # Create bars
    bars1 = ax.bar(
        x - width,
        dqn_rates,
        width,
        label="DQN Agent (Proposed)",
        color="#1f77b4",
        edgecolor="black",
        linewidth=1.5,
        alpha=0.85,
    )
    bars2 = ax.bar(
        x,
        smart_rates,
        width,
        label="SF-Aware Greedy V2",
        color="#d62728",
        edgecolor="black",
        linewidth=1.5,
        alpha=0.85,
    )
    bars3 = ax.bar(
        x + width,
        nearest_rates,
        width,
        label="Nearest Sensor Greedy",
        color="#808080",
        edgecolor="black",
        linewidth=1.5,
        alpha=0.85,
    )

    # Highlight starved sensors (collection rate < 20%)
    starvation_threshold = 20.0
    for i, rate in enumerate(dqn_rates):
        if rate < starvation_threshold:
            ax.axvspan(
                i - width - 0.15, i - width + 0.15, alpha=0.2, color="red", zorder=0
            )

    # Formatting
    ax.set_xlabel("Sensor ID", fontsize=13, fontweight="bold")
    ax.set_ylabel("Data Collection Rate (%)", fontsize=13, fontweight="bold")
    ax.set_title(
        "Fairness Analysis: Per-Sensor Data Collection Comparison",
        fontsize=15,
        fontweight="bold",
        pad=20,
    )
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{i}" for i in sensor_ids], fontsize=10)
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=11, framealpha=0.95)
    ax.grid(axis="y", alpha=0.5, linestyle="--")

    # Add fairness threshold line
    ax.axhline(
        y=starvation_threshold,
        color="red",
        linestyle="--",
        linewidth=2,
        alpha=0.6,
        label=f"Starvation Threshold ({starvation_threshold}%)",
    )

    # Add average lines for each agent
    ax.axhline(
        y=np.mean(dqn_rates), color="#1f77b4", linestyle=":", linewidth=2, alpha=0.5
    )
    ax.axhline(
        y=np.mean(smart_rates), color="#d62728", linestyle=":", linewidth=2, alpha=0.5
    )
    ax.axhline(
        y=np.mean(nearest_rates), color="#808080", linestyle=":", linewidth=2, alpha=0.5
    )

    plt.tight_layout()
    output_file = results_dir / "fairness_bar_chart.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✓ Fairness bar chart saved to {output_file}")
    plt.show()


def plot_fairness_metrics_table(dqn_metrics, smart_metrics, nearest_metrics):
    """
    Generate a summary table of fairness metrics.
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.axis("tight")
    ax.axis("off")

    # Prepare table data
    metrics_names = [
        "Jain's Fairness Index (↑)",
        "Coefficient of Variation (↓)",
        "Min/Max Ratio (↑)",
        "Min Collection Rate (%)",
        "Max Collection Rate (%)",
        "Mean Collection Rate (%)",
        "Std Deviation (%)",
    ]

    table_data = []
    for metric_name in metrics_names:
        metric_key = (
            metric_name.split("(")[0].strip().lower().replace("'", "").replace(" ", "_")
        )
        if "jain" in metric_key:
            metric_key = "jains_fairness_index"
        elif "coefficient" in metric_key:
            metric_key = "coefficient_of_variation"
        elif "min/max" in metric_key:
            metric_key = "min_max_ratio"
        elif "std" in metric_key:
            metric_key = "std_collection_rate"

        dqn_val = dqn_metrics.get(metric_key, 0)
        smart_val = smart_metrics.get(metric_key, 0)
        nearest_val = nearest_metrics.get(metric_key, 0)

        table_data.append(
            [
                metric_name,
                f"{dqn_val:.4f}" if isinstance(dqn_val, float) else f"{dqn_val:.2f}",
                (
                    f"{smart_val:.4f}"
                    if isinstance(smart_val, float)
                    else f"{smart_val:.2f}"
                ),
                (
                    f"{nearest_val:.4f}"
                    if isinstance(nearest_val, float)
                    else f"{nearest_val:.2f}"
                ),
            ]
        )

    table = ax.table(
        cellText=table_data,
        colLabels=["Metric", "DQN Agent", "Smart Greedy V2", "Nearest Greedy"],
        cellLoc="center",
        loc="center",
        colWidths=[0.4, 0.2, 0.2, 0.2],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Color header
    for i in range(4):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Highlight best values in each row
    for i in range(1, len(table_data) + 1):
        # Extract numeric values (skip first column which is the metric name)
        row_vals = []
        for j in range(1, 4):
            try:
                row_vals.append(float(table[(i, j)].get_text().get_text()))
            except:
                row_vals.append(0)

        # Determine best based on metric (higher or lower is better)
        if "↑" in table_data[i - 1][0]:  # Higher is better
            best_idx = np.argmax(row_vals) + 1
        else:  # Lower is better
            best_idx = np.argmin(row_vals) + 1

        table[(i, best_idx)].set_facecolor("#90EE90")
        table[(i, best_idx)].set_text_props(weight="bold")

    plt.title("Fairness Metrics Comparison", fontsize=14, fontweight="bold", pad=20)

    output_file = results_dir / "fairness_metrics_table.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✓ Fairness metrics table saved to {output_file}")
    plt.show()


# ==================== MAIN ====================


def main():
    print("=" * 100)
    print("FAIRNESS ANALYSIS - SENSOR-BY-SENSOR COMPARISON")
    print("=" * 100)

    # NOTE: This script assumes you have sensor-level data saved during evaluation
    # You'll need to modify your evaluation script to save this data

    # For demonstration, we'll create synthetic data
    # In practice, load from JSON snapshots saved by AnalysisUAVEnv

    print("\n⚠ NOTE: This script requires sensor-level data from evaluation runs.")
    print("You need to modify your evaluation script to save sensor data snapshots.")
    print("\nExpected file format: JSON with structure:")
    print("""
    {
        "sensor_data": [
            {
                "sensor_id": 0,
                "total_data_generated": 50000,
                "total_data_transmitted": 45000,
                "total_data_lost": 5000
            },
            ...
        ]
    }
    """)

    # Create synthetic example data for demonstration
    num_sensors = 20

    # DQN: Good overall but neglects some peripheral sensors
    dqn_collection_rates = np.random.uniform(40, 95, num_sensors)
    dqn_collection_rates[[2, 5, 17, 19]] = np.random.uniform(
        5, 20, 4
    )  # Starved sensors

    # Smart Greedy: More balanced but lower average
    smart_collection_rates = np.random.uniform(50, 85, num_sensors)

    # Nearest Greedy: Highly variable
    nearest_collection_rates = np.random.uniform(20, 90, num_sensors)

    # Create DataFrames
    dqn_sensors = pd.DataFrame(
        {
            "sensor_id": range(num_sensors),
            "data_generated": np.random.uniform(40000, 60000, num_sensors),
            "data_collected": dqn_collection_rates
            * np.random.uniform(40000, 60000, num_sensors)
            / 100,
            "collection_rate": dqn_collection_rates,
        }
    )

    smart_sensors = pd.DataFrame(
        {
            "sensor_id": range(num_sensors),
            "data_generated": np.random.uniform(40000, 60000, num_sensors),
            "data_collected": smart_collection_rates
            * np.random.uniform(40000, 60000, num_sensors)
            / 100,
            "collection_rate": smart_collection_rates,
        }
    )

    nearest_sensors = pd.DataFrame(
        {
            "sensor_id": range(num_sensors),
            "data_generated": np.random.uniform(40000, 60000, num_sensors),
            "data_collected": nearest_collection_rates
            * np.random.uniform(40000, 60000, num_sensors)
            / 100,
            "collection_rate": nearest_collection_rates,
        }
    )

    # Calculate fairness metrics
    print("\nCalculating fairness metrics...")
    dqn_metrics = calculate_fairness_metrics(dqn_sensors)
    smart_metrics = calculate_fairness_metrics(smart_sensors)
    nearest_metrics = calculate_fairness_metrics(nearest_sensors)

    print("\n" + "-" * 100)
    print("DQN Agent Fairness Metrics:")
    for key, val in dqn_metrics.items():
        print(f"  {key}: {val:.4f}")

    print("\n" + "-" * 100)
    print("Smart Greedy V2 Fairness Metrics:")
    for key, val in smart_metrics.items():
        print(f"  {key}: {val:.4f}")

    print("\n" + "-" * 100)
    print("Nearest Greedy Fairness Metrics:")
    for key, val in nearest_metrics.items():
        print(f"  {key}: {val:.4f}")

    # Generate plots
    print("\n" + "=" * 100)
    print("Generating fairness visualizations...")
    plot_fairness_comparison(dqn_sensors, smart_sensors, nearest_sensors)
    plot_fairness_metrics_table(dqn_metrics, smart_metrics, nearest_metrics)

    print("\n✓ Fairness analysis complete.")
    print("\n" + "=" * 100)
    print("INTERPRETATION GUIDE:")
    print("=" * 100)
    print(
        "• Jain's Fairness Index: Range [0,1]. Higher = more fair. 1.0 = perfect fairness."
    )
    print(
        "• Coefficient of Variation: Lower = more consistent. <0.3 = good, >0.5 = concerning."
    )
    print(
        "• Min/Max Ratio: Range [0,1]. Higher = better. Shows worst vs. best sensor treatment."
    )
    print(
        "• Starvation Threshold: Sensors below 20% collection rate are considered starved."
    )
    print("\nYour dissertation narrative should highlight:")
    print("1. Which sensors DQN favors (spatial patterns)")
    print("2. Why starvation penalty is needed in reward function")
    print("3. Trade-off between total data collected and fairness")
    print("=" * 100)


if __name__ == "__main__":
    main()
