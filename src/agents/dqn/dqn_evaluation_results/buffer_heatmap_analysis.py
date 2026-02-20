"""
Buffer State Heatmap - Network Health Visualization
Generates spatial heatmap showing which sensors are starved (full buffers)
vs. well-serviced (empty buffers) across the 500m x 500m grid.

Author: ATILADE GABRIEL OKE
Modified: February 2026
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy.interpolate import griddata
from pathlib import Path

# ==================== PATHS ====================
script_dir = Path(__file__).resolve().parent
results_dir = script_dir / "baseline_results"

print(f"Results directory: {results_dir}")

# ==================== CONSTANTS ====================
GRID_SIZE = (100, 100)  # 100m x 100m from your config
SENSOR_BUFFER_CAPACITY = 10000  # Adjust based on your sensor config


# ==================== DATA LOADING ====================


def load_sensor_snapshot_data(snapshot_path):
    """
    Load sensor-level snapshot data.
    Expected format: JSON with sensor positions and buffer states.

    Returns DataFrame with columns:
    - sensor_id
    - x_position
    - y_position
    - buffer_fullness (0-1, where 1 = full buffer = starved)
    - data_lost
    """
    import json

    if not snapshot_path.exists():
        print(f"⚠ WARNING: Snapshot file not found at {snapshot_path}")
        return None

    with open(snapshot_path, "r") as f:
        data = json.load(f)

    sensor_data = []
    for sensor in data.get("sensor_data", []):
        sensor_id = sensor["sensor_id"]
        position = sensor["position"]
        buffer = sensor.get("data_buffer", 0)
        data_lost = sensor.get("total_data_lost", 0)

        # Calculate buffer fullness ratio (0 = empty, 1 = full)
        buffer_fullness = (
            min(buffer / SENSOR_BUFFER_CAPACITY, 1.0)
            if SENSOR_BUFFER_CAPACITY > 0
            else 0
        )

        sensor_data.append(
            {
                "sensor_id": sensor_id,
                "x_position": position[0],
                "y_position": position[1],
                "buffer_fullness": buffer_fullness,
                "data_lost": data_lost,
                "buffer_bytes": buffer,
            }
        )

    return pd.DataFrame(sensor_data)


# ==================== HEATMAP GENERATION ====================


def create_interpolated_heatmap(sensor_df, grid_resolution=50):
    """
    Create smooth heatmap using interpolation.

    Args:
        sensor_df: DataFrame with sensor positions and buffer states
        grid_resolution: Number of grid points (higher = smoother)

    Returns:
        X, Y, Z grids for heatmap plotting
    """
    # Extract sensor data
    x = sensor_df["x_position"].values
    y = sensor_df["y_position"].values
    z = sensor_df["buffer_fullness"].values

    # Create interpolation grid
    grid_x = np.linspace(0, GRID_SIZE[0], grid_resolution)
    grid_y = np.linspace(0, GRID_SIZE[1], grid_resolution)
    grid_X, grid_Y = np.meshgrid(grid_x, grid_y)

    # Interpolate buffer states across grid
    grid_Z = griddata(
        points=(x, y),
        values=z,
        xi=(grid_X, grid_Y),
        method="cubic",  # Smooth interpolation
        fill_value=0.0,  # Default for areas far from sensors
    )

    # Clip to valid range [0, 1]
    grid_Z = np.clip(grid_Z, 0, 1)

    return grid_X, grid_Y, grid_Z


def create_voronoi_heatmap(sensor_df, grid_resolution=100):
    """
    Create heatmap using Voronoi regions (each pixel takes nearest sensor value).
    """
    from scipy.spatial import distance_matrix

    # Extract sensor data
    sensor_positions = sensor_df[["x_position", "y_position"]].values
    buffer_values = sensor_df["buffer_fullness"].values

    # Create grid
    grid_x = np.linspace(0, GRID_SIZE[0], grid_resolution)
    grid_y = np.linspace(0, GRID_SIZE[1], grid_resolution)
    grid_X, grid_Y = np.meshgrid(grid_x, grid_y)

    # Flatten grid
    grid_points = np.column_stack([grid_X.ravel(), grid_Y.ravel()])

    # Find nearest sensor for each grid point
    distances = distance_matrix(grid_points, sensor_positions)
    nearest_sensor_idx = np.argmin(distances, axis=1)

    # Assign buffer values
    grid_Z = buffer_values[nearest_sensor_idx].reshape(grid_X.shape)

    return grid_X, grid_Y, grid_Z


# ==================== PLOTTING ====================


def plot_buffer_heatmap(sensor_df, agent_name, use_voronoi=True):
    """
    Generate spatial heatmap of sensor buffer states.

    Red = Full buffers (starved sensors)
    Blue = Empty buffers (well-serviced sensors)
    """
    plt.style.use("seaborn-v0_8-white")
    fig, ax = plt.subplots(figsize=(12, 10))

    # Generate heatmap data
    if use_voronoi:
        grid_X, grid_Y, grid_Z = create_voronoi_heatmap(sensor_df, grid_resolution=100)
        interpolation = "nearest"
    else:
        grid_X, grid_Y, grid_Z = create_interpolated_heatmap(
            sensor_df, grid_resolution=80
        )
        interpolation = "bilinear"

    # Custom colormap: Blue (good) -> Yellow (warning) -> Red (critical)
    colors = ["#0000FF", "#00FFFF", "#FFFF00", "#FF8C00", "#FF0000"]
    n_bins = 100
    cmap = LinearSegmentedColormap.from_list("buffer_state", colors, N=n_bins)

    # Plot heatmap
    im = ax.imshow(
        grid_Z,
        extent=[0, GRID_SIZE[0], 0, GRID_SIZE[1]],
        origin="lower",
        cmap=cmap,
        alpha=0.8,
        interpolation=interpolation,
        vmin=0,
        vmax=1,
    )

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(
        "Buffer Fullness (0=Empty, 1=Full)",
        fontsize=12,
        fontweight="bold",
        rotation=270,
        labelpad=25,
    )

    # Overlay sensor positions
    ax.scatter(
        sensor_df["x_position"],
        sensor_df["y_position"],
        c="white",
        s=200,
        marker="s",
        edgecolors="black",
        linewidth=2,
        label="Sensor Locations",
        zorder=10,
    )

    # Annotate sensors with IDs and buffer states
    for _, sensor in sensor_df.iterrows():
        x, y = sensor["x_position"], sensor["y_position"]
        sensor_id = int(sensor["sensor_id"])
        buffer_pct = sensor["buffer_fullness"] * 100

        # Color code text based on buffer state
        text_color = (
            "red" if buffer_pct > 70 else "orange" if buffer_pct > 40 else "green"
        )

        ax.annotate(
            f"S{sensor_id}\n{buffer_pct:.0f}%",
            (x, y),
            fontsize=8,
            fontweight="bold",
            color=text_color,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="black", alpha=0.9),
        )

    # Add problem zone highlights
    starved_sensors = sensor_df[sensor_df["buffer_fullness"] > 0.7]
    if not starved_sensors.empty:
        ax.scatter(
            starved_sensors["x_position"],
            starved_sensors["y_position"],
            s=1000,
            facecolors="none",
            edgecolors="red",
            linewidth=3,
            linestyle="--",
            label="Starved Sensors (>70% full)",
            zorder=5,
        )

    # Formatting
    ax.set_xlim(0, GRID_SIZE[0])
    ax.set_ylim(0, GRID_SIZE[1])
    ax.set_xlabel("X Position (m)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Y Position (m)", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Buffer State Heatmap: {agent_name}\n(Network Health Visualization)",
        fontsize=14,
        fontweight="bold",
        pad=20,
    )
    ax.legend(loc="upper right", fontsize=10, framealpha=0.95)
    ax.grid(True, alpha=0.3, linestyle="--", color="white", linewidth=1)
    ax.set_aspect("equal")

    plt.tight_layout()
    output_file = (
        results_dir / f"buffer_heatmap_{agent_name.lower().replace(' ', '_')}.png"
    )
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✓ Heatmap saved to {output_file}")
    plt.show()


def plot_comparative_heatmaps(dqn_df, smart_df, nearest_df):
    """
    Generate side-by-side heatmaps for all agents.
    """
    plt.style.use("seaborn-v0_8-white")
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # Custom colormap
    colors = ["#0000FF", "#00FFFF", "#FFFF00", "#FF8C00", "#FF0000"]
    cmap = LinearSegmentedColormap.from_list("buffer_state", colors, N=100)

    agents = [
        (dqn_df, "DQN Agent", axes[0]),
        (smart_df, "Smart Greedy V2", axes[1]),
        (nearest_df, "Nearest Greedy", axes[2]),
    ]

    for sensor_df, agent_name, ax in agents:
        if sensor_df is None or sensor_df.empty:
            ax.text(
                0.5,
                0.5,
                "No Data Available",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=14,
                fontweight="bold",
            )
            continue

        # Generate heatmap
        grid_X, grid_Y, grid_Z = create_voronoi_heatmap(sensor_df, grid_resolution=100)

        # Plot heatmap
        im = ax.imshow(
            grid_Z,
            extent=[0, GRID_SIZE[0], 0, GRID_SIZE[1]],
            origin="lower",
            cmap=cmap,
            alpha=0.8,
            interpolation="nearest",
            vmin=0,
            vmax=1,
        )

        # Overlay sensor positions
        ax.scatter(
            sensor_df["x_position"],
            sensor_df["y_position"],
            c="white",
            s=150,
            marker="s",
            edgecolors="black",
            linewidth=2,
            zorder=10,
        )

        # Annotate sensors
        for _, sensor in sensor_df.iterrows():
            x, y = sensor["x_position"], sensor["y_position"]
            sensor_id = int(sensor["sensor_id"])
            buffer_pct = sensor["buffer_fullness"] * 100

            text_color = (
                "red" if buffer_pct > 70 else "orange" if buffer_pct > 40 else "green"
            )

            ax.annotate(
                f"S{sensor_id}",
                (x, y),
                fontsize=7,
                fontweight="bold",
                color=text_color,
                ha="center",
                va="center",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", alpha=0.8),
            )

        # Formatting
        ax.set_xlim(0, GRID_SIZE[0])
        ax.set_ylim(0, GRID_SIZE[1])
        ax.set_xlabel("X Position (m)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Y Position (m)", fontsize=11, fontweight="bold")
        ax.set_title(agent_name, fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3, linestyle="--", color="white", linewidth=0.8)
        ax.set_aspect("equal")

    # Add shared colorbar
    fig.colorbar(
        im, ax=axes, fraction=0.03, pad=0.02, label="Buffer Fullness (0=Empty, 1=Full)"
    )

    plt.suptitle(
        "Comparative Buffer State Analysis: Network Health Across Agents",
        fontsize=15,
        fontweight="bold",
        y=1.02,
    )

    plt.tight_layout()
    output_file = results_dir / "buffer_heatmap_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✓ Comparative heatmap saved to {output_file}")
    plt.show()


def plot_buffer_statistics(dqn_df, smart_df, nearest_df):
    """
    Generate statistical summary of buffer states.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Buffer Fullness Distribution
    ax1 = axes[0, 0]
    for df, label, color in [
        (dqn_df, "DQN", "#1f77b4"),
        (smart_df, "Smart Greedy", "#d62728"),
        (nearest_df, "Nearest Greedy", "#808080"),
    ]:
        if df is not None and not df.empty:
            ax1.hist(
                df["buffer_fullness"] * 100,
                bins=20,
                alpha=0.6,
                label=label,
                color=color,
                edgecolor="black",
            )

    ax1.axvline(
        x=70, color="red", linestyle="--", linewidth=2, label="Critical Threshold"
    )
    ax1.set_xlabel("Buffer Fullness (%)", fontweight="bold")
    ax1.set_ylabel("Number of Sensors", fontweight="bold")
    ax1.set_title("Buffer Fullness Distribution", fontweight="bold")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Data Loss Comparison
    ax2 = axes[0, 1]
    agents = ["DQN", "Smart Greedy", "Nearest Greedy"]
    data_loss = []
    colors_list = ["#1f77b4", "#d62728", "#808080"]

    for df in [dqn_df, smart_df, nearest_df]:
        if df is not None and not df.empty:
            data_loss.append(df["data_lost"].sum())
        else:
            data_loss.append(0)

    bars = ax2.bar(
        agents, data_loss, color=colors_list, edgecolor="black", linewidth=1.5
    )
    ax2.set_ylabel("Total Data Lost (Bytes)", fontweight="bold")
    ax2.set_title("Total Data Loss by Agent", fontweight="bold")
    ax2.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax2.grid(axis="y", alpha=0.3)

    # Annotate bars
    for bar, val in zip(bars, data_loss):
        height = bar.get_height()
        ax2.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val:.0f}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # 3. Spatial Coverage (Sensor Visit Count)
    ax3 = axes[1, 0]
    ax3.text(
        0.5,
        0.5,
        "Spatial Coverage Analysis\n(Requires visit count data)",
        ha="center",
        va="center",
        transform=ax3.transAxes,
        fontsize=12,
        fontweight="bold",
        style="italic",
    )
    ax3.axis("off")

    # 4. Problem Zone Identification
    ax4 = axes[1, 1]

    # Count starved sensors per agent
    starved_counts = []
    for df in [dqn_df, smart_df, nearest_df]:
        if df is not None and not df.empty:
            starved = len(df[df["buffer_fullness"] > 0.7])
            starved_counts.append(starved)
        else:
            starved_counts.append(0)

    bars = ax4.bar(
        agents, starved_counts, color=colors_list, edgecolor="black", linewidth=1.5
    )
    ax4.set_ylabel("Number of Starved Sensors", fontweight="bold")
    ax4.set_title("Starved Sensor Count (>70% buffer full)", fontweight="bold")
    ax4.grid(axis="y", alpha=0.3)

    # Annotate bars
    for bar, val in zip(bars, starved_counts):
        height = bar.get_height()
        ax4.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            f"{val}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=12,
        )

    plt.suptitle("Buffer State Statistics Summary", fontsize=15, fontweight="bold")
    plt.tight_layout()

    output_file = results_dir / "buffer_statistics.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"✓ Buffer statistics plot saved to {output_file}")
    plt.show()


# ==================== MAIN ====================


def main():
    print("=" * 100)
    print("BUFFER STATE HEATMAP - NETWORK HEALTH VISUALIZATION")
    print("=" * 100)

    print("\n⚠ NOTE: This script requires sensor-level buffer state data.")
    print("You need to modify your evaluation script to save sensor snapshots.")
    print("\nExpected file format: JSON with sensor positions and buffer states.")

    # Create synthetic example data for demonstration
    num_sensors = 20
    np.random.seed(42)

    # Generate random sensor positions
    sensor_positions = np.random.uniform(10, 90, (num_sensors, 2))

    # DQN: Some sensors starved (high buffer), especially in corners
    dqn_buffer_fullness = np.random.uniform(0.1, 0.5, num_sensors)
    # Make corner sensors starved
    for i, pos in enumerate(sensor_positions):
        if pos[0] > 70 and pos[1] > 70:  # Top-right corner
            dqn_buffer_fullness[i] = np.random.uniform(0.7, 0.95)
        elif pos[0] < 20 and pos[1] < 20:  # Bottom-left corner
            dqn_buffer_fullness[i] = np.random.uniform(0.6, 0.85)

    # Smart Greedy: More balanced
    smart_buffer_fullness = np.random.uniform(0.2, 0.6, num_sensors)

    # Nearest Greedy: Highly variable
    nearest_buffer_fullness = np.random.uniform(0.1, 0.9, num_sensors)

    # Create DataFrames
    dqn_df = pd.DataFrame(
        {
            "sensor_id": range(num_sensors),
            "x_position": sensor_positions[:, 0],
            "y_position": sensor_positions[:, 1],
            "buffer_fullness": dqn_buffer_fullness,
            "data_lost": dqn_buffer_fullness
            * np.random.uniform(1000, 5000, num_sensors),
            "buffer_bytes": dqn_buffer_fullness * SENSOR_BUFFER_CAPACITY,
        }
    )

    smart_df = pd.DataFrame(
        {
            "sensor_id": range(num_sensors),
            "x_position": sensor_positions[:, 0],
            "y_position": sensor_positions[:, 1],
            "buffer_fullness": smart_buffer_fullness,
            "data_lost": smart_buffer_fullness
            * np.random.uniform(1000, 5000, num_sensors),
            "buffer_bytes": smart_buffer_fullness * SENSOR_BUFFER_CAPACITY,
        }
    )

    nearest_df = pd.DataFrame(
        {
            "sensor_id": range(num_sensors),
            "x_position": sensor_positions[:, 0],
            "y_position": sensor_positions[:, 1],
            "buffer_fullness": nearest_buffer_fullness,
            "data_lost": nearest_buffer_fullness
            * np.random.uniform(1000, 5000, num_sensors),
            "buffer_bytes": nearest_buffer_fullness * SENSOR_BUFFER_CAPACITY,
        }
    )

    # Generate visualizations
    print("\n" + "=" * 100)
    print("Generating buffer state visualizations...")
    print("=" * 100)

    plot_buffer_heatmap(dqn_df, "DQN Agent (Proposed)", use_voronoi=True)
    plot_buffer_heatmap(smart_df, "Smart Greedy V2", use_voronoi=True)
    plot_buffer_heatmap(nearest_df, "Nearest Greedy", use_voronoi=True)

    plot_comparative_heatmaps(dqn_df, smart_df, nearest_df)
    plot_buffer_statistics(dqn_df, smart_df, nearest_df)

    print("\n✓ Buffer state analysis complete.")
    print("\n" + "=" * 100)
    print("INTERPRETATION GUIDE:")
    print("=" * 100)
    print("• Red regions: High buffer fullness (sensors are starved, data being lost)")
    print("• Blue regions: Low buffer fullness (sensors well-serviced)")
    print("• Yellow/Orange: Warning zones (approaching critical buffer levels)")
    print("• Problem Zones: Spatial clusters of starved sensors (red areas)")
    print("\nYour dissertation narrative should highlight:")
    print("1. Spatial patterns of starvation (e.g., 'DQN neglects top-right corner')")
    print("2. Comparison with greedy agents (more balanced coverage?)")
    print("3. Justification for reward function modifications (fairness penalty)")
    print("4. Connection to trajectory plots (why certain areas are under-visited)")
    print("=" * 100)


if __name__ == "__main__":
    main()
