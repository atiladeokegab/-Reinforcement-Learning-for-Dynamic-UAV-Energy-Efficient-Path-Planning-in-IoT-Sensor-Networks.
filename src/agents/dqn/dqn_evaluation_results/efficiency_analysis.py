"""
Efficiency Analysis - Pareto Front Visualization
Generates scatter plot showing data collected vs. energy consumed.
Demonstrates multi-objective optimization: maximize data, minimize energy.

Author: ATILADE GABRIEL OKE
Modified: February 2026
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from pathlib import Path
import ieee_style
ieee_style.apply()

# ==================== PATHS ====================
script_dir = Path(__file__).resolve().parent
results_dir = script_dir / "baseline_results"

print(f"Results directory: {results_dir}")

# ==================== CONSTANTS ====================
MAX_BATTERY = 274.0  # Wh (from your config)


# ==================== LOAD DATA ====================


def load_agent_results(agent_name):
    """Load CSV results for a specific agent."""
    csv_path = results_dir / f"{agent_name}_results.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    else:
        print(f"⚠ WARNING: Could not find {csv_path}")
        return None


def calculate_energy_consumed(battery_wh):
    """Calculate energy consumed from battery level."""
    return MAX_BATTERY - battery_wh


def calculate_efficiency(data_collected, energy_consumed):
    """Calculate bytes per Wh efficiency metric."""
    return data_collected / energy_consumed if energy_consumed > 0 else 0


# ==================== ANALYSIS ====================


def extract_efficiency_trajectory(df, agent_name):
    """
    Extract (energy_consumed, data_collected) pairs from trajectory.
    Returns DataFrame with efficiency metrics.
    """
    if df is None or df.empty:
        return None

    # Calculate energy consumed at each step
    df["energy_consumed"] = df["battery_wh"].apply(calculate_energy_consumed)

    # Calculate instantaneous efficiency
    df["efficiency"] = df.apply(
        lambda row: calculate_efficiency(
            row["total_data_collected"], row["energy_consumed"]
        ),
        axis=1,
    )

    # Add agent label
    df["agent"] = agent_name

    return df[
        ["step", "energy_consumed", "total_data_collected", "efficiency", "agent"]
    ]


def find_pareto_frontier(energy, data):
    """
    Find Pareto-optimal points (minimize energy, maximize data).
    Returns indices of Pareto-optimal points.
    """
    points = np.column_stack([energy, data])
    n_points = len(points)
    is_pareto = np.ones(n_points, dtype=bool)

    for i in range(n_points):
        for j in range(n_points):
            if i != j:
                # Point j dominates point i if:
                # - j uses less or equal energy AND collects more data, OR
                # - j collects equal or more data AND uses less energy
                if (points[j][0] <= points[i][0] and points[j][1] > points[i][1]) or (
                    points[j][1] >= points[i][1] and points[j][0] < points[i][0]
                ):
                    is_pareto[i] = False
                    break

    return np.where(is_pareto)[0]


# ==================== PLOTTING ====================


def plot_efficiency_scatter(dqn_df, smart_df, nearest_df):
    """
    Generate Pareto front scatter plot showing efficiency trade-offs.
    """
    ieee_style.apply()
    fig, ax = plt.subplots(figsize=(14, 9))

    # Plot data points with progression coloring
    agents = [
        (dqn_df, "DQN Agent (Proposed)", "#1b9e77", "o"),
        (smart_df, "SF-Aware Greedy V2", "#d95f02", "s"),
        (nearest_df, "Nearest Sensor Greedy", "#7570b3", "^"),
    ]

    all_pareto_indices = []

    for df, label, color, marker in agents:
        if df is None or df.empty:
            continue

        energy = df["energy_consumed"].values
        data = df["total_data_collected"].values

        # Create color gradient based on simulation progression
        n_points = len(energy)
        colors = (
            plt.cm.Blues(np.linspace(0.3, 0.9, n_points))
            if "DQN" in label
            else (
                plt.cm.Reds(np.linspace(0.3, 0.9, n_points))
                if "Smart" in label
                else plt.cm.Greys(np.linspace(0.3, 0.9, n_points))
            )
        )

        # Plot trajectory as scatter with gradient
        scatter = ax.scatter(
            energy,
            data,
            c=colors,
            marker=marker,
            s=80,
            alpha=0.6,
            edgecolors="black",
            linewidth=0.5,
            label=label,
        )

        # Highlight final point
        ax.scatter(
            energy[-1],
            data[-1],
            c=color,
            marker=marker,
            s=300,
            edgecolors="black",
            linewidth=2.5,
            zorder=10,
            label=f"{label} (Final)",
        )

        # Find and plot Pareto frontier for this agent
        pareto_indices = find_pareto_frontier(energy, data)
        all_pareto_indices.extend([(energy[i], data[i], label) for i in pareto_indices])

        if len(pareto_indices) > 1:
            # Sort Pareto points by energy
            sorted_indices = pareto_indices[np.argsort(energy[pareto_indices])]
            pareto_energy = energy[sorted_indices]
            pareto_data = data[sorted_indices]

            # Draw Pareto frontier line
            ax.plot(
                pareto_energy,
                pareto_data,
                color=color,
                linestyle="--",
                linewidth=2,
                alpha=0.7,
                zorder=5,
            )

    # Annotate key regions
    ax.text(
        0.05,
        0.95,
        "IDEAL REGION\n(High Data, Low Energy)",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        verticalalignment="top",
        horizontalalignment="left",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.7),
    )

    ax.text(
        0.95,
        0.05,
        "INEFFICIENT REGION\n(Low Data, High Energy)",
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        verticalalignment="bottom",
        horizontalalignment="right",
        bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.7),
    )

    # Formatting
    ax.set_xlabel("Energy Consumed (Wh)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Data Collected (Bytes)", fontsize=13, fontweight="bold")
    ax.set_title(
        "Efficiency Analysis: Data Collection vs. Energy Consumption (Pareto Front)",
        fontsize=15,
        fontweight="bold",
        pad=20,
    )

    # Format y-axis for large numbers
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    ax.legend(loc="lower right", fontsize=10, framealpha=0.95, ncol=2)
    ax.grid(True, alpha=0.4, linestyle="--")

    plt.tight_layout()
    ieee_style.clean_figure(plt.gcf())
    ieee_style.save(plt.gcf(), str(results_dir / "efficiency_pareto_front"))
    plt.close()


def plot_efficiency_progression(dqn_df, smart_df, nearest_df):
    """
    Plot efficiency (bytes/Wh) over time for each agent.
    """
    ieee_style.apply()
    fig, ax = plt.subplots(figsize=(14, 7))

    agents = [
        (dqn_df, "DQN Agent (Proposed)", "#1b9e77", "-",  "o"),
        (smart_df, "SF-Aware Greedy V2", "#d95f02", "--", "s"),
        (nearest_df, "Nearest Sensor Greedy", "#7570b3", ":", "^"),
    ]

    for df, label, color, linestyle, marker in agents:
        if df is None or df.empty:
            continue

        # Smooth efficiency with rolling average
        window = max(1, len(df) // 20)  # Adaptive window
        efficiency_smooth = df["efficiency"].rolling(window=window, center=True).mean()

        ax.plot(
            df["step"],
            efficiency_smooth,
            color=color,
            linestyle=linestyle,
            linewidth=2.5,
            marker=marker,
            markersize=5,
            markevery=max(1, len(df) // 15),
            label=label,
            alpha=0.85,
        )

    # Formatting
    ax.set_xlabel("Simulation Step (t)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Efficiency (Bytes/Wh)", fontsize=13, fontweight="bold")
    ax.set_title(
        "Efficiency Progression: Data Collection Rate per Energy Unit",
        fontsize=15,
        fontweight="bold",
        pad=20,
    )

    ax.legend(loc="best", fontsize=11, framealpha=0.95)
    ax.grid(True, alpha=0.5, linestyle="--")

    plt.tight_layout()
    ieee_style.clean_figure(plt.gcf())
    ieee_style.save(plt.gcf(), str(results_dir / "efficiency_progression"))
    plt.close()


def plot_efficiency_comparison_table(dqn_df, smart_df, nearest_df):
    """
    Generate summary table comparing efficiency metrics.
    """
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis("tight")
    ax.axis("off")

    def calculate_metrics(df, agent_name):
        if df is None or df.empty:
            return [agent_name, "N/A", "N/A", "N/A", "N/A", "N/A"]

        final_energy = df["energy_consumed"].iloc[-1]
        final_data = df["total_data_collected"].iloc[-1]
        final_efficiency = df["efficiency"].iloc[-1]
        avg_efficiency = df["efficiency"].mean()
        peak_efficiency = df["efficiency"].max()

        return [
            agent_name,
            f"{final_energy:.2f}",
            f"{final_data:.0f}",
            f"{final_efficiency:.2f}",
            f"{avg_efficiency:.2f}",
            f"{peak_efficiency:.2f}",
        ]

    # Prepare table data
    table_data = [
        calculate_metrics(dqn_df, "DQN Agent"),
        calculate_metrics(smart_df, "Smart Greedy V2"),
        calculate_metrics(nearest_df, "Nearest Greedy"),
    ]

    col_labels = [
        "Agent",
        "Final Energy\nConsumed (Wh)",
        "Final Data\nCollected (Bytes)",
        "Final Efficiency\n(Bytes/Wh)",
        "Average Efficiency\n(Bytes/Wh)",
        "Peak Efficiency\n(Bytes/Wh)",
    ]

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colWidths=[0.2, 0.16, 0.16, 0.16, 0.16, 0.16],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.8)

    # Color header
    for i in range(6):
        table[(0, i)].set_facecolor("#4CAF50")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Highlight best values in each column (except agent name)
    for col_idx in range(2, 6):  # Skip agent name and energy consumed
        col_vals = []
        for row_idx in range(1, 4):
            try:
                val_str = table[(row_idx, col_idx)].get_text().get_text()
                col_vals.append(float(val_str))
            except:
                col_vals.append(0)

        if col_vals:
            best_row = np.argmax(col_vals) + 1
            table[(best_row, col_idx)].set_facecolor("#90EE90")
            table[(best_row, col_idx)].set_text_props(weight="bold")

    # Highlight lowest energy consumed
    energy_vals = []
    for row_idx in range(1, 4):
        try:
            val_str = table[(row_idx, 1)].get_text().get_text()
            energy_vals.append(float(val_str))
        except:
            energy_vals.append(np.inf)

    if energy_vals:
        best_row = np.argmin(energy_vals) + 1
        table[(best_row, 1)].set_facecolor("#90EE90")
        table[(best_row, 1)].set_text_props(weight="bold")

    plt.title("Efficiency Metrics Summary", fontsize=14, fontweight="bold", pad=20)

    ieee_style.save(plt.gcf(), str(results_dir / "efficiency_metrics_table"))
    plt.close()


# ==================== MAIN ====================


def main():
    print("=" * 100)
    print("EFFICIENCY ANALYSIS - PARETO FRONT VISUALIZATION")
    print("=" * 100)

    # Load agent results
    print("\nLoading agent results...")
    dqn_raw = load_agent_results("dqn_agent_fresh")
    smart_raw = load_agent_results("greedy_smart_v2")
    nearest_raw = load_agent_results("greedy_nearest")

    # Extract efficiency trajectories
    print("\nProcessing efficiency trajectories...")
    dqn_df = extract_efficiency_trajectory(dqn_raw, "DQN")
    smart_df = extract_efficiency_trajectory(smart_raw, "Smart Greedy")
    nearest_df = extract_efficiency_trajectory(nearest_raw, "Nearest Greedy")

    if dqn_df is not None:
        print(f"✓ DQN: {len(dqn_df)} data points")
    if smart_df is not None:
        print(f"✓ Smart Greedy: {len(smart_df)} data points")
    if nearest_df is not None:
        print(f"✓ Nearest Greedy: {len(nearest_df)} data points")

    # Generate visualizations
    print("\n" + "=" * 100)
    print("Generating efficiency visualizations...")
    print("=" * 100)

    plot_efficiency_scatter(dqn_df, smart_df, nearest_df)
    plot_efficiency_progression(dqn_df, smart_df, nearest_df)
    plot_efficiency_comparison_table(dqn_df, smart_df, nearest_df)

    # Summary statistics
    print("\n" + "=" * 100)
    print("EFFICIENCY SUMMARY")
    print("=" * 100)

    for df, name in [
        (dqn_df, "DQN"),
        (smart_df, "Smart Greedy"),
        (nearest_df, "Nearest Greedy"),
    ]:
        if df is None or df.empty:
            continue

        final_efficiency = df["efficiency"].iloc[-1]
        avg_efficiency = df["efficiency"].mean()
        peak_efficiency = df["efficiency"].max()

        print(f"\n{name}:")
        print(f"  Final Efficiency: {final_efficiency:>10.2f} bytes/Wh")
        print(f"  Average Efficiency: {avg_efficiency:>8.2f} bytes/Wh")
        print(f"  Peak Efficiency: {peak_efficiency:>11.2f} bytes/Wh")

    print("\n✓ Efficiency analysis complete.")
    print("\n" + "=" * 100)
    print("INTERPRETATION GUIDE:")
    print("=" * 100)
    print("• Pareto Front: Points on the frontier represent optimal trade-offs.")
    print("• Top-Left Region: High data collection with low energy (IDEAL).")
    print("• Bottom-Right Region: Low data with high energy (INEFFICIENT).")
    print("• Final Efficiency: Overall bytes/Wh at mission completion.")
    print("• Peak Efficiency: Best instantaneous performance during flight.")
    print("\nYour dissertation narrative should highlight:")
    print("1. DQN's position relative to the Pareto frontier")
    print("2. Energy efficiency gains compared to baseline agents")
    print("3. Trade-offs between data maximization and energy conservation")
    print("=" * 100)


if __name__ == "__main__":
    main()
