"""
Master Analysis Orchestrator
Runs all three dissertation analysis scripts in sequence and generates
a comprehensive summary report.

Author: ATILADE GABRIEL OKE
Modified: February 2026
"""

import sys
import subprocess
from pathlib import Path
import time
import json

# ==================== CONFIGURATION ====================
script_dir = Path(__file__).resolve().parent
results_dir = script_dir / "baseline_results"

# Analysis scripts to run
ANALYSIS_SCRIPTS = [
    ("fairness_analysis.py", "Fairness Analysis"),
    ("efficiency_analysis.py", "Efficiency Analysis"),
    ("buffer_heatmap_analysis.py", "Buffer State Heatmap"),
]


# ==================== VERIFICATION ====================


def verify_data_files():
    """Check that all required data files exist."""
    print("=" * 100)
    print("DATA VERIFICATION")
    print("=" * 100)

    required_csvs = [
        "dqn_agent_fresh_results.csv",
        "greedy_smart_v2_results.csv",
        "greedy_nearest_results.csv",
    ]

    optional_jsons = [
        "dqn_sensor_snapshot.json",
        "greedy_smart_sensor_snapshot.json",
        "greedy_nearest_sensor_snapshot.json",
    ]

    all_good = True

    # Check CSV files (required for efficiency analysis)
    print("\nRequired CSV files:")
    for csv_file in required_csvs:
        csv_path = results_dir / csv_file
        exists = csv_path.exists()
        status = "✓" if exists else "✗"
        print(f"  {status} {csv_file}")
        if not exists:
            all_good = False

    # Check JSON snapshots (needed for fairness and heatmap)
    print("\nOptional JSON snapshots (for fairness & heatmap analysis):")
    json_count = 0
    for json_file in optional_jsons:
        json_path = results_dir / json_file
        exists = json_path.exists()
        status = "✓" if exists else "○"
        print(f"  {status} {json_file}")
        if exists:
            json_count += 1

    print("\n" + "=" * 100)
    if not all_good:
        print("⚠ WARNING: Missing required CSV files!")
        print("Run your evaluation script (multi_compare_agents.py) first.")
        return False

    if json_count == 0:
        print("⚠ NOTE: No JSON snapshots found.")
        print(
            "Fairness and heatmap analyses will use synthetic data for demonstration."
        )
    else:
        print(
            f"✓ Found {json_count}/3 JSON snapshots - analyses will use real sensor data!"
        )

    print("=" * 100)
    return True


# ==================== SCRIPT EXECUTION ====================


def run_analysis_script(script_name, display_name):
    """Run a single analysis script."""
    print("\n" + "=" * 100)
    print(f"RUNNING: {display_name}")
    print("=" * 100)

    script_path = script_dir / script_name

    if not script_path.exists():
        print(f"✗ ERROR: Script not found at {script_path}")
        return False

    start_time = time.time()

    try:
        # Run the script
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=str(script_dir),
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout
        )

        # Print output
        if result.stdout:
            print(result.stdout)

        if result.stderr and result.returncode != 0:
            print(f"⚠ STDERR:\n{result.stderr}")

        elapsed = time.time() - start_time

        if result.returncode == 0:
            print(f"\n✓ {display_name} completed successfully in {elapsed:.1f}s")
            return True
        else:
            print(f"\n✗ {display_name} failed with return code {result.returncode}")
            return False

    except subprocess.TimeoutExpired:
        print(f"\n✗ {display_name} timed out after 5 minutes")
        return False
    except Exception as e:
        print(f"\n✗ {display_name} failed with error: {e}")
        return False


# ==================== OUTPUT INVENTORY ====================


def generate_output_inventory():
    """List all generated plot files."""
    print("\n" + "=" * 100)
    print("GENERATED OUTPUT FILES")
    print("=" * 100)

    plot_extensions = [".png", ".pdf", ".jpg"]
    plots = []

    for ext in plot_extensions:
        plots.extend(results_dir.glob(f"*{ext}"))

    # Group by analysis type
    fairness_plots = [p for p in plots if "fairness" in p.name.lower()]
    efficiency_plots = [p for p in plots if "efficiency" in p.name.lower()]
    heatmap_plots = [
        p for p in plots if "heatmap" in p.name.lower() or "buffer" in p.name.lower()
    ]
    other_plots = [
        p for p in plots if p not in fairness_plots + efficiency_plots + heatmap_plots
    ]

    print("\n📊 Fairness Analysis Plots:")
    for plot in sorted(fairness_plots):
        print(f"  • {plot.name}")

    print("\n⚡ Efficiency Analysis Plots:")
    for plot in sorted(efficiency_plots):
        print(f"  • {plot.name}")

    print("\n🗺️  Buffer State Heatmap Plots:")
    for plot in sorted(heatmap_plots):
        print(f"  • {plot.name}")

    if other_plots:
        print("\n📈 Other Visualization Plots:")
        for plot in sorted(other_plots):
            print(f"  • {plot.name}")

    total = len(plots)
    print(f"\n✓ Total: {total} visualization files generated")
    print(f"📁 Location: {results_dir}")
    print("=" * 100)

    return plots


# ==================== SUMMARY REPORT ====================


def generate_summary_report(success_results):
    """Generate a summary report of the analysis run."""
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "script_directory": str(script_dir),
        "results_directory": str(results_dir),
        "analyses_run": [],
        "success_count": 0,
        "failure_count": 0,
    }

    for script, display_name, success in success_results:
        report["analyses_run"].append(
            {"script": script, "name": display_name, "success": success}
        )
        if success:
            report["success_count"] += 1
        else:
            report["failure_count"] += 1

    # Save report
    report_path = results_dir / "analysis_run_summary.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    print("\n" + "=" * 100)
    print("ANALYSIS RUN SUMMARY")
    print("=" * 100)
    print(f"✓ Successful: {report['success_count']}/{len(success_results)}")
    print(f"✗ Failed: {report['failure_count']}/{len(success_results)}")
    print(f"\n📄 Summary saved to: {report_path}")
    print("=" * 100)


# ==================== MAIN ====================


def main():
    print("\n" + "=" * 100)
    print("MASTER ANALYSIS ORCHESTRATOR")
    print("Dissertation Analysis Pipeline - Running All Scripts")
    print("=" * 100)
    print(f"\nScript Directory: {script_dir}")
    print(f"Results Directory: {results_dir}")
    print(f"Analysis Scripts: {len(ANALYSIS_SCRIPTS)}")

    # Step 1: Verify data files
    if not verify_data_files():
        print("\n✗ Data verification failed. Please ensure evaluation has been run.")
        print("Exiting...")
        return

    # Step 2: Run each analysis script
    success_results = []

    for script_name, display_name in ANALYSIS_SCRIPTS:
        success = run_analysis_script(script_name, display_name)
        success_results.append((script_name, display_name, success))

        # Brief pause between scripts
        time.sleep(1)

    # Step 3: Generate output inventory
    plots = generate_output_inventory()

    # Step 4: Generate summary report
    generate_summary_report(success_results)

    # Final message
    print("\n" + "=" * 100)
    print("COMPLETE!")
    print("=" * 100)
    all_success = all(success for _, _, success in success_results)

    if all_success:
        print("✓ All analyses completed successfully!")
        print(f"✓ {len(plots)} visualization files ready for your dissertation")
        print("\nNext steps:")
        print("  1. Review plots in baseline_results/")
        print("  2. Use interpretation guides in each script's output")
        print("  3. Insert figures into dissertation Chapter 4")
    else:
        print("⚠ Some analyses encountered issues - see details above")
        print("  Check that all required data files exist")
        print("  Review error messages for specific issues")

    print("\n💡 Pro tip: Re-run individual scripts to regenerate specific plots:")
    print("  python fairness_analysis.py")
    print("  python efficiency_analysis.py")
    print("  python buffer_heatmap_analysis.py")
    print("=" * 100)


if __name__ == "__main__":
    main()
