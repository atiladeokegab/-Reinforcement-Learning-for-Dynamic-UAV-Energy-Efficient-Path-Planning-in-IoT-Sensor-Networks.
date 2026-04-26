"""
Relational RL vs TSP Oracle — Multi-Condition Sweep
====================================================
Evaluates both agents across:
  - Grid sizes : 200x200, 400x400, 600x600
  - Sensor counts : 10, 20, 30
  - Seeds : 42, 123, 456, 789, 1337

Metrics per episode:
  - data_collected  (bytes)
  - jains_index     (Jain's fairness on per-sensor collection rates)
  - efficiency      (bytes / Wh consumed)
  - avg_buffer_occ  (mean buffer occupancy = urgency proxy; lower is better)
  - ndr_pct         (% sensors visited = coverage)

Outputs:
  baseline_results/relational_vs_tsp_sweep.csv    raw episode rows
  baseline_results/relational_vs_tsp_sweep.png    slide-ready 5-metric figure

Usage:
    PYTHONIOENCODING=utf-8 uv run python relational_vs_tsp_sweep.py
    PYTHONIOENCODING=utf-8 uv run python relational_vs_tsp_sweep.py --plots-only

Author: ATILADE GABRIEL OKE
"""

from __future__ import annotations

import argparse
import sys
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Path bootstrap ────────────────────────────────────────────────────────────
_HERE = Path(__file__).resolve().parent
_SRC  = _HERE.parent.parent.parent
for _p in (str(_SRC), str(_HERE.parent), str(_HERE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from environment.uav_env import UAVEnvironment
from greedy_agents import TSPOracleAgent
from relational_rl_runner import (
    InferenceRelationalUAVEnv,
    load_relational_rl_module,
)

# ── Constants ─────────────────────────────────────────────────────────────────
GRID_SIZES    = [(200, 200), (400, 400), (600, 600)]
SENSOR_COUNTS = [10, 20, 30]
SEEDS         = [42, 123, 456, 789, 1337]
MAX_STEPS     = 2100
MAX_BATTERY   = 274.0
N_MAX         = 50

_CHECKPOINT = (
    _HERE.parent / "models" / "relational_rl"
    / "results" / "checkpoints" / "stage_3" / "final"
)

OUT_DIR = _HERE / "baseline_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUT_DIR / "relational_vs_tsp_sweep.csv"
FIG_PATH = OUT_DIR / "relational_vs_tsp_sweep.png"

# Colours consistent with the rest of the dissertation figures
COLOURS = {
    "Relational RL": "#7570b3",
    "TSP Oracle":    "#1b9e77",
}
MARKERS = {
    "Relational RL": "o",
    "TSP Oracle":    "s",
}

# ── Metric helpers ────────────────────────────────────────────────────────────

def _sensor_metrics(sensors) -> dict:
    rates = []
    buf_occ = []
    for s in sensors:
        gen = float(s.total_data_generated)
        tx  = float(s.total_data_transmitted)
        rates.append((tx / gen * 100) if gen > 0 else 0.0)
        buf_occ.append(float(s.data_buffer) / float(s.max_buffer_size))
    n  = len(rates)
    s2 = sum(r ** 2 for r in rates)
    jain = (sum(rates) ** 2 / (n * s2)) if n > 0 and s2 > 0 else 0.0
    return {
        "jains_index":   round(jain, 4),
        "avg_buffer_occ": round(sum(buf_occ) / n if n else 0.0, 4),
    }


# ── TSP Oracle episode ────────────────────────────────────────────────────────

def run_tsp_episode(grid_size, n_sensors, seed) -> dict:
    np.random.seed(seed)
    random.seed(seed)

    env = UAVEnvironment(
        grid_size=grid_size,
        num_sensors=n_sensors,
        max_steps=MAX_STEPS,
        max_battery=MAX_BATTERY,
        path_loss_exponent=3.8,
        rssi_threshold=-85.0,
        sensor_duty_cycle=10.0,
        render_mode=None,
    )
    agent = TSPOracleAgent(env)

    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        action = agent.select_action(obs)
        obs, _, done, truncated, _ = env.step(action)
        if truncated:
            break

    energy = MAX_BATTERY - float(env.uav.battery)
    ndr    = len(env.sensors_visited) / n_sensors * 100
    sm     = _sensor_metrics(env.sensors)
    result = {
        "agent":          "TSP Oracle",
        "grid_w":         grid_size[0],
        "n_sensors":      n_sensors,
        "seed":           seed,
        "data_collected": round(float(env.total_data_collected), 2),
        "ndr_pct":        round(ndr, 2),
        "energy_wh":      round(energy, 4),
        "efficiency":     round(float(env.total_data_collected) / energy if energy > 0 else 0.0, 4),
        **sm,
    }
    env.close()
    return result


# ── Relational RL episode ─────────────────────────────────────────────────────

def run_relational_episode(rl_module, grid_size, n_sensors, seed) -> dict:
    import torch
    from ray.rllib.core.columns import Columns

    np.random.seed(seed)
    random.seed(seed)

    env = InferenceRelationalUAVEnv(
        n_max=N_MAX,
        grid_size=grid_size,
        num_sensors=n_sensors,
        max_steps=MAX_STEPS,
        max_battery=MAX_BATTERY,
        path_loss_exponent=3.8,
        rssi_threshold=-85.0,
        sensor_duty_cycle=10.0,
        render_mode=None,
    )

    obs, _ = env.reset(seed=seed)
    done = False
    while not done:
        batch = {Columns.OBS: {
            k: torch.as_tensor(np.asarray(v)).unsqueeze(0)
            for k, v in obs.items()
        }}
        with torch.no_grad():
            out    = rl_module._forward_inference(batch)
            action = int(torch.argmax(out[Columns.ACTION_DIST_INPUTS], dim=-1).item())

        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    energy = MAX_BATTERY - float(env.uav.battery)
    ndr    = len(env.sensors_visited) / n_sensors * 100
    sm     = _sensor_metrics(env.sensors)
    result = {
        "agent":          "Relational RL",
        "grid_w":         grid_size[0],
        "n_sensors":      n_sensors,
        "seed":           seed,
        "data_collected": round(float(env.total_data_collected), 2),
        "ndr_pct":        round(ndr, 2),
        "energy_wh":      round(energy, 4),
        "efficiency":     round(float(env.total_data_collected) / energy if energy > 0 else 0.0, 4),
        **sm,
    }
    env.close()
    return result


# ── Plotting ──────────────────────────────────────────────────────────────────

def _condition_label(gw, n):
    return f"{gw}×{gw}\nn={n}"


def plot_sweep(df: pd.DataFrame, path: Path):
    try:
        import ieee_style
        ieee_style.apply()
    except ImportError:
        pass

    conditions = [(gw, n) for gw in [200, 400, 600] for n in [10, 20, 30]]
    xlabels    = [_condition_label(gw, n) for gw, n in conditions]
    x          = np.arange(len(conditions))
    w          = 0.32
    agents     = ["Relational RL", "TSP Oracle"]

    metrics = [
        ("data_collected",  "Data collected (bytes)",       False),
        ("efficiency",      "Energy efficiency (bytes/Wh)", False),
        ("jains_index",     "Jain's fairness index",        False),
        ("avg_buffer_occ",  "Avg buffer occupancy",         True),   # lower=better
        ("ndr_pct",         "Coverage (%)",                 False),
    ]

    fig, axes = plt.subplots(1, len(metrics), figsize=(22, 5))
    fig.suptitle(
        "Relational RL vs Oracle TSP — Full Metric Comparison\n"
        f"({len(SEEDS)} seeds × {len(GRID_SIZES)} grid sizes × {len(SENSOR_COUNTS)} sensor counts)",
        fontsize=12, fontweight="bold",
    )

    for ax, (col, ylabel, lower_better) in zip(axes, metrics):
        for i, agent in enumerate(agents):
            sub    = df[df["agent"] == agent]
            means, errs = [], []
            for gw, n in conditions:
                vals = sub[(sub["grid_w"] == gw) & (sub["n_sensors"] == n)][col].dropna()
                means.append(vals.mean() if len(vals) else np.nan)
                errs.append(vals.std()  if len(vals) else 0.0)

            offset = (i - 0.5) * w
            bars = ax.bar(
                x + offset, means, w,
                yerr=errs, capsize=3,
                color=COLOURS[agent], alpha=0.82,
                edgecolor="white", linewidth=0.8,
                error_kw={"elinewidth": 1.2},
                label=agent,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=7)
        ax.set_ylabel(ylabel, fontsize=9)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

        if lower_better:
            ax.set_title(ylabel + "\n(lower = better)", fontsize=8, fontweight="bold")
        else:
            ax.set_title(ylabel, fontsize=8, fontweight="bold")

        if col == "data_collected":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    axes[0].legend(fontsize=8, loc="upper left", framealpha=0.85)
    plt.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Figure saved: {path.name}")


def print_summary(df: pd.DataFrame):
    metrics = [
        ("data_collected", "Data (bytes)"),
        ("efficiency",     "Eff (B/Wh)"),
        ("jains_index",    "Jain's"),
        ("avg_buffer_occ", "Buf occ"),
        ("ndr_pct",        "NDR %"),
    ]
    print(f"\n{'='*80}")
    print(f"AGGREGATE RESULTS  (mean ± std across {len(SEEDS)} seeds × {len(GRID_SIZES)} grids × {len(SENSOR_COUNTS)} N)")
    print(f"{'='*80}")
    hdr = f"{'Agent':<18}" + "".join(f"{lbl:>18}" for _, lbl in metrics)
    print(hdr)
    print("-" * len(hdr))
    for agent in ["Relational RL", "TSP Oracle"]:
        sub  = df[df["agent"] == agent]
        row  = f"{agent:<18}"
        for col, _ in metrics:
            vals = sub[col].dropna()
            row += f"  {vals.mean():>7.2f}±{vals.std():>6.2f}"
        print(row)

    print(f"\n--- Gap: Relational RL vs TSP Oracle (relative) ---")
    for col, lbl in metrics:
        rel_m = df[df["agent"] == "Relational RL"][col].mean()
        tsp_m = df[df["agent"] == "TSP Oracle"][col].mean()
        gap   = (rel_m / tsp_m - 1) * 100 if tsp_m != 0 else float("nan")
        arrow = "↑" if gap > 0 else "↓"
        print(f"  {lbl:<18}: {gap:+.1f}%  {arrow}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plots-only", action="store_true",
                        help="Skip evaluation, regenerate plots from existing CSV")
    parser.add_argument("--resume", action="store_true",
                        help="Skip rows already present in the CSV")
    args = parser.parse_args()

    if args.plots_only:
        if not CSV_PATH.exists():
            print(f"ERROR: {CSV_PATH} not found. Run without --plots-only first.")
            return
        df = pd.read_csv(CSV_PATH)
        plot_sweep(df, FIG_PATH)
        print_summary(df)
        return

    # ── Load Relational RL module ─────────────────────────────────────────────
    rl_module = load_relational_rl_module(_CHECKPOINT)

    # ── Resume: find already-completed rows ──────────────────────────────────
    completed: set[tuple] = set()
    if args.resume and CSV_PATH.exists():
        done_df = pd.read_csv(CSV_PATH)
        for _, r in done_df.iterrows():
            completed.add((r["agent"], int(r["grid_w"]), int(r["n_sensors"]), int(r["seed"])))
        print(f"Resuming: {len(completed)} rows already done.")

    # ── Run all episodes ──────────────────────────────────────────────────────
    total = 2 * len(GRID_SIZES) * len(SENSOR_COUNTS) * len(SEEDS)
    done_count = len(completed)
    print(f"\n{'='*70}")
    print(f"Relational RL vs TSP Oracle sweep")
    print(f"  Grids: {GRID_SIZES}  |  Sensors: {SENSOR_COUNTS}  |  Seeds: {SEEDS}")
    print(f"  Total episodes: {total}  |  Already done: {done_count}")
    print(f"{'='*70}\n")

    rows: list[dict] = []
    ep_idx = 0

    for grid_size in GRID_SIZES:
        for n_sensors in SENSOR_COUNTS:
            for seed in SEEDS:
                ep_idx += 1

                # TSP Oracle
                key_tsp = ("TSP Oracle", grid_size[0], n_sensors, seed)
                if key_tsp not in completed:
                    try:
                        r = run_tsp_episode(grid_size, n_sensors, seed)
                        rows.append(r)
                        completed.add(key_tsp)
                        pd.DataFrame(rows).to_csv(CSV_PATH, index=False)
                        print(
                            f"  [TSP {ep_idx:>3}/{total}] "
                            f"{grid_size[0]}×{grid_size[0]} n={n_sensors} s={seed} | "
                            f"data={r['data_collected']:>8.0f}  "
                            f"jain={r['jains_index']:.3f}  "
                            f"buf={r['avg_buffer_occ']:.3f}  "
                            f"ndr={r['ndr_pct']:.1f}%"
                        )
                    except Exception as e:
                        print(f"  ✗ TSP {grid_size} n={n_sensors} s={seed}: {e}")

                # Relational RL
                key_rel = ("Relational RL", grid_size[0], n_sensors, seed)
                if key_rel not in completed:
                    try:
                        r = run_relational_episode(rl_module, grid_size, n_sensors, seed)
                        rows.append(r)
                        completed.add(key_rel)
                        pd.DataFrame(rows).to_csv(CSV_PATH, index=False)
                        print(
                            f"  [REL {ep_idx:>3}/{total}] "
                            f"{grid_size[0]}×{grid_size[0]} n={n_sensors} s={seed} | "
                            f"data={r['data_collected']:>8.0f}  "
                            f"jain={r['jains_index']:.3f}  "
                            f"buf={r['avg_buffer_occ']:.3f}  "
                            f"ndr={r['ndr_pct']:.1f}%"
                        )
                    except Exception as e:
                        print(f"  ✗ REL {grid_size} n={n_sensors} s={seed}: {e}")

    # ── Load full CSV and plot ─────────────────────────────────────────────────
    if CSV_PATH.exists():
        df = pd.read_csv(CSV_PATH)
        plot_sweep(df, FIG_PATH)
        print_summary(df)
    else:
        print("No results written — check for errors above.")


if __name__ == "__main__":
    main()
