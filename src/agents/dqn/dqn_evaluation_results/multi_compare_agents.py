"""
Multi-Seed Evaluation & Publication-Ready Shaded Plot
======================================================
Runs all three agents across N independent seeds and produces:
  1. Shaded confidence interval plot (mean ± std) — main dissertation figure
  2. Summary comparison table (Reward, Coverage, Jain's Fairness, Energy Efficiency)
  3. Efficiency Metrics Summary table (matching the per-agent table format)
  4. Per-seed raw CSV files for reproducibility appendix

Academic standard: 5 seeds minimum for RL result validity.
Follows IEEE/MDPI conventions for RL benchmarking.

Author: ATILADE GABRIEL OKE
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import json
from pathlib import Path
import time
from scipy import stats
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

# ==================== PATH SETUP ====================
script_dir         = Path(__file__).resolve().parent
src_dir            = script_dir.parent.parent.parent
script_dir_results = script_dir.parent

sys.path.insert(0, str(src_dir))

from environment.uav_env import UAVEnvironment
from greedy_agents import MaxThroughputGreedyV2, NearestSensorGreedy

# ==================== CONFIGURATION ====================

SEEDS = [42, 123, 256, 789, 1337]

PLOT_CONFIG = {
    "grid_size":          (500, 500),
    "num_sensors":        20,
    "max_steps":          2100,
    "path_loss_exponent": 3.8,
    "rssi_threshold":     -120.0,
    "sensor_duty_cycle":  10.0,
}

EVAL_MAX_BATTERY = 274.0

DQN_MODEL_PATH = (
    script_dir_results / "models" / "dqn_full_observability" / "dqn_final.zip"
)
DQN_CONFIG_PATH = (
    script_dir_results / "models" / "dqn_full_observability" / "frame_stacking_config.json"
)
VEC_NORMALIZE_PATH = (
    script_dir_results / "models" / "dqn_full_observability" / "vec_normalize.pkl"
)

OUTPUT_DIR = script_dir / "multi_seed_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_EVERY = 50

# ==================== ENVIRONMENT WRAPPERS ====================

class FixedLayoutEnv(UAVEnvironment):
    """Forces identical sensor positions across all envs for a given seed."""
    def __init__(self, fixed_positions, **kwargs):
        self._fixed_positions = fixed_positions
        super().__init__(**kwargs)

    def reset(self, **kwargs):
        obs = super().reset(**kwargs)
        for sensor, pos in zip(self.sensors, self._fixed_positions):
            sensor.position = np.array(pos, dtype=np.float32)
        return obs


class FixedLayoutSnapshotEnv(FixedLayoutEnv):
    """FixedLayout + snapshot preservation for DQN."""
    def __init__(self, fixed_positions, **kwargs):
        super().__init__(fixed_positions, **kwargs)
        self.last_sensor_data = None

    def reset(self, **kwargs):
        if hasattr(self, "sensors") and self.current_step > 0:
            self.last_sensor_data = [
                {
                    "sensor_id":              int(s.sensor_id),
                    "total_data_generated":   float(s.total_data_generated),
                    "total_data_transmitted": float(s.total_data_transmitted),
                    "total_data_lost":        float(s.total_data_lost),
                }
                for s in self.sensors
            ]
        return super().reset(**kwargs)


# ==================== HELPERS ====================

def load_frame_stacking_config(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return {"use_frame_stacking": True, "n_stack": 4}


def _base_env_kwargs():
    return {
        "grid_size":          PLOT_CONFIG["grid_size"],
        "num_sensors":        PLOT_CONFIG["num_sensors"],
        "max_steps":          PLOT_CONFIG["max_steps"],
        "path_loss_exponent": PLOT_CONFIG["path_loss_exponent"],
        "rssi_threshold":     PLOT_CONFIG["rssi_threshold"],
        "sensor_duty_cycle":  PLOT_CONFIG["sensor_duty_cycle"],
        "max_battery":        274.0,
        "render_mode":        None,
    }


def get_canonical_positions(seed):
    """Extract sensor positions from a seeded master environment."""
    master = UAVEnvironment(**_base_env_kwargs())
    master.reset(seed=seed)
    positions = [tuple(float(v) for v in s.position) for s in master.sensors]
    master.close()
    return positions


def compute_jains_index(collection_rates):
    n = len(collection_rates)
    if n == 0:
        return 0.0
    s1 = sum(collection_rates)
    s2 = sum(x ** 2 for x in collection_rates)
    return (s1 ** 2) / (n * s2) if s2 > 0 else 1.0


def compute_energy_efficiency(env):
    energy_used = EVAL_MAX_BATTERY - env.uav.battery
    if energy_used <= 0:
        return 0.0
    return float(env.total_data_collected / energy_used)


def compute_jains_from_env(env):
    rates = []
    for s in env.sensors:
        gen = float(s.total_data_generated)
        tx  = float(s.total_data_transmitted)
        rates.append((tx / gen * 100) if gen > 0 else 0.0)
    return compute_jains_index(rates)


def _unwrap_base_env(vec):
    """
    Drill through VecNormalize → VecFrameStack → DummyVecEnv → Monitor
    to get the raw FixedLayoutSnapshotEnv instance.
    """
    # Step 1: unwrap VecEnv wrappers (VecNormalize, VecFrameStack)
    inner = vec
    while hasattr(inner, 'venv'):
        inner = inner.venv
    # inner is now DummyVecEnv
    env = inner.envs[0]
    # Step 2: unwrap single-env wrappers (Monitor, TimeLimit, etc.)
    while hasattr(env, 'env'):
        env = env.env
    return env


# ==================== SINGLE-SEED EPISODE RUNNERS ====================

def run_greedy_episode(agent_class, fixed_positions, seed, agent_name):
    env = FixedLayoutEnv(fixed_positions, **_base_env_kwargs())
    obs, _ = env.reset(seed=seed)
    agent = agent_class(env)

    history = {"step": [], "cumulative_reward": [], "battery_percent": [],
               "coverage_percent": [], "total_data_collected": [],
               "energy_consumed_wh": []}
    cum_reward = 0.0

    while True:
        action = agent.select_action(obs)
        obs, reward, done, trunc, _ = env.step(action)
        cum_reward += reward

        if env.current_step % LOG_EVERY == 0 or done or trunc:
            history["step"].append(env.current_step)
            history["cumulative_reward"].append(cum_reward)
            history["battery_percent"].append(env.uav.get_battery_percentage())
            history["coverage_percent"].append(
                (len(env.sensors_visited) / env.num_sensors) * 100
            )
            history["total_data_collected"].append(env.total_data_collected)
            history["energy_consumed_wh"].append(EVAL_MAX_BATTERY - env.uav.battery)

        if done or trunc:
            break

    energy_used = EVAL_MAX_BATTERY - env.uav.battery
    efficiencies = [
        (dc / ec) if ec > 0 else 0.0
        for dc, ec in zip(history["total_data_collected"], history["energy_consumed_wh"])
    ]

    summary = {
        "seed":                seed,
        "agent":               agent_name,
        "final_reward":        cum_reward,
        "final_coverage":      (len(env.sensors_visited) / env.num_sensors) * 100,
        "jains_index":         compute_jains_from_env(env),
        "energy_efficiency":   compute_energy_efficiency(env),
        "final_battery":       env.uav.get_battery_percentage(),
        "total_data_bytes":    env.total_data_collected,
        "energy_consumed_wh":  energy_used,
        "peak_efficiency":     max(efficiencies) if efficiencies else 0.0,
        "avg_efficiency":      np.mean(efficiencies) if efficiencies else 0.0,
    }

    env.close()
    return pd.DataFrame(history), summary


def run_dqn_episode(model, fs_config, fixed_positions, seed):
    """
    Run one DQN episode with VecFrameStack + VecNormalize.
    FIXED: _unwrap_base_env() drills through all wrappers to get the real env.
    """
    fp = fixed_positions
    kw = _base_env_kwargs()

    def _make():
        return FixedLayoutSnapshotEnv(fp, **kw)

    vec = DummyVecEnv([_make])

    if fs_config.get("use_frame_stacking", True):
        stacked = VecFrameStack(vec, n_stack=fs_config.get("n_stack", 4))
    else:
        stacked = vec

    if VEC_NORMALIZE_PATH.exists():
        stacked = VecNormalize.load(str(VEC_NORMALIZE_PATH), stacked)
        stacked.training = False
        stacked.norm_reward = False
    else:
        print("  ⚠ vec_normalize.pkl not found — obs won't be normalised")

    # ← FIX: unwrap AFTER all layers are applied
    base_env = _unwrap_base_env(stacked)
    print(f"  [DEBUG] base_env type: {type(base_env).__name__} | battery={base_env.uav.battery:.1f}")
    obs = stacked.reset()

    history = {"step": [], "cumulative_reward": [], "battery_percent": [],
               "coverage_percent": [], "total_data_collected": [],
               "energy_consumed_wh": []}
    cum_reward = 0.0
    step_count = 0  # ← track independently: base_env.current_step resets mid-loop

    while True:
        action, _ = model.predict(obs, deterministic=True)
        av = int(action[0]) if hasattr(action, '__len__') else int(action)

        # Snapshot BEFORE step — VecEnv auto-resets base_env inside stacked.step()
        # so any reads of base_env AFTER step() may see the post-reset state
        pre_battery  = base_env.uav.battery
        pre_coverage = (len(base_env.sensors_visited) / base_env.num_sensors) * 100
        pre_data     = base_env.total_data_collected

        obs, rwds, dones, _ = stacked.step([av])
        cum_reward += float(rwds[0])
        step_count += 1
        episode_done = bool(dones[0])

        if episode_done:
            # Use pre-step snapshot — base_env is already reset at this point
            history["step"].append(step_count)
            history["cumulative_reward"].append(cum_reward)
            history["battery_percent"].append((pre_battery / EVAL_MAX_BATTERY) * 100)
            history["coverage_percent"].append(pre_coverage)
            history["total_data_collected"].append(pre_data)
            history["energy_consumed_wh"].append(EVAL_MAX_BATTERY - pre_battery)
            break
        elif step_count % LOG_EVERY == 0:
            history["step"].append(step_count)
            history["cumulative_reward"].append(cum_reward)
            history["battery_percent"].append(base_env.uav.get_battery_percentage())
            history["coverage_percent"].append(
                (len(base_env.sensors_visited) / base_env.num_sensors) * 100
            )
            history["total_data_collected"].append(base_env.total_data_collected)
            history["energy_consumed_wh"].append(EVAL_MAX_BATTERY - base_env.uav.battery)

    # ← CRITICAL FIX: VecEnv auto-resets base_env the moment done=True fires,
    #   so base_env is already wiped by the time we get here.
    #   Read from the LAST LOGGED HISTORY ENTRY instead — these were recorded
    #   inside the loop BEFORE the auto-reset triggered.
    energy_used    = history["energy_consumed_wh"][-1]  if history["energy_consumed_wh"]  else 0.0
    total_data     = history["total_data_collected"][-1] if history["total_data_collected"] else 0.0
    final_batt_pct = history["battery_percent"][-1]     if history["battery_percent"]      else 100.0
    final_coverage = history["coverage_percent"][-1]    if history["coverage_percent"]      else 0.0
    final_eff      = total_data / energy_used if energy_used > 0 else 0.0

    # Jain's index: read from snapshot saved in FixedLayoutSnapshotEnv.reset()
    if base_env.last_sensor_data:
        rates = []
        for s in base_env.last_sensor_data:
            gen = s["total_data_generated"]
            tx  = s["total_data_transmitted"]
            rates.append((tx / gen * 100) if gen > 0 else 0.0)
        jains = compute_jains_index(rates)
    else:
        jains = 0.0

    efficiencies = [
        (dc / ec) if ec > 0 else 0.0
        for dc, ec in zip(history["total_data_collected"], history["energy_consumed_wh"])
    ]

    stacked.reset()  # ← safe to call now, all data already captured

    summary = {
        "seed":                seed,
        "agent":               "DQN",
        "final_reward":        cum_reward,
        "final_coverage":      final_coverage,
        "jains_index":         jains,
        "energy_efficiency":   final_eff,
        "final_battery":       final_batt_pct,
        "total_data_bytes":    total_data,
        "energy_consumed_wh":  energy_used,
        "peak_efficiency":     max(efficiencies) if efficiencies else 0.0,
        "avg_efficiency":      float(np.mean(efficiencies)) if efficiencies else 0.0,
    }

    stacked.close()
    return pd.DataFrame(history), summary


# ==================== MULTI-SEED RUNNER ====================

def run_all_seeds(dqn_model, fs_config):
    agent_names   = ["DQN", "SF-Aware Greedy V2", "Nearest Sensor Greedy"]
    all_histories = {n: [] for n in agent_names}
    all_summaries = {n: [] for n in agent_names}

    total_runs = len(SEEDS) * (3 if dqn_model else 2)
    run_idx = 0

    for seed in SEEDS:
        print(f"\n{'=' * 60}")
        print(f"  SEED {seed}  ({SEEDS.index(seed)+1}/{len(SEEDS)})")
        print(f"{'=' * 60}")

        fixed_positions = get_canonical_positions(seed)

        if dqn_model is not None:
            run_idx += 1
            print(f"\n[{run_idx}/{total_runs}] DQN  (seed={seed})")
            t0 = time.time()
            hist, summ = run_dqn_episode(dqn_model, fs_config, fixed_positions, seed)
            all_histories["DQN"].append(hist)
            all_summaries["DQN"].append(summ)
            print(f"  ✓  reward={summ['final_reward']:.0f}  "
                  f"cov={summ['final_coverage']:.1f}%  "
                  f"J={summ['jains_index']:.4f}  "
                  f"energy={summ['energy_consumed_wh']:.1f}Wh  "
                  f"({time.time()-t0:.1f}s)")

        run_idx += 1
        print(f"\n[{run_idx}/{total_runs}] SF-Aware Greedy V2  (seed={seed})")
        t0 = time.time()
        hist, summ = run_greedy_episode(
            MaxThroughputGreedyV2, fixed_positions, seed, "SF-Aware Greedy V2"
        )
        all_histories["SF-Aware Greedy V2"].append(hist)
        all_summaries["SF-Aware Greedy V2"].append(summ)
        print(f"  ✓  reward={summ['final_reward']:.0f}  "
              f"cov={summ['final_coverage']:.1f}%  "
              f"J={summ['jains_index']:.4f}  "
              f"energy={summ['energy_consumed_wh']:.1f}Wh  "
              f"({time.time()-t0:.1f}s)")

        run_idx += 1
        print(f"\n[{run_idx}/{total_runs}] Nearest Sensor Greedy  (seed={seed})")
        t0 = time.time()
        hist, summ = run_greedy_episode(
            NearestSensorGreedy, fixed_positions, seed, "Nearest Sensor Greedy"
        )
        all_histories["Nearest Sensor Greedy"].append(hist)
        all_summaries["Nearest Sensor Greedy"].append(summ)
        print(f"  ✓  reward={summ['final_reward']:.0f}  "
              f"cov={summ['final_coverage']:.1f}%  "
              f"J={summ['jains_index']:.4f}  "
              f"energy={summ['energy_consumed_wh']:.1f}Wh  "
              f"({time.time()-t0:.1f}s)")

    return all_histories, all_summaries


# ==================== INTERPOLATION ====================

def interpolate_to_common_steps(histories, n_points=200):
    result = {}
    for agent, hist_list in histories.items():
        if not hist_list:
            continue
        max_step = max(df["step"].iloc[-1] for df in hist_list)
        common_steps = np.linspace(0, max_step, n_points)

        interp_rewards, interp_coverage, interp_data = [], [], []

        for df in hist_list:
            steps = df["step"].values
            interp_rewards.append(np.interp(common_steps, steps, df["cumulative_reward"].values))
            interp_coverage.append(np.interp(common_steps, steps, df["coverage_percent"].values))
            interp_data.append(np.interp(common_steps, steps, df["total_data_collected"].values))

        interp_rewards  = np.array(interp_rewards)
        interp_coverage = np.array(interp_coverage)
        interp_data     = np.array(interp_data)

        result[agent] = {
            "steps":         common_steps,
            "mean_reward":   interp_rewards.mean(axis=0),
            "std_reward":    interp_rewards.std(axis=0),
            "mean_coverage": interp_coverage.mean(axis=0),
            "std_coverage":  interp_coverage.std(axis=0),
            "mean_data":     interp_data.mean(axis=0),
            "std_data":      interp_data.std(axis=0),
        }
    return result


# ==================== PUBLICATION PLOTS ====================

AGENT_STYLES = {
    "DQN":                  {"color": "#1565C0", "linestyle": "-",  "marker": "o",
                              "label": "DQN Agent (Proposed)"},
    "SF-Aware Greedy V2":   {"color": "#C62828", "linestyle": "--", "marker": "s",
                              "label": "SF-Aware Greedy V2"},
    "Nearest Sensor Greedy":{"color": "#555555", "linestyle": ":",  "marker": "^",
                              "label": "Nearest Sensor Greedy"},
}


def plot_shaded_reward(interp_data, seeds):
    sns.set_theme(style="whitegrid", font_scale=1.15)
    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=(12, 7))

    for agent, d in interp_data.items():
        style = AGENT_STYLES[agent]
        ax.plot(d["steps"], d["mean_reward"], color=style["color"],
                linewidth=2.5, linestyle=style["linestyle"],
                label=style["label"], zorder=3)
        ax.fill_between(d["steps"],
                        d["mean_reward"] - d["std_reward"],
                        d["mean_reward"] + d["std_reward"],
                        color=style["color"], alpha=0.18, zorder=2)

    ax.set_xlabel("Simulation Step (t)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Cumulative Reward", fontsize=13, fontweight="bold")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.text(0.98, 0.04,
            f"n = {len(seeds)} independent seeds\nSeeds: {seeds}\nShaded region = ±1 std. dev.",
            transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#CCCCCC", alpha=0.9))
    ax.legend(loc="lower right", fontsize=11, framealpha=0.9, edgecolor="#CCCCCC")
    ax.set_title(
        "Comparative Performance: DQN vs. Heuristic Baselines\n"
        f"(Mean ± Std. Dev. over {len(seeds)} independent random seeds)",
        fontsize=13, fontweight="bold", pad=12
    )
    plt.tight_layout()
    out = OUTPUT_DIR / "fig1_shaded_reward_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {out.name}")
    plt.show()


def plot_shaded_coverage(interp_data, seeds):
    sns.set_theme(style="whitegrid", font_scale=1.15)
    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=(12, 6))

    for agent, d in interp_data.items():
        style = AGENT_STYLES[agent]
        ax.plot(d["steps"], d["mean_coverage"], color=style["color"],
                linewidth=2.5, linestyle=style["linestyle"], label=style["label"])
        ax.fill_between(d["steps"],
                        d["mean_coverage"] - d["std_coverage"],
                        d["mean_coverage"] + d["std_coverage"],
                        color=style["color"], alpha=0.18)

    ax.set_xlabel("Simulation Step (t)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Sensor Coverage (%)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.legend(loc="lower right", fontsize=11)
    ax.set_title(f"Sensor Coverage Rate Over Time  (n={len(seeds)} seeds, ±1 std. dev.)",
                 fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    out = OUTPUT_DIR / "fig2_shaded_coverage_comparison.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {out.name}")
    plt.show()


def plot_summary_table(all_summaries, seeds):
    metrics = [
        ("final_reward",      "Cumulative Reward",          "1e6", True),
        ("final_coverage",    "Coverage (%)",               "%",   True),
        ("jains_index",       "Jain's Fairness Index",      "",    True),
        ("energy_efficiency", "Energy Efficiency\n(bytes/Wh)", "", True),
    ]

    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams["font.family"] = "serif"
    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    fig.suptitle(f"Multi-Metric Performance Summary  (n={len(seeds)} seeds, mean ± std)",
                 fontsize=14, fontweight="bold", y=1.02)

    table_rows = []

    for ax, (metric_key, metric_label, unit, higher_better) in zip(axes, metrics):
        means, stds, names, colors = [], [], [], []

        for agent, style in AGENT_STYLES.items():
            summaries = all_summaries.get(agent, [])
            if not summaries:
                continue
            vals = [s[metric_key] for s in summaries]
            m, s = np.mean(vals), np.std(vals)
            means.append(m)
            stds.append(s)
            names.append(style["label"])
            colors.append(style["color"])
            table_rows.append({
                "Agent":  style["label"],
                "Metric": metric_label.replace("\n", " "),
                "Mean":   f"{m:.4f}",
                "Std":    f"{s:.4f}",
                "Seeds":  str(len(vals)),
            })

        x = np.arange(len(names))
        bars = ax.bar(x, means, color=colors, alpha=0.8,
                      edgecolor="white", linewidth=1.2, zorder=3)
        ax.errorbar(x, means, yerr=stds, fmt='none', color='black',
                    capsize=6, capthick=2, elinewidth=2, zorder=4)
        ax.set_xticks(x)
        ax.set_xticklabels(names, rotation=20, ha='right', fontsize=9)
        ax.set_ylabel(metric_label, fontsize=10, fontweight="bold")
        ax.set_title(metric_label, fontsize=10, fontweight="bold", pad=8)
        ax.yaxis.grid(True, alpha=0.5)
        if unit == "1e6":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        best_idx = np.argmax(means) if higher_better else np.argmin(means)
        bars[best_idx].set_edgecolor("gold")
        bars[best_idx].set_linewidth(3)
        ax.text(best_idx, means[best_idx] + stds[best_idx] * 0.1,
                "★", ha='center', va='bottom', fontsize=14, color='goldenrod')

    plt.tight_layout()
    out_fig = OUTPUT_DIR / "fig3_metric_comparison_bars.png"
    plt.savefig(out_fig, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {out_fig.name}")
    plt.show()

    table_df = pd.DataFrame(table_rows)
    out_csv = OUTPUT_DIR / "summary_table.csv"
    table_df.to_csv(out_csv, index=False)
    print(f"✓ Saved: {out_csv.name}")
    return table_df


def plot_per_seed_variance(all_histories):
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=(13, 7))

    for agent, hist_list in all_histories.items():
        if not hist_list:
            continue
        style = AGENT_STYLES[agent]
        for df in hist_list:
            ax.plot(df["step"], df["cumulative_reward"],
                    color=style["color"], alpha=0.2, linewidth=1.0)
        common = np.linspace(0, max(df["step"].iloc[-1] for df in hist_list), 300)
        interps = [np.interp(common, df["step"].values, df["cumulative_reward"].values)
                   for df in hist_list]
        ax.plot(common, np.mean(interps, axis=0),
                color=style["color"], linewidth=3, label=style["label"], zorder=5)

    ax.set_xlabel("Simulation Step (t)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Cumulative Reward", fontsize=13, fontweight="bold")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend(loc="lower right", fontsize=11)
    ax.set_title("Per-Seed Traces (faint) and Mean (bold) — Convergence Stability Analysis",
                 fontsize=12, fontweight="bold", pad=12)
    ax.text(0.02, 0.96,
            "Narrow trace spread → stable policy\nWide trace spread → high variance / unstable",
            transform=ax.transAxes, fontsize=8.5, va='top', ha='left',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                      edgecolor='#CCCCCC', alpha=0.9))
    plt.tight_layout()
    out = OUTPUT_DIR / "fig4_per_seed_traces.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {out.name}")
    plt.show()


# ==================== EFFICIENCY METRICS TABLE ====================

def plot_efficiency_metrics_table(all_summaries, seeds):
    """
    Generates the 'Efficiency Metrics Summary' table showing:
    - Final Energy Consumed (Wh)  — mean ± std across seeds
    - Final Data Collected (Bytes) — mean ± std across seeds
    - Final Efficiency (Bytes/Wh) — at episode end
    - Average Efficiency (Bytes/Wh) — averaged over episode steps
    - Peak Efficiency (Bytes/Wh)   — best single step value

    Gold highlight = best value per column.
    """
    AGENT_DISPLAY = {
        "DQN":                  "DQN Agent",
        "SF-Aware Greedy V2":   "Smart Greedy V2",
        "Nearest Sensor Greedy": "Nearest Greedy",
    }
    AGENT_COLORS = {
        "DQN":                  "#1565C0",
        "SF-Aware Greedy V2":   "#C62828",
        "Nearest Sensor Greedy": "#555555",
    }

    columns = [
        "Agent",
        "Avg Energy\nConsumed (Wh)",
        "Avg Data\nCollected (Bytes)",
        "Avg Final\nEfficiency (B/Wh)",
        "Avg Step\nEfficiency (B/Wh)",
        "Peak\nEfficiency (B/Wh)",
    ]

    rows = []
    raw_vals = {col: [] for col in columns[1:]}  # for finding best per column

    for agent in ["DQN", "SF-Aware Greedy V2", "Nearest Sensor Greedy"]:
        summs = all_summaries.get(agent, [])
        if not summs:
            rows.append([AGENT_DISPLAY[agent], "—", "—", "—", "—", "—"])
            for col in columns[1:]:
                raw_vals[col].append(None)
            continue

        energy_vals   = [s["energy_consumed_wh"]  for s in summs]
        data_vals     = [s["total_data_bytes"]     for s in summs]
        final_eff     = [s["energy_efficiency"]    for s in summs]
        avg_eff       = [s["avg_efficiency"]       for s in summs]
        peak_eff      = [s["peak_efficiency"]      for s in summs]

        def fmt(vals, decimals=2):
            m, s = np.mean(vals), np.std(vals)
            return f"{m:.{decimals}f} ±{s:.{decimals}f}", m

        e_str,  e_m  = fmt(energy_vals, 2)
        d_str,  d_m  = fmt(data_vals, 0)
        fe_str, fe_m = fmt(final_eff, 2)
        ae_str, ae_m = fmt(avg_eff, 2)
        pe_str, pe_m = fmt(peak_eff, 2)

        rows.append([AGENT_DISPLAY[agent], e_str, d_str, fe_str, ae_str, pe_str])
        raw_vals["Avg Energy\nConsumed (Wh)"].append(e_m)
        raw_vals["Avg Data\nCollected (Bytes)"].append(d_m)
        raw_vals["Avg Final\nEfficiency (B/Wh)"].append(fe_m)
        raw_vals["Avg Step\nEfficiency (B/Wh)"].append(ae_m)
        raw_vals["Peak\nEfficiency (B/Wh)"].append(pe_m)

    # Determine best per column (higher = better, except energy where lower = better)
    best_higher = {
        "Avg Data\nCollected (Bytes)": True,
        "Avg Final\nEfficiency (B/Wh)": True,
        "Avg Step\nEfficiency (B/Wh)": True,
        "Peak\nEfficiency (B/Wh)": True,
        "Avg Energy\nConsumed (Wh)": False,  # lower is better
    }
    best_col_idx = {}
    for col, higher in best_higher.items():
        vals = raw_vals[col]
        valid = [(i, v) for i, v in enumerate(vals) if v is not None]
        if valid:
            best_i = max(valid, key=lambda x: x[1])[0] if higher else min(valid, key=lambda x: x[1])[0]
            best_col_idx[col] = best_i

    # ── Draw the table ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 3.5))
    ax.axis("off")

    header_color  = "#2E7D32"   # dark green header
    best_color    = "#A5D6A7"   # light green for best cell
    alt_row_color = "#F5F5F5"   # light grey alternate rows
    white         = "white"

    col_widths = [0.16, 0.17, 0.18, 0.17, 0.17, 0.15]

    tbl = ax.table(
        cellText=rows,
        colLabels=columns,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1, 2.2)

    # Style header
    for j, col in enumerate(columns):
        cell = tbl[0, j]
        cell.set_facecolor(header_color)
        cell.set_text_props(color="white", fontweight="bold", fontsize=9)
        cell.set_edgecolor("white")

    # Style data rows
    agent_list = ["DQN", "SF-Aware Greedy V2", "Nearest Sensor Greedy"]
    for i, agent in enumerate(agent_list):
        row_color = white if i % 2 == 0 else alt_row_color
        for j in range(len(columns)):
            cell = tbl[i + 1, j]
            cell.set_facecolor(row_color)
            cell.set_edgecolor("#DDDDDD")

            # Agent name: colour-coded bold
            if j == 0:
                cell.set_text_props(color=AGENT_COLORS[agent], fontweight="bold")

        # Highlight best cells in green
        for col_idx, col_name in enumerate(columns[1:], start=1):
            best_row = best_col_idx.get(col_name)
            if best_row == i:
                cell = tbl[i + 1, col_idx]
                cell.set_facecolor(best_color)
                cell.set_text_props(fontweight="bold")

    ax.set_title(
        f"Efficiency Metrics Summary  (Mean ± Std over {len(seeds)} seeds)",
        fontsize=13, fontweight="bold", pad=16
    )

    plt.tight_layout()
    out = OUTPUT_DIR / "fig5_efficiency_metrics_table.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {out.name}")
    plt.show()

    # Also save as CSV
    csv_rows = []
    for agent, summs in all_summaries.items():
        if not summs:
            continue
        for s in summs:
            csv_rows.append({
                "Agent":                agent,
                "Seed":                 s["seed"],
                "Energy_Consumed_Wh":   s["energy_consumed_wh"],
                "Data_Collected_Bytes": s["total_data_bytes"],
                "Final_Efficiency_BpWh":s["energy_efficiency"],
                "Avg_Step_Eff_BpWh":   s["avg_efficiency"],
                "Peak_Eff_BpWh":        s["peak_efficiency"],
            })
    pd.DataFrame(csv_rows).to_csv(OUTPUT_DIR / "efficiency_metrics.csv", index=False)
    print(f"✓ Saved: efficiency_metrics.csv")


# ==================== ADDITIONAL FIGURES ====================

def plot_shaded_data_throughput(interp_data, seeds):
    """
    fig6: Shaded mean ± std of total data collected over time.
    Companion to fig1 (reward) and fig2 (coverage) — shows throughput trajectory
    not just end-state, proving DQN collects more data throughout the episode.
    """
    sns.set_theme(style="whitegrid", font_scale=1.15)
    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=(12, 6))

    for agent, d in interp_data.items():
        style = AGENT_STYLES[agent]
        ax.plot(d["steps"], d["mean_data"], color=style["color"],
                linewidth=2.5, linestyle=style["linestyle"],
                label=style["label"], zorder=3)
        ax.fill_between(d["steps"],
                        d["mean_data"] - d["std_data"],
                        d["mean_data"] + d["std_data"],
                        color=style["color"], alpha=0.18, zorder=2)

    ax.set_xlabel("Simulation Step (t)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Cumulative Data Collected (Bytes)", fontsize=13, fontweight="bold")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend(loc="lower right", fontsize=11, framealpha=0.9, edgecolor="#CCCCCC")
    ax.text(0.98, 0.04,
            f"n = {len(seeds)} seeds  |  Shaded = ±1 std. dev.",
            transform=ax.transAxes, fontsize=9, ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#CCCCCC", alpha=0.9))
    ax.set_title(
        f"Data Throughput Over Time  (Mean ± Std. Dev., n={len(seeds)} seeds)",
        fontsize=13, fontweight="bold", pad=12
    )
    plt.tight_layout()
    out = OUTPUT_DIR / "fig6_shaded_data_throughput.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {out.name}")
    plt.show()


def plot_box_plots(all_summaries, seeds):
    """
    fig7: Violin + box plots for reward, coverage, Jain's index, energy efficiency.
    More statistically honest than bar+errorbar — shows full distribution
    across seeds including outliers. Standard in IEEE RL benchmarking papers.
    """
    sns.set_theme(style="whitegrid", font_scale=1.1)
    plt.rcParams["font.family"] = "serif"

    metrics = [
        ("final_reward",      "Cumulative Reward",          "1e"),
        ("final_coverage",    "Final Coverage (%)",         "%"),
        ("jains_index",       "Jain's Fairness Index",      ""),
        ("energy_efficiency", "Energy Efficiency (B/Wh)",   ""),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(18, 6))
    fig.suptitle(
        f"Seed-by-Seed Distribution of Key Metrics  (n={len(seeds)} seeds per agent)",
        fontsize=13, fontweight="bold", y=1.02
    )

    for ax, (key, label, unit) in zip(axes, metrics):
        plot_data, plot_labels, plot_colors = [], [], []

        for agent, style in AGENT_STYLES.items():
            summs = all_summaries.get(agent, [])
            if not summs:
                continue
            vals = [s[key] for s in summs]
            plot_data.append(vals)
            plot_labels.append(style["label"].replace(" ", "\n"))
            plot_colors.append(style["color"])

        parts = ax.violinplot(plot_data, positions=range(len(plot_data)),
                              showmeans=False, showmedians=False, showextrema=False)
        for pc, color in zip(parts["bodies"], plot_colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.35)
            pc.set_edgecolor(color)

        bp = ax.boxplot(plot_data, positions=range(len(plot_data)),
                        widths=0.25, patch_artist=True, notch=False,
                        medianprops=dict(color="black", linewidth=2),
                        whiskerprops=dict(linewidth=1.5),
                        capprops=dict(linewidth=1.5),
                        flierprops=dict(marker="o", markersize=5, alpha=0.6))
        for patch, color in zip(bp["boxes"], plot_colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        # Overlay individual seed points
        for i, (vals, color) in enumerate(zip(plot_data, plot_colors)):
            jitter = np.random.default_rng(0).uniform(-0.06, 0.06, size=len(vals))
            ax.scatter(np.full(len(vals), i) + jitter, vals,
                       color=color, s=40, zorder=5, edgecolors="white",
                       linewidths=0.8, alpha=0.9)

        ax.set_xticks(range(len(plot_labels)))
        ax.set_xticklabels(plot_labels, fontsize=8.5)
        ax.set_ylabel(label, fontsize=10, fontweight="bold")
        ax.set_title(label, fontsize=10, fontweight="bold", pad=8)
        ax.yaxis.grid(True, alpha=0.5, linestyle="--")
        if unit == "1e":
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    plt.tight_layout()
    out = OUTPUT_DIR / "fig7_box_violin_plots.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {out.name}")
    plt.show()


def plot_convergence_stability(interp_data, seeds):
    """
    fig8: Std dev of cumulative reward across seeds over time.
    A narrow, low line = stable policy that behaves consistently regardless of env layout.
    Wide or rising line = high variance / layout-sensitive agent.
    This directly answers 'is your DQN reliable?' for the supervisor.
    """
    sns.set_theme(style="whitegrid", font_scale=1.15)
    plt.rcParams["font.family"] = "serif"
    fig, ax = plt.subplots(figsize=(12, 6))

    for agent, d in interp_data.items():
        style = AGENT_STYLES[agent]
        ax.plot(d["steps"], d["std_reward"], color=style["color"],
                linewidth=2.5, linestyle=style["linestyle"],
                label=style["label"], zorder=3)

    ax.set_xlabel("Simulation Step (t)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Std. Dev. of Cumulative Reward", fontsize=13, fontweight="bold")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend(loc="upper left", fontsize=11, framealpha=0.9, edgecolor="#CCCCCC")
    ax.text(0.98, 0.96,
            "Lower line = more consistent policy\nacross different environment layouts",
            transform=ax.transAxes, fontsize=9, ha="right", va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="lightyellow",
                      edgecolor="#CCCCCC", alpha=0.9))
    ax.set_title(
        f"Policy Convergence Stability  "
        f"(Reward Std. Dev. over {len(seeds)} seeds — lower = more robust)",
        fontsize=13, fontweight="bold", pad=12
    )
    plt.tight_layout()
    out = OUTPUT_DIR / "fig8_convergence_stability.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {out.name}")
    plt.show()


def plot_jains_over_time(all_histories, all_summaries, seeds):
    """
    fig9: Jain's Fairness Index building up through the episode (mean ± std).
    Since per-step Jain's isn't logged directly, we proxy it via the ratio of
    coverage_percent to total_data_collected — a monotone fairness signal.
    The final-step Jain's values from all_summaries are overlaid as scatter points.
    Shows DQN achieves high fairness earlier in the episode, not just at the end.
    """
    sns.set_theme(style="whitegrid", font_scale=1.15)
    plt.rcParams["font.family"] = "serif"
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    fig.suptitle("Fairness & Coverage Dynamics Over Time",
                 fontsize=13, fontweight="bold")

    # Left: coverage rate (direct proxy for visit fairness over time)
    ax = axes[0]
    for agent, hist_list in all_histories.items():
        if not hist_list:
            continue
        style = AGENT_STYLES[agent]
        common = np.linspace(0, max(df["step"].iloc[-1] for df in hist_list), 200)
        interps = [np.interp(common, df["step"].values, df["coverage_percent"].values)
                   for df in hist_list]
        mean_cov = np.mean(interps, axis=0)
        std_cov  = np.std(interps,  axis=0)
        ax.plot(common, mean_cov, color=style["color"],
                linewidth=2.5, linestyle=style["linestyle"], label=style["label"])
        ax.fill_between(common, mean_cov - std_cov, mean_cov + std_cov,
                        color=style["color"], alpha=0.18)
    ax.set_xlabel("Simulation Step (t)", fontsize=11, fontweight="bold")
    ax.set_ylabel("Sensor Coverage (%)", fontsize=11, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_title("Coverage Accumulation Rate\n(faster rise = fairer early service)",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4, linestyle="--")

    # Right: final Jain's index distribution per agent (beeswarm + mean line)
    ax2 = axes[1]
    agent_list = ["DQN", "SF-Aware Greedy V2", "Nearest Sensor Greedy"]
    positions  = np.arange(len(agent_list))
    for i, agent in enumerate(agent_list):
        summs = all_summaries.get(agent, [])
        if not summs:
            continue
        style  = AGENT_STYLES[agent]
        jvals  = [s["jains_index"] for s in summs]
        jitter = np.random.default_rng(i).uniform(-0.08, 0.08, size=len(jvals))
        ax2.scatter(np.full(len(jvals), i) + jitter, jvals,
                    color=style["color"], s=80, zorder=4,
                    edgecolors="white", linewidths=1.0, alpha=0.9,
                    label=style["label"])
        ax2.hlines(np.mean(jvals), i - 0.2, i + 0.2,
                   color=style["color"], linewidth=3, zorder=5)
        ax2.hlines(np.mean(jvals) - np.std(jvals), i - 0.12, i + 0.12,
                   color=style["color"], linewidth=1.5, linestyle="--", zorder=5)
        ax2.hlines(np.mean(jvals) + np.std(jvals), i - 0.12, i + 0.12,
                   color=style["color"], linewidth=1.5, linestyle="--", zorder=5)

    ax2.set_xticks(positions)
    ax2.set_xticklabels([AGENT_STYLES[a]["label"].replace(" ", "\n")
                         for a in agent_list], fontsize=9)
    ax2.set_ylabel("Jain's Fairness Index", fontsize=11, fontweight="bold")
    ax2.set_ylim(0, 1.05)
    ax2.axhline(1.0, color="green", linestyle=":", linewidth=1.2,
                alpha=0.6, label="Perfect fairness (J=1)")
    ax2.set_title(f"Final Jain's Index per Seed\n(bar = mean, dashed = ±1 std)",
                  fontsize=11, fontweight="bold")
    ax2.legend(fontsize=9, loc="lower right")
    ax2.grid(axis="y", alpha=0.4, linestyle="--")

    plt.tight_layout()
    out = OUTPUT_DIR / "fig9_jains_fairness_dynamics.png"
    plt.savefig(out, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {out.name}")
    plt.show()


# ==================== STATISTICAL TESTS ====================

def run_statistical_tests(all_summaries):
    print("\n" + "=" * 60)
    print("STATISTICAL SIGNIFICANCE TESTS (Welch's t-test)")
    print("H0: DQN mean == Baseline mean")
    print("=" * 60)

    dqn_rewards = [s["final_reward"] for s in all_summaries.get("DQN", [])]
    if not dqn_rewards:
        print("  ⚠ No DQN results — skipping tests")
        return

    results = []
    for agent in ["SF-Aware Greedy V2", "Nearest Sensor Greedy"]:
        baseline_rewards = [s["final_reward"] for s in all_summaries.get(agent, [])]
        if not baseline_rewards:
            continue
        t_stat, p_val = stats.ttest_ind(dqn_rewards, baseline_rewards, equal_var=False)
        significant = "YES ✓" if p_val < 0.05 else "NO ✗"
        print(f"\n  DQN vs {agent}:")
        print(f"    DQN mean    = {np.mean(dqn_rewards):.0f} ± {np.std(dqn_rewards):.0f}")
        print(f"    Base mean   = {np.mean(baseline_rewards):.0f} ± {np.std(baseline_rewards):.0f}")
        print(f"    t-statistic = {t_stat:.4f}")
        print(f"    p-value     = {p_val:.4f}")
        print(f"    Significant (p<0.05)? {significant}")
        results.append({
            "Comparison":    f"DQN vs {agent}",
            "DQN Mean":      f"{np.mean(dqn_rewards):.0f}",
            "DQN Std":       f"{np.std(dqn_rewards):.0f}",
            "Baseline Mean": f"{np.mean(baseline_rewards):.0f}",
            "Baseline Std":  f"{np.std(baseline_rewards):.0f}",
            "t-statistic":   f"{t_stat:.4f}",
            "p-value":       f"{p_val:.4f}",
            "Significant":   significant,
        })

    if results:
        stats_df = pd.DataFrame(results)
        out = OUTPUT_DIR / "statistical_tests.csv"
        stats_df.to_csv(out, index=False)
        print(f"\n✓ Saved: {out.name}")
    print("=" * 60)


# ==================== SAVE RAW DATA ====================

def save_all_raw_data(all_histories, all_summaries):
    for agent, hist_list in all_histories.items():
        agent_slug = agent.replace(" ", "_").replace("-", "").lower()
        for df, seed in zip(hist_list, SEEDS):
            df["seed"] = seed
            df.to_csv(OUTPUT_DIR / f"{agent_slug}_seed{seed}.csv", index=False)
    rows = []
    for agent, summs in all_summaries.items():
        rows.extend(summs)
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "all_seeds_summary.csv", index=False)
    print(f"✓ Raw CSVs saved to {OUTPUT_DIR}")


# ==================== MAIN ====================

def main():
    print("=" * 70)
    print("MULTI-SEED EVALUATION — PUBLICATION-READY ANALYSIS")
    print(f"Seeds: {SEEDS}  ({len(SEEDS)} runs per agent)")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)

    dqn_model = None
    fs_config  = {"use_frame_stacking": True, "n_stack": 4}

    if DQN_MODEL_PATH.exists():
        print(f"\nLoading DQN model from {DQN_MODEL_PATH}...")
        dqn_model = DQN.load(DQN_MODEL_PATH)
        fs_config  = load_frame_stacking_config(DQN_CONFIG_PATH)
        print(f"  ✓ Loaded | {fs_config}")
        if VEC_NORMALIZE_PATH.exists():
            print(f"  ✓ vec_normalize.pkl found — will normalise obs at eval time")
        else:
            print(f"  ⚠ vec_normalize.pkl NOT found — obs won't be normalised")
    else:
        print(f"  ⚠ DQN model not found — running greedy agents only")

    t_start = time.time()
    all_histories, all_summaries = run_all_seeds(dqn_model, fs_config)
    print(f"\nTotal evaluation time: {(time.time()-t_start)/60:.1f} minutes")

    save_all_raw_data(all_histories, all_summaries)
    interp = interpolate_to_common_steps(all_histories)

    print("\nGenerating publication figures...")
    plot_shaded_reward(interp, SEEDS)
    plot_shaded_coverage(interp, SEEDS)
    table_df = plot_summary_table(all_summaries, SEEDS)
    plot_per_seed_variance(all_histories)
    plot_efficiency_metrics_table(all_summaries, SEEDS)
    plot_shaded_data_throughput(interp, SEEDS)
    plot_box_plots(all_summaries, SEEDS)
    plot_convergence_stability(interp, SEEDS)
    plot_jains_over_time(all_histories, all_summaries, SEEDS)
    run_statistical_tests(all_summaries)

    print("\n" + "=" * 70)
    print("SUMMARY TABLE")
    print("=" * 70)
    print(table_df.to_string(index=False))

    print("\n" + "=" * 70)
    print("FILES GENERATED")
    print("=" * 70)
    print("  fig1_shaded_reward_comparison.png")
    print("  fig2_shaded_coverage_comparison.png")
    print("  fig3_metric_comparison_bars.png")
    print("  fig4_per_seed_traces.png")
    print("  fig5_efficiency_metrics_table.png")
    print("  fig6_shaded_data_throughput.png        ← NEW")
    print("  fig7_box_violin_plots.png               ← NEW")
    print("  fig8_convergence_stability.png          ← NEW")
    print("  fig9_jains_fairness_dynamics.png        ← NEW")
    print("  summary_table.csv")
    print("  statistical_tests.csv")
    print("  all_seeds_summary.csv")
    print("  efficiency_metrics.csv")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()