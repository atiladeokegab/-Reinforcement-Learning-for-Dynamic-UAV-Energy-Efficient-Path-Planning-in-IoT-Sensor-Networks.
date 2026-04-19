"""
Multi-Model Evaluation: DQN v5-v8 vs Greedy Agents
====================================================
Sweeps 3 grid sizes x 3 sensor counts x 8 start positions x 5 seeds = 360 episodes
per agent (6 agents -> 2,160 total episodes).

GPU inference + 4 parallel workers for ~2-hour runtime.

Usage:
    PYTHONIOENCODING=utf-8 uv run python multi_model_eval.py
    PYTHONIOENCODING=utf-8 uv run python multi_model_eval.py --dry-run   # 1 seed, ~432 ep
    PYTHONIOENCODING=utf-8 uv run python multi_model_eval.py --resume    # skip done rows
    PYTHONIOENCODING=utf-8 uv run python multi_model_eval.py --no-plots  # episodes only
    PYTHONIOENCODING=utf-8 uv run python multi_model_eval.py --cpu       # force CPU
    PYTHONIOENCODING=utf-8 uv run python multi_model_eval.py --workers 2

Author: ATILADE GABRIEL OKE
"""

import sys
import os
import json
import random
import time
import argparse
import traceback
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import gymnasium

# ─── Canonical paths (accessible in both main and worker processes) ───────────
SCRIPT_DIR = Path(__file__).resolve().parent           # dqn_evaluation_results/
SRC_DIR    = SCRIPT_DIR.parent.parent.parent            # src/
MODEL_BASE = SCRIPT_DIR.parent / "models"               # src/agents/dqn/models/
OUTPUT_DIR = SCRIPT_DIR / "multi_model_results"

# ─── Evaluation constants ─────────────────────────────────────────────────────
MODEL_REGISTRY = [
    {"name": "DQN-v5", "model_dir": "dqn_v5_gnn"},
    {"name": "DQN-v6", "model_dir": "dqn_v6_gnn_pos"},
    {"name": "DQN-v7", "model_dir": "dqn_v7_tuned"},
    {"name": "DQN-v8", "model_dir": "dqn_v8_tuned_nopos"},
]
GREEDY_NAMES    = ["SmartGreedyV2", "NearestGreedy"]
ALL_AGENTS      = [m["name"] for m in MODEL_REGISTRY] + GREEDY_NAMES

GRID_SIZES      = [(200, 200), (400, 400), (600, 600)]
SENSOR_COUNTS   = [10, 20, 30]
EVAL_SEEDS      = [42, 123, 456, 789, 1337]
RANDOM_POS_SEED = 999
MAX_BATTERY     = 274.0
MAX_STEPS       = 2100
N_WORKERS       = 4
STARVATION_THR  = 20.0   # % collection rate below which a sensor is "starved"

AGENT_COLORS = {
    "DQN-v5":        "#1b9e77",
    "DQN-v6":        "#d95f02",
    "DQN-v7":        "#7570b3",
    "DQN-v8":        "#e7298a",
    "SmartGreedyV2": "#66a61e",
    "NearestGreedy": "#e6ab02",
}
AGENT_MARKERS = {
    "DQN-v5": "o", "DQN-v6": "s", "DQN-v7": "^", "DQN-v8": "D",
    "SmartGreedyV2": "v", "NearestGreedy": "P",
}


# ─── Worker-side path bootstrap ───────────────────────────────────────────────
def _setup_paths():
    """Called at the top of every worker function to fix sys.path after spawn."""
    for p in [str(SRC_DIR), str(SCRIPT_DIR.parent), str(SCRIPT_DIR)]:
        if p not in sys.path:
            sys.path.insert(0, p)
    for candidate in SRC_DIR.rglob("gnn_extractor.py"):
        gnn_dir = str(candidate.parent)
        if gnn_dir not in sys.path:
            sys.path.insert(0, gnn_dir)
        break


# ─── Config loader ────────────────────────────────────────────────────────────
def _load_config(config_path: Path) -> dict:
    defaults = {
        "use_frame_stacking":      True,
        "n_stack":                 10,
        "max_sensors_limit":       50,
        "features_per_sensor":     3,
        "include_sensor_positions": False,
    }
    try:
        with open(config_path) as f:
            return {**defaults, **json.load(f)}
    except FileNotFoundError:
        return defaults


# ─── Start-position table (fixed per grid size) ───────────────────────────────
def get_start_positions(grid_size: tuple) -> dict:
    G   = grid_size[0]
    rng = np.random.RandomState(RANDOM_POS_SEED)
    rp  = [(int(rng.randint(20, G - 20)), int(rng.randint(20, G - 20))) for _ in range(3)]
    return {
        "center":    (G // 2, G // 2),
        "corner_00": (0,      0),
        "corner_0G": (0,      G),
        "corner_G0": (G,      0),
        "corner_GG": (G,      G),
        "random_0":  rp[0],
        "random_1":  rp[1],
        "random_2":  rp[2],
    }


# ─── Zero-padded env wrapper ──────────────────────────────────────────────────
class PaddedUAVEnv:
    """
    Thin wrapper factory. Returns a UAVEnvironment subclass that zero-pads the
    observation vector to max_sensors_limit so DQN models with fixed input size
    can evaluate on any N <= max_sensors_limit without retraining.
    """
    @staticmethod
    def make(max_sensors_limit: int = 50):
        from environment.uav_env import UAVEnvironment

        class _Padded(UAVEnvironment):
            _MSL = max_sensors_limit

            def __init__(self, **kwargs):
                super().__init__(**kwargs)
                fps        = self._features_per_sensor
                pad_n      = (self._MSL - self.num_sensors) * fps
                padded_dim = self.observation_space.shape[0] + pad_n
                self._pad_n = pad_n
                self.observation_space = gymnasium.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(padded_dim,), dtype=np.float32
                )

            def _pad(self, obs: np.ndarray) -> np.ndarray:
                if self._pad_n == 0:
                    return obs
                return np.concatenate([obs, np.zeros(self._pad_n, dtype=np.float32)])

            def reset(self, **kwargs):
                obs, info = super().reset(**kwargs)
                return self._pad(obs), info

            def step(self, action):
                obs, r, term, trunc, info = super().step(action)
                return self._pad(obs), r, term, trunc, info

        return _Padded


# ─── Sensor metrics helper ────────────────────────────────────────────────────
def _compute_sensor_metrics(sensor_dicts: list) -> dict:
    """sensor_dicts: list of {total_data_generated, total_data_transmitted}."""
    rates = []
    for s in sensor_dicts:
        gen = float(s.get("total_data_generated", 0))
        tx  = float(s.get("total_data_transmitted", 0))
        rates.append((tx / gen * 100) if gen > 0 else 0.0)
    n  = len(rates)
    s2 = sum(r ** 2 for r in rates)
    jain = (sum(rates) ** 2 / (n * s2)) if n > 0 and s2 > 0 else 0.0
    return {
        "jains_index":          round(jain, 4),
        "min_collection_rate":  round(min(rates) if rates else 0.0, 2),
        "mean_collection_rate": round(sum(rates) / n if n else 0.0, 2),
        "starved_sensors":      sum(1 for r in rates if r < STARVATION_THR),
    }


# ─── Single DQN episode ───────────────────────────────────────────────────────
def _run_dqn_episode(model, config, model_name,
                     grid_size, n_sensors, sp_name, start_pos, seed) -> dict:
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

    np.random.seed(seed)
    random.seed(seed)

    PaddedEnv = PaddedUAVEnv.make(config["max_sensors_limit"])
    env_kwargs = dict(
        grid_size=grid_size,
        num_sensors=n_sensors,
        max_steps=MAX_STEPS,
        path_loss_exponent=3.8,
        rssi_threshold=-85.0,
        sensor_duty_cycle=10.0,
        max_battery=MAX_BATTERY,
        uav_start_position=start_pos,
        include_sensor_positions=config.get("include_sensor_positions", False),
        render_mode=None,
    )

    vec_env = DummyVecEnv([lambda: PaddedEnv(**env_kwargs)])
    vec_env = VecFrameStack(vec_env, n_stack=config.get("n_stack", 10))

    # Seed base env directly
    base_env = vec_env.venv.envs[0]
    try:
        base_env.np_random, _ = gymnasium.utils.seeding.np_random(seed)
    except Exception:
        base_env.np_random = np.random.RandomState(seed)

    obs = vec_env.reset()
    trajectory         = []
    pre_sensor_snap    = None
    cumulative_reward  = 0.0
    step_count         = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        action    = int(action[0]) if isinstance(action, np.ndarray) else int(action)

        # Snapshot BEFORE step (VecEnv auto-resets on done, overwriting base_env state)
        pre_battery  = float(base_env.uav.battery)
        pre_coverage = (len(base_env.sensors_visited) / n_sensors) * 100
        pre_data     = float(base_env.total_data_collected)
        pre_n_vis    = len(base_env.sensors_visited)
        pre_pos      = tuple(float(x) for x in base_env.uav.position)
        pre_sensor_snap = [
            {"total_data_generated":   float(s.total_data_generated),
             "total_data_transmitted": float(s.total_data_transmitted)}
            for s in base_env.sensors
        ]

        obs, rewards, dones, _ = vec_env.step([action])
        cumulative_reward += float(rewards[0])
        step_count        += 1
        trajectory.append(pre_pos)

        if bool(dones[0]):
            energy_consumed = MAX_BATTERY - pre_battery
            break

    vec_env.close()

    sensor_metrics = _compute_sensor_metrics(pre_sensor_snap) if pre_sensor_snap else {}

    return {
        "model":           model_name,
        "grid_w":          grid_size[0],
        "grid_h":          grid_size[1],
        "n_sensors":       n_sensors,
        "start_pos_name":  sp_name,
        "start_x":         start_pos[0],
        "start_y":         start_pos[1],
        "seed":            seed,
        "total_reward":    round(cumulative_reward, 2),
        "data_collected":  round(pre_data, 0),
        "ndr_pct":         round(pre_coverage, 2),
        "battery_pct":     round((pre_battery / MAX_BATTERY) * 100, 2),
        "energy_consumed": round(energy_consumed, 4),
        "efficiency":      round(pre_data / energy_consumed if energy_consumed > 0 else 0.0, 4),
        "steps":           step_count,
        **sensor_metrics,
        "_trajectory":     trajectory,
    }


# ─── Single greedy episode ────────────────────────────────────────────────────
def _run_greedy_episode(agent_name,
                        grid_size, n_sensors, sp_name, start_pos, seed) -> dict:
    from environment.uav_env import UAVEnvironment
    from greedy_agents import MaxThroughputGreedyV2, NearestSensorGreedy

    np.random.seed(seed)
    random.seed(seed)

    env = UAVEnvironment(
        grid_size=grid_size,
        num_sensors=n_sensors,
        max_steps=MAX_STEPS,
        path_loss_exponent=3.8,
        rssi_threshold=-85.0,
        sensor_duty_cycle=10.0,
        max_battery=MAX_BATTERY,
        uav_start_position=start_pos,
        render_mode=None,
    )

    AgentCls = MaxThroughputGreedyV2 if agent_name == "SmartGreedyV2" else NearestSensorGreedy
    agent    = AgentCls(env)

    obs, _ = env.reset(seed=seed)
    trajectory        = []
    cumulative_reward = 0.0
    done              = False

    while not done:
        trajectory.append(tuple(float(x) for x in env.uav.position))
        action = agent.select_action(obs)
        obs, reward, done, truncated, _ = env.step(action)
        cumulative_reward += reward
        if truncated:
            break

    energy_consumed = MAX_BATTERY - env.uav.battery
    sensor_dicts    = [
        {"total_data_generated":   float(s.total_data_generated),
         "total_data_transmitted": float(s.total_data_transmitted)}
        for s in env.sensors
    ]
    ndr = (len(env.sensors_visited) / n_sensors) * 100

    result = {
        "model":           agent_name,
        "grid_w":          grid_size[0],
        "grid_h":          grid_size[1],
        "n_sensors":       n_sensors,
        "start_pos_name":  sp_name,
        "start_x":         start_pos[0],
        "start_y":         start_pos[1],
        "seed":            seed,
        "total_reward":    round(cumulative_reward, 2),
        "data_collected":  round(float(env.total_data_collected), 0),
        "ndr_pct":         round(ndr, 2),
        "battery_pct":     round(float(env.uav.get_battery_percentage()), 2),
        "energy_consumed": round(float(energy_consumed), 4),
        "efficiency":      round(
            float(env.total_data_collected) / float(energy_consumed)
            if energy_consumed > 0 else 0.0, 4),
        "steps":           env.current_step,
        **_compute_sensor_metrics(sensor_dicts),
        "_trajectory":     trajectory,
    }
    env.close()
    return result


# ─── Worker functions (spawned by ProcessPoolExecutor) ────────────────────────
def dqn_worker(args):
    model_info, episodes, device_str, traj_dir_str = args
    _setup_paths()
    from stable_baselines3 import DQN as SB3_DQN

    model_path = MODEL_BASE / model_info["model_dir"] / "dqn_final.zip"
    config     = _load_config(MODEL_BASE / model_info["model_dir"] / "training_config.json")
    traj_dir   = Path(traj_dir_str)

    try:
        model = SB3_DQN.load(str(model_path), device=device_str)
    except Exception:
        model = SB3_DQN.load(str(model_path), device="cpu")

    results = []
    for grid_size, n_sensors, sp_name, start_pos, seed in episodes:
        tag = f"{model_info['name']} {grid_size[0]}x{grid_size[1]} n={n_sensors} {sp_name} s={seed}"
        try:
            r     = _run_dqn_episode(model, config, model_info["name"],
                                     grid_size, n_sensors, sp_name, start_pos, seed)
            traj  = np.array(r.pop("_trajectory"), dtype=np.float32)
            fname = (f"{model_info['name']}_{grid_size[0]}x{grid_size[1]}"
                     f"_n{n_sensors}_{sp_name}_{seed}.npy")
            np.save(traj_dir / fname, traj)
            r["trajectory_file"] = fname
            results.append(r)
            print(f"  ✓ {tag} | reward={r['total_reward']:.0f} "
                  f"ndr={r['ndr_pct']:.1f}% data={r['data_collected']:.0f} "
                  f"jain={r.get('jains_index', 0):.3f}", flush=True)
        except Exception as exc:
            results.append({
                "model": model_info["name"], "grid_w": grid_size[0], "grid_h": grid_size[1],
                "n_sensors": n_sensors, "start_pos_name": sp_name,
                "start_x": start_pos[0], "start_y": start_pos[1],
                "seed": seed, "error": str(exc),
            })
            print(f"  ✗ {tag}: {exc}", flush=True)
            traceback.print_exc()
    return results


def greedy_worker(args):
    agent_name, episodes, traj_dir_str = args
    _setup_paths()
    traj_dir = Path(traj_dir_str)

    results = []
    for grid_size, n_sensors, sp_name, start_pos, seed in episodes:
        tag = f"{agent_name} {grid_size[0]}x{grid_size[1]} n={n_sensors} {sp_name} s={seed}"
        try:
            r     = _run_greedy_episode(agent_name,
                                        grid_size, n_sensors, sp_name, start_pos, seed)
            traj  = np.array(r.pop("_trajectory"), dtype=np.float32)
            fname = (f"{agent_name}_{grid_size[0]}x{grid_size[1]}"
                     f"_n{n_sensors}_{sp_name}_{seed}.npy")
            np.save(traj_dir / fname, traj)
            r["trajectory_file"] = fname
            results.append(r)
            print(f"  ✓ {tag} | ndr={r['ndr_pct']:.1f}% "
                  f"data={r['data_collected']:.0f} jain={r.get('jains_index', 0):.3f}",
                  flush=True)
        except Exception as exc:
            results.append({
                "model": agent_name, "grid_w": grid_size[0], "grid_h": grid_size[1],
                "n_sensors": n_sensors, "start_pos_name": sp_name,
                "start_x": start_pos[0], "start_y": start_pos[1],
                "seed": seed, "error": str(exc),
            })
            print(f"  ✗ {tag}: {exc}", flush=True)
    return results


# ─── Episode list builders ────────────────────────────────────────────────────
def build_all_episodes(seeds=None) -> list:
    if seeds is None:
        seeds = EVAL_SEEDS
    episodes = []
    for grid_size in GRID_SIZES:
        start_positions = get_start_positions(grid_size)
        for n_sensors in SENSOR_COUNTS:
            for sp_name, sp in start_positions.items():
                for seed in seeds:
                    episodes.append((grid_size, n_sensors, sp_name, sp, seed))
    return episodes


def chunk(lst: list, n: int) -> list:
    if not lst:
        return []
    size = max(1, (len(lst) + n - 1) // n)
    return [lst[i:i + size] for i in range(0, len(lst), size)]


# ─── Resume helpers ───────────────────────────────────────────────────────────
def load_completed(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    if "error" in df.columns:
        df = df[df["error"].isna()]
    df = df.dropna(subset=["model", "grid_w", "grid_h", "n_sensors", "start_pos_name", "seed"])
    return set(zip(
        df["model"],
        df["grid_w"].astype(int),
        df["grid_h"].astype(int),
        df["n_sensors"].astype(int),
        df["start_pos_name"],
        df["seed"].astype(int),
    ))


def filter_remaining(episodes: list, completed: set, agent_name: str) -> list:
    return [
        ep for ep in episodes
        if (agent_name, ep[0][0], ep[0][1], ep[1], ep[2], ep[4]) not in completed
    ]


def append_to_csv(rows: list, csv_path: Path):
    clean = [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]
    df    = pd.DataFrame(clean)
    df.to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)


# ─── Plot helpers ─────────────────────────────────────────────────────────────
def _try_ieee():
    try:
        import ieee_style
        ieee_style.apply()
        return ieee_style
    except ImportError:
        return None


def plot_ndr_grouped_bar(df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    fig.suptitle("Mean NDR (%) by Agent and Grid Size\n(error bars = std across seeds and start positions)",
                 fontsize=13, fontweight="bold")
    for ax, (gw, gh) in zip(axes, [(200, 200), (400, 400), (600, 600)]):
        sub = df[(df["grid_w"] == gw) & (df["grid_h"] == gh)]
        x     = np.arange(len(ALL_AGENTS))
        means = [sub[sub["model"] == a]["ndr_pct"].mean() for a in ALL_AGENTS]
        stds  = [sub[sub["model"] == a]["ndr_pct"].std()  for a in ALL_AGENTS]
        bars  = ax.bar(x, means, yerr=stds, capsize=4, width=0.6,
                       color=[AGENT_COLORS[a] for a in ALL_AGENTS],
                       alpha=0.85, edgecolor="white", error_kw={"elinewidth": 1.5})
        ax.set_xticks(x)
        ax.set_xticklabels(ALL_AGENTS, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"{gw}\u00d7{gh}", fontweight="bold")
        ax.set_ylabel("NDR (%)" if ax == axes[0] else "")
        ax.set_ylim(0, 115)
        ax.grid(axis="y", alpha=0.35, linestyle="--")
        for bar, mean in zip(bars, means):
            if not np.isnan(mean):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                        f"{mean:.1f}", ha="center", fontsize=7, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "ndr_grouped_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  \u2713 ndr_grouped_bar.png")


def plot_data_boxplot(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots(figsize=(14, 6))
    data_groups = [df[df["model"] == a]["data_collected"].dropna().values for a in ALL_AGENTS]
    bp = ax.boxplot(data_groups, tick_labels=ALL_AGENTS, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker="o", markersize=3, alpha=0.4))
    for patch, agent in zip(bp["boxes"], ALL_AGENTS):
        patch.set_facecolor(AGENT_COLORS[agent])
        patch.set_alpha(0.75)
    ax.set_ylabel("Data Collected (Bytes)", fontsize=11, fontweight="bold")
    ax.set_title("Data Collected Distribution Across All Conditions\n"
                 "(all grid sizes, sensor counts, start positions, seeds)",
                 fontsize=12, fontweight="bold")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.grid(axis="y", alpha=0.35, linestyle="--")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    fig.savefig(out_dir / "data_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  \u2713 data_boxplot.png")


def plot_fairness_heatmap(df: pd.DataFrame, out_dir: Path):
    cond_keys = [(gw, n) for gw in [200, 400, 600] for n in [10, 20, 30]]
    cond_labels = [f"{gw}\u00d7{gw}\nn={n}" for gw, n in cond_keys]

    matrix = np.full((len(ALL_AGENTS), len(cond_keys)), np.nan)
    for j, (gw, n) in enumerate(cond_keys):
        for i, agent in enumerate(ALL_AGENTS):
            sub = df[(df["model"] == agent) & (df["grid_w"] == gw) & (df["n_sensors"] == n)]
            if len(sub) > 0 and "jains_index" in sub.columns:
                matrix[i, j] = sub["jains_index"].mean()

    fig, ax = plt.subplots(figsize=(15, 5))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Mean Jain's Fairness Index", fraction=0.03, pad=0.02)
    ax.set_xticks(range(len(cond_labels)))
    ax.set_xticklabels(cond_labels, fontsize=8)
    ax.set_yticks(range(len(ALL_AGENTS)))
    ax.set_yticklabels(ALL_AGENTS, fontsize=9)
    for i in range(len(ALL_AGENTS)):
        for j in range(len(cond_keys)):
            val = matrix[i, j]
            if not np.isnan(val):
                color = "black" if 0.25 < val < 0.75 else "white"
                ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                        fontsize=7, color=color, fontweight="bold")
    ax.set_title("Mean Jain's Fairness Index — Agent \u00d7 Condition\n"
                 "(green = equitable collection, red = sensor starvation)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    fig.savefig(out_dir / "fairness_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  \u2713 fairness_heatmap.png")


def plot_pareto_scatter(df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Pareto Frontier: Energy Consumed vs Data Collected\n"
                 "(mean \u00b1 std across seeds and start positions, per sensor count)",
                 fontsize=12, fontweight="bold")
    for ax, n_sensors in zip(axes, [10, 20, 30]):
        sub = df[df["n_sensors"] == n_sensors]
        for agent in ALL_AGENTS:
            a = sub[sub["model"] == agent]
            if a.empty:
                continue
            xm, ym = a["energy_consumed"].mean(), a["data_collected"].mean()
            xs, ys = a["energy_consumed"].std(),  a["data_collected"].std()
            ax.errorbar(xm, ym, xerr=xs, yerr=ys,
                        fmt="none", color=AGENT_COLORS[agent], alpha=0.35, capsize=3)
            ax.scatter(xm, ym, s=220, color=AGENT_COLORS[agent],
                       marker=AGENT_MARKERS[agent], zorder=5,
                       edgecolors="white", linewidths=1.5, label=agent)
            ax.annotate(agent, (xm, ym), textcoords="offset points",
                        xytext=(7, 4), fontsize=7, color=AGENT_COLORS[agent],
                        path_effects=[pe.withStroke(linewidth=1.5, foreground="white")])

        # Highlight Pareto-efficient points (min energy for max data)
        pts = [(sub[sub["model"] == a]["energy_consumed"].mean(),
                sub[sub["model"] == a]["data_collected"].mean()) for a in ALL_AGENTS
               if not sub[sub["model"] == a].empty]
        if pts:
            pts_sorted = sorted(pts, key=lambda p: p[0])
            pareto_x, pareto_y, best_y = [], [], -np.inf
            for x_p, y_p in pts_sorted:
                if y_p > best_y:
                    pareto_x.append(x_p)
                    pareto_y.append(y_p)
                    best_y = y_p
            ax.plot(pareto_x, pareto_y, "k--", linewidth=1.2, alpha=0.5,
                    label="Pareto frontier", zorder=1)

        ax.set_xlabel("Energy Consumed (Wh)", fontweight="bold")
        ax.set_ylabel("Data Collected (Bytes)" if ax == axes[0] else "")
        ax.set_title(f"N = {n_sensors} sensors", fontweight="bold")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.grid(alpha=0.35, linestyle="--")
        handles, lbls = ax.get_legend_handles_labels()
        if handles and ax == axes[-1]:
            ax.legend(handles, lbls, fontsize=7, loc="lower right", framealpha=0.9)
    plt.tight_layout()
    fig.savefig(out_dir / "pareto_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  \u2713 pareto_scatter.png")


def plot_starved_sensor_bar(df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"Mean Starved Sensors (<{STARVATION_THR:.0f}% collection rate)",
                 fontsize=13, fontweight="bold")
    for ax, n_sensors in zip(axes, [10, 20, 30]):
        sub = df[df["n_sensors"] == n_sensors]
        x     = np.arange(len(ALL_AGENTS))
        means = [sub[sub["model"] == a]["starved_sensors"].mean() for a in ALL_AGENTS]
        stds  = [sub[sub["model"] == a]["starved_sensors"].std()  for a in ALL_AGENTS]
        ax.bar(x, means, yerr=stds, capsize=4, width=0.6,
               color=[AGENT_COLORS[a] for a in ALL_AGENTS],
               alpha=0.85, edgecolor="white", error_kw={"elinewidth": 1.5})
        ax.set_xticks(x)
        ax.set_xticklabels(ALL_AGENTS, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"N = {n_sensors} sensors", fontweight="bold")
        ax.set_ylabel("Starved sensor count" if ax == axes[0] else "")
        ax.axhline(0, color="black", linewidth=0.8)
        ax.grid(axis="y", alpha=0.35, linestyle="--")
    plt.tight_layout()
    fig.savefig(out_dir / "starved_sensor_bar.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  \u2713 starved_sensor_bar.png")


def plot_summary_table(df: pd.DataFrame, out_dir: Path):
    metrics = [
        ("total_reward",    "Reward"),
        ("data_collected",  "Data (B)"),
        ("ndr_pct",         "NDR %"),
        ("battery_pct",     "Batt %"),
        ("efficiency",      "Eff (B/Wh)"),
        ("jains_index",     "Jain's"),
        ("starved_sensors", "Starved"),
    ]
    col_labels = ["Agent"] + [label for _, label in metrics]
    rows = []
    for agent in ALL_AGENTS:
        sub  = df[df["model"] == agent]
        row  = [agent]
        for col, _ in metrics:
            if col in sub.columns:
                vals = sub[col].dropna()
                row.append(f"{vals.mean():.2f}\u00b1{vals.std():.2f}" if len(vals) else "N/A")
            else:
                row.append("N/A")
        rows.append(row)

    fig, ax = plt.subplots(figsize=(17, 3.8))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 2.2)
    for j in range(len(col_labels)):
        cell = tbl[0, j]
        cell.set_facecolor("#1b9e77")
        cell.set_text_props(color="white", fontweight="bold")
        cell.set_edgecolor("white")
    for i, agent in enumerate(ALL_AGENTS):
        bg = "white" if i % 2 == 0 else "#F5F5F5"
        for j in range(len(col_labels)):
            tbl[i + 1, j].set_facecolor(bg)
            tbl[i + 1, j].set_edgecolor("#DDDDDD")
        tbl[i + 1, 0].set_text_props(color=AGENT_COLORS[agent], fontweight="bold")
    ax.set_title(
        "Summary Statistics: Mean \u00b1 Std across all 360 episodes per agent\n"
        "(3 grid sizes \u00d7 3 sensor counts \u00d7 8 start positions \u00d7 5 seeds)",
        fontweight="bold", pad=14, fontsize=10)
    plt.tight_layout()
    fig.savefig(out_dir / "summary_table.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  \u2713 summary_table.png")


def plot_per_condition_trajectories(df: pd.DataFrame, traj_dir: Path, out_dir: Path):
    """One 2x3 figure per (grid, n_sensors, start_pos): mean path + per-seed faded lines."""
    conditions = (df[["grid_w", "grid_h", "n_sensors", "start_pos_name"]]
                  .drop_duplicates()
                  .values.tolist())
    total = len(conditions)
    for idx, (gw, gh, n_sensors, sp_name) in enumerate(conditions, 1):
        gw, gh, n_sensors = int(gw), int(gh), int(n_sensors)
        cond_key = f"{gw}x{gh}_n{n_sensors}_{sp_name}"
        cond_dir = out_dir / "per_condition" / cond_key
        cond_dir.mkdir(parents=True, exist_ok=True)
        out_file = cond_dir / "trajectories.png"
        if out_file.exists():
            continue

        fig, axes = plt.subplots(2, 3, figsize=(18, 11))
        fig.suptitle(f"UAV Trajectories — {gw}\u00d7{gh}, N={n_sensors}, Start={sp_name}\n"
                     f"(faded = individual seeds, bold = mean path)",
                     fontsize=12, fontweight="bold")

        for ax, agent in zip(axes.flatten(), ALL_AGENTS):
            sub = df[(df["model"] == agent) & (df["grid_w"] == gw) &
                     (df["n_sensors"] == n_sensors) & (df["start_pos_name"] == sp_name)]
            ax.set_xlim(0, gw)
            ax.set_ylim(0, gh)
            ax.set_aspect("equal")
            ax.set_title(agent, fontweight="bold", color=AGENT_COLORS[agent], fontsize=10)
            ax.set_xlabel("X (m)", fontsize=8)
            ax.set_ylabel("Y (m)", fontsize=8)
            ax.grid(alpha=0.25, linestyle="--")

            trajs = []
            for _, ep_row in sub.iterrows():
                fname = ep_row.get("trajectory_file")
                if pd.isna(fname):
                    continue
                fp = traj_dir / str(fname)
                if fp.exists():
                    trajs.append(np.load(fp))

            if not trajs:
                ax.text(0.5, 0.5, "No data", ha="center", va="center",
                        transform=ax.transAxes, color="gray", fontsize=11)
                continue

            for traj in trajs:
                if len(traj) > 1:
                    ax.plot(traj[:, 0], traj[:, 1],
                            color=AGENT_COLORS[agent], alpha=0.18, linewidth=0.9)

            max_len = max(len(t) for t in trajs)
            padded  = [np.pad(t, ((0, max_len - len(t)), (0, 0)), mode="edge")
                       for t in trajs]
            mean_t  = np.mean(padded, axis=0)
            ax.plot(mean_t[:, 0], mean_t[:, 1],
                    color=AGENT_COLORS[agent], linewidth=2.5, alpha=0.95)
            ax.scatter(mean_t[0,  0], mean_t[0,  1], c="#2ca02c", s=130,
                       marker="^", edgecolors="darkgreen", linewidth=1.2,
                       zorder=6, label="Start")
            ax.scatter(mean_t[-1, 0], mean_t[-1, 1], c="#d62728", s=130,
                       marker="*", edgecolors="darkred", linewidth=1.2,
                       zorder=6, label="End")
            ax.legend(fontsize=7, loc="lower right")

            stats_txt = (f"seeds={len(trajs)}\n"
                         f"ndr={sub['ndr_pct'].mean():.1f}%\n"
                         f"data={sub['data_collected'].mean():.2e}")
            ax.text(0.02, 0.98, stats_txt, transform=ax.transAxes,
                    fontsize=7, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="#CCCCCC", alpha=0.85))

        plt.tight_layout()
        fig.savefig(out_file, dpi=110, bbox_inches="tight")
        plt.close()
        if idx % 10 == 0 or idx == total:
            print(f"  \u2713 Trajectory plots: {idx}/{total}", flush=True)


# ─── Main orchestration ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Multi-model evaluation sweep")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Run 1 seed per condition (~432 episodes)")
    parser.add_argument("--smoke-test", action="store_true",
                        help="1 condition (200x200, n=10, center, seed=42) per agent — 6 episodes, ~2 min")
    parser.add_argument("--resume",     action="store_true",
                        help="Skip rows already present in all_results.csv")
    parser.add_argument("--no-plots",   action="store_true",
                        help="Run episodes only; skip plot generation")
    parser.add_argument("--workers",    type=int, default=N_WORKERS,
                        help="Number of parallel worker processes (default 4)")
    parser.add_argument("--cpu",        action="store_true",
                        help="Force CPU inference (avoids CUDA in workers)")
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    traj_dir   = OUTPUT_DIR / "trajectories";  traj_dir.mkdir(exist_ok=True)
    summary_dir = OUTPUT_DIR / "summary";      summary_dir.mkdir(exist_ok=True)

    device    = "cpu" if args.cpu else "cuda"
    n_workers = args.workers
    csv_path  = OUTPUT_DIR / "all_results.csv"
    completed = load_completed(csv_path) if args.resume else set()

    if args.smoke_test:
        sp = get_start_positions((200, 200))
        all_episodes = [((200, 200), 10, "center", sp["center"], 42)]
    elif args.dry_run:
        all_episodes = build_all_episodes(EVAL_SEEDS[:1])
    else:
        all_episodes = build_all_episodes()
    total_ep     = (len(MODEL_REGISTRY) + len(GREEDY_NAMES)) * len(all_episodes)

    print(f"\n{'='*80}")
    print(f"MULTI-MODEL EVALUATION SWEEP")
    print(f"  DQN models : {[m['name'] for m in MODEL_REGISTRY]}")
    print(f"  Greedy     : {GREEDY_NAMES}")
    print(f"  Grid sizes : {GRID_SIZES}")
    print(f"  Sensors    : {SENSOR_COUNTS}")
    print(f"  Start pos  : 8 (center, 4 corners, 3 random)")
    active_seeds = [42] if args.smoke_test else (EVAL_SEEDS[:1] if args.dry_run else EVAL_SEEDS)
    print(f"  Seeds      : {active_seeds}")
    print(f"  Episodes   : {len(all_episodes)} per agent  |  {total_ep} total")
    print(f"  Workers    : {n_workers}  |  Device: {device}")
    print(f"  Dry-run    : {args.dry_run}  |  Resume: {args.resume}")
    print(f"  Output     : {OUTPUT_DIR}")
    print(f"{'='*80}\n")

    t0 = time.time()

    # ── Greedy agents (pure CPU, safe to fully parallelise) ───────────────────
    for agent_name in GREEDY_NAMES:
        remaining = filter_remaining(all_episodes, completed, agent_name)
        if not remaining:
            print(f"[{agent_name}] All done, skipping.\n")
            continue
        print(f"\n[{agent_name}]  {len(remaining)} episodes -> {n_workers} workers")
        chunks     = chunk(remaining, n_workers)
        w_args     = [(agent_name, c, str(traj_dir)) for c in chunks]
        all_rows: list = []
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(greedy_worker, a): a for a in w_args}
            for fut in as_completed(futs):
                rows = fut.result()
                all_rows.extend(rows)
                append_to_csv(rows, csv_path)
        ok = [r for r in all_rows if "error" not in r]
        completed.update(
            (r["model"], r["grid_w"], r["grid_h"], r["n_sensors"],
             r["start_pos_name"], r["seed"]) for r in ok
        )
        print(f"  [{agent_name}] {len(ok)}/{len(all_rows)} episodes OK  "
              f"({time.time()-t0:.0f}s elapsed)")

    # ── DQN models (GPU inference, 4 workers each share the same GPU) ─────────
    for model_info in MODEL_REGISTRY:
        remaining = filter_remaining(all_episodes, completed, model_info["name"])
        if not remaining:
            print(f"\n[{model_info['name']}] All done, skipping.")
            continue
        print(f"\n[{model_info['name']}]  {len(remaining)} episodes -> {n_workers} workers")
        chunks = chunk(remaining, n_workers)
        w_args = [(model_info, c, device, str(traj_dir)) for c in chunks]
        all_rows = []
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futs = {ex.submit(dqn_worker, a): a for a in w_args}
            for fut in as_completed(futs):
                rows = fut.result()
                all_rows.extend(rows)
                append_to_csv(rows, csv_path)
        ok = [r for r in all_rows if "error" not in r]
        completed.update(
            (r["model"], r["grid_w"], r["grid_h"], r["n_sensors"],
             r["start_pos_name"], r["seed"]) for r in ok
        )
        print(f"  [{model_info['name']}] {len(ok)}/{len(all_rows)} episodes OK  "
              f"({(time.time()-t0)/60:.1f} min elapsed)")

    # ── Generate plots ────────────────────────────────────────────────────────
    if not args.no_plots:
        if not csv_path.exists():
            print("\nNo results CSV found — nothing to plot.")
        else:
            print(f"\n{'='*80}\nGenerating summary plots from {csv_path.name}...")
            df = pd.read_csv(csv_path)
            if "error" in df.columns:
                n_err = df["error"].notna().sum()
                if n_err:
                    print(f"  Warning: {n_err} failed episodes in CSV (excluded from plots)")
                df = df[df["error"].isna()]
            _try_ieee()
            plot_ndr_grouped_bar(df, summary_dir)
            plot_data_boxplot(df, summary_dir)
            plot_fairness_heatmap(df, summary_dir)
            plot_pareto_scatter(df, summary_dir)
            plot_starved_sensor_bar(df, summary_dir)
            plot_summary_table(df, summary_dir)
            print("\nGenerating per-condition trajectory plots (this may take a few minutes)...")
            plot_per_condition_trajectories(df, traj_dir, OUTPUT_DIR)
            print(f"\n\u2713 All plots saved to {summary_dir}")

    elapsed = time.time() - t0
    print(f"\n{'='*80}")
    print(f"\u2713 SWEEP COMPLETE  --  {elapsed/60:.1f} min total")
    print(f"  Master CSV : {csv_path}")
    print(f"  Plots      : {summary_dir}")
    print(f"  Trajectories: {traj_dir}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
