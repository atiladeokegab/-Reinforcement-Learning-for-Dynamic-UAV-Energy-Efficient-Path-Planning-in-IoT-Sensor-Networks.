"""
Multi-seed evaluation: dqn_v3_fixed_3M vs SF-Aware Greedy V2 vs Nearest Greedy.

5 seeds, N=40 sensors, 500x500 grid. UAV start position is seeded-random
(reproducible per seed, never fixed at 0,0). Each seed produces:
  - Per-agent CSV:  results/seed_{S}_{agent}.csv
  - Trajectory plot: results/trajectory_seed_{S}.pdf

After all seeds:
  - Summary CSV:             results/summary_all_seeds.csv
  - DQN 5-panel trajectories: results/dqn_trajectories_all_seeds.pdf

Author: ATILADE GABRIEL OKE
"""

import sys
import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import gymnasium
from pathlib import Path
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecNormalize

script_dir = Path(__file__).resolve().parent
src_dir    = script_dir.parent.parent.parent
sys.path.insert(0, str(script_dir.parent))
sys.path.insert(0, str(src_dir))

import ieee_style
ieee_style.apply()

from environment.uav_env import UAVEnvironment
from greedy_agents import MaxThroughputGreedyV2, NearestSensorGreedy

# ── Config ────────────────────────────────────────────────────────────────────

SEEDS      = [42, 123, 256, 789, 1337]
N_SENSORS  = 40
GRID_SIZE  = (500, 500)
MAX_STEPS  = 2100
MAX_BATT   = 274.0

def _parse_model_dir() -> Path:
    """Return model directory from --model CLI arg, or fall back to default."""
    for i, arg in enumerate(sys.argv[1:], 1):
        if arg == "--model" and i < len(sys.argv):
            return Path(sys.argv[i + 1])
        if arg.startswith("--model="):
            return Path(arg.split("=", 1)[1])
    return script_dir.parent / "models" / "dqn_v3_fixed_3M" / "dqn_v3_fixed"

_MODEL_BASE    = _parse_model_dir()
# prefer best_metric_model if present, otherwise fall back to dqn_final
_metric_zip    = _MODEL_BASE / "best_metric_model.zip"
MODEL_PATH     = _metric_zip if _metric_zip.exists() else _MODEL_BASE / "dqn_final.zip"
CONFIG_PATH    = _MODEL_BASE / "training_config.json"
VEC_NORM_PATH  = _MODEL_BASE / "vec_normalize.pkl"

_model_tag  = _MODEL_BASE.parent.name  # e.g. "dqn_smoke_test"
OUTPUT_DIR  = script_dir / f"multiseed_results_{_model_tag}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

AGENT_STYLES = {
    "DQN":          {"color": ieee_style.AGENT_COLORS["DQN Agent"],       "marker": "o", "ls": "-"},
    "SmartGreedy":  {"color": ieee_style.AGENT_COLORS["Smart Greedy V2"], "marker": "s", "ls": "--"},
    "NearestGreedy":{"color": ieee_style.AGENT_COLORS["Nearest Greedy"],  "marker": "^", "ls": ":"},
}
AGENT_LABELS = {
    "DQN":           "DQN Agent (v3, 3M)",
    "SmartGreedy":   "SF-Aware Greedy V2",
    "NearestGreedy": "Nearest Greedy",
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config():
    defaults = {"use_frame_stacking": True, "n_stack": 4, "max_sensors_limit": 50}
    try:
        with open(CONFIG_PATH) as f:
            return {**defaults, **json.load(f)}
    except FileNotFoundError:
        print(f"  Config not found — using defaults")
        return defaults


def uav_start_for_seed(seed: int) -> tuple:
    """Reproducible random start per seed, never at (0,0)."""
    rng = np.random.default_rng(seed + 99991)   # offset avoids overlap with env seeding
    gw, gh = GRID_SIZE
    return (float(rng.uniform(0.05 * gw, 0.95 * gw)),
            float(rng.uniform(0.05 * gh, 0.95 * gh)))


def reposition_pct(traj: np.ndarray) -> float:
    """% of steps where the UAV actually moved (position changed vs hover/collect)."""
    if len(traj) < 2:
        return 0.0
    diffs = np.abs(np.diff(traj, axis=0)).sum(axis=1)
    return float((diffs > 0).sum() / len(diffs) * 100)


def _unwrap_base(vec):
    inner = vec
    while hasattr(inner, "venv"):
        inner = inner.venv
    env = inner.envs[0]
    while hasattr(env, "env"):
        env = env.env
    return env


# ── AnalysisUAVEnv (zero-padding wrapper) ────────────────────────────────────

class AnalysisUAVEnv(UAVEnvironment):
    def __init__(self, max_sensors_limit: int = 50, **kwargs):
        self.max_sensors_limit = max_sensors_limit
        super().__init__(**kwargs)
        raw = self.observation_space.shape[0]
        self._fps = 0
        for uav_f in range(raw + 1):
            rem = raw - uav_f
            if rem > 0 and rem % self.num_sensors == 0:
                self._fps = rem // self.num_sensors
                break
        if self._fps == 0:
            raise ValueError(f"Cannot infer features_per_sensor from obs={raw}, N={self.num_sensors}")
        padded = raw + (self.max_sensors_limit - self.num_sensors) * self._fps
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(padded,), dtype=np.float32)
        self.last_sensor_data = None

    def _pad(self, obs):
        extra = np.zeros((self.max_sensors_limit - self.num_sensors) * self._fps, dtype=np.float32)
        return np.concatenate([obs, extra]).astype(np.float32)

    def reset(self, **kwargs):
        if hasattr(self, "sensors") and self.current_step > 0:
            self.last_sensor_data = [
                {"sensor_id": s.sensor_id, "total_data_transmitted": float(s.total_data_transmitted),
                 "total_data_generated": float(s.total_data_generated)}
                for s in self.sensors
            ]
        obs, info = super().reset(**kwargs)
        return self._pad(obs), info

    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        return self._pad(obs), r, term, trunc, info


# ── Environment factory ───────────────────────────────────────────────────────

def make_dqn_env(cfg, seed, uav_start):
    env_kw = dict(
        grid_size=GRID_SIZE, num_sensors=N_SENSORS, max_steps=MAX_STEPS,
        max_sensors_limit=cfg["max_sensors_limit"],
        path_loss_exponent=3.8, rssi_threshold=-85.0, sensor_duty_cycle=10.0,
        max_battery=MAX_BATT, uav_start_position=uav_start,
    )
    np.random.seed(seed); random.seed(seed)
    vec = DummyVecEnv([lambda: AnalysisUAVEnv(**env_kw)])
    if cfg.get("use_frame_stacking", True):
        vec = VecFrameStack(vec, n_stack=cfg.get("n_stack", 4))
    if VEC_NORM_PATH.exists():
        try:
            vec = VecNormalize.load(str(VEC_NORM_PATH), vec)
            vec.training = False; vec.norm_reward = False
        except AssertionError:
            pass
    base = _unwrap_base(vec)
    try:
        base.np_random, _ = gymnasium.utils.seeding.np_random(seed)
    except Exception:
        base.np_random = np.random.RandomState(seed)
    np.random.seed(seed); random.seed(seed)
    return vec, base


def make_greedy_env(seed, uav_start):
    np.random.seed(seed); random.seed(seed)
    return UAVEnvironment(
        grid_size=GRID_SIZE, num_sensors=N_SENSORS, max_steps=MAX_STEPS,
        path_loss_exponent=3.8, rssi_threshold=-85.0, sensor_duty_cycle=10.0,
        max_battery=MAX_BATT, uav_start_position=uav_start,
    )


# ── Run agents ────────────────────────────────────────────────────────────────

def run_dqn(model, vec, base, seed):
    obs = vec.reset()
    traj, history = [], []
    cum_r = 0.0
    pre_snap = None

    while True:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action[0]) if isinstance(action, np.ndarray) and action.ndim > 0 else int(action)

        pre_batt     = base.uav.battery
        pre_cov      = len(base.sensors_visited) / base.num_sensors * 100
        pre_data     = base.total_data_collected
        pre_n_vis    = len(base.sensors_visited)
        pre_pos      = tuple(base.uav.position)
        pre_snap = [
            {"sensor_id": s.sensor_id,
             "total_data_generated":   float(s.total_data_generated),
             "total_data_transmitted": float(s.total_data_transmitted)}
            for s in base.sensors
        ]

        obs, rewards, dones, _ = vec.step([action])
        cum_r += float(rewards[0])
        done   = bool(dones[0])
        traj.append(pre_pos)

        step = len(traj)
        if done or step % 50 == 0:
            energy = MAX_BATT - pre_batt
            history.append({
                "seed": seed, "step": step,
                "cumulative_reward": cum_r,
                "battery_pct": pre_batt / MAX_BATT * 100,
                "coverage_pct": pre_cov,
                "sensors_visited": pre_n_vis,
                "data_collected": pre_data,
                "efficiency": pre_data / energy if energy > 0 else 0.0,
            })
        if done:
            break

    rates = [s["total_data_transmitted"] / s["total_data_generated"] * 100
             if s["total_data_generated"] > 0 else 0.0
             for s in pre_snap]
    n = len(rates); s2 = sum(r**2 for r in rates)
    jains = (sum(rates)**2 / (n * s2)) if n > 0 and s2 > 0 else 0.0
    traj_arr = np.array(traj)
    repo_pct = reposition_pct(traj_arr)
    return pd.DataFrame(history), traj_arr, jains, repo_pct


def run_greedy(agent, env, seed, name):
    obs, _ = env.reset(seed=seed)
    traj, history = [], []
    cum_r = 0.0
    done  = False
    step  = 0

    while not done:
        action = agent.select_action(obs)
        obs, reward, done, truncated, _ = env.step(action)
        cum_r += reward
        step  += 1
        traj.append(tuple(env.uav.position))
        done = done or truncated

        if done or step % 50 == 0:
            energy = MAX_BATT - env.uav.battery
            history.append({
                "seed": seed, "step": step,
                "cumulative_reward": cum_r,
                "battery_pct": env.uav.get_battery_percentage(),
                "coverage_pct": len(env.sensors_visited) / env.num_sensors * 100,
                "sensors_visited": len(env.sensors_visited),
                "data_collected": env.total_data_collected,
                "efficiency": env.total_data_collected / energy if energy > 0 else 0.0,
            })

    rates = [s.total_data_transmitted / s.total_data_generated * 100
             if s.total_data_generated > 0 else 0.0
             for s in env.sensors]
    n = len(rates); s2 = sum(r**2 for r in rates)
    jains = (sum(rates)**2 / (n * s2)) if n > 0 and s2 > 0 else 0.0
    traj_arr = np.array(traj)
    repo_pct = reposition_pct(traj_arr)
    return pd.DataFrame(history), traj_arr, jains, repo_pct


# ── Plotting ──────────────────────────────────────────────────────────────────

def _draw_trajectory(ax, traj, sensor_positions, color, title, grid_size):
    ax.set_xlim(0, grid_size[0]); ax.set_ylim(0, grid_size[1])
    ax.set_aspect("equal")
    ax.scatter(sensor_positions[:, 0], sensor_positions[:, 1],
               s=80, c="#d95f02", marker="s", edgecolors="#7a2b00",
               linewidth=0.8, zorder=3, label="Sensor")
    if len(traj) > 1:
        ax.plot(traj[:, 0], traj[:, 1], color=color, lw=1.0, alpha=0.6, zorder=2)
        ax.scatter(*traj[0],  s=120, c="#2ca02c", marker="^", edgecolors="darkgreen",
                   lw=1.0, zorder=4, label="Start")
        ax.scatter(*traj[-1], s=120, c=color,     marker="*", edgecolors="black",
                   lw=0.8, zorder=4, label="End")
        dist     = float(np.sum(np.sqrt(np.sum(np.diff(traj, axis=0)**2, axis=1))))
        repo_pct = reposition_pct(traj)
        ax.text(0.02, 0.97,
                f"{dist:.0f} m | {len(traj)} steps\nReposition: {repo_pct:.1f}%",
                transform=ax.transAxes, fontsize=7, va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#ccc", alpha=0.85))
    ax.set_title(title, fontsize=9, fontweight="bold")
    ax.set_xlabel("x (m)", fontsize=8); ax.set_ylabel("y (m)", fontsize=8)
    ieee_style.clean_axes(ax)


def plot_seed_trajectories(seed, uav_start, sensor_pos, dqn_traj, smart_traj, near_traj,
                           dqn_jains, smart_jains, near_jains,
                           repo_dqn, repo_smart, repo_near):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Seed {seed} | N={N_SENSORS} | UAV start ({uav_start[0]:.0f}, {uav_start[1]:.0f})",
                 fontsize=11, fontweight="bold")

    data = [
        (axes[0], dqn_traj,   AGENT_STYLES["DQN"]["color"],
         f"DQN Agent (v3, 3M)\nJain's={dqn_jains:.3f}  |  Reposition={repo_dqn:.1f}%"),
        (axes[1], smart_traj, AGENT_STYLES["SmartGreedy"]["color"],
         f"SF-Aware Greedy V2\nJain's={smart_jains:.3f}  |  Reposition={repo_smart:.1f}%"),
        (axes[2], near_traj,  AGENT_STYLES["NearestGreedy"]["color"],
         f"Nearest Greedy\nJain's={near_jains:.3f}  |  Reposition={repo_near:.1f}%"),
    ]
    for ax, traj, color, title in data:
        _draw_trajectory(ax, traj, sensor_pos, color, title, GRID_SIZE)

    plt.tight_layout()
    out = OUTPUT_DIR / f"trajectory_seed_{seed}"
    ieee_style.save(fig, str(out))
    print(f"  Saved: {out.name}.pdf")
    plt.close()


def plot_dqn_all_seeds(seed_trajs, seed_sensor_pos, seed_starts):
    """5-panel figure: DQN trajectory for each seed."""
    fig, axes = plt.subplots(1, 5, figsize=(25, 5))
    fig.suptitle(f"DQN Agent (v3, 3M) — Trajectories Across 5 Seeds | N={N_SENSORS}",
                 fontsize=12, fontweight="bold")
    color = AGENT_STYLES["DQN"]["color"]
    for ax, seed in zip(axes, SEEDS):
        traj   = seed_trajs[seed]
        s_pos  = seed_sensor_pos[seed]
        start  = seed_starts[seed]
        _draw_trajectory(ax, traj, s_pos, color,
                         f"Seed {seed}\n({start[0]:.0f}, {start[1]:.0f})", GRID_SIZE)
    plt.tight_layout()
    out = OUTPUT_DIR / "dqn_trajectories_all_seeds"
    ieee_style.save(fig, str(out))
    print(f"  Saved: {out.name}.pdf")
    plt.close()


def plot_summary_bars(summary_df):
    """Bar chart comparing mean ± std across seeds for key metrics."""
    metrics = ["coverage_pct", "data_collected", "efficiency", "jains_index", "reposition_pct"]
    titles  = ["Sensor Coverage (%)", "Data Collected (bytes)",
               "Efficiency (bytes/Wh)", "Jain's Fairness Index", "Repositioning (%)"]

    fig, axes = plt.subplots(1, 5, figsize=(22, 5))
    fig.suptitle(f"5-Seed Aggregate Comparison | N={N_SENSORS} | 500×500 grid",
                 fontsize=12, fontweight="bold")

    agents = ["DQN", "SmartGreedy", "NearestGreedy"]
    x = np.arange(len(agents))
    width = 0.6

    for ax, metric, title in zip(axes, metrics, titles):
        means, stds, colors = [], [], []
        for agent in agents:
            sub = summary_df[summary_df["agent"] == agent]
            means.append(sub[metric].mean())
            stds.append(sub[metric].std())
            colors.append(AGENT_STYLES[agent]["color"])
        bars = ax.bar(x, means, width, yerr=stds, capsize=5,
                      color=colors, alpha=0.85, edgecolor="white", linewidth=1.2,
                      error_kw={"elinewidth": 1.5, "ecolor": "black"})
        ax.set_xticks(x)
        ax.set_xticklabels([AGENT_LABELS[a].replace(" ", "\n") for a in agents], fontsize=8)
        ax.set_title(title, fontsize=10, fontweight="bold")
        if "data" in metric:
            ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ieee_style.clean_axes(ax)

    plt.tight_layout()
    out = OUTPUT_DIR / "summary_bars"
    ieee_style.save(fig, str(out))
    print(f"  Saved: {out.name}.pdf")
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 80)
    print(f"Multi-seed evaluation: dqn_v3_fixed_3M vs greedy baselines")
    print(f"Seeds: {SEEDS}  |  N={N_SENSORS}  |  Grid: {GRID_SIZE}")
    print(f"Model: {MODEL_PATH}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 80)

    if not MODEL_PATH.exists():
        print(f"ERROR: model not found at {MODEL_PATH}")
        return

    cfg = load_config()
    print(f"Config: max_sensors_limit={cfg['max_sensors_limit']}, n_stack={cfg.get('n_stack', 4)}\n")

    model = DQN.load(str(MODEL_PATH))
    print("DQN model loaded.\n")

    summary_rows = []
    seed_dqn_trajs    = {}
    seed_sensor_pos   = {}
    seed_uav_starts   = {}

    for seed in SEEDS:
        print(f"\n{'─'*60}")
        uav_start = uav_start_for_seed(seed)
        seed_uav_starts[seed] = uav_start
        print(f"Seed {seed} | UAV start: ({uav_start[0]:.1f}, {uav_start[1]:.1f})")

        # ── DQN ──────────────────────────────────────────────────────────────
        print("  Running DQN...")
        vec, base = make_dqn_env(cfg, seed, uav_start)
        base.reset(seed=seed)
        # Re-create to apply seed properly
        vec.close()
        vec, base = make_dqn_env(cfg, seed, uav_start)
        df_dqn, traj_dqn, jains_dqn, repo_dqn = run_dqn(model, vec, base, seed)
        vec.close()

        sensor_pos = np.array([s.position for s in base.sensors])
        seed_sensor_pos[seed] = sensor_pos
        seed_dqn_trajs[seed]  = traj_dqn

        dqn_final = df_dqn.iloc[-1]
        print(f"    DQN    | NDR={dqn_final.coverage_pct:.1f}% | "
              f"Data={dqn_final.data_collected:.0f}B | Jain's={jains_dqn:.3f} | "
              f"Reposition={repo_dqn:.1f}%")

        # ── Greedy agents (shared env so sensor layout matches) ───────────────
        print("  Running greedy agents...")
        g_env = make_greedy_env(seed, uav_start)
        g_env.reset(seed=seed)

        agent_smart = MaxThroughputGreedyV2(g_env)
        df_smart, traj_smart, jains_smart, repo_smart = run_greedy(agent_smart, g_env, seed, "SmartGreedy")
        smart_final = df_smart.iloc[-1]
        print(f"    Smart  | NDR={smart_final.coverage_pct:.1f}% | "
              f"Data={smart_final.data_collected:.0f}B | Jain's={jains_smart:.3f} | "
              f"Reposition={repo_smart:.1f}%")

        agent_near = NearestSensorGreedy(g_env)
        df_near, traj_near, jains_near, repo_near = run_greedy(agent_near, g_env, seed, "NearestGreedy")
        near_final = df_near.iloc[-1]
        print(f"    Nearest| NDR={near_final.coverage_pct:.1f}% | "
              f"Data={near_final.data_collected:.0f}B | Jain's={jains_near:.3f} | "
              f"Reposition={repo_near:.1f}%")
        g_env.close()

        # ── Save per-seed CSVs ────────────────────────────────────────────────
        for tag, df in [("dqn", df_dqn), ("smart_greedy", df_smart), ("nearest_greedy", df_near)]:
            path = OUTPUT_DIR / f"seed_{seed}_{tag}.csv"
            df.to_csv(path, index=False)
        print(f"  CSVs saved: seed_{seed}_dqn.csv  seed_{seed}_smart_greedy.csv  seed_{seed}_nearest_greedy.csv")

        # ── Per-seed trajectory plot ──────────────────────────────────────────
        plot_seed_trajectories(seed, uav_start, sensor_pos,
                               traj_dqn, traj_smart, traj_near,
                               jains_dqn, jains_smart, jains_near,
                               repo_dqn, repo_smart, repo_near)

        # ── Accumulate summary ────────────────────────────────────────────────
        for tag, df, jains, repo in [
            ("DQN",           df_dqn,   jains_dqn,   repo_dqn),
            ("SmartGreedy",   df_smart, jains_smart, repo_smart),
            ("NearestGreedy", df_near,  jains_near,  repo_near),
        ]:
            final = df.iloc[-1]
            summary_rows.append({
                "seed":           seed,
                "agent":          tag,
                "coverage_pct":   final.coverage_pct,
                "data_collected": final.data_collected,
                "efficiency":     final.efficiency,
                "final_reward":   final.cumulative_reward,
                "battery_pct":    final.battery_pct,
                "jains_index":    jains,
                "reposition_pct": repo,
            })

    # ── Cross-seed plots ──────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("Generating cross-seed figures...")
    plot_dqn_all_seeds(seed_dqn_trajs, seed_sensor_pos, seed_uav_starts)

    summary_df = pd.DataFrame(summary_rows)
    plot_summary_bars(summary_df)

    # ── Summary CSV ───────────────────────────────────────────────────────────
    summary_path = OUTPUT_DIR / "summary_all_seeds.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"  Saved: summary_all_seeds.csv")

    # ── Print aggregate table ─────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print(f"AGGREGATE RESULTS (mean ± std across {len(SEEDS)} seeds)")
    print(f"{'='*80}")
    jains_hdr = "Jain's"
    print(f"{'Agent':<18} {'NDR (%)':>10} {'Data (B)':>14} {'Eff (B/Wh)':>12} {jains_hdr:>8}")
    print("-" * 70)
    for agent in ["DQN", "SmartGreedy", "NearestGreedy"]:
        sub = summary_df[summary_df["agent"] == agent]
        print(
            f"  {AGENT_LABELS[agent]:<20}"
            f"  {sub['coverage_pct'].mean():>6.1f}±{sub['coverage_pct'].std():>4.1f}"
            f"  {sub['data_collected'].mean():>10.0f}±{sub['data_collected'].std():>7.0f}"
            f"  {sub['efficiency'].mean():>8.2f}±{sub['efficiency'].std():>6.2f}"
            f"  {sub['jains_index'].mean():>6.3f}±{sub['jains_index'].std():>5.3f}"
        )
    print(f"\nAll outputs written to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
