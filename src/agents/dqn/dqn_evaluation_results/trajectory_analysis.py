"""
Trajectory & SF-Distribution Analysis
======================================
Generates three figures that visually demonstrate the DQN's protocol-aware
repositioning behaviour versus the greedy baselines:

  fig_trajectory_comparison.png  -- side-by-side UAV paths on sensor field
                                     (seed 42, N=20, 500x500)
  fig_position_heatmap.png       -- 2D position density across 5 seeds
  fig_sf_dynamics.png            -- SF distribution over time: DQN vs SF-Aware

These figures provide the visual evidence that the UAV's physical position is
an active control variable for the LoRaWAN protocol, rather than a passive
navigation parameter.

Author: ATILADE GABRIEL OKE
"""

import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from pathlib import Path
import ieee_style
ieee_style.apply()
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import gymnasium

script_dir = Path(__file__).resolve().parent
src_dir    = script_dir.parent.parent.parent
sys.path.insert(0, str(src_dir))

from environment.uav_env import UAVEnvironment
from greedy_agents import MaxThroughputGreedyV2, NearestSensorGreedy, LawnmowerAgent

# ==================== CONFIGURATION ====================

SEEDS          = [42, 123, 256, 789, 1337]
VIZ_SEED       = 42          # seed used for trajectory comparison figure
NUM_SENSORS    = 20
GRID_SIZE      = (1000, 1000)
MAX_STEPS      = 2100
N_STACK        = 4
MAX_SENSORS_LIMIT = 50

OUTPUT_DIR = script_dir / "trajectory_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BASE_ENV_KWARGS = {
    "grid_size":          GRID_SIZE,
    "num_sensors":        NUM_SENSORS,
    "max_steps":          MAX_STEPS,
    "path_loss_exponent": 3.8,
    "rssi_threshold":     -85.0,
    "sensor_duty_cycle":  10.0,
}

AGENT_STYLES = {
    "DQN":             {"color": ieee_style.AGENT_COLORS["DQN"],             "label": "DQN"},
    "SF-Aware Greedy": {"color": ieee_style.AGENT_COLORS["SF-Aware Greedy"], "label": "SF-Aware Greedy"},
    "Nearest Greedy":  {"color": ieee_style.AGENT_COLORS["Nearest Greedy"],  "label": "Nearest Greedy"},
}

SF_COLORS = ieee_style.SF_COLORS

_MAIN_MODEL_CANDIDATES = [
    script_dir.parent.parent.parent.parent / "models" / "dqn_domain_rand" / "best_model" / "best_model.zip",
    script_dir.parent.parent.parent.parent / "models" / "dqn_domain_rand" / "dqn_final.zip",
    script_dir.parent / "models" / "dqn_domain_rand"  / "dqn_final.zip",
]

# ==================== HELPERS ====================

def load_model():
    for path in _MAIN_MODEL_CANDIDATES:
        if path.exists():
            print(f"  Model: {path}")
            return DQN.load(str(path))
    raise FileNotFoundError("DQN model not found.")


def _unwrap_base_env(vec):
    inner = vec
    while hasattr(inner, "venv"):
        inner = inner.venv
    env = inner.envs[0]
    while hasattr(env, "env"):
        env = env.env
    return env


def get_canonical_positions(seed):
    """Return sensor positions for a given seed."""
    env = UAVEnvironment(**BASE_ENV_KWARGS)
    env.reset(seed=seed)
    positions = [(float(s.position[0]), float(s.position[1])) for s in env.sensors]
    initial_sfs = [int(s.spreading_factor) for s in env.sensors]
    env.close()
    return positions, initial_sfs


# ==================== ENV WRAPPERS ====================

class FixedLayoutEnv(UAVEnvironment):
    def __init__(self, fixed_positions, **kwargs):
        self._fixed_positions = fixed_positions
        super().__init__(**kwargs)

    def reset(self, **kwargs):
        W, H = float(self.grid_size[0]), float(self.grid_size[1])
        self.uav.start_position = np.array([
            float(np.random.uniform(0.1 * W, 0.9 * W)),
            float(np.random.uniform(0.1 * H, 0.9 * H)),
        ], dtype=np.float32)
        obs, info = super().reset(**kwargs)
        for sensor, pos in zip(self.sensors, self._fixed_positions):
            sensor.position = np.array(pos, dtype=np.float32)
        # Recompute obs so frame stack sees correct sensor positions from step 0
        obs = self._get_observation()
        return obs, info


class FixedLayoutPaddedEnv(FixedLayoutEnv):
    def __init__(self, fixed_positions, **kwargs):
        super().__init__(fixed_positions, **kwargs)
        raw = self.observation_space.shape[0]
        self._fps = 0
        for uav_f in range(raw + 1):
            rem = raw - uav_f
            if rem > 0 and rem % self.num_sensors == 0:
                self._fps = rem // self.num_sensors
                break
        padded = raw + (MAX_SENSORS_LIMIT - self.num_sensors) * self._fps
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(padded,), dtype=np.float32
        )

    def _pad(self, raw):
        pad = np.zeros(
            (MAX_SENSORS_LIMIT - self.num_sensors) * self._fps, dtype=np.float32
        )
        return np.concatenate([raw, pad]).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        return self._pad(obs), info

    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        return self._pad(obs), r, term, trunc, info


# ==================== EPISODE RUNNERS ====================

def run_greedy_trajectory(agent_class, fixed_positions, seed):
    """Run one greedy episode; return per-step trajectory dict."""
    env = FixedLayoutEnv(fixed_positions, **BASE_ENV_KWARGS)
    obs, _ = env.reset(seed=seed)
    agent = agent_class(env)
    if hasattr(agent, "reset"):
        agent.reset()

    steps, xs, ys, actions, sf_dists = [], [], [], [], []
    step = 0
    while True:
        action = agent.select_action(obs)
        obs, _, done, trunc, _ = env.step(action)
        step += 1
        xs.append(float(env.uav.position[0]))
        ys.append(float(env.uav.position[1]))
        actions.append(int(action))
        sf_dists.append({sf: sum(1 for s in env.sensors if s.spreading_factor == sf)
                         for sf in range(7, 13)})
        steps.append(step)
        if done or trunc:
            break
    final_sfs = [int(s.spreading_factor) for s in env.sensors]
    env.close()
    return {"steps": steps, "x": xs, "y": ys, "action": actions,
            "sf_dist": sf_dists, "final_sfs": final_sfs}


def run_dqn_trajectory(model, fixed_positions, seed):
    """Run one DQN episode; return per-step trajectory dict."""
    env = FixedLayoutPaddedEnv(fixed_positions, **BASE_ENV_KWARGS)
    vec_env  = VecFrameStack(DummyVecEnv([lambda: env]), n_stack=N_STACK)
    base_env = _unwrap_base_env(vec_env)

    obs = vec_env.reset()
    steps, xs, ys, actions, sf_dists = [], [], [], [], []
    step = 0
    snap_sfs = [int(s.spreading_factor) for s in base_env.sensors]

    while True:
        snap_sfs = [int(s.spreading_factor) for s in base_env.sensors]
        action, _ = model.predict(obs, deterministic=True)
        obs, _, done_arr, _ = vec_env.step(action)
        step += 1
        xs.append(float(base_env.uav.position[0]))
        ys.append(float(base_env.uav.position[1]))
        actions.append(int(action[0]))
        sf_dists.append({sf: sum(1 for s in base_env.sensors if s.spreading_factor == sf)
                         for sf in range(7, 13)})
        steps.append(step)
        if bool(done_arr[0]):
            break

    vec_env.close()
    return {"steps": steps, "x": xs, "y": ys, "action": actions,
            "sf_dist": sf_dists, "final_sfs": snap_sfs}


# ==================== FIGURE 1: TRAJECTORY COMPARISON ====================

def plot_trajectory_comparison(model, positions, initial_sfs):
    """Side-by-side trajectory plots: DQN | SF-Aware | Nearest on seed 42."""
    print("  Running trajectory episodes (seed 42)...")
    dqn_traj  = run_dqn_trajectory(model, positions, VIZ_SEED)
    sfa_traj  = run_greedy_trajectory(MaxThroughputGreedyV2, positions, VIZ_SEED)
    nrst_traj = run_greedy_trajectory(NearestSensorGreedy,   positions, VIZ_SEED)
    lawn_traj = run_greedy_trajectory(LawnmowerAgent,        positions, VIZ_SEED)

    trajs = [
        ("DQN",             dqn_traj,  "#1b9e77"),
        ("SF-Aware Greedy", sfa_traj,  "#d95f02"),
        ("Nearest Greedy",  nrst_traj, "#7570b3"),
        ("Lawnmower",       lawn_traj, "#e6ab02"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))

    for ax, (name, traj, col) in zip(axes, trajs):
        xs = np.array(positions)[:, 0]
        ys = np.array(positions)[:, 1]

        # Sensor scatter coloured by final SF (after ADR adaptation during episode)
        for xi, yi, sf in zip(xs, ys, traj["final_sfs"]):
            ax.scatter(xi, yi, c=SF_COLORS[sf], s=70, zorder=5,
                       edgecolors='black', linewidths=0.4)

        # Trajectory: colour by action (hover=orange, move=agent colour)
        x_arr = np.array(traj["x"])
        y_arr = np.array(traj["y"])
        act   = np.array(traj["action"])

        # Step-style line shows discrete grid movements clearly
        ax.step(x_arr, y_arr, where="post", color=col,
                alpha=0.55, linewidth=0.9, zorder=2)

        # Highlight hover steps — low alpha so moves pop
        collect_mask = act == 4
        ax.scatter(x_arr[collect_mask],  y_arr[collect_mask],
                   c="#FF7F00", s=8, alpha=0.3, zorder=3, label="Collect")
        ax.scatter(x_arr[~collect_mask], y_arr[~collect_mask],
                   c=col, s=5, alpha=0.6, zorder=3, label="Move")

        # Start and end markers
        ax.scatter(x_arr[0],  y_arr[0],  c="black", s=80, marker="^",
                   zorder=6, label="Start")
        ax.scatter(x_arr[-1], y_arr[-1], c="red",   s=80, marker="s",
                   zorder=6, label="End")

        collect_pct = 100.0 * collect_mask.sum() / len(act)
        ax.set_title(f"{name}\n{collect_pct:.1f}\\% collect", fontsize=10)
        ax.set_xlim(0, GRID_SIZE[0])
        ax.set_ylim(0, GRID_SIZE[1])
        ax.set_xlabel("$x$ (m)")
        ax.set_ylabel("$y$ (m)")
        ieee_style.clean_axes(ax)

    # SF legend
    sf_patches = [mpatches.Patch(color=SF_COLORS[sf], label=f"SF{sf}")
                  for sf in range(7, 13)]
    action_patches = [
        mpatches.Patch(color="#FF7F00", label="Hover step"),
        mpatches.Patch(color="#888888", label="Move step"),
    ]
    axes[2].legend(handles=sf_patches + action_patches,
                   fontsize=8, loc="upper right", ncol=2,
                   title="Sensor SF / Action")

    fig.suptitle(
        "UAV Trajectories: DQN vs Greedy Baselines\n"
        r"($N=20$, $500\times500$ grid, seed 42; sensors coloured by final SF after ADR adaptation)",
        fontsize=10
    )
    plt.tight_layout()
    ieee_style.save(fig, OUTPUT_DIR / "fig_trajectory_comparison")
    plt.close()
    return dqn_traj, sfa_traj


# ==================== FIGURE 2: POSITION HEATMAP ====================

def plot_position_heatmap(model):
    """2D density heatmap of UAV positions across 5 seeds."""
    print("  Running heatmap episodes (5 seeds)...")
    agent_data = {"DQN": [], "SF-Aware Greedy": [], "Nearest Greedy": [], "Lawnmower": []}

    for seed in SEEDS:
        positions, _ = get_canonical_positions(seed)
        dqn  = run_dqn_trajectory(model, positions, seed)
        sfa  = run_greedy_trajectory(MaxThroughputGreedyV2, positions, seed)
        nrst = run_greedy_trajectory(NearestSensorGreedy,   positions, seed)
        lawn = run_greedy_trajectory(LawnmowerAgent,        positions, seed)
        agent_data["DQN"].append(dqn)
        agent_data["SF-Aware Greedy"].append(sfa)
        agent_data["Nearest Greedy"].append(nrst)
        agent_data["Lawnmower"].append(lawn)

    fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))
    cmaps = ["Blues", "Reds", "Greens", "Oranges"]

    for ax, (name, trajs), cmap in zip(axes, agent_data.items(), cmaps):
        all_x = np.concatenate([t["x"] for t in trajs])
        all_y = np.concatenate([t["y"] for t in trajs])
        h, xedge, yedge = np.histogram2d(all_x, all_y, bins=40,
                                          range=[[0, GRID_SIZE[0]],
                                                 [0, GRID_SIZE[1]]])
        ax.pcolormesh(xedge, yedge, h.T, cmap=cmap)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("$x$ (m)")
        ax.set_ylabel("$y$ (m)")
        # No grid overlay on heatmaps — only remove spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(False)

    fig.suptitle(
        "UAV Position Density Across 5 Seeds\n"
        r"($N=20$, $500\times500$; brighter = more time spent)",
        fontsize=10
    )
    plt.tight_layout()
    ieee_style.save(fig, OUTPUT_DIR / "fig_position_heatmap")
    plt.close()
    return agent_data


# ==================== FIGURE 3: SF DISTRIBUTION OVER TIME ====================

def plot_sf_dynamics(dqn_traj, sfa_traj):
    """Stacked area: number of sensors per SF over episode steps."""
    print("  Generating SF dynamics figure...")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

    for ax, (name, traj) in zip(axes,
            [("DQN", dqn_traj), ("SF-Aware Greedy", sfa_traj)]):
        sfs = range(7, 13)
        sf_series = {sf: [] for sf in sfs}
        for sf_d in traj["sf_dist"]:
            for sf in sfs:
                sf_series[sf].append(sf_d.get(sf, 0))
        steps = traj["steps"]
        stack = np.array([sf_series[sf] for sf in sfs])
        ax.stackplot(steps, stack,
                     labels=[f"SF{sf}" for sf in sfs],
                     colors=[SF_COLORS[sf] for sf in sfs],
                     alpha=0.85)
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("Episode step")
        ax.set_ylabel("Number of sensors")
        ax.set_xlim(0, max(steps))
        ax.set_ylim(0, NUM_SENSORS)
        ieee_style.clean_axes(ax)

    axes[1].legend(loc="upper right", fontsize=8, ncol=2)
    fig.suptitle(
        "Spreading Factor Distribution Over Episode\n"
        r"($N=20$, $500\times500$, seed 42; shows how UAV movement reshapes SF assignments)",
        fontsize=10
    )
    plt.tight_layout()
    ieee_style.save(fig, OUTPUT_DIR / "fig_sf_dynamics")
    plt.close()


# ==================== MAIN ====================

def main():
    print("=" * 60)
    print("Trajectory & SF Distribution Analysis")
    print("=" * 60)

    model = load_model()
    positions, initial_sfs = get_canonical_positions(VIZ_SEED)

    print("\n[1/3] Trajectory comparison...")
    dqn_traj, sfa_traj = plot_trajectory_comparison(model, positions, initial_sfs)

    print("\n[2/3] Position heatmap...")
    plot_position_heatmap(model)

    print("\n[3/3] SF dynamics...")
    plot_sf_dynamics(dqn_traj, sfa_traj)

    print("\nDone. Figures saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
