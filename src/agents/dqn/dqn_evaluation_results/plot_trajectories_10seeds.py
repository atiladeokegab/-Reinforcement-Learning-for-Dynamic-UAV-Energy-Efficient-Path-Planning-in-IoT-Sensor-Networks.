"""
Recreates trajectories_10seeds.png:
  Top row    — DQN-400 (dqn_400_reposition_test, 200k steps)
  Bottom row — Smart Greedy V2
  10 columns — one per seed

400×400 grid, 20 sensors.
Path coloured by time (viridis). Green triangle=start, red X=end.
Blue squares=visited sensors, orange squares=unvisited.

Output: baseline_results/trajectories_10seeds.png
"""

import sys
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import gymnasium
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

script_dir = Path(__file__).resolve().parent
dqn_dir    = script_dir.parent
src_dir    = dqn_dir.parent.parent

sys.path.insert(0, str(dqn_dir))
sys.path.insert(0, str(src_dir))

from environment.uav_env import UAVEnvironment
from greedy_agents import MaxThroughputGreedyV2

OUTPUT_DIR    = script_dir / "baseline_results"
MODEL_DIR     = dqn_dir / "models" / "dqn_400_reposition_test"
GRID          = (400, 400)
N_SENSORS     = 20
MAX_STEPS     = 2100
MAX_BATTERY   = 274.0
MAX_SENSORS_LIMIT = 50
SEEDS         = [42, 123, 456, 789, 1337, 2024, 3141, 5555, 8888, 9999]


# ── Env wrapper ───────────────────────────────────────────────────────────────

class EvalEnv(UAVEnvironment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        raw = self.observation_space.shape[0]
        fps = next(
            (rem // self.num_sensors
             for uav_f in range(raw + 1)
             if (rem := raw - uav_f) > 0 and rem % self.num_sensors == 0),
            0,
        )
        if fps == 0:
            raise ValueError(f"Cannot detect fps: raw={raw}, n={self.num_sensors}")
        self._fps = fps
        padded = raw + (MAX_SENSORS_LIMIT - self.num_sensors) * fps
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(padded,), dtype=np.float32
        )

    def _pad(self, obs):
        return np.concatenate(
            [obs, np.zeros((MAX_SENSORS_LIMIT - self.num_sensors) * self._fps,
                           dtype=np.float32)]
        ).astype(np.float32)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        # Randomise UAV start position after env reset
        W, H = self.grid_size
        self.uav.position = np.array([
            float(self.np_random.uniform(0, W)),
            float(self.np_random.uniform(0, H)),
        ])
        # Rebuild obs with new UAV position
        obs = self._get_observation()
        return self._pad(obs), info

    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        return self._pad(obs), r, term, trunc, info


def _make_env():
    return EvalEnv(
        grid_size=GRID, num_sensors=N_SENSORS, max_steps=MAX_STEPS,
        path_loss_exponent=3.8, rssi_threshold=-85.0,
        sensor_duty_cycle=10.0, max_battery=MAX_BATTERY, render_mode=None,
    )


# ── DQN run ───────────────────────────────────────────────────────────────────

def run_dqn(model, n_stack, seed):
    np.random.seed(seed)
    random.seed(seed)

    vec = DummyVecEnv([_make_env])
    vec = VecFrameStack(vec, n_stack=n_stack)

    base = vec
    while hasattr(base, "venv"):
        base = base.venv
    base_env = base.envs[0]
    while hasattr(base_env, "env"):
        base_env = base_env.env

    base_env.np_random, _ = gymnasium.utils.seeding.np_random(seed)
    np.random.seed(seed)
    random.seed(seed)

    obs = vec.reset()
    positions       = [tuple(base_env.uav.position)]
    sensor_positions = [tuple(s.position) for s in base_env.sensors]
    visited_ids     = set()

    while True:
        action, _ = model.predict(obs, deterministic=True)
        action     = int(action[0]) if isinstance(action, np.ndarray) else int(action)
        pre_pos    = tuple(base_env.uav.position)
        pre_snap   = {s.sensor_id: float(s.total_data_transmitted)
                      for s in base_env.sensors}

        obs, _, dones, _ = vec.step([action])
        done = bool(dones[0])

        positions.append(pre_pos)
        for s in base_env.sensors:
            if s.total_data_transmitted > 0 or pre_snap.get(s.sensor_id, 0) > 0:
                visited_ids.add(s.sensor_id)

        if done:
            # capture final visited set before reset
            final_visited = {s.sensor_id for s in base_env.sensors
                             if s.total_data_transmitted > 0}
            visited_ids |= final_visited
            vec.close()
            break

    return np.array(positions), sensor_positions, visited_ids


# ── Greedy run ────────────────────────────────────────────────────────────────

def run_greedy(seed):
    np.random.seed(seed)
    random.seed(seed)

    env = _make_env()
    obs, _ = env.reset(seed=seed)
    agent  = MaxThroughputGreedyV2(env)
    positions = [tuple(env.uav.position)]

    done = False
    while not done:
        action = agent.select_action(obs)
        obs, _, done, trunc, _ = env.step(action)
        positions.append(tuple(env.uav.position))
        if trunc:
            done = True

    sensor_positions = [tuple(s.position) for s in env.sensors]
    visited_ids      = {s.sensor_id for s in env.sensors
                        if s.total_data_transmitted > 0}
    return np.array(positions), sensor_positions, visited_ids


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot(dqn_results, greedy_results, seeds):
    n_seeds = len(seeds)
    fig, axes = plt.subplots(2, n_seeds, figsize=(3.2 * n_seeds, 7))
    cmap = cm.get_cmap("viridis")
    W, H = GRID

    rows = [
        (dqn_results,    "DQN-400 (200k steps)"),
        (greedy_results, "Smart Greedy V2"),
    ]

    for row_idx, (results, row_label) in enumerate(rows):
        for col_idx, (seed, (pos, s_pos, visited)) in enumerate(zip(seeds, results)):
            ax = axes[row_idx][col_idx]
            ax.set_xlim(0, W)
            ax.set_ylim(0, H)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])

            # Sensors — blue if visited, orange if not
            for i, (sx, sy) in enumerate(s_pos):
                color = "#1f77b4" if i in visited else "#ff7f0e"
                ax.scatter(sx, sy, c=color, s=45, marker="s",
                           zorder=3, linewidths=0)

            # Path coloured by time
            if len(pos) > 1:
                n = len(pos) - 1
                for i in range(n):
                    ax.plot([pos[i, 0], pos[i+1, 0]],
                            [pos[i, 1], pos[i+1, 1]],
                            color=cmap(i / n), linewidth=0.9, alpha=0.75)
                ax.scatter(*pos[0],  c="green", s=80, marker="^",
                           zorder=5, linewidths=0)
                ax.scatter(*pos[-1], c="red",   s=80, marker="X",
                           zorder=5, linewidths=0)

            unique = len(set(map(tuple, pos.tolist())))
            cov    = len(visited) / N_SENSORS * 100
            ax.set_title(
                f"seed={seed}\nNDR:{cov:.0f}%  pos:{unique}",
                fontsize=7, pad=3,
            )

            if col_idx == 0:
                ax.set_ylabel(row_label, fontsize=9, fontweight="bold")

    # Shared legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor="green",
               markersize=8, label="Start"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="red",
               markersize=8, label="End"),
        Patch(facecolor="#1f77b4", label="Visited sensor"),
        Patch(facecolor="#ff7f0e", label="Unvisited sensor"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle(
        "DQN-400 vs Smart Greedy V2 — Trajectories across 10 seeds (400×400, 20 sensors)",
        fontsize=12, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    out = OUTPUT_DIR / "trajectories_10seeds.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    import json
    print("Loading DQN model...")
    cfg_path = MODEL_DIR / "training_config.json"
    with open(cfg_path) as f:
        cfg = json.load(f)
    n_stack = cfg.get("n_stack", 4)
    model   = DQN.load(str(MODEL_DIR / "dqn_final.zip"))
    print(f"  n_stack={n_stack}")

    dqn_results    = []
    greedy_results = []

    for i, seed in enumerate(SEEDS):
        print(f"Seed {seed} ({i+1}/{len(SEEDS)})...", end=" ", flush=True)
        pos_d, sp_d, vis_d = run_dqn(model, n_stack, seed)
        pos_g, sp_g, vis_g = run_greedy(seed)
        dqn_results.append((pos_d, sp_d, vis_d))
        greedy_results.append((pos_g, sp_g, vis_g))
        print(f"DQN NDR={len(vis_d)/N_SENSORS*100:.0f}%  Greedy NDR={len(vis_g)/N_SENSORS*100:.0f}%")

    print("\nPlotting...")
    plot(dqn_results, greedy_results, SEEDS)
    print("Done.")


if __name__ == "__main__":
    main()
