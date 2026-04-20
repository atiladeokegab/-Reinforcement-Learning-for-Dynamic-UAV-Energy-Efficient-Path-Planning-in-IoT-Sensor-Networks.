"""
Baseline validation for the existing dqn_v3 (3M step model, pre-fix training)
using the corrected evaluation code:
  - Random UAV start position
  - CR bounded to [0,1] (total_data_generated fix)
  - Jain's Index from pre-step sensor snapshot (not post-reset)

Compares dqn_v3 against Smart Greedy V2.
Seeds: 42, 123, 777, 2024, 9999
Grid:  500×500, 20 sensors (standard dissertation evaluation conditions)

Output: CLI table + trajectory plot
"""

import sys
import random
import json
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

OUTPUT_DIR  = script_dir / "baseline_results"
MODEL_DIR   = dqn_dir / "models" / "dqn_v3"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

GRID            = (500, 500)
N_SENSORS       = 20
MAX_STEPS       = 2100
MAX_BATTERY     = 274.0
MAX_SENSORS_LIMIT = 50
SEEDS           = [42, 123, 777, 2024, 9999]


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
        W, H = self.grid_size
        self.uav.position = np.array([
            float(self.np_random.uniform(0, W)),
            float(self.np_random.uniform(0, H)),
        ])
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


def _jains(rates):
    if not rates or all(r == 0 for r in rates):
        return 0.0
    n = len(rates); s = sum(rates); s2 = sum(r*r for r in rates)
    return (s*s) / (n*s2) if s2 > 0 else 0.0


def _sensor_stats(snap):
    crs = [(s["total_data_transmitted"] / s["total_data_generated"] * 100)
           if s["total_data_generated"] > 0 else 0.0
           for s in snap]
    return _jains(crs), sum(1 for c in crs if c < 20), min(crs), max(crs)


# ── DQN runner ────────────────────────────────────────────────────────────────

def run_dqn(model, n_stack, seed):
    np.random.seed(seed); random.seed(seed)
    vec = DummyVecEnv([_make_env])
    vec = VecFrameStack(vec, n_stack=n_stack)

    base = vec
    while hasattr(base, "venv"): base = base.venv
    base_env = base.envs[0]
    while hasattr(base_env, "env"): base_env = base_env.env

    base_env.np_random, _ = gymnasium.utils.seeding.np_random(seed)
    np.random.seed(seed); random.seed(seed)

    obs = vec.reset()
    positions        = [tuple(base_env.uav.position)]
    sensor_positions = [tuple(s.position) for s in base_env.sensors]

    while True:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action[0]) if isinstance(action, np.ndarray) else int(action)

        pre_pos      = tuple(base_env.uav.position)
        pre_battery  = base_env.uav.battery
        pre_coverage = len(base_env.sensors_visited) / base_env.num_sensors * 100
        pre_data     = base_env.total_data_collected
        pre_snap     = [{"sensor_id": int(s.sensor_id),
                         "total_data_generated":   float(s.total_data_generated),
                         "total_data_transmitted": float(s.total_data_transmitted),
                         "total_data_lost":        float(s.total_data_lost)}
                        for s in base_env.sensors]

        obs, _, dones, _ = vec.step([action])
        done = bool(dones[0])
        positions.append(pre_pos)

        if done:
            energy = MAX_BATTERY - pre_battery
            jains, starved, min_cr, max_cr = _sensor_stats(pre_snap)
            visited_ids = {s["sensor_id"] for s in pre_snap
                           if s["total_data_transmitted"] > 0}
            vec.close()
            return {
                "coverage": pre_coverage, "bytes": pre_data,
                "bph": pre_data / energy if energy > 0 else 0.0,
                "jains": jains, "starved": starved,
                "min_cr": min_cr, "max_cr": max_cr,
                "positions": positions,
                "sensor_positions": sensor_positions,
                "visited_ids": visited_ids, "grid": GRID,
            }


# ── Greedy runner ─────────────────────────────────────────────────────────────

def run_greedy(seed):
    np.random.seed(seed); random.seed(seed)
    env = _make_env()
    obs, _ = env.reset(seed=seed)
    agent  = MaxThroughputGreedyV2(env)
    positions = [tuple(env.uav.position)]
    done = False

    while not done:
        action = agent.select_action(obs)
        obs, _, done, trunc, _ = env.step(action)
        positions.append(tuple(env.uav.position))
        if trunc: done = True

    snap = [{"sensor_id": int(s.sensor_id),
             "total_data_generated":   float(s.total_data_generated),
             "total_data_transmitted": float(s.total_data_transmitted),
             "total_data_lost":        float(s.total_data_lost)}
            for s in env.sensors]
    energy = MAX_BATTERY - env.uav.battery
    jains, starved, min_cr, max_cr = _sensor_stats(snap)
    return {
        "coverage": len(env.sensors_visited) / env.num_sensors * 100,
        "bytes": env.total_data_collected,
        "bph": env.total_data_collected / energy if energy > 0 else 0.0,
        "jains": jains, "starved": starved,
        "min_cr": min_cr, "max_cr": max_cr,
        "positions": positions,
        "sensor_positions": [tuple(s.position) for s in env.sensors],
        "visited_ids": {s["sensor_id"] for s in snap if s["total_data_transmitted"] > 0},
        "grid": GRID,
    }


# ── Table ─────────────────────────────────────────────────────────────────────

def print_table(seeds, dqn_results, greedy_results):
    TOP  = "┌──────┬────────────────┬──────────┬────────┬───────┬────────┬───────┬───────┬─────────┐"
    HDR  = "│ Seed │ Agent          │ Coverage │  Data  │ B/Wh  │ Jain's │ MinCR │ MaxCR │ Starved │"
    SEP  = "├──────┼────────────────┼──────────┼────────┼───────┼────────┼───────┼───────┼─────────┤"
    BOT  = "└──────┴────────────────┴──────────┴────────┴───────┴────────┴───────┴───────┴─────────┘"

    print(TOP); print(HDR); print(SEP)

    for i, seed in enumerate(seeds):
        d, g = dqn_results[i], greedy_results[i]
        for label, r in [("DQN-v3 (3M)", d), ("Smart Greedy", g)]:
            seed_col = f" {seed:<4} " if label == "DQN-v3 (3M)" else "      "
            print(f"│{seed_col}│ {label:<14} │"
                  f" {r['coverage']:>7.0f}% │ {r['bytes']:>6.0f} │"
                  f" {r['bph']:>5.1f} │ {r['jains']:>6.3f} │"
                  f" {r['min_cr']:>5.1f}% │ {r['max_cr']:>5.1f}% │"
                  f" {r['starved']:>2}/{N_SENSORS}   │")
        if i < len(seeds) - 1:
            print(SEP)

    print(SEP)
    for label, results in [("DQN-v3 (3M)", dqn_results), ("Smart Greedy", greedy_results)]:
        seed_col = " AVG  " if label == "DQN-v3 (3M)" else "      "
        print(f"│{seed_col}│ {label:<14} │"
              f" {np.mean([r['coverage'] for r in results]):>7.1f}% │"
              f" {np.mean([r['bytes']    for r in results]):>6.0f} │"
              f" {np.mean([r['bph']      for r in results]):>5.1f} │"
              f" {np.mean([r['jains']    for r in results]):>6.3f} │"
              f" {np.mean([r['min_cr']   for r in results]):>5.1f}% │"
              f" {np.mean([r['max_cr']   for r in results]):>5.1f}% │"
              f" {np.mean([r['starved']  for r in results]):>4.1f}/20 │")
    print(BOT)


# ── Trajectory plot ───────────────────────────────────────────────────────────

def plot_trajectories(seeds, dqn_results, greedy_results):
    fig, axes = plt.subplots(2, len(seeds), figsize=(3.2 * len(seeds), 7))
    cmap = matplotlib.colormaps["viridis"]
    W, H = GRID

    for row, (results, label) in enumerate([(dqn_results, "DQN-v3 (3M steps)"),
                                             (greedy_results, "Smart Greedy V2")]):
        for col, (seed, res) in enumerate(zip(seeds, results)):
            ax = axes[row][col]
            ax.set_xlim(0, W); ax.set_ylim(0, H)
            ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])

            s_pos    = np.array(res["sensor_positions"])
            visited  = res["visited_ids"]
            s_colors = ["#1f77b4" if i in visited else "#ff7f0e"
                        for i in range(len(s_pos))]
            ax.scatter(s_pos[:, 0], s_pos[:, 1], c=s_colors, s=40,
                       marker="s", zorder=3, linewidths=0)

            pos = np.array(res["positions"])
            if len(pos) > 1:
                n = len(pos) - 1
                for i in range(n):
                    ax.plot([pos[i,0], pos[i+1,0]], [pos[i,1], pos[i+1,1]],
                            color=cmap(i/n), linewidth=0.9, alpha=0.75)
                ax.scatter(*pos[0],  c="green", s=80, marker="^", zorder=5)
                ax.scatter(*pos[-1], c="red",   s=80, marker="X", zorder=5)

            ax.set_title(f"seed={seed}\nNDR:{res['coverage']:.0f}%  J:{res['jains']:.2f}",
                         fontsize=7, pad=3)
            if col == 0:
                ax.set_ylabel(label, fontsize=9, fontweight="bold")

    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    fig.legend(handles=[
        Line2D([0],[0], marker="^", color="w", markerfacecolor="green", markersize=8, label="Start"),
        Line2D([0],[0], marker="X", color="w", markerfacecolor="red",   markersize=8, label="End"),
        Patch(facecolor="#1f77b4", label="Visited sensor"),
        Patch(facecolor="#ff7f0e", label="Unvisited sensor"),
    ], loc="lower center", ncol=4, fontsize=9, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("DQN-v3 (3M steps, fixed eval) vs Smart Greedy V2\n"
                 "500×500 grid, 20 sensors, random UAV start",
                 fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    out = OUTPUT_DIR / "dqn_v3_baseline_trajectories.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"Trajectory plot saved: {out}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 60)
    print("BASELINE VALIDATION — dqn_v3 (3M steps, fixed eval code)")
    print(f"Grid: {GRID}  |  Sensors: {N_SENSORS}  |  Seeds: {SEEDS}")
    print("=" * 60)

    cfg = json.load(open(MODEL_DIR / "training_config.json"))
    n_stack = cfg.get("n_stack", 4)
    model   = DQN.load(str(MODEL_DIR / "dqn_final.zip"))
    print(f"Model loaded: dqn_v3  (n_stack={n_stack})\n")

    dqn_results, greedy_results = [], []
    for i, seed in enumerate(SEEDS):
        print(f"Seed {seed} ({i+1}/{len(SEEDS)})...", end=" ", flush=True)
        d = run_dqn(model, n_stack, seed)
        g = run_greedy(seed)
        dqn_results.append(d)
        greedy_results.append(g)
        print(f"DQN: {d['coverage']:.0f}% / {d['bytes']:.0f}B / J={d['jains']:.3f}  |  "
              f"Greedy: {g['coverage']:.0f}% / {g['bytes']:.0f}B / J={g['jains']:.3f}")

    print()
    print_table(SEEDS, dqn_results, greedy_results)
    plot_trajectories(SEEDS, dqn_results, greedy_results)
    print("\nDone.")


if __name__ == "__main__":
    main()
