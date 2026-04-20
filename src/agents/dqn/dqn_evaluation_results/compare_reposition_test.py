"""
Three-way comparison: DQN-v1 (no starvation penalty) vs DQN-v2 (with starvation penalty)
vs Smart Greedy V2.

Both DQN models trained on 400×400, 20 sensors, 200k steps.
  v1: models/dqn_400_reposition_test/       (buffer_size/batch_size in TRAINING_CONFIG — bug)
  v2: models/dqn_400_reposition_test_v2/    (fixed: in HYPERPARAMS + penalty_starved=-30k)

Evaluation: 10 seeds, 400×400, 20 sensors.
Outputs:
  - CLI table (all metrics per seed + averages)
  - Trajectory PNG: 3 rows (agents) × 10 cols (seeds)
  - Per-seed bar chart: starved sensors

Author: ATILADE GABRIEL OKE
"""

import sys
import random
import json
import time
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import gymnasium
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

# ── Paths ─────────────────────────────────────────────────────────────────────
script_dir = Path(__file__).resolve().parent          # dqn_evaluation_results/
dqn_dir    = script_dir.parent                        # agents/dqn/
src_dir    = dqn_dir.parent.parent                    # src/

sys.path.insert(0, str(dqn_dir))
sys.path.insert(0, str(src_dir))

from environment.uav_env import UAVEnvironment
from greedy_agents import MaxThroughputGreedyV2

OUTPUT_DIR = script_dir / "baseline_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_V1_DIR = dqn_dir / "models" / "dqn_400_reposition_test"
MODEL_V2_DIR = dqn_dir / "models" / "dqn_400_reposition_test_v2"
MODEL_V3_DIR = dqn_dir / "models" / "dqn_400_reposition_test_v3"

# ── Eval config ───────────────────────────────────────────────────────────────
GRID        = (400, 400)
N_SENSORS   = 20
MAX_STEPS   = 2100
MAX_BATTERY = 274.0
SEEDS       = [42, 123, 777, 2024, 9999]

MAX_SENSORS_LIMIT = 50   # must match training


# ── Env wrapper (zero-padding + pre-reset snapshot) ───────────────────────────

class EvalEnv(UAVEnvironment):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        raw = self.observation_space.shape[0]
        fps = 0
        for uav_f in range(raw + 1):
            rem = raw - uav_f
            if rem > 0 and rem % self.num_sensors == 0:
                fps = rem // self.num_sensors
                break
        if fps == 0:
            raise ValueError(f"Cannot detect fps: raw={raw}, n={self.num_sensors}")
        self._fps = fps
        padded = raw + (MAX_SENSORS_LIMIT - self.num_sensors) * fps
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(padded,), dtype=np.float32
        )
        self._raw = raw
        self.last_sensor_data = None

    def _pad(self, obs):
        return np.concatenate(
            [obs, np.zeros((MAX_SENSORS_LIMIT - self.num_sensors) * self._fps, dtype=np.float32)]
        ).astype(np.float32)

    def reset(self, **kwargs):
        if hasattr(self, "sensors") and self.current_step > 0:
            self.last_sensor_data = [
                {
                    "sensor_id":              int(s.sensor_id),
                    "position":               list(s.position),
                    "total_data_generated":   float(s.total_data_generated),
                    "total_data_transmitted": float(s.total_data_transmitted),
                    "total_data_lost":        float(s.total_data_lost),
                    "data_buffer":            float(s.data_buffer),
                    "max_buffer_size":        float(s.max_buffer_size),
                }
                for s in self.sensors
            ]
        obs, info = super().reset(**kwargs)
        # Random UAV start position every episode
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


def _jains(rates):
    if not rates or all(r == 0 for r in rates):
        return 0.0
    n  = len(rates)
    s  = sum(rates)
    s2 = sum(r * r for r in rates)
    return (s * s) / (n * s2) if s2 > 0 else 0.0


def _sensor_stats(sensor_data):
    """Return (jains, starved_count, min_cr, max_cr) from sensor snapshot list."""
    crs = []
    for s in sensor_data:
        gen = s["total_data_generated"]
        tx  = s["total_data_transmitted"]
        crs.append((tx / gen * 100.0) if gen > 0 else 0.0)
    starved = sum(1 for c in crs if c < 20.0)
    return _jains(crs), starved, min(crs) if crs else 0.0, max(crs) if crs else 0.0


# ── DQN runner ────────────────────────────────────────────────────────────────

def load_dqn(model_dir: Path):
    config_path = model_dir / "training_config.json"
    try:
        with open(config_path) as f:
            cfg = json.load(f)
        n_stack = cfg.get("n_stack", 4)
    except FileNotFoundError:
        print(f"  ⚠ No training_config.json in {model_dir.name} — using n_stack=4")
        n_stack = 4

    model_path = model_dir / "dqn_final.zip"
    if not model_path.exists():
        model_path = model_dir / "best_model" / "best_model.zip"
    if not model_path.exists():
        raise FileNotFoundError(f"No model zip found in {model_dir}")

    print(f"  Loading {model_path.name} from {model_dir.name}")
    model = DQN.load(str(model_path))
    return model, n_stack


def run_dqn(model, n_stack, seed):
    np.random.seed(seed)
    random.seed(seed)

    def _make():
        e = EvalEnv(
            grid_size=GRID, num_sensors=N_SENSORS, max_steps=MAX_STEPS,
            path_loss_exponent=3.8, rssi_threshold=-85.0,
            sensor_duty_cycle=10.0, max_battery=MAX_BATTERY, render_mode=None,
        )
        return e

    vec = DummyVecEnv([_make])
    vec = VecFrameStack(vec, n_stack=n_stack)

    # Get base env reference
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
    positions = [tuple(base_env.uav.position)]
    sensor_positions = [list(s.position) for s in base_env.sensors]
    final_snapshot = None
    steps = 0

    while True:
        action, _ = model.predict(obs, deterministic=True)
        action = int(action[0]) if isinstance(action, np.ndarray) else int(action)

        pre_pos      = tuple(base_env.uav.position)
        pre_battery  = base_env.uav.battery
        pre_coverage = len(base_env.sensors_visited) / base_env.num_sensors * 100
        pre_data     = base_env.total_data_collected

        pre_snap = [
            {
                "sensor_id":              int(s.sensor_id),
                "position":               list(s.position),
                "total_data_generated":   float(s.total_data_generated),
                "total_data_transmitted": float(s.total_data_transmitted),
                "total_data_lost":        float(s.total_data_lost),
                "data_buffer":            float(s.data_buffer),
                "max_buffer_size":        float(s.max_buffer_size),
            }
            for s in base_env.sensors
        ]

        obs, _, dones, _ = vec.step([action])
        done = bool(dones[0])
        steps += 1
        positions.append(pre_pos)

        if done:
            final_snapshot = pre_snap
            energy = MAX_BATTERY - pre_battery
            efficiency = (pre_data / energy) if energy > 0 else 0.0
            jains, starved, min_cr, max_cr = _sensor_stats(final_snapshot)
            visited_ids = {s["sensor_id"] for s in final_snapshot
                           if s["total_data_transmitted"] > 0}
            vec.close()
            return {
                "coverage": pre_coverage,
                "bytes":    pre_data,
                "bph":      efficiency,
                "jains":    jains,
                "starved":  starved,
                "min_cr":   min_cr,
                "max_cr":   max_cr,
                "positions": positions,
                "sensor_positions": sensor_positions,
                "visited_ids": visited_ids,
                "grid":     GRID,
            }


# ── Greedy runner ─────────────────────────────────────────────────────────────

def run_greedy(seed):
    np.random.seed(seed)
    random.seed(seed)

    env = EvalEnv(
        grid_size=GRID, num_sensors=N_SENSORS, max_steps=MAX_STEPS,
        path_loss_exponent=3.8, rssi_threshold=-85.0,
        sensor_duty_cycle=10.0, max_battery=MAX_BATTERY, render_mode=None,
    )
    obs, _ = env.reset(seed=seed)
    agent = MaxThroughputGreedyV2(env)
    done = False
    positions = [tuple(env.uav.position)]

    while not done:
        action = agent.select_action(obs)
        obs, _, done, trunc, _ = env.step(action)
        positions.append(tuple(env.uav.position))
        if trunc:
            done = True

    sensor_data = [
        {
            "sensor_id":              int(s.sensor_id),
            "total_data_generated":   float(s.total_data_generated),
            "total_data_transmitted": float(s.total_data_transmitted),
            "total_data_lost":        float(s.total_data_lost),
        }
        for s in env.sensors
    ]
    energy = MAX_BATTERY - env.uav.battery
    data   = env.total_data_collected
    coverage = len(env.sensors_visited) / env.num_sensors * 100

    jains, starved, min_cr, max_cr = _sensor_stats(sensor_data)
    sensor_positions = [list(s.position) for s in env.sensors]
    return {
        "coverage": coverage,
        "bytes":    data,
        "bph":      data / energy if energy > 0 else 0.0,
        "jains":    jains,
        "starved":  starved,
        "min_cr":   min_cr,
        "max_cr":   max_cr,
        "positions": positions,
        "sensor_positions": sensor_positions,
        "visited_ids": {s["sensor_id"] for s in sensor_data
                        if s["total_data_transmitted"] > 0},
        "grid":     GRID,
    }


# ── Trajectory plot ───────────────────────────────────────────────────────────

def plot_trajectories(all_results, seeds, labels, colors):
    """
    N rows (agents) × 10 cols (seeds).
    Path coloured by time (viridis). Green=start, red=end.
    Sensors: blue=visited (CR>0), orange=unvisited.
    """
    n_agents = len(all_results)
    n_seeds  = len(seeds)
    fig, axes = plt.subplots(n_agents, n_seeds,
                             figsize=(3.2 * n_seeds, 3.8 * n_agents))

    cmap = matplotlib.colormaps["viridis"]

    for row, (res_list, label, color) in enumerate(zip(all_results, labels, colors)):
        for col, (res, seed) in enumerate(zip(res_list, seeds)):
            ax = axes[row][col]
            W, H = res["grid"]
            ax.set_xlim(0, W)
            ax.set_ylim(0, H)
            ax.set_aspect("equal")
            ax.set_xticks([])
            ax.set_yticks([])

            # Sensors: blue=visited, orange=unvisited
            s_pos     = np.array(res["sensor_positions"])
            s_visited = res.get("visited_ids", set())
            s_colors  = ["#1f77b4" if i in s_visited else "#ff7f0e"
                         for i in range(len(s_pos))]
            ax.scatter(s_pos[:, 0], s_pos[:, 1],
                       c=s_colors, s=40, marker="s", zorder=3, linewidths=0)

            # Trajectory coloured by time
            pos = np.array(res["positions"])
            if len(pos) > 1:
                n = len(pos) - 1
                for i in range(n):
                    ax.plot([pos[i, 0], pos[i+1, 0]],
                            [pos[i, 1], pos[i+1, 1]],
                            color=cmap(i / n), linewidth=0.8, alpha=0.7)
                ax.scatter(*pos[0],  c="green", s=70, zorder=5, marker="^")
                ax.scatter(*pos[-1], c="red",   s=70, zorder=5, marker="X")

            ax.set_title(
                f"s={seed}\nCov:{res['coverage']:.0f}% St:{res['starved']}",
                fontsize=7
            )

            if col == 0:
                ax.set_ylabel(label, fontsize=9, fontweight="bold")

    # Shared legend
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    legend_elements = [
        Line2D([0], [0], marker="^", color="w", markerfacecolor="green",  markersize=8, label="Start"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="red",    markersize=8, label="End"),
        Patch(facecolor="#1f77b4", label="Visited sensor"),
        Patch(facecolor="#ff7f0e", label="Unvisited sensor"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, -0.01))

    plt.suptitle(
        "DQN-v1 vs DQN-v2 vs DQN-v3 vs Smart Greedy V2 — Trajectories (400×400, 20 sensors, random UAV start)",
        fontsize=11, fontweight="bold", y=1.01
    )
    plt.tight_layout()
    out = OUTPUT_DIR / "reposition_4way_trajectories.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nTrajectory plot saved: {out}")


# ── CLI table ─────────────────────────────────────────────────────────────────

def print_table(seeds, results_v1, results_v2, results_v3, results_g):
    SEP  = "├──────┼──────────────────────────┼──────────┼────────┼───────┼────────┼───────┼───────┼─────────┤"
    TOP  = "┌──────┬──────────────────────────┬──────────┬────────┬───────┬────────┬───────┬───────┬─────────┐"
    BOT  = "└──────┴──────────────────────────┴──────────┴────────┴───────┴────────┴───────┴───────┴─────────┘"
    HDR  = "│ Seed │ Agent                    │ Coverage │  Data  │ B/Wh  │ Jain's │ MinCR │ MaxCR │ Starved │"
    BSEP = "├──────┼──────────────────────────┼──────────┼────────┼───────┼────────┼───────┼───────┼─────────┤"

    agents = [
        ("DQN-v1 (no pen, def HP)", results_v1),
        ("DQN-v2 (pen, small HP) ", results_v2),
        ("DQN-v3 (pen, def HP)   ", results_v3),
        ("Smart Greedy V2        ", results_g),
    ]

    print(TOP)
    print(HDR)
    print(BSEP)

    all_results = list(zip(results_v1, results_v2, results_v3, results_g))

    for i, seed in enumerate(seeds):
        for j, (label, res_list) in enumerate(agents):
            r = res_list[i]
            seed_col = f" {seed:<4} " if j == 0 else "      "
            print(
                f"│{seed_col}│ {label} │"
                f" {r['coverage']:>7.0f}% │"
                f" {r['bytes']:>6.0f} │"
                f" {r['bph']:>5.1f} │"
                f" {r['jains']:>6.3f} │"
                f" {r['min_cr']:>5.1f}% │"
                f" {r['max_cr']:>5.1f}% │"
                f" {r['starved']:>2}/{N_SENSORS}   │"
            )
        if i < len(seeds) - 1:
            print(SEP)

    print(BSEP)

    # Averages
    for j, (label, res_list) in enumerate(agents):
        avg_cov  = np.mean([r['coverage'] for r in res_list])
        avg_b    = np.mean([r['bytes']    for r in res_list])
        avg_bph  = np.mean([r['bph']      for r in res_list])
        avg_j    = np.mean([r['jains']    for r in res_list])
        avg_min  = np.mean([r['min_cr']   for r in res_list])
        avg_max  = np.mean([r['max_cr']   for r in res_list])
        avg_st   = np.mean([r['starved']  for r in res_list])
        seed_col = " AVG  " if j == 0 else "      "
        print(
            f"│{seed_col}│ {label} │"
            f" {avg_cov:>7.1f}% │"
            f" {avg_b:>6.0f} │"
            f" {avg_bph:>5.1f} │"
            f" {avg_j:>6.3f} │"
            f" {avg_min:>5.1f}% │"
            f" {avg_max:>5.1f}% │"
            f" {avg_st:>4.1f}/20 │"
        )

    print(BOT)
    print(
        "\nv1 = no starvation penalty, default HP (buffer=150k, batch=256)"
        "\nv2 = starvation penalty -30k, small HP  (buffer=50k,  batch=64)"
        "\nv3 = starvation penalty -30k, default HP (buffer=150k, batch=256)"
    )


# ── Starved bar chart ─────────────────────────────────────────────────────────

def plot_starved_bars(seeds, results_v1, results_v2, results_v3, results_g):
    x = np.arange(len(seeds))
    w = 0.20
    fig, ax = plt.subplots(figsize=(14, 4))

    ax.bar(x - 1.5*w, [r["starved"] for r in results_v1], w, label="DQN-v1 (no penalty, default HP)", color="#4e79a7")
    ax.bar(x - 0.5*w, [r["starved"] for r in results_v2], w, label="DQN-v2 (penalty, small HP)",      color="#f28e2b")
    ax.bar(x + 0.5*w, [r["starved"] for r in results_v3], w, label="DQN-v3 (penalty, default HP)",    color="#e15759")
    ax.bar(x + 1.5*w, [r["starved"] for r in results_g],  w, label="Smart Greedy V2",                 color="#59a14f")

    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seeds], rotation=30, fontsize=9)
    ax.set_xlabel("Random Seed")
    ax.set_ylabel("Sensors with CR < 20%")
    ax.set_title("Starved Sensors (<20% CR) — v1 vs v2 vs v3 vs Smart Greedy V2")
    ax.legend(fontsize=8)
    ax.set_ylim(0, N_SENSORS + 1)
    plt.tight_layout()

    out = OUTPUT_DIR / "reposition_starved_bars.png"
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"Starved bar chart saved: {out}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("THREE-WAY COMPARISON: DQN-v1 vs DQN-v2 vs Smart Greedy V2")
    print(f"Grid: {GRID}  |  Sensors: {N_SENSORS}  |  Seeds: {len(SEEDS)}")
    print("=" * 70)

    # Load both DQN models
    print("\n[1/3] Loading DQN models...")
    model_v1, n_stack_v1 = load_dqn(MODEL_V1_DIR)
    if not MODEL_V2_DIR.exists():
        print(f"\n⚠  {MODEL_V2_DIR.name} does not exist yet — training still running?")
        print("   Re-run this script once training completes.")
        return

    model_v2, n_stack_v2 = load_dqn(MODEL_V2_DIR)

    print("\n[2/4] Loading DQN-v3...")
    if not MODEL_V3_DIR.exists():
        print(f"  ⚠  {MODEL_V3_DIR.name} does not exist yet — training still running?")
        print("     Re-run once training completes.")
        return
    model_v3, n_stack_v3 = load_dqn(MODEL_V3_DIR)

    print("\n[3/4] Running evaluations across 10 seeds...")
    results_v1, results_v2, results_v3, results_g = [], [], [], []

    for i, seed in enumerate(SEEDS):
        print(f"\n  Seed {seed} ({i+1}/{len(SEEDS)})")
        print(f"    DQN-v1...", end=" ", flush=True)
        r1 = run_dqn(model_v1, n_stack_v1, seed)
        print(f"Cov={r1['coverage']:.0f}%  Bytes={r1['bytes']:.0f}  Starved={r1['starved']}")

        print(f"    DQN-v2...", end=" ", flush=True)
        r2 = run_dqn(model_v2, n_stack_v2, seed)
        print(f"Cov={r2['coverage']:.0f}%  Bytes={r2['bytes']:.0f}  Starved={r2['starved']}")

        print(f"    DQN-v3...", end=" ", flush=True)
        r3 = run_dqn(model_v3, n_stack_v3, seed)
        print(f"Cov={r3['coverage']:.0f}%  Bytes={r3['bytes']:.0f}  Starved={r3['starved']}")

        print(f"    Greedy...", end=" ", flush=True)
        rg = run_greedy(seed)
        print(f"Cov={rg['coverage']:.0f}%  Bytes={rg['bytes']:.0f}  Starved={rg['starved']}")

        results_v1.append(r1)
        results_v2.append(r2)
        results_v3.append(r3)
        results_g.append(rg)

    print("\n[4/4] Generating outputs...")
    print_table(SEEDS, results_v1, results_v2, results_v3, results_g)

    plot_starved_bars(SEEDS, results_v1, results_v2, results_v3, results_g)
    plot_trajectories(
        [results_v1, results_v2, results_v3, results_g],
        SEEDS,
        labels=[
            "DQN-v1\n(no penalty, default HP)",
            "DQN-v2\n(penalty, small HP)",
            "DQN-v3\n(penalty, default HP)",
            "Smart Greedy V2",
        ],
        colors=["#4e79a7", "#f28e2b", "#e15759", "#59a14f"],
    )

    print("\nDone. Output files:")
    print(f"  {OUTPUT_DIR / 'reposition_3way_trajectories.png'}")
    print(f"  {OUTPUT_DIR / 'reposition_starved_bars.png'}")


if __name__ == "__main__":
    main()
