"""
Model Shootout (1000×1000) — all available DQN models vs greedy baselines on large grid.

Tests generalisation to the hardest condition where hover strategy fails.
Auto-detects each model's expected observation size from its policy network.

Author: ATILADE GABRIEL OKE
"""

import sys
import random
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import time

script_dir = Path(__file__).resolve().parent
src_dir    = script_dir.parent.parent.parent
sys.path.insert(0, str(src_dir))

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import gymnasium
import ieee_style
ieee_style.apply()

from environment.uav_env import UAVEnvironment
from greedy_agents import MaxThroughputGreedyV2, NearestSensorGreedy

# ==================== CONFIG ====================

GRID_SIZE    = (1000, 1000)
N_SENSORS    = 20
MAX_STEPS    = 2100
PROJECT_ROOT = src_dir.parent
BASE_OUTPUT  = script_dir / "shootout_results_1000"
BASE_OUTPUT.mkdir(parents=True, exist_ok=True)

# 5 runs: (seed, start_position) — scaled for 1000×1000 grid
RUNS = [
    (42,   (500, 500)),   # centre
    (123,  (100, 100)),   # bottom-left corner
    (256,  (900, 900)),   # top-right corner
    (789,  (500, 100)),   # bottom-centre
    (1337, (100, 500)),   # left-centre
]

# Set per-run (overwritten in main loop)
SEED      = RUNS[0][0]
START_POS = RUNS[0][1]
OUTPUT_DIR = BASE_OUTPUT

# All "final" / "best" DQN checkpoints to evaluate
CANDIDATE_MODELS = [
    ("DQN-7 (Attention best)",    PROJECT_ROOT / "models/dqn_attention/best_model/best_model.zip",
                                  PROJECT_ROOT / "models/dqn_attention/training_config.json"),
    ("DQN-7 (Attention final)",   PROJECT_ROOT / "models/dqn_attention/dqn_final.zip",
                                  PROJECT_ROOT / "models/dqn_attention/training_config.json"),
    ("DQN DR-v2 (best)",          PROJECT_ROOT / "models/dqn_domain_rand/best_model/best_model.zip",
                                  PROJECT_ROOT / "models/dqn_domain_rand/training_config.json"),
    ("DQN DR-v2 (final)",         PROJECT_ROOT / "models/dqn_domain_rand/dqn_final.zip",
                                  PROJECT_ROOT / "models/dqn_domain_rand/training_config.json"),
    ("DQN DR-v1 (best)",          PROJECT_ROOT / "models/dqn_domain_rand_v1/best_model/best_model.zip",
                                  PROJECT_ROOT / "models/dqn_domain_rand_v1/training_config.json"),
    ("DQN DR-v1 (final)",         PROJECT_ROOT / "models/dqn_domain_rand_v1/dqn_final.zip",
                                  PROJECT_ROOT / "models/dqn_domain_rand_v1/training_config.json"),
    ("DQN Full-Obs (final)",      PROJECT_ROOT / "models/dqn_full_observability/dqn_final.zip",
                                  PROJECT_ROOT / "models/dqn_full_observability/frame_stacking_config.json"),
]


# ==================== ENV WRAPPER ====================

class ShootoutEnv(UAVEnvironment):
    """UAVEnvironment with zero-padding to match any model's obs size."""

    def __init__(self, target_obs_size: int, **kwargs):
        self.target_obs_size = target_obs_size
        super().__init__(**kwargs)
        raw = self.observation_space.shape[0]
        self._pad_len = max(0, target_obs_size - raw)
        padded = raw + self._pad_len
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(padded,), dtype=np.float32
        )
        print(f"  [ShootoutEnv] raw={raw}  pad={self._pad_len}  total={padded}")

    def _pad(self, obs):
        if self._pad_len == 0:
            return obs.astype(np.float32)
        return np.concatenate([obs, np.zeros(self._pad_len, dtype=np.float32)])

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        return self._pad(obs), info

    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        return self._pad(obs), r, term, trunc, info


# ==================== HELPERS ====================

def _unwrap(vec):
    inner = vec
    while hasattr(inner, "venv"):
        inner = inner.venv
    env = inner.envs[0]
    while hasattr(env, "env"):
        env = env.env
    return env


def _load_config(config_path):
    defaults = {"use_frame_stacking": True, "n_stack": 4, "max_sensors_limit": None}
    if Path(config_path).exists():
        with open(config_path) as f:
            cfg = json.load(f)
        defaults.update(cfg)
    return defaults


def _infer_obs_size(model_path):
    """Load model just to read its policy obs size, then discard."""
    m = DQN.load(str(model_path))
    size = m.observation_space.shape[0]
    del m
    return size


class TrajectoryTracker:
    def __init__(self):
        self.positions = []

    def record(self, x, y):
        self.positions.append((float(x), float(y)))

    def path_length(self):
        if len(self.positions) < 2:
            return 0.0
        arr = np.array(self.positions)
        return float(np.sum(np.sqrt(np.sum(np.diff(arr, axis=0) ** 2, axis=1))))

    def array(self):
        return np.array(self.positions) if self.positions else np.zeros((0, 2))


# ==================== RUN FUNCTIONS ====================

def run_dqn(label, model_path, config_path):
    cfg     = _load_config(config_path)
    n_stack = cfg.get("n_stack", 4)

    # Infer expected obs size directly from the saved model
    stacked_obs_size = _infer_obs_size(model_path)
    raw_target = stacked_obs_size // n_stack

    env_kwargs = dict(
        target_obs_size      = raw_target,
        grid_size            = GRID_SIZE,
        num_sensors          = N_SENSORS,
        max_steps            = MAX_STEPS,
        path_loss_exponent   = 3.8,
        rssi_threshold       = -85.0,
        sensor_duty_cycle    = 10.0,
        uav_start_position   = START_POS,
    )

    np.random.seed(SEED)
    random.seed(SEED)
    vec = DummyVecEnv([lambda: ShootoutEnv(**env_kwargs)])
    vec = VecFrameStack(vec, n_stack=n_stack)
    base = _unwrap(vec)
    base.np_random, _ = gymnasium.utils.seeding.np_random(SEED)
    np.random.seed(SEED)
    random.seed(SEED)

    model = DQN.load(str(model_path))
    obs   = vec.reset()
    tracker  = TrajectoryTracker()
    cum_rew  = 0.0
    steps    = 0
    done     = False

    # Pre-episode snapshot defaults (overwritten each step before auto-reset fires)
    snap_visited      = set()
    snap_data         = 0.0
    snap_battery_left = base.uav.max_battery
    snap_sensors      = []

    while not done:
        pre_pos = tuple(base.uav.position)
        tracker.record(*pre_pos)

        # Capture state BEFORE step (auto-reset fires inside vec.step on done)
        snap_visited      = set(base.sensors_visited)
        snap_data         = base.total_data_collected
        snap_battery_left = base.uav.battery
        snap_sensors      = [
            (s.total_data_generated, s.total_data_transmitted)
            for s in base.sensors
        ]

        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, _ = vec.step(action)
        cum_rew += float(rewards[0])
        steps   += 1
        done     = bool(dones[0])

    ndr   = len(snap_visited) / base.num_sensors * 100
    rates = []
    for gen, tx in snap_sensors:
        rates.append(tx / gen * 100 if gen > 0 else 0.0)
    j      = sum(r for r in rates) ** 2 / (len(rates) * sum(r ** 2 for r in rates)) if any(r > 0 for r in rates) else 0.0
    energy = base.uav.max_battery - snap_battery_left
    eff    = snap_data / energy if energy > 0 else 0.0
    path   = tracker.path_length()

    return {
        "label":   label,
        "reward":  cum_rew,
        "ndr":     ndr,
        "jains":   j,
        "data":    snap_data,
        "eff":     eff,
        "path":    path,
        "steps":   steps,
        "tracker": tracker,
        "sensors": [(s.position[0], s.position[1]) for s in base.sensors],
    }


def run_greedy(label, AgentClass):
    np.random.seed(SEED)
    random.seed(SEED)
    env = UAVEnvironment(
        grid_size          = GRID_SIZE,
        num_sensors        = N_SENSORS,
        max_steps          = MAX_STEPS,
        path_loss_exponent = 3.8,
        rssi_threshold     = -85.0,
        sensor_duty_cycle  = 10.0,
        uav_start_position = START_POS,
    )
    obs, _ = env.reset(seed=SEED)
    agent   = AgentClass(env)
    tracker = TrajectoryTracker()
    cum_rew = 0.0
    done    = False

    while not done:
        tracker.record(*env.uav.position)
        action = agent.select_action(obs)
        obs, reward, term, trunc, _ = env.step(action)
        cum_rew += reward
        done = term or trunc

    ndr   = len(env.sensors_visited) / env.num_sensors * 100
    rates = []
    for s in env.sensors:
        if s.total_data_generated > 0:
            rates.append(s.total_data_transmitted / s.total_data_generated * 100)
        else:
            rates.append(0.0)
    j = sum(r for r in rates) ** 2 / (len(rates) * sum(r ** 2 for r in rates)) if any(r > 0 for r in rates) else 0.0
    energy = env.uav.max_battery - env.uav.battery
    eff    = env.total_data_collected / energy if energy > 0 else 0.0
    path   = tracker.path_length()

    return {
        "label":   label,
        "reward":  cum_rew,
        "ndr":     ndr,
        "jains":   j,
        "data":    env.total_data_collected,
        "eff":     eff,
        "path":    path,
        "steps":   len(tracker.positions),
        "tracker": tracker,
        "sensors": [(s.position[0], s.position[1]) for s in env.sensors],
    }


# ==================== PLOTTING ====================

COLORS = [
    "#1b9e77", "#d95f02", "#7570b3", "#e7298a",
    "#66a61e", "#e6ab02", "#a6761d", "#666666",
    "#1f78b4", "#33a02c",
]


def plot_trajectories(results):
    n   = len(results)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.5 * rows))
    axes = np.array(axes).flatten()

    sensor_positions = results[0]["sensors"]
    sx = [p[0] for p in sensor_positions]
    sy = [p[1] for p in sensor_positions]

    for i, (res, ax, color) in enumerate(zip(results, axes, COLORS)):
        ax.scatter(sx, sy, c="#d95f02", s=50, marker="s", zorder=2)
        traj = res["tracker"].array()
        if len(traj) > 1:
            ax.plot(traj[:, 0], traj[:, 1], color=color, lw=1.2,
                    drawstyle="steps-post", alpha=0.85)
        if len(traj) > 0:
            ax.scatter(*traj[0], marker="^", s=120, color="green",
                       zorder=5, label="Start")
            ax.scatter(*traj[-1], marker="*", s=180, color=color,
                       zorder=5, label="End")
        ax.set_xlim(0, GRID_SIZE[0])
        ax.set_ylim(0, GRID_SIZE[1])
        ax.set_aspect("equal")
        ax.set_title(res["label"], fontsize=8)
        ax.text(0.02, 0.98,
                f"Path: {res['path']:.0f}m\nNDR: {res['ndr']:.0f}%\nJ: {res['jains']:.3f}",
                transform=ax.transAxes, fontsize=7, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
        ax.set_xlabel("x (m)", fontsize=7)
        ax.set_ylabel("y (m)", fontsize=7)

    for ax in axes[len(results):]:
        ax.set_visible(False)

    fig.suptitle(
        f"Model Shootout — 1000×1000, N=20, seed={SEED}, start=({START_POS[0]},{START_POS[1]})",
        fontsize=10,
    )
    fig.tight_layout()
    ieee_style.save(fig, OUTPUT_DIR / "shootout_trajectories")
    plt.close(fig)
    print(f"  Saved: shootout_trajectories.png")


def plot_bar_comparison(results):
    labels  = [r["label"] for r in results]
    ndrs    = [r["ndr"]    for r in results]
    jains   = [r["jains"]  for r in results]
    paths   = [r["path"]   for r in results]

    x    = np.arange(len(labels))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, vals, title, unit in zip(
        axes,
        [ndrs, jains, paths],
        ["Network Discovery Rate", "Jain's Fairness Index", "Path Length"],
        ["%", "", "m"],
    ):
        bars = ax.bar(x, vals, color=COLORS[:len(labels)], edgecolor="white", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=7)
        ax.set_title(title, fontsize=9)
        ax.set_ylabel(unit, fontsize=8)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005 * max(vals),
                    f"{v:.1f}", ha="center", va="bottom", fontsize=6.5)

    fig.tight_layout()
    ieee_style.save(fig, OUTPUT_DIR / "shootout_bars")
    plt.close(fig)
    print(f"  Saved: shootout_bars.png")


# ==================== MAIN ====================

def run_one(seed, start_pos, output_dir):
    global SEED, START_POS, OUTPUT_DIR
    SEED       = seed
    START_POS  = start_pos
    OUTPUT_DIR = output_dir
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"SEED={seed}  START={start_pos}  → {output_dir.name}")
    print("=" * 70)

    results = []

    for label, model_path, config_path in CANDIDATE_MODELS:
        if not model_path.exists():
            print(f"  [SKIP] {label}")
            continue
        print(f"\n  {label}...")
        t0 = time.time()
        try:
            res = run_dqn(label, model_path, config_path)
            results.append(res)
            print(f"    reward={res['reward']:.0f}  NDR={res['ndr']:.0f}%  "
                  f"J={res['jains']:.4f}  path={res['path']:.0f}m  ({time.time()-t0:.1f}s)")
        except Exception as e:
            print(f"    [ERROR] {e}")

    for label, AgentClass in [
        ("SF-Aware Greedy V2",    MaxThroughputGreedyV2),
        ("Nearest Sensor Greedy", NearestSensorGreedy),
    ]:
        print(f"\n  {label}...")
        t0 = time.time()
        res = run_greedy(label, AgentClass)
        results.append(res)
        print(f"    reward={res['reward']:.0f}  NDR={res['ndr']:.0f}%  "
              f"J={res['jains']:.4f}  path={res['path']:.0f}m  ({time.time()-t0:.1f}s)")

    print("\n" + "-" * 70)
    print(f"{'Agent':<28} {'NDR':>6} {'Jains':>7} {'Data(B)':>9} {'Path(m)':>8}")
    print("-" * 70)
    for r in results:
        print(f"{r['label']:<28} {r['ndr']:>5.1f}% {r['jains']:>7.4f} "
              f"{r['data']:>9.0f} {r['path']:>8.0f}")

    plot_trajectories(results)
    plot_bar_comparison(results)
    return results


def main():
    all_runs = []
    for i, (seed, start_pos) in enumerate(RUNS, 1):
        folder = BASE_OUTPUT / f"run_{i:02d}_seed{seed}_start{start_pos[0]}_{start_pos[1]}"
        results = run_one(seed, start_pos, folder)
        all_runs.append((seed, start_pos, results))
        print(f"\n✓ Run {i}/5 complete — saved to {folder.name}\n")

    print("\n" + "=" * 70)
    print("ALL RUNS COMPLETE")
    print(f"Results in: {BASE_OUTPUT}")


if __name__ == "__main__":
    main()
