"""
Dwell-time trajectory visualisation.

For each agent, plots the path taken overlaid on a dwell-time heatmap:
  - Each unique grid cell is shaded by how many timesteps the UAV spent there.
  - Darker cell  →  UAV lingered / hovered longer.
  - Thin line    →  actual trajectory (steps-post style).
  - Sensors shown as squares; visited sensors filled, unvisited hollow.

Configurable: GRID_SIZE, N_SENSORS, SEED, START_POS.

Author: ATILADE GABRIEL OKE
"""

import sys
import json
import random
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from collections import defaultdict

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

GRID_SIZE  = (500, 500)
N_SENSORS  = 20
MAX_STEPS  = 2100
SEED       = 42
START_POS  = (GRID_SIZE[0] // 2, GRID_SIZE[1] // 2)
CELL_SIZE  = max(1, GRID_SIZE[0] // 50)

RANDOM_START_SEED = 99   # fixed seed for reproducible random start positions
_START_LABEL      = "centre"


def get_start_positions(grid_size, n_random=10):
    """Return list of (label, (x, y)) for all start positions."""
    W, H  = grid_size
    m     = max(5, W // 20)   # margin so UAV starts just inside boundary
    positions = [
        ("corner_BL", (m,     m    )),
        ("corner_BR", (W - m, m    )),
        ("corner_TL", (m,     H - m)),
        ("corner_TR", (W - m, H - m)),
        ("wall_S",    (W // 2, m    )),
        ("wall_N",    (W // 2, H - m)),
        ("wall_W",    (m,      H // 2)),
        ("wall_E",    (W - m,  H // 2)),
    ]
    rng = np.random.default_rng(RANDOM_START_SEED)
    for i in range(n_random):
        x = int(rng.integers(m, W - m))
        y = int(rng.integers(m, H - m))
        positions.append((f"random_{i+1:02d}", (x, y)))
    return positions

PROJECT_ROOT = src_dir.parent
OUTPUT_DIR   = script_dir / "dwell_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CANDIDATE_MODELS = [
    ("DQN-Movement (best)",  PROJECT_ROOT / "models/dqn_movement/best_model/best_model.zip",
                              PROJECT_ROOT / "models/dqn_movement/training_config.json"),
    ("DQN-7 (Attn best)",    PROJECT_ROOT / "models/dqn_attention/best_model/best_model.zip",
                              PROJECT_ROOT / "models/dqn_attention/training_config.json"),
    ("DQN DR-v2 (best)",     PROJECT_ROOT / "models/dqn_domain_rand/best_model/best_model.zip",
                              PROJECT_ROOT / "models/dqn_domain_rand/training_config.json"),
    ("DQN DR-v2 (final)",    PROJECT_ROOT / "models/dqn_domain_rand/dqn_final.zip",
                              PROJECT_ROOT / "models/dqn_domain_rand/training_config.json"),
    ("DQN DR-v1 (best)",     PROJECT_ROOT / "models/dqn_domain_rand_v1/best_model/best_model.zip",
                              PROJECT_ROOT / "models/dqn_domain_rand_v1/training_config.json"),
    ("DQN Full-Obs",         PROJECT_ROOT / "models/dqn_full_observability/dqn_final.zip",
                              PROJECT_ROOT / "models/dqn_full_observability/frame_stacking_config.json"),
]

# ==================== ENV WRAPPER ====================

class PaddedEnv(UAVEnvironment):
    def __init__(self, target_obs_size, **kwargs):
        self.target_obs_size = target_obs_size
        super().__init__(**kwargs)
        raw = self.observation_space.shape[0]
        self._pad = max(0, target_obs_size - raw)
        padded = raw + self._pad
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(padded,), dtype=np.float32)

    def _p(self, obs):
        if self._pad == 0:
            return obs.astype(np.float32)
        return np.concatenate([obs, np.zeros(self._pad, dtype=np.float32)])

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        return self._p(obs), info

    def step(self, action):
        obs, r, term, trunc, info = super().step(action)
        return self._p(obs), r, term, trunc, info


# ==================== HELPERS ====================

def _unwrap(vec):
    inner = vec
    while hasattr(inner, "venv"):
        inner = inner.venv
    env = inner.envs[0]
    while hasattr(env, "env"):
        env = env.env
    return env


def _load_config(path):
    defaults = {"use_frame_stacking": True, "n_stack": 4, "max_sensors_limit": None}
    if Path(path).exists():
        with open(path) as f:
            defaults.update(json.load(f))
    return defaults


def _infer_obs_size(model_path):
    m = DQN.load(str(model_path))
    size = m.observation_space.shape[0]
    del m
    return size


def _build_dwell(positions, grid_size, cell_size):
    """Return 2-D dwell count array (rows=y-cells, cols=x-cells)."""
    nx = grid_size[0] // cell_size + 1
    ny = grid_size[1] // cell_size + 1
    grid = np.zeros((ny, nx), dtype=np.float32)
    for x, y in positions:
        xi = int(x) // cell_size
        yi = int(y) // cell_size
        xi = min(xi, nx - 1)
        yi = min(yi, ny - 1)
        grid[yi, xi] += 1
    return grid


# ==================== RUN FUNCTIONS ====================

def run_dqn(label, model_path, config_path):
    cfg     = _load_config(config_path)
    n_stack = cfg.get("n_stack", 4)
    stacked = _infer_obs_size(model_path)
    raw_target = stacked // n_stack

    env_kwargs = dict(
        target_obs_size    = raw_target,
        grid_size          = GRID_SIZE,
        num_sensors        = N_SENSORS,
        max_steps          = MAX_STEPS,
        path_loss_exponent = 3.8,
        rssi_threshold     = -85.0,
        sensor_duty_cycle  = 10.0,
        uav_start_position = START_POS,
    )

    np.random.seed(SEED); random.seed(SEED)
    vec  = DummyVecEnv([lambda: PaddedEnv(**env_kwargs)])
    vec  = VecFrameStack(vec, n_stack=n_stack)
    base = _unwrap(vec)
    base.np_random, _ = gymnasium.utils.seeding.np_random(SEED)
    np.random.seed(SEED); random.seed(SEED)

    model    = DQN.load(str(model_path))
    obs      = vec.reset()
    path     = []
    cum_rew  = 0.0
    done     = False

    snap_visited = set()
    snap_sensors = []

    while not done:
        path.append(tuple(base.uav.position))
        snap_visited = set(base.sensors_visited)
        snap_sensors = [(s.position, s.spreading_factor) for s in base.sensors]
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, _ = vec.step(action)
        cum_rew += float(rewards[0])
        done = bool(dones[0])

    return dict(label=label, path=path, visited=snap_visited,
                sensors=snap_sensors, reward=cum_rew,
                ndr=len(snap_visited)/N_SENSORS*100)


def run_greedy(label, AgentClass):
    np.random.seed(SEED); random.seed(SEED)
    env = UAVEnvironment(
        grid_size=GRID_SIZE, num_sensors=N_SENSORS, max_steps=MAX_STEPS,
        path_loss_exponent=3.8, rssi_threshold=-85.0,
        sensor_duty_cycle=10.0, uav_start_position=START_POS)
    obs, _ = env.reset(seed=SEED)
    agent  = AgentClass(env)
    path   = []
    cum_rew = 0.0
    done   = False

    while not done:
        path.append(tuple(env.uav.position))
        action = agent.select_action(obs)
        obs, reward, term, trunc, _ = env.step(action)
        cum_rew += reward
        done = term or trunc

    sensors = [(s.position, s.spreading_factor) for s in env.sensors]
    visited = env.sensors_visited
    return dict(label=label, path=path, visited=visited,
                sensors=sensors, reward=cum_rew,
                ndr=len(visited)/N_SENSORS*100)


# ==================== PLOTTING ====================

def plot_dwell(results):
    n    = len(results)
    cols = min(4, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 5 * rows))
    axes = np.array(axes).flatten()

    cmap_dwell = matplotlib.colormaps.get_cmap("Blues")    # white→dark blue, darker = more dwell
    cmap_dwell.set_under("white")

    extent = [0, GRID_SIZE[0], 0, GRID_SIZE[1]]

    for ax, res in zip(axes, results):
        path = np.array(res["path"])

        # ── Dwell heatmap ──────────────────────────────────────────────
        dwell = _build_dwell(res["path"], GRID_SIZE, CELL_SIZE)
        vmax  = max(dwell.max(), 1)
        im = ax.imshow(
            dwell,
            origin="lower",
            extent=extent,
            cmap=cmap_dwell,
            vmin=0.5,      # anything < 0.5 renders white (never visited)
            vmax=vmax,
            aspect="auto",
            alpha=0.75,
            interpolation="nearest",
        )
        plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02,
                     label="Steps at cell")

        # ── Trajectory path (magenta — distinct from blue dwell + green sensors) ──
        if len(path) > 1:
            ax.plot(path[:, 0], path[:, 1],
                    color="#e31a1c", lw=1.0, alpha=0.75,
                    drawstyle="steps-post", zorder=3)

        # ── Start / end markers ────────────────────────────────────────
        ax.scatter(*path[0],  marker="^", s=130, color="#f768a1",
                   zorder=6, label="Start", edgecolors="white", linewidths=0.5)
        ax.scatter(*path[-1], marker="*", s=200, color="#ffff33",
                   zorder=6, label="End",   edgecolors="black", linewidths=0.4)

        # ── Sensors: green filled if visited, grey hollow if not ───────
        visited_ids = res["visited"]
        for i, (pos, sf) in enumerate(res["sensors"]):
            visited = i in visited_ids
            ax.scatter(pos[0], pos[1],
                       marker="s", s=60,
                       facecolors="#33a02c" if visited else "none",
                       edgecolors="#33a02c" if visited else "#888888",
                       linewidths=1.4,
                       zorder=5)

        ax.set_xlim(0, GRID_SIZE[0])
        ax.set_ylim(0, GRID_SIZE[1])
        ax.set_aspect("equal")
        ax.set_title(res["label"], fontsize=8, fontweight="bold")

        path_len = sum(
            np.sqrt((b[0]-a[0])**2 + (b[1]-a[1])**2)
            for a, b in zip(res["path"], res["path"][1:])
        )
        hover_steps = int(dwell.max())
        ax.text(0.02, 0.98,
                f"NDR: {res['ndr']:.0f}%\n"
                f"Path: {path_len:.0f}m\n"
                f"Max dwell: {hover_steps} steps",
                transform=ax.transAxes, fontsize=7, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.85))
        ax.set_xlabel("x (m)", fontsize=7)
        ax.set_ylabel("y (m)", fontsize=7)

    for ax in axes[len(results):]:
        ax.set_visible(False)

    fig.suptitle(
        f"Dwell-Time Trajectory — {GRID_SIZE[0]}×{GRID_SIZE[1]}, N={N_SENSORS}, "
        f"seed={SEED}, start=({START_POS[0]},{START_POS[1]})\n"
        f"Green squares = visited sensors, grey hollow = unvisited. "
        f"Red line = path. Darker blue cell = UAV hovered longer.",
        fontsize=9,
    )
    fig.tight_layout()
    start_tag = globals().get("_START_LABEL", f"seed{SEED}")
    out = OUTPUT_DIR / f"dwell_start_{start_tag}"
    ieee_style.save(fig, out)
    plt.close(fig)
    print(f"Saved: {out}.png")


# ==================== MAIN ====================

def run_one_condition(grid_size, n_sensors, start_label, start_pos, seed=42):
    global GRID_SIZE, N_SENSORS, SEED, START_POS, CELL_SIZE
    GRID_SIZE  = grid_size
    N_SENSORS  = n_sensors
    SEED       = seed
    START_POS  = start_pos
    CELL_SIZE  = max(1, grid_size[0] // 50)

    results = []

    for label, model_path, config_path in CANDIDATE_MODELS:
        if not model_path.exists():
            continue
        try:
            res = run_dqn(label, model_path, config_path)
            results.append(res)
        except Exception as e:
            print(f"    [ERROR] {label}: {e}")

    for label, AgentClass in [
        ("SF-Aware Greedy V2",    MaxThroughputGreedyV2),
        ("Nearest Sensor Greedy", NearestSensorGreedy),
    ]:
        res = run_greedy(label, AgentClass)
        results.append(res)

    if results:
        plot_dwell(results)


def main():
    GRID_SIZES    = [(100, 100), (300, 300), (500, 500), (1000, 1000)]
    SENSOR_COUNTS = [10, 20, 30, 40]

    total = sum(
        len(get_start_positions(gs)) for gs in GRID_SIZES
    ) * len(SENSOR_COUNTS)
    done  = 0

    for gs in GRID_SIZES:
        starts = get_start_positions(gs)
        for ns in SENSOR_COUNTS:
            for start_label, start_pos in starts:
                # Save into per-condition subfolder
                global OUTPUT_DIR
                OUTPUT_DIR = (
                    script_dir / "dwell_results"
                    / f"{gs[0]}x{gs[1]}_N{ns}"
                )
                OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

                # Embed start label in filename via SEED+label trick:
                # override plot filename by patching SEED temporarily
                global SEED
                SEED = 42  # keep env seed fixed, only start varies

                # Patch plot output name
                global _START_LABEL
                _START_LABEL = start_label

                print(f"  {gs[0]}x{gs[1]} N={ns} start={start_label} {start_pos}")
                run_one_condition(gs, ns, start_label, start_pos)
                done += 1
                print(f"  [{done}/{total}] done")

    print("\nAll conditions done.")


if __name__ == "__main__":
    main()
