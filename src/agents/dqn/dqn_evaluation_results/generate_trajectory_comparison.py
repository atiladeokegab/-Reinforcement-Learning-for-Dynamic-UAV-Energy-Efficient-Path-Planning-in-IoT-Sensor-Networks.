"""
Generate a 2-panel DQN vs PPO+Relational trajectory figure for the dissertation.

Condition: 300x300 grid, N=30 — the condition where the DQN's perimeter
strategy fails most visibly and the relational encoder recovers fully.

Output: baseline_results/trajectory_dqn_vs_relational.png
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_HERE = Path(__file__).resolve().parent
_DQN  = _HERE.parent
_SRC  = _HERE.parents[2]
for _p in (str(_SRC), str(_SRC / "environment"), str(_DQN),
           str(_SRC / "experiments" / "relational_policy")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
import gymnasium

from environment.uav_env import UAVEnvironment
from relational_rl_runner import InferenceRelationalUAVEnv, load_relational_rl_module
from ray.rllib.core.columns import Columns

# ── Config ────────────────────────────────────────────────────────────────────
SEED       = 9
GRID_SIZE  = (400, 400)
N_SENSORS  = 30
MAX_STEPS  = 2100
MAX_BATTERY = 274.0
N_MAX      = 50
N_STACK    = 4
MAX_SENSORS_LIMIT = 50

DQN_MODEL_PATH = _DQN / "models" / "dqn_v3_retrain" / "dqn_dr_5000000_steps.zip"
REL_CKPT_DIR   = _DQN / "models" / "relational_rl" / "results" / "checkpoints" / "stage_4" / "final"
OUT_PATH       = _HERE / "baseline_results" / "trajectory_dqn_vs_relational.png"

ENV_BASE = {
    "max_steps":          MAX_STEPS,
    "path_loss_exponent": 3.8,
    "rssi_threshold":     -85.0,
    "sensor_duty_cycle":  10.0,
    "max_battery":        MAX_BATTERY,
    "render_mode":        None,
}


# ── Padded DQN env ────────────────────────────────────────────────────────────

class _PaddedEnv(UAVEnvironment):
    def __init__(self, **kw):
        self._positions: list[tuple] = []
        self._snap_positions: list[tuple] = []
        self._snap_sensors: list = []
        self._snap_visited: set = set()
        self._snap_ndr: float = 0.0
        super().__init__(**kw)
        raw = self.observation_space.shape[0]
        self._fps = 0
        for uav_f in range(raw + 1):
            rem = raw - uav_f
            if rem > 0 and rem % self.num_sensors == 0:
                self._fps = rem // self.num_sensors
                break
        padded = raw + (MAX_SENSORS_LIMIT - self.num_sensors) * self._fps
        self.observation_space = gymnasium.spaces.Box(
            low=-np.inf, high=np.inf, shape=(padded,), dtype=np.float32)

    def _pad(self, obs):
        pad = np.zeros((MAX_SENSORS_LIMIT - self.num_sensors) * self._fps, dtype=np.float32)
        return np.concatenate([obs, pad]).astype(np.float32)

    def reset(self, **kw):
        obs, info = super().reset(**kw)
        self._path_history = [(float(self.uav.position[0]), float(self.uav.position[1]))]
        return self._pad(obs), info

    def step(self, action):
        obs, r, te, tr, info = super().step(action)
        pos = (float(self.uav.position[0]), float(self.uav.position[1]))
        if not self._path_history or self._path_history[-1] != pos:
            self._path_history.append(pos)
        if te or tr:
            self._snap_positions = list(self._path_history)
            self._snap_sensors   = list(self.sensors)
            self._snap_visited   = set(self.sensors_visited)
            self._snap_ndr       = len(self.sensors_visited) / self.num_sensors * 100.0
        return self._pad(obs), r, te, tr, info


# ── Run DQN episode ───────────────────────────────────────────────────────────

def run_dqn(seed):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = DQN.load(str(DQN_MODEL_PATH), device=device)
    model.policy.set_training_mode(False)

    env_kw = {**ENV_BASE, "grid_size": GRID_SIZE, "num_sensors": N_SENSORS}
    vec = DummyVecEnv([lambda: _PaddedEnv(**env_kw)])
    vec = VecFrameStack(vec, n_stack=N_STACK)

    inner = vec
    while hasattr(inner, "venv"):
        inner = inner.venv
    base: _PaddedEnv = inner.envs[0]

    import random; random.seed(seed); np.random.seed(seed)
    obs = vec.reset()
    done = np.array([False])
    while not done[0]:
        with torch.no_grad():
            action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, _ = vec.step(action)
        done = dones
    vec.close()
    return base._snap_positions, base._snap_sensors, base._snap_visited, base._snap_ndr


# ── Run Relational episode ────────────────────────────────────────────────────

def run_relational(seed, fixed_sensor_positions=None):
    module = load_relational_rl_module(REL_CKPT_DIR)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        module = module.to(device)
    module.eval()

    env = InferenceRelationalUAVEnv(
        n_max=N_MAX, grid_size=GRID_SIZE, num_sensors=N_SENSORS,
        sensor_positions=fixed_sensor_positions, **ENV_BASE)

    import random; random.seed(seed); np.random.seed(seed)
    obs, _ = env.reset()
    positions = [(float(env.uav.position[0]), float(env.uav.position[1]))]
    done = False
    while not done:
        batch = {Columns.OBS: {
            k: torch.as_tensor(np.asarray(v), dtype=torch.float32).unsqueeze(0)
            for k, v in obs.items()
        }}
        if device == "cuda":
            batch = {Columns.OBS: {k: v.cuda() for k, v in batch[Columns.OBS].items()}}
        with torch.no_grad():
            out = module._forward_inference(batch)
        action = int(torch.argmax(out[Columns.ACTION_DIST_INPUTS], dim=-1).item())
        obs, _, terminated, truncated, _ = env.step(action)
        pos = (float(env.uav.position[0]), float(env.uav.position[1]))
        if positions[-1] != pos:
            positions.append(pos)
        done = terminated or truncated
    ndr = len(env.sensors_visited) / env.num_sensors * 100.0
    env.close()
    return positions, env.sensors, env.sensors_visited, ndr


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot(dqn_data, rel_data):
    dqn_pos,  dqn_sensors,  dqn_visited,  dqn_ndr  = dqn_data
    rel_pos,  rel_sensors,  rel_visited,  rel_ndr  = rel_data

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.2), constrained_layout=True)
    W, H = GRID_SIZE

    for ax, positions, sensors, visited, ndr, label in [
        (axes[0], dqn_pos,  dqn_sensors,  dqn_visited,  dqn_ndr,  "DQN (flat MLP)"),
        (axes[1], rel_pos,  rel_sensors,  rel_visited,  rel_ndr,  "PPO (relational)"),
    ]:
        xs = [p[0] for p in positions]
        ys = [p[1] for p in positions]

        # Trajectory with alpha fade (recent = darker)
        n = len(xs)
        for i in range(n - 1):
            alpha = 0.25 + 0.65 * (i / max(n - 1, 1))
            ax.plot(xs[i:i+2], ys[i:i+2], color="#2166ac", lw=1.0, alpha=alpha)

        # Start / end markers
        ax.plot(xs[0],  ys[0],  marker="s", ms=6, color="green",  zorder=5, label="Start")
        ax.plot(xs[-1], ys[-1], marker="^", ms=6, color="red",    zorder=5, label="End")

        # Sensors — visited is a set of sensor_id ints
        for s in sensors:
            sx, sy = float(s.position[0]), float(s.position[1])
            color = "#1a9641" if s.sensor_id in visited else "#d7191c"
            ax.scatter(sx, sy, c=color, s=40, zorder=4, edgecolors="white", linewidths=0.4)

        ax.set_xlim(0, W)
        ax.set_ylim(0, H)
        ax.set_xlabel("$x$ (grid units)", fontsize=9)
        ax.set_ylabel("$y$ (grid units)", fontsize=9)
        ax.set_title(f"{label}\nNDR = {ndr:.0f}%", fontsize=9)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=8)

    # Shared legend
    legend_elements = [
        mpatches.Patch(color="#1a9641", label="Sensor visited"),
        mpatches.Patch(color="#d7191c", label="Sensor missed"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor="green",  ms=6, label="UAV start"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor="red",    ms=6, label="UAV end"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               fontsize=8, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle(
        f"UAV trajectories: {GRID_SIZE[0]}x{GRID_SIZE[1]} grid, N={N_SENSORS}, seed {SEED}",
        fontsize=10, y=1.01
    )
    fig.savefig(str(OUT_PATH), dpi=180, bbox_inches="tight")
    print(f"Saved: {OUT_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Running DQN on seed {SEED}...")
    dqn_data = run_dqn(SEED)
    print(f"  NDR = {dqn_data[3]:.1f}%")

    # Pass the DQN's sensor positions to the relational env so both agents
    # operate on the identical sensor layout.
    fixed_positions = [(float(s.position[0]), float(s.position[1])) for s in dqn_data[1]]
    print(f"Running Relational on seed {SEED} (same sensor layout)...")
    rel_data = run_relational(SEED, fixed_sensor_positions=fixed_positions)
    print(f"  NDR = {rel_data[3]:.1f}%")

    plot(dqn_data, rel_data)
