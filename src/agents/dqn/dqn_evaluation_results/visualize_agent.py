"""
Live visualization of an agent in the UAV environment.

Usage examples:
    # DQN checkpoint (framestack auto-detected)
    uv run python src/agents/dqn/dqn_evaluation_results/visualize_agent.py \
        --agent src/agents/dqn/models/dqn_domain_rand/dqn_final.zip

    # Greedy baselines
    uv run python .../visualize_agent.py --agent nearest
    uv run python .../visualize_agent.py --agent max_throughput
    uv run python .../visualize_agent.py --agent tsp
    uv run python .../visualize_agent.py --agent random

Flags:
    --grid 500          grid size (square)
    --sensors 20        number of sensors
    --seed 42           RNG seed
    --max-steps 2100
    --fps 8             render pace
    --framestack auto|1|4   override frame-stack depth

Always prefix with PYTHONIOENCODING=utf-8 on Windows.
fast and human w
Author: ATILADE GABRIEL OKE
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np

# Path setup — this file lives in src/agents/dqn/dqn_evaluation_results/
_HERE = Path(__file__).resolve().parent
_SRC  = _HERE.parent.parent.parent
sys.path.insert(0, str(_SRC))

from environment.uav_env import UAVEnvironment
from agents.dqn.dqn_evaluation_results.greedy_agents import (
    NearestSensorGreedy,
    MaxThroughputGreedyV2,
    TSPOracleAgent,
)
from agents.dqn.dqn_evaluation_results.relational_rl_runner import (
    InferenceRelationalUAVEnv,
    load_relational_rl_module,
)

DEFAULT_RELATIONAL_CKPT = (
    _SRC / "agents" / "dqn" / "models" / "relational_rl"
    / "results" / "checkpoints" / "stage_4" / "final"
)


# ─────────────────────────────────────────────────────────────────────────────
# Agent wrappers — unify .predict(obs) -> action
# ─────────────────────────────────────────────────────────────────────────────

class RelationalRLWrapper:
    """Wraps a Ray RLlib RLModule that expects Dict obs {uav, sensors, mask}."""

    def __init__(self, rl_module):
        import torch
        from ray.rllib.core.columns import Columns
        self._torch = torch
        self._Columns = Columns
        self.rl_module = rl_module

    def predict(self, obs):
        torch = self._torch
        Columns = self._Columns
        batch = {Columns.OBS: {
            k: torch.as_tensor(np.asarray(v)).unsqueeze(0)
            for k, v in obs.items()
        }}
        with torch.no_grad():
            out = self.rl_module._forward_inference(batch)
        logits = out[Columns.ACTION_DIST_INPUTS]
        return int(torch.argmax(logits, dim=-1).item())


class RandomAgent:
    def __init__(self, env):
        self.env = env
    def predict(self, obs):
        return int(self.env.action_space.sample())


class GreedyWrapper:
    def __init__(self, greedy):
        self.greedy = greedy
    def predict(self, obs):
        return int(self.greedy.select_action(obs))


class DQNWrapper:
    """
    Wraps an SB3 DQN model. Handles:
      - zero-padding per-frame obs to the model's max_sensors_limit
      - frame-stacking (k frames concatenated)
    """

    FEATURES_PER_SENSOR = 3  # matches training (features_per_sensor=3)

    def __init__(self, model, env_num_sensors: int, max_sensors: int,
                 stack_k: int):
        self.model = model
        self.env_num_sensors = env_num_sensors
        self.max_sensors = max_sensors
        self.stack_k = stack_k
        self.pad_width = (max_sensors - env_num_sensors) * self.FEATURES_PER_SENSOR
        self.buf: deque = deque(maxlen=stack_k)

    def _pad(self, obs: np.ndarray) -> np.ndarray:
        if self.pad_width <= 0:
            return obs.astype(np.float32)
        padding = np.zeros(self.pad_width, dtype=np.float32)
        return np.concatenate([obs, padding]).astype(np.float32)

    def reset(self, first_obs: np.ndarray):
        padded = self._pad(first_obs)
        self.buf.clear()
        for _ in range(self.stack_k):
            self.buf.append(padded)

    def predict(self, obs):
        padded = self._pad(obs)
        self.buf.append(padded)
        if self.stack_k == 1:
            stacked = padded
        else:
            stacked = np.concatenate(list(self.buf), axis=-1).astype(np.float32)
        action, _ = self.model.predict(stacked, deterministic=True)
        return int(action)


# ─────────────────────────────────────────────────────────────────────────────
# Agent factory
# ─────────────────────────────────────────────────────────────────────────────

_START_PRESETS = {
    # corners
    "bottom-left":  (0.0, 0.0),
    "bottom-right": (1.0, 0.0),
    "top-left":     (0.0, 1.0),
    "top-right":    (1.0, 1.0),
    # edge midpoints
    "mid-left":     (0.0, 0.5),
    "mid-right":    (1.0, 0.5),
    "mid-top":      (0.5, 1.0),
    "mid-bottom":   (0.5, 0.0),
    # centre
    "center":       (0.5, 0.5),
}


def build_env(agent_spec: str, args) -> UAVEnvironment:
    """Build the env — InferenceRelationalUAVEnv for relational, standard otherwise."""
    if args.start_pos and args.start_pos != "origin":
        fx, fy = _START_PRESETS[args.start_pos]
        # Nudge slightly inside the boundary so boundary-hit penalties don't trigger
        # on step 0 when starting exactly on an edge.
        inset = max(args.grid * 0.01, 1.0)
        sx = min(max(fx * args.grid, inset), args.grid - inset)
        sy = min(max(fy * args.grid, inset), args.grid - inset)
        start = (float(sx), float(sy))
        print(f"UAV start preset '{args.start_pos}' → ({start[0]:.1f}, {start[1]:.1f})")
    elif args.random_start:
        rng = np.random.default_rng(args.seed)
        start = (float(rng.uniform(0, args.grid)), float(rng.uniform(0, args.grid)))
        print(f"Random UAV start: ({start[0]:.1f}, {start[1]:.1f})")
    else:
        start = (0, 0)

    common = dict(
        grid_size=(args.grid, args.grid),
        num_sensors=args.sensors,
        uav_start_position=start,
        max_steps=args.max_steps,
        render_mode="human",
    )
    if agent_spec.lower() == "relational":
        return InferenceRelationalUAVEnv(n_max=args.n_max, **common)
    return UAVEnvironment(**common)


def build_agent(spec: str, env: UAVEnvironment, framestack_override: str,
                relational_ckpt: str):
    key = spec.lower()

    if key == "random":
        return RandomAgent(env), None
    if key == "nearest":
        return GreedyWrapper(NearestSensorGreedy(env)), None
    if key in ("max_throughput", "maxthroughput", "max"):
        return GreedyWrapper(MaxThroughputGreedyV2(env)), None
    if key == "tsp":
        g = TSPOracleAgent(env)
        return GreedyWrapper(g), g
    if key == "relational":
        ckpt = Path(relational_ckpt) if relational_ckpt else DEFAULT_RELATIONAL_CKPT
        if not ckpt.exists():
            raise FileNotFoundError(f"Relational RL checkpoint not found: {ckpt}")
        rl_module = load_relational_rl_module(ckpt)
        return RelationalRLWrapper(rl_module), None

    # Otherwise treat as a path to an SB3 .zip
    from stable_baselines3 import DQN

    path = Path(spec)
    if not path.exists():
        raise FileNotFoundError(f"Model path not found: {path}")

    print(f"Loading DQN from: {path}")
    model = DQN.load(str(path), device="cpu")

    model_obs_dim = int(model.observation_space.shape[0])
    env_obs_dim   = int(env.observation_space.shape[0])
    features = DQNWrapper.FEATURES_PER_SENSOR

    # Auto-detect (k, max_sensors) such that:
    #   model_obs_dim == k * (3 + features * max_sensors)
    #   max_sensors >= env.num_sensors
    #   k*env_obs_dim <= model_obs_dim  (padding fills the gap)
    candidates = []
    for k in (4, 1, 2, 8):
        if framestack_override != "auto" and k != int(framestack_override):
            continue
        if model_obs_dim % k != 0:
            continue
        per_frame = model_obs_dim // k
        rem = per_frame - 3
        if rem < 0 or rem % features != 0:
            continue
        max_s = rem // features
        if max_s < env.num_sensors:
            continue
        candidates.append((k, max_s))

    if not candidates:
        raise ValueError(
            f"Could not reconcile model obs dim {model_obs_dim} with env "
            f"(num_sensors={env.num_sensors}, features={features}). "
            f"Try a different --framestack or --sensors."
        )

    stack_k, max_sensors = candidates[0]
    print(f"  Env obs dim: {env_obs_dim} (N={env.num_sensors}) | "
          f"Model obs dim: {model_obs_dim} | "
          f"framestack k={stack_k} | pad-to-max-sensors={max_sensors}")

    return DQNWrapper(model, env.num_sensors, max_sensors, stack_k), None


# ─────────────────────────────────────────────────────────────────────────────
# Episode loop
# ─────────────────────────────────────────────────────────────────────────────

def run_episode(agent, env: UAVEnvironment, fps: float, tsp_agent=None):
    obs, info = env.reset(seed=args.seed)

    if isinstance(agent, DQNWrapper):
        agent.reset(obs)
    if tsp_agent is not None:
        tsp_agent.reset()

    env.render()
    action_names = ["UP", "DOWN", "LEFT", "RIGHT", "COLLECT"]
    sleep_s = 1.0 / max(fps, 0.1)

    total_reward = 0.0
    step = 0
    render_every = max(1, int(getattr(args, "render_every", 1)))
    while True:
        action = agent.predict(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1

        if step % render_every == 0 or terminated or truncated:
            env.render()
            if sleep_s > 0:
                time.sleep(sleep_s)

        if step % 50 == 0:
            print(
                f"Step {step:4d} | {action_names[action]:7s} | "
                f"Pos=({info['uav_position'][0]:.1f},{info['uav_position'][1]:.1f}) | "
                f"Bat={info['battery_percent']:5.1f}% | "
                f"Cov={info['coverage_percentage']:5.1f}% | "
                f"R={total_reward:+.1f}"
            )

        if terminated or truncated:
            break

    print("\n" + "=" * 70)
    print("Episode finished")
    print("=" * 70)
    print(f"  Steps             : {info['current_step']}")
    print(f"  Total reward      : {info['total_reward']:.2f}")
    print(f"  Coverage          : {info['coverage_percentage']:.1f}%")
    print(f"  Data collected    : {info['total_data_collected']:.1f} bytes")
    print(f"  Battery remaining : {info['battery']:.1f} Wh "
          f"({info['battery_percent']:.1f}%)")
    print(f"  Max / Avg urgency : {info['max_urgency']:.2f} / "
          f"{info['avg_urgency']:.2f}")
    print("=" * 70)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--agent", required=True,
                   help="Path to .zip OR one of: random, nearest, max_throughput, tsp, relational")
    p.add_argument("--relational-ckpt", default="",
                   help="Checkpoint dir for relational agent (default: stage_4/final)")
    p.add_argument("--n-max", type=int, default=50,
                   help="n_max for InferenceRelationalUAVEnv padding (default 50)")
    p.add_argument("--grid", type=int, default=500)
    p.add_argument("--sensors", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--random-start", action="store_true",
                   help="Spawn UAV at a random (seeded) position instead of (0,0)")
    p.add_argument("--start-pos", default="origin",
                   choices=["origin", "center",
                            "top-left", "top-right",
                            "bottom-left", "bottom-right",
                            "mid-top", "mid-bottom", "mid-left", "mid-right"],
                   help="Preset UAV start position (overrides --random-start if given)")
    p.add_argument("--max-steps", type=int, default=2100)
    p.add_argument("--fps", type=float, default=20.0,
                   help="Frames per second (default 20 — fast but watchable)")
    p.add_argument("--render-every", type=int, default=2,
                   help="Render every Nth env step (default 2 for speed)")
    p.add_argument("--framestack", default="auto",
                   help="auto | 1 | 4  (DQN only)")
    p.add_argument("--hold", type=float, default=5.0,
                   help="Seconds to keep window open after episode ends")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    env = build_env(args.agent, args)
    agent, tsp_agent = build_agent(
        args.agent, env, args.framestack, args.relational_ckpt
    )
    run_episode(agent, env, fps=args.fps, tsp_agent=tsp_agent)

    import matplotlib.pyplot as plt
    if args.hold > 0:
        print(f"\nHolding window for {args.hold:.0f}s (Ctrl+C to exit)...")
        try:
            time.sleep(args.hold)
        except KeyboardInterrupt:
            pass
    env.close()
    plt.ioff()
