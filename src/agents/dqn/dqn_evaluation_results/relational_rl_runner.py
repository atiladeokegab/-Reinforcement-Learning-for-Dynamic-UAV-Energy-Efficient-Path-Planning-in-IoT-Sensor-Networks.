"""Load + run the Relational RL (Ray RLlib PPO) policy for compare_agents.py.

Keeps raw env rewards (no REWARD_SCALE, no potential shaping, no dwell bonus)
so cumulative_reward is directly comparable with the DQN / greedy / TSP agents.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import gymnasium
from gymnasium import spaces

_HERE = Path(__file__).resolve().parent
_SRC  = _HERE.parent.parent.parent
_ROOT = _SRC.parent
for _p in (str(_SRC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from environment.uav_env import UAVEnvironment

_RLMODULE_CACHE: dict[str, Any] = {}

N_MAX_DEFAULT = 50


class InferenceRelationalUAVEnv(UAVEnvironment):
    """UAVEnvironment subclass exposing the Dict observation the policy expects.

    Observation (matches RelationalUAVEnv):
        "uav"     : (3,)          [x/W, y/H, battery/max_battery]
        "sensors" : (n_max, 5)    [rel_x, rel_y, buf_fill, urgency, visited]
        "mask"    : (n_max,)      1=real sensor slot, 0=padding

    Rewards, physics, and sensor logic are inherited unchanged.
    """

    def __init__(self, n_max: int = N_MAX_DEFAULT, **kwargs: Any):
        self.n_max = n_max
        kwargs.setdefault("include_sensor_positions", True)
        super().__init__(**kwargs)

        assert self.num_sensors <= n_max, (
            f"num_sensors={self.num_sensors} exceeds n_max={n_max}"
        )

        self.observation_space = spaces.Dict({
            "uav":     spaces.Box(-1.0, 1.0, (3,),         dtype=np.float32),
            "sensors": spaces.Box(-1.0, 1.0, (n_max, 5),   dtype=np.float32),
            "mask":    spaces.Box( 0.0, 1.0, (n_max,),     dtype=np.float32),
        })

    def reset(self, **kwargs):
        _, info = super().reset(**kwargs)
        return self._build_obs(), info

    def step(self, action: int):
        _, reward, terminated, truncated, info = super().step(action)
        return self._build_obs(), reward, terminated, truncated, info

    def _build_obs(self) -> dict[str, np.ndarray]:
        W = float(self.grid_size[0])
        H = float(self.grid_size[1])
        uav_x = float(self.uav.position[0])
        uav_y = float(self.uav.position[1])

        uav_obs = np.array(
            [uav_x / W, uav_y / H, self.uav.battery / self.uav.max_battery],
            dtype=np.float32,
        )

        sensor_obs = np.zeros((self.n_max, 5), dtype=np.float32)
        mask_obs   = np.zeros(self.n_max,      dtype=np.float32)

        for i, sensor in enumerate(self.sensors):
            if i >= self.n_max:
                break
            sx = float(sensor.position[0])
            sy = float(sensor.position[1])
            sensor_obs[i] = [
                np.clip((sx - uav_x) / W, -1.0, 1.0),
                np.clip((sy - uav_y) / H, -1.0, 1.0),
                sensor.data_buffer / sensor.max_buffer_size,
                self._calculate_urgency(sensor),
                1.0 if sensor.sensor_id in self.sensors_visited else 0.0,
            ]
            mask_obs[i] = 1.0

        return {"uav": uav_obs, "sensors": sensor_obs, "mask": mask_obs}


def _resolve_policy_dir(checkpoint_dir: Path | str) -> Path:
    """Accept either the stage root (stage_4/final) or the policy dir directly."""
    p = Path(checkpoint_dir)
    candidates = [
        p,
        p / "learner_group" / "learner" / "rl_module" / "default_policy",
        p / "final" / "learner_group" / "learner" / "rl_module" / "default_policy",
    ]
    for c in candidates:
        if (c / "module_state.pkl").exists():
            return c
    raise FileNotFoundError(
        f"No module_state.pkl found under {p}. "
        f"Tried: {[str(c) for c in candidates]}"
    )


def load_relational_rl_module(checkpoint_dir: Path | str):
    """Load the RelationalUAVModule from a Ray RLlib checkpoint.

    checkpoint_dir may point at either:
      * the default_policy dir (contains module_state.pkl), or
      * the stage dir (e.g. stage_4/final).
    """
    from ray.rllib.core.rl_module.rl_module import RLModule

    policy_dir = _resolve_policy_dir(checkpoint_dir)
    cache_key = str(policy_dir)
    if cache_key in _RLMODULE_CACHE:
        return _RLMODULE_CACHE[cache_key]

    print(f"Loading Relational RL module from {policy_dir}")
    rl_module = RLModule.from_checkpoint(str(policy_dir))
    try:
        rl_module.eval()
    except AttributeError:
        pass
    _RLMODULE_CACHE[cache_key] = rl_module
    print("✓ Relational RL module loaded")
    return rl_module


def run_relational_rl_agent_for_plot(
    rl_module,
    env,
    name: str = "Relational RL",
    seed: int | None = None,
    max_battery: float = 274.0,
    log_interval: int = 50,
):
    """Run the RLModule deterministically on `env` for one episode."""
    import torch
    from ray.rllib.core.columns import Columns

    print(f"\nRunning {name}...")
    obs, _ = env.reset(seed=seed) if seed is not None else env.reset()

    history = {
        "step": [], "cumulative_reward": [], "battery_percent": [],
        "battery_wh": [], "coverage_percent": [], "sensors_visited": [],
        "total_data_collected": [], "efficiency": [],
    }
    positions: list[tuple[float, float]] = []
    cumulative_reward = 0.0
    step_count = 0

    while True:
        batch = {Columns.OBS: {
            k: torch.as_tensor(np.asarray(v)).unsqueeze(0)
            for k, v in obs.items()
        }}
        with torch.no_grad():
            out = rl_module._forward_inference(batch)
        logits = out[Columns.ACTION_DIST_INPUTS]
        action = int(torch.argmax(logits, dim=-1).item())

        obs, reward, terminated, truncated, info = env.step(action)
        cumulative_reward += float(reward)
        step_count += 1
        positions.append((float(env.uav.position[0]), float(env.uav.position[1])))

        done = terminated or truncated
        if env.current_step % log_interval == 0 or done:
            battery_pct  = env.uav.get_battery_percentage()
            coverage_pct = (
                (len(env.sensors_visited) / env.num_sensors) * 100
                if hasattr(env, "sensors_visited") else 0.0
            )
            energy = max_battery - env.uav.battery
            eff    = (env.total_data_collected / energy) if energy > 0 else 0.0
            history["step"].append(env.current_step)
            history["cumulative_reward"].append(cumulative_reward)
            history["battery_percent"].append(battery_pct)
            history["battery_wh"].append(env.uav.battery)
            history["coverage_percent"].append(coverage_pct)
            history["sensors_visited"].append(len(env.sensors_visited))
            history["total_data_collected"].append(env.total_data_collected)
            history["efficiency"].append(eff)
            print(
                f"  Step {env.current_step:>4}: Reward={cumulative_reward:>10.1f}, "
                f"Battery={battery_pct:>5.1f}%, NDR={coverage_pct:>5.1f}%, "
                f"Data={env.total_data_collected:>8.0f}bytes"
            )

        if done:
            break

    return (
        pd.DataFrame(history),
        step_count,
        np.array(positions) if positions else np.array([]),
    )
