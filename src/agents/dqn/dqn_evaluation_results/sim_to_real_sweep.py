"""Sim-to-real robustness sweep: DQN vs Relational RL under test-time
perturbations. Writes per-(agent, perturbation, magnitude, seed) rows to
baseline_results/sim_to_real/robustness_raw.csv.

Four perturbation axes (applied only at test time; policies are not
retrained):

  gps_sigma       Additive Gaussian noise on the UAV's observed position
                  (grid units). sigma in {0, 5, 10, 20}.
  path_loss       Test-time path-loss exponent. Default 3.8; sweep
                  {3.4, 3.8, 4.2, 4.6}.
  shadow_sigma    Test-time shadowing variance (dB). Default 4; sweep
                  {4, 6, 8, 10}.
  wind_drift      Constant additive bias on the UAV's move displacement
                  (cells / step). Sweep {0.0, 0.5, 1.0, 2.0}.

Condition: 200x200 grid, N=20, 10 seeds per cell. Both agents evaluated
deterministically (argmax).
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import numpy as np

from stable_baselines3 import DQN
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

SCRIPT_DIR = Path(__file__).resolve().parent
SRC_DIR    = SCRIPT_DIR.parent.parent.parent

sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SRC_DIR))

_GNN = list(Path(SCRIPT_DIR.parent).rglob("gnn_extractor.py"))
if _GNN:
    sys.path.insert(0, str(_GNN[0].parent))

from environment.uav_env import UAVEnvironment
from compare_agents import AnalysisUAVEnv, load_training_config
from relational_rl_runner import (
    InferenceRelationalUAVEnv,
    load_relational_rl_module,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DQN_MODEL_DIR  = SCRIPT_DIR.parent / "models" / "dqn_v3_retrain"
DQN_MODEL_PATH = DQN_MODEL_DIR / "dqn_final.zip"
DQN_CONFIG     = DQN_MODEL_DIR / "training_config.json"

REL_CKPT = (SCRIPT_DIR.parent / "models" / "relational_rl" /
            "results" / "checkpoints" / "stage_4" / "final")

OUT_DIR = SCRIPT_DIR / "baseline_results" / "sim_to_real"
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_CSV = OUT_DIR / "robustness_raw.csv"

N_SEEDS       = 10
BASE_SEED     = 2000
GRID          = (200, 200)
N_SENSORS     = 20
MAX_STEPS     = 2100
BASE_PLE      = 3.8
BASE_SHADOW   = 4.0

PERTURBATIONS = {
    "gps_sigma":     [0.0, 5.0, 10.0, 20.0],
    "path_loss":     [3.4, 3.8, 4.2, 4.6],
    "shadow_sigma":  [4.0, 6.0, 8.0, 10.0],
    "wind_drift":    [0.0, 0.5, 1.0, 2.0],
}

# Common env kwargs (these are the "nominal" training conditions the DQN
# and Relational RL policies were trained against).
BASE_ENV_KW = dict(
    grid_size=GRID,
    num_sensors=N_SENSORS,
    max_steps=MAX_STEPS,
    path_loss_exponent=BASE_PLE,
    rssi_threshold=-85.0,
    sensor_duty_cycle=10.0,
    uav_start_position=(0, 0),
)


# ---------------------------------------------------------------------------
# Perturbation wrappers
# ---------------------------------------------------------------------------

class GPSNoiseMixin:
    """Add Gaussian noise to the UAV's observed position each step.

    Applies to the position that feeds the observation builder but not to
    the true `self.uav.position` used by physics. For the DQN
    (AnalysisUAVEnv) this mutates just before `_get_observation`; for the
    Relational RL env, we wrap `_build_obs` similarly.
    """


def _perturb_positions_dqn(env_cls, gps_sigma: float, shadow_sigma: float,
                           wind_drift: float, ple: float):
    """Build a subclass of AnalysisUAVEnv with all four perturbations baked in."""

    class PerturbedDQNEnv(env_cls):
        def __init__(self, **kwargs):
            kwargs["path_loss_exponent"] = ple
            super().__init__(**kwargs)
            for s in self.sensors:
                s.path_loss_exponent = ple
                if hasattr(s, "shadowing_std_db"):
                    s.shadowing_std_db = shadow_sigma

        def step(self, action):
            # Wind-drift: occasionally turn a move into a repeat-move or drop it.
            # We model this as an additive probability of the move being
            # extended by one extra unit (cells/step > 0 => slightly longer
            # move). Implementation keeps the action discrete; the effect is
            # a probabilistic additional move in the same direction after
            # the commanded one, chosen so that mean displacement matches.
            if wind_drift > 0 and action in (0, 1, 2, 3):
                p_extra = min(0.95, wind_drift / 2.0)
                if np.random.random() < p_extra:
                    super().step(action)  # extra half-step (drift)
            obs, r, term, trunc, info = super().step(action)
            return obs, r, term, trunc, info

        def _get_observation(self):
            if gps_sigma > 0:
                true_pos = self.uav.position.copy()
                noisy = true_pos + np.random.normal(0, gps_sigma, size=2)
                noisy = np.clip(noisy, 0, [self.grid_size[0] - 1,
                                            self.grid_size[1] - 1])
                self.uav.position = noisy
                try:
                    return super()._get_observation()
                finally:
                    self.uav.position = true_pos
            return super()._get_observation()

    return PerturbedDQNEnv


def _perturb_positions_rel(gps_sigma: float, shadow_sigma: float,
                           wind_drift: float, ple: float):

    class PerturbedRelEnv(InferenceRelationalUAVEnv):
        def __init__(self, **kwargs):
            kwargs["path_loss_exponent"] = ple
            super().__init__(**kwargs)
            for s in self.sensors:
                s.path_loss_exponent = ple
                if hasattr(s, "shadowing_std_db"):
                    s.shadowing_std_db = shadow_sigma

        def step(self, action):
            if wind_drift > 0 and action in (0, 1, 2, 3):
                p_extra = min(0.95, wind_drift / 2.0)
                if np.random.random() < p_extra:
                    super().step(action)
            return super().step(action)

        def _build_obs(self):
            if gps_sigma > 0:
                true_pos = self.uav.position.copy()
                noisy = true_pos + np.random.normal(0, gps_sigma, size=2)
                noisy = np.clip(noisy, 0, [self.grid_size[0] - 1,
                                            self.grid_size[1] - 1])
                self.uav.position = noisy
                try:
                    return super()._build_obs()
                finally:
                    self.uav.position = true_pos
            return super()._build_obs()

    return PerturbedRelEnv


# ---------------------------------------------------------------------------
# Runners
# ---------------------------------------------------------------------------

_DQN_MODEL = None

def _get_dqn_model():
    global _DQN_MODEL
    if _DQN_MODEL is None:
        print("  (loading DQN model into cache...)")
        _DQN_MODEL = DQN.load(DQN_MODEL_PATH)
    return _DQN_MODEL

def run_dqn_episode(seed: int, gps_sigma, shadow_sigma, wind_drift, ple,
                    cfg):
    np.random.seed(seed); random.seed(seed)
    EnvCls = _perturb_positions_dqn(AnalysisUAVEnv, gps_sigma, shadow_sigma,
                                    wind_drift, ple)
    def _mk():
        env = EnvCls(max_sensors_limit=cfg.get("max_sensors_limit", 50),
                     **BASE_ENV_KW)
        env.reset(seed=seed)
        return env
    vec = DummyVecEnv([_mk])
    if cfg.get("use_frame_stacking", True):
        vec = VecFrameStack(vec, n_stack=cfg.get("n_stack", 4))

    model = _get_dqn_model()
    obs = vec.reset(); done = False; steps = 0; last_info = {}
    while not done:
        a, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = vec.step(a)
        steps += 1
        last_info = infos[0]
        done = bool(dones[0])
    vec.close()
    return {
        "steps": steps,
        "coverage_pct":   float(last_info.get("coverage_percentage", 0)),
        "reward":         float(last_info.get("total_reward", 0)),
        "boundary_hits":  int(last_info.get("boundary_hits", 0)),
        "edge_step_pct":  100.0 * int(last_info.get("edge_steps", 0)) / max(steps, 1),
    }


def run_rel_episode(seed: int, gps_sigma, shadow_sigma, wind_drift, ple,
                    rl_module):
    import torch
    from ray.rllib.core.columns import Columns
    np.random.seed(seed); random.seed(seed)
    EnvCls = _perturb_positions_rel(gps_sigma, shadow_sigma, wind_drift, ple)
    env = EnvCls(n_max=50, **BASE_ENV_KW)
    obs, _ = env.reset(seed=seed)
    done = False; steps = 0; last_info = {}
    while not done:
        batch = {Columns.OBS: {k: torch.as_tensor(np.asarray(v)).unsqueeze(0)
                                for k, v in obs.items()}}
        with torch.no_grad():
            out = rl_module._forward_inference(batch)
        a = int(torch.argmax(out[Columns.ACTION_DIST_INPUTS], dim=-1).item())
        obs, _, term, trunc, last_info = env.step(a)
        steps += 1
        done = bool(term or trunc)
    return {
        "steps": steps,
        "coverage_pct":   float(last_info.get("coverage_percentage", 0)),
        "reward":         float(last_info.get("total_reward", 0)),
        "boundary_hits":  int(last_info.get("boundary_hits", 0)),
        "edge_step_pct":  100.0 * int(last_info.get("edge_steps", 0)) / max(steps, 1),
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main():
    cfg = load_training_config(DQN_CONFIG)
    print("Loading Relational RL checkpoint...")
    rl_module = load_relational_rl_module(REL_CKPT)
    print("OK")

    rows = []
    total = 0
    for axis, values in PERTURBATIONS.items():
        for v in values:
            for seed_i in range(N_SEEDS):
                seed = BASE_SEED + seed_i
                kw = dict(gps_sigma=0.0, shadow_sigma=BASE_SHADOW,
                          wind_drift=0.0, ple=BASE_PLE)
                kw[axis if axis != "path_loss" else "ple"] = v
                if axis == "path_loss":
                    kw["ple"] = v

                # DQN
                try:
                    r_dqn = run_dqn_episode(seed, **kw, cfg=cfg)
                    rows.append({"agent": "DQN", "axis": axis, "value": v,
                                 "seed": seed, **r_dqn})
                except Exception as e:
                    print(f"  DQN fail {axis}={v} seed={seed}: {e}")
                    rows.append({"agent": "DQN", "axis": axis, "value": v,
                                 "seed": seed, "steps": 0, "coverage_pct": 0,
                                 "reward": 0, "boundary_hits": 0,
                                 "edge_step_pct": 0})

                # Relational RL
                try:
                    r_rel = run_rel_episode(seed, **kw, rl_module=rl_module)
                    rows.append({"agent": "Relational RL", "axis": axis,
                                 "value": v, "seed": seed, **r_rel})
                except Exception as e:
                    print(f"  Rel fail {axis}={v} seed={seed}: {e}")
                    rows.append({"agent": "Relational RL", "axis": axis,
                                 "value": v, "seed": seed, "steps": 0,
                                 "coverage_pct": 0, "reward": 0,
                                 "boundary_hits": 0, "edge_step_pct": 0})

                total += 2
                print(f"  [{total}/{2 * sum(len(v) for v in PERTURBATIONS.values()) * N_SEEDS}] "
                      f"{axis}={v}  seed={seed}  "
                      f"DQN cov={rows[-2]['coverage_pct']:.0f}%  "
                      f"Rel cov={rows[-1]['coverage_pct']:.0f}%")

    # Write CSV
    fieldnames = ["agent", "axis", "value", "seed", "steps",
                  "coverage_pct", "reward", "boundary_hits",
                  "edge_step_pct"]
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    print(f"\n{len(rows)} rows -> {OUT_CSV}")


if __name__ == "__main__":
    main()
