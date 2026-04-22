"""
train_transformer_stage1.py
===========================
Stage-1 kickstart variant. Restores the GTrXL algorithm from the canonical
Stage-0 checkpoint produced by train_transformer.py and begins training
directly on Stage 1 (200x200, 20 sensors) with an elevated entropy
coefficient to shock the policy out of the Stage-0 local optimum.

Rationale
---------
The canonical run plateaued near 88% NDR on Stage 0 with entropy collapsed
to ~0.008. The policy had learned an efficient 9-sensor loop and would not
explore enough to discover the 10th. Rather than keep burning compute on a
saturated stage, we forcibly advance to Stage 1 and re-inflate exploration.

Outputs
-------
* Checkpoints:  models/transformer_gtrxl_stage1/
* Ray tmp dir:  /workspace/ray_tmp_stage1

Usage
-----
    PYTHONIOENCODING=utf-8 uv run python \\
        src/experiments/transformer_policy/train_transformer_stage1.py

Curriculum gate (unchanged from relaxed variant)
------------------------------------------------
NDR >= 90%, JFI >= 0.85, 50-episode rolling window, 200k min timesteps.
"""

from __future__ import annotations

import os
import sys
import pathlib
import logging
from collections import deque
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[3]
_SRC  = _HERE.parents[2]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks

from experiments.transformer_policy.transformer_model import register_model
from experiments.transformer_policy.env_wrapper import TransformerObsWrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hardware detection
# ---------------------------------------------------------------------------
try:
    import torch
    _CUDA_COUNT = torch.cuda.device_count()
except ImportError:
    _CUDA_COUNT = 0

NUM_GPUS: int = min(_CUDA_COUNT, 2)
log.info("Detected %d CUDA GPU(s). Using %d for training.", _CUDA_COUNT, NUM_GPUS)

# ---------------------------------------------------------------------------
# Stage-1 kickstart configuration
# ---------------------------------------------------------------------------
START_STAGE: int = 1
RESTORE_FROM: str = "/workspace/uav/models/transformer_gtrxl/stage_0_progress"

# ---------------------------------------------------------------------------
# Curriculum stages
# ---------------------------------------------------------------------------
CURRICULUM_STAGES: list[dict[str, Any]] = [
    {"grid_size": (100, 100), "num_sensors": 10,  "name": "Stage-0 · 100×100  · 10 sensors"},
    {"grid_size": (200, 200), "num_sensors": 20,  "name": "Stage-1 · 200×200  · 20 sensors"},
    {"grid_size": (300, 300), "num_sensors": 30,  "name": "Stage-2 · 300×300  · 30 sensors"},
    {"grid_size": (400, 400), "num_sensors": 40,  "name": "Stage-3 · 400×400  · 40 sensors"},
    {"grid_size": (500, 500), "num_sensors": 50,  "name": "Stage-4 · 500×500  · 50 sensors"},
]

ADVANCE_CRITERIA: dict[str, float] = {
    "ndr_pct": 90.0,
    "jains":   0.85,
    "window":  50,
    "min_timesteps_per_stage": 200_000,
}

# Per-stage entropy coefficient. Stage 1 bumped to 0.05 to break out of the
# Stage-0 local optimum; decays toward the canonical 0.01 in later stages.
ENTROPY_SCHEDULE: list[float] = [0.01, 0.05, 0.03, 0.02, 0.01]

# ---------------------------------------------------------------------------
# PPO hyper-parameters (match relaxed variant)
# ---------------------------------------------------------------------------
TRAIN_BATCH_SIZE    = 8_192
MINIBATCH_SIZE      = 512
NUM_SGD_ITER        = 10
CLIP_PARAM          = 0.2
LR                  = 2.5e-4
GAMMA               = 0.99
GAE_LAMBDA          = 0.95
VF_LOSS_COEFF       = 0.5
GRAD_CLIP           = 1.0
NUM_ROLLOUT_WORKERS = 4

CHECKPOINT_ROOT = _ROOT / "models" / "transformer_gtrxl_stage1"


# ---------------------------------------------------------------------------
# Metrics callback
# ---------------------------------------------------------------------------
class MetricsCallback(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode,
                       env_index: int, **kwargs) -> None:
        try:
            info: dict = episode.last_info_for()
        except Exception:
            return
        ndr = info.get("ndr")
        if ndr is None:
            return
        episode.custom_metrics["ndr_pct"]       = float(ndr) * 100.0
        episode.custom_metrics["jain_fairness"] = float(info.get("jains", 0.0))


# ---------------------------------------------------------------------------
# Algorithm builder
# ---------------------------------------------------------------------------
def build_algorithm(stage_cfg: dict[str, Any], model_cfg: dict,
                    entropy_coeff: float) -> Any:
    env_cfg = {
        "grid_size":   stage_cfg["grid_size"],
        "num_sensors": stage_cfg["num_sensors"],
    }

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(env="UAVTransformerEnv", env_config=env_cfg)
        .training(
            model=model_cfg,
            train_batch_size=TRAIN_BATCH_SIZE,
            minibatch_size=MINIBATCH_SIZE,
            num_epochs=NUM_SGD_ITER,
            clip_param=CLIP_PARAM,
            lr=LR,
            gamma=GAMMA,
            lambda_=GAE_LAMBDA,
            entropy_coeff=entropy_coeff,
            vf_loss_coeff=VF_LOSS_COEFF,
            grad_clip=GRAD_CLIP,
        )
        .env_runners(
            num_env_runners=NUM_ROLLOUT_WORKERS,
            rollout_fragment_length=512,
            batch_mode="truncate_episodes",
            sample_timeout_s=300,
        )
        .resources(num_gpus=NUM_GPUS, num_cpus_per_worker=1)
        .callbacks(MetricsCallback)
        .reporting(min_sample_timesteps_per_iteration=TRAIN_BATCH_SIZE)
        .debugging(log_level="WARNING")
    )
    return config.build_algo()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _advance_workers(algo: Any, stage_cfg: dict[str, Any]) -> None:
    grid   = stage_cfg["grid_size"]
    n_sens = stage_cfg["num_sensors"]
    algo.workers.foreach_env(lambda e: e.reconfigure(grid, n_sens))
    algo.workers.local_worker().foreach_env(lambda e: e.reconfigure(grid, n_sens))
    log.info("All workers reconfigured → grid=%s, num_sensors=%d", grid, n_sens)


def _set_entropy(algo: Any, coeff: float) -> None:
    """Best-effort mutation of entropy_coeff across all policies.

    RLlib's PPO TorchPolicyV2 stores the active coefficient on the policy
    instance. We mutate both the policy attribute and the config dict on
    every rollout + local worker so subsequent SGD steps pick it up.
    """
    def setter(policy, pid):
        try:
            policy.entropy_coeff = coeff
        except Exception as e:
            log.warning("Failed to set policy.entropy_coeff: %s", e)
        try:
            if hasattr(policy, "config") and isinstance(policy.config, dict):
                policy.config["entropy_coeff"] = coeff
        except Exception:
            pass

    try:
        algo.workers.foreach_policy(setter)
    except Exception as e:
        log.warning("foreach_policy entropy update failed: %s", e)

    try:
        p = algo.get_policy()
        if p is not None:
            setter(p, "default_policy")
    except Exception:
        pass

    log.info("Entropy coefficient set to %.4f across all policies", coeff)


def _extract_metrics(result: dict) -> tuple[float, float]:
    env_r = result.get("env_runners", {})
    cm = env_r.get("custom_metrics") or result.get("custom_metrics", {})
    ndr  = cm.get("ndr_pct_mean",       cm.get("ndr_pct",       0.0))
    jain = cm.get("jain_fairness_mean", cm.get("jain_fairness", 0.0))
    return float(ndr), float(jain)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train() -> None:
    os.environ["PYTHONPATH"] = str(_SRC)
    os.environ["RAY_TRAIN_ENABLE_LIBUV"] = "0"
    os.environ.pop("VIRTUAL_ENV", None)
    os.environ["UV_PROJECT_ENVIRONMENT"] = "/workspace/uav/.venv"
    os.environ["UV_NO_SYNC"] = "1"

    ray.init(
        ignore_reinit_error=True,
        num_cpus=8,
        num_gpus=NUM_GPUS,
        _temp_dir="/workspace/ray_tmp_stage1",
        runtime_env={
            "env_vars": {
                "PYTHONPATH": str(_SRC),
                "UV_PROJECT_ENVIRONMENT": "/workspace/uav/.venv",
                "UV_NO_SYNC": "1",
            }
        },
    )
    tune.register_env(
        "UAVTransformerEnv",
        lambda cfg: TransformerObsWrapper(cfg),
    )
    model_cfg = register_model()

    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    log.info("Checkpoints → %s", CHECKPOINT_ROOT)

    # ── Build algo at the starting stage, then restore Stage-0 weights ────
    start_cfg     = CURRICULUM_STAGES[START_STAGE]
    start_entropy = ENTROPY_SCHEDULE[START_STAGE]
    log.info("Building algo for %s with entropy=%.4f",
             start_cfg["name"], start_entropy)
    algo = build_algorithm(start_cfg, model_cfg, start_entropy)

    log.info("Restoring weights from %s", RESTORE_FROM)
    algo.restore(RESTORE_FROM)
    log.info("Restore complete. Reconfiguring workers to Stage %d env.",
             START_STAGE)
    _advance_workers(algo, start_cfg)
    # Re-assert entropy after restore (restore may reload policy config).
    _set_entropy(algo, start_entropy)

    stage_idx = START_STAGE
    total_timesteps: int = 0
    ndr_window:  deque[float] = deque(maxlen=int(ADVANCE_CRITERIA["window"]))
    jain_window: deque[float] = deque(maxlen=int(ADVANCE_CRITERIA["window"]))
    stage_timesteps: int = 0

    iteration: int = 0
    window_size = int(ADVANCE_CRITERIA["window"])
    min_ts      = int(ADVANCE_CRITERIA["min_timesteps_per_stage"])

    while stage_idx < len(CURRICULUM_STAGES):
        stage_name = CURRICULUM_STAGES[stage_idx]["name"]

        result = algo.train()
        iteration += 1

        ts_this_iter: int = result.get("num_env_steps_sampled", TRAIN_BATCH_SIZE)
        total_timesteps  += ts_this_iter
        stage_timesteps  += ts_this_iter

        ndr, jain = _extract_metrics(result)
        ndr_window.append(ndr)
        jain_window.append(jain)

        mean_ndr  = float(np.mean(ndr_window))  if ndr_window  else 0.0
        mean_jain = float(np.mean(jain_window)) if jain_window else 0.0

        if iteration % 10 == 0:
            log.info(
                "[%s] iter=%d | Σts=%s | NDR=%.1f%% (roll=%.1f%%) | "
                "JFI=%.3f (roll=%.3f) | reward=%.1f",
                stage_name, iteration,
                f"{total_timesteps:,}",
                ndr, mean_ndr,
                jain, mean_jain,
                (
                    result.get("env_runners", {}).get(
                        "episode_return_mean",
                        result.get("env_runners", {}).get(
                            "episode_reward_mean",
                            result.get(
                                "episode_return_mean",
                                result.get("episode_reward_mean", float("nan")),
                            ),
                        ),
                    )
                ),
            )
            import shutil
            stage_ckpt = CHECKPOINT_ROOT / f"stage_{stage_idx}_progress"
            stage_ckpt.mkdir(parents=True, exist_ok=True)
            algo.save(str(stage_ckpt))
            for old in sorted(stage_ckpt.glob("checkpoint_*"))[:-1]:
                shutil.rmtree(old, ignore_errors=True)

        window_full   = len(ndr_window) >= window_size
        ndr_gate_met  = mean_ndr  >= ADVANCE_CRITERIA["ndr_pct"]
        jain_gate_met = mean_jain >= ADVANCE_CRITERIA["jains"]
        timestep_gate = stage_timesteps >= min_ts

        if window_full and ndr_gate_met and jain_gate_met and timestep_gate:
            ckpt_path = CHECKPOINT_ROOT / f"stage_{stage_idx}_final"
            algo.save(str(ckpt_path))
            log.info(
                "Stage %d COMPLETE → NDR=%.1f%%, JFI=%.3f | checkpoint saved to %s",
                stage_idx, mean_ndr, mean_jain, ckpt_path,
            )

            stage_idx += 1
            if stage_idx >= len(CURRICULUM_STAGES):
                log.info("All curriculum stages complete. Training finished.")
                break

            next_cfg     = CURRICULUM_STAGES[stage_idx]
            next_entropy = ENTROPY_SCHEDULE[stage_idx]
            log.info("Advancing to %s (entropy=%.4f)",
                     next_cfg["name"], next_entropy)
            _advance_workers(algo, next_cfg)
            _set_entropy(algo, next_entropy)

            ndr_window.clear()
            jain_window.clear()
            stage_timesteps = 0

    final_path = CHECKPOINT_ROOT / "final"
    algo.save(str(final_path))
    log.info("Final model saved → %s", final_path)
    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    train()
