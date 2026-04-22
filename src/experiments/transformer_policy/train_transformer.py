"""
train_transformer.py
====================
PPO + GTrXL training script for UAV IoT data-collection with a 5-stage
competence-based curriculum.

Usage
-----
    PYTHONIOENCODING=utf-8 uv run python \\
        src/experiments/transformer_policy/train_transformer.py

Hardware assumption
-------------------
Two CUDA GPUs (e.g. dual RTX 3090 / A100).  The script falls back to CPU or
a single GPU automatically if fewer are available.

Architecture note
-----------------
GTrXLNet is a TorchModelV2 (legacy model API) and therefore requires
`enable_rl_module_and_learner=False`.  Data-parallel multi-GPU training is
activated by setting `num_gpus=NUM_GPUS` in resources() and confirmed by
RLlib's internal TorchPolicy trainer.

To migrate to the full New API Stack (TorchRLModule + LearnerGroup) wrap
GTrXLNet in a TorchStatefulEncoderRLModule and replace the resources() call
with:
    .api_stack(enable_rl_module_and_learner=True,
               enable_env_runner_and_connector_v2=True)
    .learners(num_learners=2, num_gpus_per_learner=1)

Curriculum
----------
Five stages advance when a rolling window of 50 consecutive evaluation
episodes satisfies BOTH:
    NDR  (New Discovery Rate)  ≥ 95 %   — fraction of sensors visited
    JFI  (Jain's Fairness)     ≥ 0.85   — equity of data collected

NDR and JFI are computed from the info dict returned by UAVEnvironment.step()
and logged as custom metrics via MetricsCallback.

Stage map
---------
  0: 100×100,  10 sensors
  1: 200×200,  20 sensors
  2: 300×300,  30 sensors
  3: 400×400,  40 sensors
  4: 500×500,  50 sensors  ← final target
"""

from __future__ import annotations

import os
import sys
import pathlib
import logging
import math
from collections import deque
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Path setup — must come before any project imports
# ---------------------------------------------------------------------------
_HERE = pathlib.Path(__file__).resolve()
_ROOT = _HERE.parents[3]   # project root  (…/project/)
_SRC  = _HERE.parents[2]   # src/ — needed for `from experiments…` and `from environment…`
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# ---------------------------------------------------------------------------
# Ray / RLlib imports
# ---------------------------------------------------------------------------
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.callbacks import DefaultCallbacks

# ---------------------------------------------------------------------------
# Project imports
# ---------------------------------------------------------------------------
from experiments.transformer_policy.transformer_model import register_model
from experiments.transformer_policy.env_wrapper import TransformerObsWrapper

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
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
# Curriculum stages
# ---------------------------------------------------------------------------
CURRICULUM_STAGES: list[dict[str, Any]] = [
    {"grid_size": (100, 100), "num_sensors": 10,  "name": "Stage-0 · 100×100  · 10 sensors"},
    {"grid_size": (200, 200), "num_sensors": 20,  "name": "Stage-1 · 200×200  · 20 sensors"},
    {"grid_size": (300, 300), "num_sensors": 30,  "name": "Stage-2 · 300×300  · 30 sensors"},
    {"grid_size": (400, 400), "num_sensors": 40,  "name": "Stage-3 · 400×400  · 40 sensors"},
    {"grid_size": (500, 500), "num_sensors": 50,  "name": "Stage-4 · 500×500  · 50 sensors"},
]

# Competence gate — advance only when both criteria hold over WINDOW episodes.
ADVANCE_CRITERIA: dict[str, float] = {
    "ndr_pct": 95.0,   # New Discovery Rate threshold (%)
    "jains":   0.85,   # Jain's Fairness Index threshold
    "window":  50,     # rolling evaluation window length
    "min_timesteps_per_stage": 200_000,  # never advance before this
}

# ---------------------------------------------------------------------------
# PPO hyper-parameters
# ---------------------------------------------------------------------------
TRAIN_BATCH_SIZE  = 32_768
MINIBATCH_SIZE    = 2_048
NUM_SGD_ITER      = 10
CLIP_PARAM        = 0.2
LR                = 2.5e-4
GAMMA             = 0.99
GAE_LAMBDA        = 0.95
ENTROPY_COEFF     = 0.01
VF_LOSS_COEFF     = 0.5
GRAD_CLIP         = 1.0
NUM_ROLLOUT_WORKERS = 4    # 4 keeps CPU headroom when running alongside relational + DQN

CHECKPOINT_ROOT = _ROOT / "models" / "transformer_gtrxl"

# ---------------------------------------------------------------------------
# Metrics callback
# ---------------------------------------------------------------------------

class MetricsCallback(DefaultCallbacks):
    """
    Computes NDR and Jain's Fairness Index from the terminal info dict and
    logs them as custom metrics so the training loop can read them from
    result["custom_metrics"]["ndr_mean"] etc.
    """

    def on_episode_end(
        self,
        *,
        worker,
        base_env,
        policies,
        episode,
        env_index: int,
        **kwargs,
    ) -> None:
        try:
            info: dict = episode.last_info_for()
        except Exception:
            return

        ratios: list[float] = info.get("sensor_collection_ratios", [])
        if not ratios:
            return

        n = len(ratios)
        visited = sum(1 for r in ratios if r > 0.0)

        ndr_pct: float = visited / n * 100.0

        sum_r  = sum(ratios)
        sum_r2 = sum(r * r for r in ratios)
        jain: float = (
            (sum_r ** 2) / (n * sum_r2 + 1e-9)
            if sum_r2 > 0 else 0.0
        )

        episode.custom_metrics["ndr_pct"] = ndr_pct
        episode.custom_metrics["jain_fairness"] = jain


# ---------------------------------------------------------------------------
# Algorithm builder
# ---------------------------------------------------------------------------

def build_algorithm(stage_cfg: dict[str, Any], model_cfg: dict) -> Any:
    """Construct a PPO algorithm for the given curriculum stage."""

    env_cfg = {
        "grid_size":   stage_cfg["grid_size"],
        "num_sensors": stage_cfg["num_sensors"],
    }

    config = (
        PPOConfig()
        # ── API stack: old learner for GTrXLNet compatibility ──────────────
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        # ── environment ───────────────────────────────────────────────────
        .environment(
            env="UAVTransformerEnv",
            env_config=env_cfg,
        )
        # ── model ─────────────────────────────────────────────────────────
        .training(
            model=model_cfg,
            train_batch_size=TRAIN_BATCH_SIZE,
            minibatch_size=MINIBATCH_SIZE,
            num_epochs=NUM_SGD_ITER,
            clip_param=CLIP_PARAM,
            lr=LR,
            gamma=GAMMA,
            lambda_=GAE_LAMBDA,
            entropy_coeff=ENTROPY_COEFF,
            vf_loss_coeff=VF_LOSS_COEFF,
            grad_clip=GRAD_CLIP,
        )
        # ── env runners (.rollouts() is removed in Ray 2.55; use .env_runners()) ──
        .env_runners(
            num_env_runners=NUM_ROLLOUT_WORKERS,
            rollout_fragment_length="auto",
            # Attention-based models require entire episodes rather than
            # fixed-length fragments to maintain recurrent state integrity.
            batch_mode="complete_episodes",
        )
        # ── resources ─────────────────────────────────────────────────────
        # num_gpus drives data-parallel multi-GPU training on the old learner
        # stack.  Replace with .learners(num_learners=2, num_gpus_per_learner=1)
        # when migrating to the New API Stack TorchRLModule path.
        .resources(
            num_gpus=NUM_GPUS,
            num_cpus_per_worker=1,
        )
        # ── callbacks ─────────────────────────────────────────────────────
        .callbacks(MetricsCallback)
        # ── reporting ─────────────────────────────────────────────────────
        .reporting(min_sample_timesteps_per_iteration=TRAIN_BATCH_SIZE)
        .debugging(log_level="WARNING")
    )

    # build_algo() is the Ray 2.x preferred alias for build()
    return config.build_algo()


# ---------------------------------------------------------------------------
# Curriculum helpers
# ---------------------------------------------------------------------------

def _advance_workers(algo: Any, stage_cfg: dict[str, Any]) -> None:
    """Push new grid/sensor config to all rollout workers."""
    grid   = stage_cfg["grid_size"]
    n_sens = stage_cfg["num_sensors"]

    algo.workers.foreach_env(lambda e: e.reconfigure(grid, n_sens))
    # Also update the local worker (used for evaluation).
    algo.workers.local_worker().foreach_env(lambda e: e.reconfigure(grid, n_sens))

    log.info(
        "All workers reconfigured → grid=%s, num_sensors=%d",
        grid, n_sens,
    )


def _extract_metrics(result: dict) -> tuple[float, float]:
    """Return (mean_ndr_pct, mean_jain) from a training result dict."""
    cm = result.get("custom_metrics", {})
    ndr  = cm.get("ndr_pct_mean",       cm.get("ndr_pct",       0.0))
    jain = cm.get("jain_fairness_mean", cm.get("jain_fairness", 0.0))
    return float(ndr), float(jain)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train() -> None:
    # Set env vars on the parent process so the raylet inherits them directly
    # (runtime_env env_vars reach worker *Python* processes but not the raylet's
    # uv invocation that happens before workers start).
    os.environ["PYTHONPATH"] = str(_SRC)
    os.environ["RAY_TRAIN_ENABLE_LIBUV"] = "0"
    os.environ.pop("VIRTUAL_ENV", None)  # prevent uv venv-detection confusion in workers
    # UV_PROJECT_ENVIRONMENT tells uv to reuse the existing venv instead of
    # creating a fresh copy in the raylet's temp dir (avoids the 500 MB copy
    # that exhausts disk quota and causes worker startup timeouts).
    os.environ["UV_PROJECT_ENVIRONMENT"] = "/workspace/uav/.venv"
    os.environ["UV_NO_SYNC"] = "1"  # skip uv sync entirely in the temp dir

    ray.init(
        ignore_reinit_error=True,
        num_cpus=8,
        num_gpus=NUM_GPUS,
        _temp_dir="/workspace/ray_tmp",
        runtime_env={
            # Propagate to worker Python processes as a belt-and-suspenders
            # measure (the os.environ above covers the raylet's uv phase).
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

    # Build the algorithm with Stage 0 config first.
    algo = build_algorithm(CURRICULUM_STAGES[0], model_cfg)
    log.info("Algorithm built.  Starting curriculum.")

    stage_idx = 0
    total_timesteps: int = 0

    # Rolling windows per stage — reset at each stage transition.
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
                result.get("episode_reward_mean", float("nan")),
            )
            # Periodic checkpoint — keep only the newest subdir to avoid
            # disk exhaustion (each save creates a new checkpoint_XXXXXX dir).
            import shutil
            stage_ckpt = CHECKPOINT_ROOT / f"stage_{stage_idx}_progress"
            stage_ckpt.mkdir(parents=True, exist_ok=True)
            algo.save(str(stage_ckpt))
            for old in sorted(stage_ckpt.glob("checkpoint_*"))[:-1]:
                shutil.rmtree(old, ignore_errors=True)

        # ── Competence gate ────────────────────────────────────────────
        window_full     = len(ndr_window)  >= window_size
        ndr_gate_met    = mean_ndr  >= ADVANCE_CRITERIA["ndr_pct"]
        jain_gate_met   = mean_jain >= ADVANCE_CRITERIA["jains"]
        timestep_gate   = stage_timesteps >= min_ts

        if window_full and ndr_gate_met and jain_gate_met and timestep_gate:
            # Save stage checkpoint.
            ckpt_path = CHECKPOINT_ROOT / f"stage_{stage_idx}_final"
            algo.save(str(ckpt_path))
            log.info(
                "Stage %d COMPLETE → NDR=%.1f%%, JFI=%.3f | "
                "checkpoint saved to %s",
                stage_idx, mean_ndr, mean_jain, ckpt_path,
            )

            stage_idx += 1
            if stage_idx >= len(CURRICULUM_STAGES):
                log.info("All curriculum stages complete.  Training finished.")
                break

            # Advance to next stage.
            log.info(
                "Advancing to %s", CURRICULUM_STAGES[stage_idx]["name"]
            )
            _advance_workers(algo, CURRICULUM_STAGES[stage_idx])

            # Reset rolling windows for the new stage.
            ndr_window.clear()
            jain_window.clear()
            stage_timesteps = 0

    # Final checkpoint.
    final_path = CHECKPOINT_ROOT / "final"
    algo.save(str(final_path))
    log.info("Final model saved → %s", final_path)
    algo.stop()
    ray.shutdown()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    train()
