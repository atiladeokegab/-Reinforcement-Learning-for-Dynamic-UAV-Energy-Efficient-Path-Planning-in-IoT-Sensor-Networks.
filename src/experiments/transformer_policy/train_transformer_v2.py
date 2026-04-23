"""
train_transformer_v2.py
=======================
Clean end-to-end curriculum run for the GTrXL transformer policy.

Differences from the stage-specific scripts
--------------------------------------------
* Restores from stage_1_progress (keeps the 24h of compute already done).
* NDR gate relaxed to 65% (physically achievable on each grid size).
* JFI gate relaxed to 0.80.
* MAX_ITERS_PER_STAGE = 600 safety valve: auto-advances when a stage stalls
  so the run always completes rather than hanging forever.
* Single script covers Stage 1 → Stage 4 (200×200 → 500×500).

Curriculum
----------
  Stage 1 : 200×200,  20 sensors  (restores from stage_1_progress)
  Stage 2 : 300×300,  30 sensors
  Stage 3 : 400×400,  40 sensors
  Stage 4 : 500×500,  50 sensors

Graduation gate (per stage)
----------------------------
  NDR rolling mean ≥ 65%  OR  stage iters ≥ 600
  JFI rolling mean ≥ 0.80 OR  stage iters ≥ 600
  50-episode rolling window, ≥ 200 k min timesteps

Outputs
-------
  models/transformer_gtrxl_v2/stage_<N>_progress   (every 10 iters)
  models/transformer_gtrxl_v2/stage_<N>_final      (on graduation)
  models/transformer_gtrxl_v2/final                (end of run)

Usage
-----
    PYTHONIOENCODING=utf-8 uv run python \\
        src/experiments/transformer_policy/train_transformer_v2.py
"""

from __future__ import annotations

import os
import sys
import shutil
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
# Hardware
# ---------------------------------------------------------------------------
try:
    import torch
    _CUDA_COUNT = torch.cuda.device_count()
except ImportError:
    _CUDA_COUNT = 0

NUM_GPUS: int = min(_CUDA_COUNT, 2)
log.info("Detected %d CUDA GPU(s). Using %d for training.", _CUDA_COUNT, NUM_GPUS)

# ---------------------------------------------------------------------------
# Restore point — the 24h Stage-1 run already in place
# ---------------------------------------------------------------------------
START_STAGE  = 1
RESTORE_FROM = "/workspace/uav/models/transformer_gtrxl_stage1/stage_1_progress"

# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------
CURRICULUM_STAGES: list[dict[str, Any]] = [
    {"grid_size": (100, 100), "num_sensors": 10, "name": "Stage-0 · 100×100 · 10 sensors"},
    {"grid_size": (200, 200), "num_sensors": 20, "name": "Stage-1 · 200×200 · 20 sensors"},
    {"grid_size": (300, 300), "num_sensors": 30, "name": "Stage-2 · 300×300 · 30 sensors"},
    {"grid_size": (400, 400), "num_sensors": 40, "name": "Stage-3 · 400×400 · 40 sensors"},
    {"grid_size": (500, 500), "num_sensors": 50, "name": "Stage-4 · 500×500 · 50 sensors"},
]

# Relaxed gates: achievable based on observed performance
ADVANCE_CRITERIA: dict[str, float] = {
    "ndr_pct":                65.0,
    "jains":                  0.80,
    "window":                 50,
    "min_timesteps_per_stage": 200_000,
}

# Safety valve: force-advance after this many iters even if gate not met.
# At ~1 min/iter this caps each stage at ~10 hours.
MAX_ITERS_PER_STAGE = 600

# Entropy decays toward canonical 0.01 as environments grow harder.
ENTROPY_SCHEDULE: list[float] = [0.01, 0.05, 0.03, 0.02, 0.01]

# ---------------------------------------------------------------------------
# PPO hyper-parameters (unchanged from working stage1 run)
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

CHECKPOINT_ROOT = _ROOT / "models" / "transformer_gtrxl_v2"


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
def _get_worker_group(algo: Any) -> Any:
    grp = getattr(algo, "env_runner_group", None)
    if grp is None:
        w = getattr(algo, "workers", None)
        grp = w() if callable(w) else w
    return grp


def _advance_workers(algo: Any, stage_cfg: dict[str, Any]) -> None:
    grid   = stage_cfg["grid_size"]
    n_sens = stage_cfg["num_sensors"]
    grp = _get_worker_group(algo)
    grp.foreach_env(lambda e: e.reconfigure(grid, n_sens))
    local = (
        getattr(grp, "local_env_runner", None)
        or (grp.local_worker() if hasattr(grp, "local_worker") else None)
    )
    if local is not None:
        local.foreach_env(lambda e: e.reconfigure(grid, n_sens))
    log.info("Workers reconfigured → grid=%s, num_sensors=%d", grid, n_sens)


def _set_entropy(algo: Any, coeff: float) -> None:
    def setter(policy, pid):
        try:
            policy.entropy_coeff = coeff
        except Exception as exc:
            log.warning("Failed to set policy.entropy_coeff: %s", exc)
        try:
            if hasattr(policy, "config") and isinstance(policy.config, dict):
                policy.config["entropy_coeff"] = coeff
        except Exception:
            pass

    try:
        _get_worker_group(algo).foreach_policy(setter)
    except Exception as exc:
        log.warning("foreach_policy entropy update failed: %s", exc)
    try:
        p = algo.get_policy()
        if p is not None:
            setter(p, "default_policy")
    except Exception:
        pass
    log.info("Entropy → %.4f", coeff)


def _extract_metrics(result: dict) -> tuple[float, float]:
    env_r = result.get("env_runners", {})
    cm = env_r.get("custom_metrics") or result.get("custom_metrics", {})
    ndr  = cm.get("ndr_pct_mean",       cm.get("ndr_pct",       0.0))
    jain = cm.get("jain_fairness_mean", cm.get("jain_fairness", 0.0))
    return float(ndr), float(jain)


def _episode_reward(result: dict) -> float:
    env_r = result.get("env_runners", {})
    for key in ("episode_return_mean", "episode_reward_mean"):
        v = env_r.get(key, result.get(key))
        if v is not None:
            return float(v)
    return float("nan")


def _save_progress(algo: Any, stage_idx: int) -> None:
    ckpt = CHECKPOINT_ROOT / f"stage_{stage_idx}_progress"
    ckpt.mkdir(parents=True, exist_ok=True)
    algo.save(str(ckpt))
    for old in sorted(ckpt.glob("checkpoint_*"))[:-1]:
        shutil.rmtree(old, ignore_errors=True)


def _graduate_stage(algo: Any, stage_idx: int, mean_ndr: float,
                    mean_jain: float, reason: str) -> None:
    ckpt_path = CHECKPOINT_ROOT / f"stage_{stage_idx}_final"
    algo.save(str(ckpt_path))
    log.info(
        "Stage %d GRADUATE (%s) → NDR=%.1f%%, JFI=%.3f | saved %s",
        stage_idx, reason, mean_ndr, mean_jain, ckpt_path,
    )


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
        _temp_dir="/workspace/ray_tmp_v2",
        runtime_env={
            "env_vars": {
                "PYTHONPATH": str(_SRC),
                "UV_PROJECT_ENVIRONMENT": "/workspace/uav/.venv",
                "UV_NO_SYNC": "1",
            }
        },
    )
    tune.register_env("UAVTransformerEnv", lambda cfg: TransformerObsWrapper(cfg))
    model_cfg = register_model()

    CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
    log.info("Checkpoints → %s", CHECKPOINT_ROOT)

    start_cfg     = CURRICULUM_STAGES[START_STAGE]
    start_entropy = ENTROPY_SCHEDULE[START_STAGE]
    log.info("Building algo for %s (entropy=%.4f)", start_cfg["name"], start_entropy)
    algo = build_algorithm(start_cfg, model_cfg, start_entropy)

    log.info("Restoring weights from %s", RESTORE_FROM)
    algo.restore(RESTORE_FROM)
    log.info("Restore complete.")
    _advance_workers(algo, start_cfg)
    _set_entropy(algo, start_entropy)

    stage_idx       = START_STAGE
    total_timesteps = 0
    iteration       = 0

    window_size = int(ADVANCE_CRITERIA["window"])
    min_ts      = int(ADVANCE_CRITERIA["min_timesteps_per_stage"])

    ndr_window:   deque[float] = deque(maxlen=window_size)
    jain_window:  deque[float] = deque(maxlen=window_size)
    stage_ts:     int = 0
    stage_iters:  int = 0

    while stage_idx < len(CURRICULUM_STAGES):
        stage_name = CURRICULUM_STAGES[stage_idx]["name"]

        result = algo.train()
        iteration   += 1
        stage_iters += 1

        ts = result.get("num_env_steps_sampled", TRAIN_BATCH_SIZE)
        total_timesteps += ts
        stage_ts        += ts

        ndr, jain = _extract_metrics(result)
        ndr_window.append(ndr)
        jain_window.append(jain)

        mean_ndr  = float(np.mean(ndr_window))  if ndr_window  else 0.0
        mean_jain = float(np.mean(jain_window)) if jain_window else 0.0

        if iteration % 10 == 0:
            log.info(
                "[%s] iter=%d (stage_iter=%d) | Σts=%s | "
                "NDR=%.1f%% (roll=%.1f%%) | JFI=%.3f (roll=%.3f) | reward=%.1f",
                stage_name, iteration, stage_iters,
                f"{total_timesteps:,}",
                ndr, mean_ndr, jain, mean_jain,
                _episode_reward(result),
            )
            _save_progress(algo, stage_idx)

        # ── Graduation logic ────────────────────────────────────────────────
        window_full  = len(ndr_window) >= window_size
        gate_met     = (
            window_full
            and mean_ndr  >= ADVANCE_CRITERIA["ndr_pct"]
            and mean_jain >= ADVANCE_CRITERIA["jains"]
            and stage_ts  >= min_ts
        )
        max_hit = stage_iters >= MAX_ITERS_PER_STAGE

        if gate_met or max_hit:
            reason = "gate" if gate_met else f"max_iters={MAX_ITERS_PER_STAGE}"
            _graduate_stage(algo, stage_idx, mean_ndr, mean_jain, reason)

            stage_idx += 1
            if stage_idx >= len(CURRICULUM_STAGES):
                log.info("All stages complete.")
                break

            next_cfg     = CURRICULUM_STAGES[stage_idx]
            next_entropy = ENTROPY_SCHEDULE[stage_idx]
            log.info("Advancing to %s (entropy=%.4f)", next_cfg["name"], next_entropy)
            _advance_workers(algo, next_cfg)
            _set_entropy(algo, next_entropy)

            ndr_window.clear()
            jain_window.clear()
            stage_ts    = 0
            stage_iters = 0

    final_path = CHECKPOINT_ROOT / "final"
    algo.save(str(final_path))
    log.info("Final model saved → %s", final_path)
    algo.stop()
    ray.shutdown()


if __name__ == "__main__":
    train()