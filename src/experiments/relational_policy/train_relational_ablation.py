"""
Training script: Relational RL Ablation Baseline (Temporal Memory experiment).

Purpose
-------
Retrain the Relational RL (PPO) model as a rigorous ablation baseline that
isolates the contribution of temporal memory (DQN frame-stack / GTrXL) from
the inductive bias of permutation-invariant attention.

Key differences from train_relational.py
-----------------------------------------
  1. N=20 sensors throughout all curriculum stages — matches the dissertation's
     primary evaluation configuration (500×500, N=20) for direct comparability.
  2. No cluster dwell bonus — any hovering near sensors is purely emergent,
     driven by LoRaWAN EMA-ADR physics (see env_wrapper_ablation.py).
  3. Per-step penalty (STEP_PENALTY=1.0) — mild pressure against aimless motion;
     quantifies the energy cost of stateless decision-making.
  4. Warm-started from the original Stage 2 checkpoint (300×300, N=20) — the
     policy already masters small-grid discovery; we train only Stages 2–4
     with the new reward signal to capture Stage 3–4 behaviour changes.
  5. All curriculum gates use NDR + Jain's (no efficiency gate) — consistent
     with the dissertation's primary metrics.

Dissertation narrative
----------------------
  Section 5.7  — "Local Controller" framing: Relational Ablation achieves high
                  Jain's (attention sees all sensor urgency) but lower NDR than
                  DQN (no temporal memory of past positions).
  Section 6.2  — Future work: GTrXL + Relational Attention fusion.

Output
------
  Checkpoints: src/agents/dqn/models/relational_rl_ablation/checkpoints/
  TensorBoard:  src/agents/dqn/models/relational_rl_ablation/tb/

Usage
-----
  uv run python src/experiments/relational_policy/train_relational_ablation.py
"""

from __future__ import annotations

import os

os.environ.setdefault("RAY_TRAIN_ENABLE_LIBUV", "0")

import sys
import logging
from pathlib import Path
from typing import Any

_HERE = Path(__file__).resolve().parent
_SRC  = _HERE.parent.parent
_ROOT = _SRC.parent
for _p in (str(_HERE), str(_SRC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from experiments.relational_policy.env_wrapper import EpisodeMetricsStore, GAMMA, N_MAX
from experiments.relational_policy.env_wrapper_ablation import AblationUAVEnv
from experiments.relational_policy.relational_module import RelationalUAVModule


# ── Metrics callback ──────────────────────────────────────────────────────────

class AblationMetricsCallback(RLlibCallback):
    """
    Reads NDR / Jain's / Efficiency / Gini / min_cr / max_cr from the terminal
    info dict and logs them via metrics_logger so Ray aggregates across workers.
    """

    def on_episode_end(
        self,
        *,
        episode,
        env_runner=None,
        metrics_logger=None,
        env=None,
        env_index=None,
        rl_module=None,
        **kwargs,
    ) -> None:
        if metrics_logger is None:
            return
        try:
            infos = episode.get_infos()
            info  = infos[-1] if infos else {}
        except Exception:
            return

        ndr = info.get("ndr")
        if ndr is None:
            return

        metrics_logger.log_value("ndr",        float(ndr),                        window=20)
        metrics_logger.log_value("jains",       float(info.get("jains",   0.0)),   window=20)
        metrics_logger.log_value("efficiency",  float(info.get("efficiency", 0.0)), window=20)
        metrics_logger.log_value("gini",        float(info.get("gini",    0.0)),   window=20)
        metrics_logger.log_value("min_cr",      float(info.get("min_cr",  0.0)),   window=20)
        metrics_logger.log_value("max_cr",      float(info.get("max_cr",  0.0)),   window=20)


# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Output paths ──────────────────────────────────────────────────────────────

_MODELS_BASE = _ROOT / "src" / "agents" / "dqn" / "models" / "relational_rl_ablation"
CKPT_DIR     = _MODELS_BASE / "checkpoints"
TB_DIR       = _MODELS_BASE / "tb"
_MODELS_BASE.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
TB_DIR.mkdir(parents=True, exist_ok=True)

# Warm-start: load Stage 2 weights from the original Relational RL training.
# Stage 2 (300×300, N=20) is the first stage where the policy encounters the
# full N=20 sensor count, so its weights are a valid prior for the ablation.
_ORIGINAL_CKPT_BASE = (
    _ROOT / "src" / "agents" / "dqn" / "models" / "relational_rl" /
    "results" / "checkpoints"
)
WARMSTART_STAGE     = 2
WARMSTART_CKPT      = _ORIGINAL_CKPT_BASE / f"stage_{WARMSTART_STAGE}" / "final"

# The ablation runs stages 2–4 only (0–1 are covered by the warm-start).
START_STAGE = 2


# ── Curriculum ────────────────────────────────────────────────────────────────
# N capped at 20 throughout — matches the dissertation evaluation config.
# Stage 0 and 1 are defined for completeness but skipped (warm-start).

ABLATION_CURRICULUM: list[tuple] = [
    (100, 100, 10,  0.05,  "100x100 N=10"),    # stage 0  — skipped
    (200, 200, 15,  0.04,  "200x200 N=15"),    # stage 1  — skipped
    (300, 300, 20,  0.02,  "300x300 N=20"),    # stage 2  — warm-start entry
    (400, 400, 20,  0.008, "400x400 N=20"),    # stage 3
    (500, 500, 20,  0.003, "500x500 N=20"),    # stage 4  — dissertation target
]


# ── Stage gates ───────────────────────────────────────────────────────────────
# All stages use NDR + Jain's — consistent with the dissertation's primary
# metrics and avoids the efficiency gate that was tuned for N=50 Stage 4.

def _gate_ndr95(m):       return m["ndr"] >= 0.95
def _gate_ndr90_jain(m):  return m["ndr"] >= 0.90 and m["jains"] >= 0.80

ABLATION_STAGE_GATES = [
    _gate_ndr95,       # stage 0
    _gate_ndr95,       # stage 1
    _gate_ndr90_jain,  # stage 2
    _gate_ndr90_jain,  # stage 3
    _gate_ndr90_jain,  # stage 4
]

MAX_ITERS_PER_STAGE = 200
TOTAL_ENV_STEPS     = 10_000_000


# ── Model hyperparameters (identical to original) ─────────────────────────────

MODULE_CONFIG: dict[str, Any] = {
    "d_model":     128,
    "n_heads":     4,
    "gru_hidden":  256,
    "dropout":     0.1,
    "max_seq_len": 20,
}

BASE_ENV_CONFIG: dict[str, Any] = {
    "max_steps":   2100,
    "max_battery": 274.0,
    "n_max":       N_MAX,
}


# ── Config builder ────────────────────────────────────────────────────────────

def build_ablation_config(
    grid_w: int,
    grid_h: int,
    n_sensors: int,
    entropy_coeff: float,
) -> PPOConfig:
    env_config = {
        **BASE_ENV_CONFIG,
        "grid_size":   (grid_w, grid_h),
        "num_sensors": n_sensors,
    }
    return (
        PPOConfig()
        .environment(env="AblationUAV", env_config=env_config)
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=RelationalUAVModule,
                model_config=MODULE_CONFIG,
            )
        )
        .training(
            gamma=GAMMA,
            lr=3e-4,
            train_batch_size=8192,
            num_epochs=20,
            minibatch_size=256,
            clip_param=0.2,
            vf_clip_param=50_000,
            vf_loss_coeff=0.5,
            entropy_coeff=entropy_coeff,
            lambda_=0.95,
            grad_clip=0.5,
        )
        .env_runners(
            num_env_runners=8,
            rollout_fragment_length="auto",
        )
        .framework("torch")
        .learners(
            num_learners=1,
            num_gpus_per_learner=1,
        )
        .callbacks(AblationMetricsCallback)
        .reporting(metrics_num_episodes_for_smoothing=20)
        .checkpointing(export_native_model_files=True)
    )


# ── Stage runner ──────────────────────────────────────────────────────────────

def _run_ablation_stage(stage_idx: int, total_steps: int) -> tuple[int, bool]:
    """Run one ablation curriculum stage in an isolated Ray session."""
    grid_w, grid_h, n_sensors, ent, label = ABLATION_CURRICULUM[stage_idx]

    log.info("=" * 60)
    log.info(f"ABLATION STAGE {stage_idx}: {label}  (step_penalty={1.0}, no dwell bonus)")
    log.info(f"  entropy_coeff = {ent}")
    log.info("=" * 60)

    EpisodeMetricsStore.reset()
    os.environ["PYTHONPATH"]               = str(_SRC)
    os.environ["RAY_TRAIN_ENABLE_LIBUV"]   = "0"
    os.environ.pop("VIRTUAL_ENV", None)
    os.environ["UV_PROJECT_ENVIRONMENT"]   = "/workspace/uav/.venv"
    os.environ["UV_NO_SYNC"]               = "1"

    ray.init(
        ignore_reinit_error=True,
        num_cpus=8,
        num_gpus=1,
        _temp_dir="/workspace/ray_tmp",
        runtime_env={
            "env_vars": {
                "PYTHONPATH":             str(_SRC),
                "UV_PROJECT_ENVIRONMENT": "/workspace/uav/.venv",
                "UV_NO_SYNC":             "1",
            }
        },
    )
    tune.register_env("AblationUAV", lambda cfg: AblationUAVEnv(**cfg))

    config = build_ablation_config(grid_w, grid_h, n_sensors, ent)
    algo   = config.build_algo()

    # ── Warm-start logic ──────────────────────────────────────────────────────
    # Stage 2 (first ablation stage): load weights from original Stage 2 ckpt.
    # Stages 3+: load weights from the previous ablation stage checkpoint.
    if stage_idx == START_STAGE:
        rl_module_ckpt = WARMSTART_CKPT / "learner_group" / "learner" / "rl_module"
        if rl_module_ckpt.exists():
            log.info(
                f"Warm-starting from original Stage {WARMSTART_STAGE} checkpoint: "
                f"{rl_module_ckpt}"
            )
            algo.learner_group.foreach_learner(
                func=lambda learner, _p=str(rl_module_ckpt): learner.module.restore_from_path(_p)
            )
            algo.env_runner_group.sync_weights(
                from_worker_or_learner_group=algo.learner_group
            )
            log.info("Warm-start weights loaded and synced to env runners")
        else:
            log.warning(
                f"Warm-start checkpoint not found at {rl_module_ckpt}. "
                f"Stage {stage_idx} starts cold."
            )
    elif stage_idx > START_STAGE:
        prev_ckpt = CKPT_DIR / f"stage_{stage_idx - 1}" / "final"
        rl_module_ckpt = prev_ckpt / "learner_group" / "learner" / "rl_module"
        if rl_module_ckpt.exists():
            log.info(f"Restoring ablation stage {stage_idx - 1} weights from {rl_module_ckpt}")
            algo.learner_group.foreach_learner(
                func=lambda learner, _p=str(rl_module_ckpt): learner.module.restore_from_path(_p)
            )
            algo.env_runner_group.sync_weights(
                from_worker_or_learner_group=algo.learner_group
            )
        else:
            log.warning(
                f"No ablation checkpoint at {rl_module_ckpt}; stage {stage_idx} starts cold"
            )

    stage_ckpt_dir = CKPT_DIR / f"stage_{stage_idx}"
    stage_ckpt_dir.mkdir(parents=True, exist_ok=True)

    gate    = ABLATION_STAGE_GATES[stage_idx]
    advanced = False

    for iteration in range(MAX_ITERS_PER_STAGE):
        result  = algo.train()
        env_r   = result.get("env_runners", {})
        ep_ret  = env_r.get("episode_return_mean", float("nan"))
        lifetime_steps = result.get("num_env_steps_sampled_lifetime", 0)
        total_steps    = lifetime_steps

        n_eps = int(env_r.get("num_episodes", 0))
        m = {
            "ndr":        float(env_r.get("ndr",        0.0)),
            "jains":      float(env_r.get("jains",      0.0)),
            "efficiency": float(env_r.get("efficiency", 0.0)),
            "n_episodes": n_eps,
        }
        gini   = float(env_r.get("gini",   0.0))
        min_cr = float(env_r.get("min_cr", 0.0))
        max_cr = float(env_r.get("max_cr", 0.0))

        log.info(
            f"[ablation stage {stage_idx} | iter {iteration:3d}] "
            f"NDR={m['ndr']:.3f}  Jain={m['jains']:.3f}  Gini={gini:.3f}  "
            f"minCR={min_cr:.3f}  maxCR={max_cr:.3f}  "
            f"Eff={m['efficiency']:.1f}B/Wh  ret={ep_ret:.2f}  "
            f"steps={int(lifetime_steps):,}"
        )

        if (iteration + 1) % 10 == 0:
            import shutil
            ckpt_path = algo.save(str(stage_ckpt_dir))
            subdirs = sorted(stage_ckpt_dir.glob("checkpoint_*"))
            for old in subdirs[:-1]:
                shutil.rmtree(old, ignore_errors=True)
            log.info(f"Checkpoint saved → {ckpt_path}")

        ready = m["n_episodes"] >= 5
        if ready and gate(m):
            log.info(
                f"  ✓ Ablation stage {stage_idx} complete — "
                f"NDR={m['ndr']:.3f}  Jain={m['jains']:.3f}  "
                f"Gini={gini:.3f}  (n={m['n_episodes']} eps)"
            )
            advanced = True
            break

    final_ckpt = stage_ckpt_dir / "final"
    final_ckpt.mkdir(parents=True, exist_ok=True)
    algo.save_to_path(str(final_ckpt))
    log.info(f"Ablation stage {stage_idx} final checkpoint → {final_ckpt}")

    algo.stop()
    ray.shutdown()

    return total_steps, advanced


# ── Entry point ───────────────────────────────────────────────────────────────

def train() -> None:
    total_steps = 0
    log.info(
        f"Ablation training: stages {START_STAGE}–{len(ABLATION_CURRICULUM) - 1}, "
        f"N=20 throughout, step_penalty=1.0, no dwell bonus."
    )
    log.info(f"Warm-start source: {WARMSTART_CKPT}")
    log.info(f"Output:            {CKPT_DIR}")

    for stage_idx in range(START_STAGE, len(ABLATION_CURRICULUM)):
        if total_steps >= TOTAL_ENV_STEPS:
            break
        total_steps, _ = _run_ablation_stage(stage_idx, total_steps)

    log.info("Ablation training complete.")


if __name__ == "__main__":
    train()
