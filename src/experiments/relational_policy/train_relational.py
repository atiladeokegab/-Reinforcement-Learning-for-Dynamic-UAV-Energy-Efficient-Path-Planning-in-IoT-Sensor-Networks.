"""
Training script: Optimal Relational Policy via Ray RLlib PPO.
(Corrected for sparse-reward curriculum learning)
"""

from __future__ import annotations

import os
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
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from env_wrapper       import RelationalUAVEnv, EpisodeMetricsStore, GAMMA, N_MAX
from relational_module import RelationalUAVModule

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_DIR = _HERE / "results"
CKPT_DIR    = RESULTS_DIR / "checkpoints"
TB_DIR      = RESULTS_DIR / "tb"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
TB_DIR.mkdir(parents=True, exist_ok=True)

# ── Curriculum ────────────────────────────────────────────────────────────────
# (grid_W, grid_H, n_sensors, entropy_coeff_start, entropy_coeff_end, label)
# Two entropy values per stage: high at stage start, low at gate threshold.
# This mirrors the "broad search → exploitation" pattern within each stage,
# not just across the full training run.
CURRICULUM: list[tuple] = [
    (100, 100, 10,  0.05,  0.01,  "100x100 N=10"),
    (200, 200, 15,  0.04,  0.01,  "200x200 N=15"),
    (300, 300, 20,  0.02,  0.005, "300x300 N=20"),
    (400, 400, 35,  0.01,  0.003, "400x400 N=35"),
    (500, 500, N_MAX, 0.005, 0.001, "500x500 N=50"),
]

def _gate_ndr95(m):      return m["ndr"]        >= 0.95
def _gate_ndr90_jain(m): return m["ndr"]        >= 0.90 and m["jains"] >= 0.80
def _gate_efficiency(m): return m["efficiency"] >= 200.0

STAGE_GATES = [
    _gate_ndr95,
    _gate_ndr95,
    _gate_ndr90_jain,
    _gate_ndr90_jain,
    _gate_efficiency,
]

MAX_ITERS_PER_STAGE = 200
TOTAL_ENV_STEPS     = 10_000_000

# ── Model — match proven local config, scale AFTER NDR > 0.5 confirmed ───────
MODULE_CONFIG: dict[str, Any] = {
    "d_model":     128,   # proven locally — do NOT increase until NDR > 0.5
    "n_heads":     4,     # must divide d_model evenly
    "gru_hidden":  256,   # proven locally
    "dropout":     0.1,
    "max_seq_len": 20,
}

BASE_ENV_CONFIG: dict[str, Any] = {
    "max_steps":   2100,
    "max_battery": 274.0,
    "n_max":       N_MAX,
}

# ── Worker count — use available CPUs, leave 2 for driver + learner ──────────
# Adjust NUM_WORKERS to your cluster's CPU count - 2.
# With 13 CPUs available: 13 - 2 = 11 workers (round down for safety → 10)
NUM_WORKERS = int(os.environ.get("RLLIB_NUM_WORKERS", "10"))


def build_config(
    grid_w: int,
    grid_h: int,
    n_sensors: int,
    entropy_start: float,
    entropy_end: float,
    current_iter: int = 0,
    max_iters: int = MAX_ITERS_PER_STAGE,
) -> PPOConfig:
    """
    Construct a PPOConfig for one curriculum stage.

    Entropy is linearly annealed from entropy_start → entropy_end
    over the stage's iteration budget, approximating an in-stage
    exploration schedule without requiring a global step counter.
    """
    # Linear interpolation for current entropy
    progress = min(current_iter / max(max_iters - 1, 1), 1.0)
    entropy_coeff = entropy_start + progress * (entropy_end - entropy_start)

    env_config = {
        **BASE_ENV_CONFIG,
        "grid_size":   (grid_w, grid_h),
        "num_sensors": n_sensors,
    }

    return (
        PPOConfig()
        .environment(
            env="RelationalUAV",
            env_config=env_config,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=RelationalUAVModule,
                model_config=MODULE_CONFIG,
            )
        )
        .training(
            gamma=GAMMA,
            lr=1e-4,              # reduced from 3e-4 — more stable for transformer
            train_batch_size=8192,    # CRITICAL: match local — sparse signal preservation
            num_epochs=10,
            minibatch_size=256,       # CRITICAL: match local
            clip_param=0.2,
            vf_clip_param=10.0,
            vf_loss_coeff=0.5,
            entropy_coeff=entropy_coeff,
            lambda_=0.95,
            grad_clip=0.5,
        )
        .env_runners(
            num_env_runners=NUM_WORKERS,   # use available CPUs
            rollout_fragment_length=128,   # explicit — tight credit assignment
            num_envs_per_env_runner=2,     # vectorise lightly per worker
        )
        .framework("torch")
        .learners(
            num_learners=1,
            num_gpus_per_learner=1,
        )
        .reporting(
            metrics_num_episodes_for_smoothing=20,
        )
        .checkpointing(
            export_native_model_files=True,
        )
    )


def _run_stage(stage_idx: int, total_steps: int) -> tuple[int, bool]:
    """
    Run one curriculum stage in a fresh Ray session.
    Each stage rebuilds the config per-iteration to apply entropy annealing.
    """
    grid_w, grid_h, n_sensors, ent_start, ent_end, label = CURRICULUM[stage_idx]

    log.info("=" * 60)
    log.info(f"CURRICULUM STAGE {stage_idx}: {label}")
    log.info(f"  entropy: {ent_start:.4f} → {ent_end:.4f} over {MAX_ITERS_PER_STAGE} iters")
    log.info("=" * 60)

    EpisodeMetricsStore.reset()
    ray.init(ignore_reinit_error=True)
    tune.register_env("RelationalUAV", lambda cfg: RelationalUAVEnv(**cfg))

    # Build initial config (iter 0 = max entropy)
    config = build_config(grid_w, grid_h, n_sensors, ent_start, ent_end, 0)
    algo   = config.build_algo()

    # Warm-start from previous stage checkpoint
    prev_ckpt = CKPT_DIR / f"stage_{stage_idx - 1}" / "final"
    if stage_idx > 0 and prev_ckpt.exists():
        log.info(f"Restoring weights from {prev_ckpt}")
        algo.restore(str(prev_ckpt))

    stage_ckpt_dir = CKPT_DIR / f"stage_{stage_idx}"
    stage_ckpt_dir.mkdir(parents=True, exist_ok=True)

    gate     = STAGE_GATES[stage_idx]
    advanced = False

    for iteration in range(MAX_ITERS_PER_STAGE):

        # Update entropy coefficient for this iteration
        progress      = iteration / max(MAX_ITERS_PER_STAGE - 1, 1)
        entropy_now   = ent_start + progress * (ent_end - ent_start)
        algo.config.training(entropy_coeff=entropy_now)   # hot-update

        result         = algo.train()
        ep_ret         = result.get("env_runners", {}).get("episode_return_mean", float("nan"))
        lifetime_steps = result.get("num_env_steps_sampled_lifetime", 0)
        total_steps    = lifetime_steps

        m = EpisodeMetricsStore.rolling_means()
        log.info(
            f"[stage {stage_idx} | iter {iteration:3d}] "
            f"NDR={m['ndr']:.3f}  Jain={m['jains']:.3f}  "
            f"Eff={m['efficiency']:.1f}B/Wh  "
            f"(n={m['n_episodes']:2d} eps)  "
            f"ret={ep_ret:.0f}  ent={entropy_now:.4f}  "
            f"steps={int(lifetime_steps):,}"
        )

        if (iteration + 1) % 10 == 0:
            ckpt_path = algo.save(str(stage_ckpt_dir))
            log.info(f"Checkpoint saved → {ckpt_path}")

        if EpisodeMetricsStore.ready() and gate(m):
            log.info(
                f"  ✓ Stage {stage_idx} complete — "
                f"NDR={m['ndr']:.3f}  Jain={m['jains']:.3f}  "
                f"Eff={m['efficiency']:.1f} B/Wh"
            )
            advanced = True
            break

    final_ckpt = stage_ckpt_dir / "final"
    final_ckpt.mkdir(parents=True, exist_ok=True)
    algo.save(str(final_ckpt))
    log.info(f"Stage {stage_idx} final checkpoint → {final_ckpt}")

    algo.stop()
    ray.shutdown()

    return total_steps, advanced


def train() -> None:
    total_steps = 0
    for stage_idx in range(len(CURRICULUM)):
        if total_steps >= TOTAL_ENV_STEPS:
            break
        total_steps, _ = _run_stage(stage_idx, total_steps)
    log.info("Training complete.")


if __name__ == "__main__":
    train()