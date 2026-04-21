"""
Training script: Optimal Relational Policy via Ray RLlib PPO.

Design choices
--------------
  Algorithm  : PPO (stable with custom RLModules and recurrent state;
               DQN's replay buffer is incompatible with GTrXL-style memory).
  Exploration: PPO explores via stochastic action sampling controlled by
               entropy_coeff.  The coefficient follows a two-phase schedule:
                 Phase 1 (small grids)  : entropy_coeff = 0.05  — broad search
                 Phase 2 (500×500)      : entropy_coeff = 0.003 — near-deterministic
               This is the PPO equivalent of setting exploration_final_eps=0.03
               for DQN: by the final curriculum stage the policy is highly
               exploitative (low entropy ≈ low effective ε).

Curriculum
----------
  Five competence-based stages advancing when episode_reward_mean exceeds
  ADVANCE_THRESHOLD.  Each stage doubles the grid area:

    Stage 0 : 100×100, N=10  sensors  (warm-up)
    Stage 1 : 200×200, N=15  sensors
    Stage 2 : 300×300, N=20  sensors
    Stage 3 : 400×400, N=35  sensors
    Stage 4 : 500×500, N=50  sensors  ← dissertation target

  Grid is capped at 500×500 (upper limit of single-UAV battery feasibility
  at 274 Wh / 500 W per the dissertation energy analysis).

Usage
-----
  # Install Ray first (isolated from main project dependencies):
  #   pip install "ray[rllib]>=2.10" gymnasium torch
  #   — or —
  #   uv add "ray[rllib]>=2.10"  (adds to pyproject.toml)

  uv run python src/experiments/relational_policy/train_relational.py

  Checkpoints are saved to:
    src/experiments/relational_policy/results/checkpoints/
  TensorBoard logs:
    src/experiments/relational_policy/results/tb/
"""

from __future__ import annotations

import os
import sys
import logging
from pathlib import Path
from typing import Any

# ── Path bootstrap (keep experiment self-contained) ───────────────────────────
_HERE = Path(__file__).resolve().parent
_SRC  = _HERE.parent.parent
_ROOT = _SRC.parent
for _p in (str(_HERE), str(_SRC), str(_ROOT)):   # _HERE first so local imports resolve
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from env_wrapper      import RelationalUAVEnv, EpisodeMetricsStore, GAMMA, N_MAX  # noqa: E402
from relational_module import RelationalUAVModule                                  # noqa: E402

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Output paths ──────────────────────────────────────────────────────────────
RESULTS_DIR = _HERE / "results"
CKPT_DIR    = RESULTS_DIR / "checkpoints"
TB_DIR      = RESULTS_DIR / "tb"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
TB_DIR.mkdir(parents=True, exist_ok=True)

# ── Curriculum ────────────────────────────────────────────────────────────────
# (grid_W, grid_H, n_sensors, entropy_coeff, label)
CURRICULUM: list[tuple] = [
    (100, 100, 10,  0.05,  "100x100 N=10"),
    (200, 200, 15,  0.04,  "200x200 N=15"),
    (300, 300, 20,  0.02,  "300x300 N=20"),
    (400, 400, 35,  0.008, "400x400 N=35"),
    (500, 500, N_MAX, 0.003, "500x500 N=50"),
]

# ── Curriculum gating criteria ────────────────────────────────────────────────
# Each gate is a callable(metrics_dict) → bool.
# Metrics dict keys: "ndr", "jains", "efficiency", "n_episodes".
#
# Stage 0–1  NDR ≥ 95 %          Basic collection mastery (all sensors visited)
# Stage 2–3  NDR ≥ 90 % + Jain's ≥ 0.80  Link-budget + fairness (SF12 range test)
# Stage 4    Efficiency ≥ 200 B/Wh        Optimal energy–throughput mix
#
# Rationale (dissertation coherence): NDR and Jain's Index are the primary
# evaluation metrics in the DQN results chapter.  Gating the curriculum with
# the same metrics creates a direct narrative link between training and evaluation.

def _gate_ndr95(m):  return m["ndr"]        >= 0.95
def _gate_ndr90_jain(m): return m["ndr"]    >= 0.90 and m["jains"] >= 0.80
def _gate_efficiency(m): return m["efficiency"] >= 200.0

STAGE_GATES = [
    _gate_ndr95,        # Stage 0: 100×100 N=10
    _gate_ndr95,        # Stage 1: 200×200 N=15
    _gate_ndr90_jain,   # Stage 2: 300×300 N=20  (first grid exceeding SF12 range)
    _gate_ndr90_jain,   # Stage 3: 400×400 N=35
    _gate_efficiency,   # Stage 4: 500×500 N=50  (dissertation target)
]

# Training budget
MAX_ITERS_PER_STAGE = 200   # hard cap per stage
TOTAL_ENV_STEPS     = 10_000_000

# ── Model hyperparameters ─────────────────────────────────────────────────────
MODULE_CONFIG: dict[str, Any] = {
    "d_model":     128,   # attention embedding dimension
    "n_heads":     4,     # must divide d_model evenly
    "gru_hidden":  256,   # GTrXL-inspired GRU hidden size
    "dropout":     0.1,
    "max_seq_len": 20,    # required by RLlib for stateful RLModules (BPTT window)
}

# ── Environment config ────────────────────────────────────────────────────────
BASE_ENV_CONFIG: dict[str, Any] = {
    "max_steps":    2100,    # ~7 min flight at 2 s/step
    "max_battery":  274.0,   # Wh (single-UAV feasibility ceiling at 500×500)
    "n_max":        N_MAX,
}


def build_config(
    grid_w: int,
    grid_h: int,
    n_sensors: int,
    entropy_coeff: float,
) -> PPOConfig:
    """Construct a PPOConfig for one curriculum stage."""

    env_config = {
        **BASE_ENV_CONFIG,
        "grid_size":    (grid_w, grid_h),
        "num_sensors":  n_sensors,
    }

    return (
        PPOConfig()
        # ── Environment ───────────────────────────────────────────────────
        .environment(
            env="RelationalUAV",
            env_config=env_config,
        )
        # ── Custom RLModule ───────────────────────────────────────────────
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=RelationalUAVModule,
                model_config=MODULE_CONFIG,
            )
        )
        # ── PPO training hyperparams ──────────────────────────────────────
        .training(
            gamma=GAMMA,          # must match env_wrapper.GAMMA
            lr=3e-4,
            train_batch_size=8192,
            num_epochs=10,
            minibatch_size=256,
            clip_param=0.2,
            vf_clip_param=10.0,
            vf_loss_coeff=0.5,
            entropy_coeff=entropy_coeff,
            lambda_=0.95,
            grad_clip=0.5,
        )
        # ── Rollout workers ───────────────────────────────────────────────
        # num_env_runners=0: env runs in the main process.
        # On Windows, Ray's multiprocessing child workers crash with an access
        # violation when algo.stop() kills them between curriculum stages.
        # Running in the main process avoids all inter-process communication.
        .env_runners(
            num_env_runners=0,
            rollout_fragment_length="auto",
        )
        # ── Framework ─────────────────────────────────────────────────────
        .framework("torch")
        # ── GPU ───────────────────────────────────────────────────────────
        .learners(
            num_learners=1,
            num_gpus_per_learner=1,   # use the RTX 3050 Ti for gradient updates
        )
        # ── Reporting ─────────────────────────────────────────────────────
        .reporting(
            metrics_num_episodes_for_smoothing=20,
        )
        # ── Checkpointing ─────────────────────────────────────────────────
        .checkpointing(
            export_native_model_files=True,
        )
    )


def _run_stage(stage_idx: int, total_steps: int) -> tuple[int, bool]:
    """
    Run one curriculum stage in a fresh Ray session.

    Returns (updated_total_steps, completed_successfully).
    Isolating each stage in its own Ray init/shutdown prevents the Windows
    raylet access-violation that occurs when workers are killed between stages.
    """
    grid_w, grid_h, n_sensors, ent, label = CURRICULUM[stage_idx]

    log.info("=" * 60)
    log.info(f"CURRICULUM STAGE {stage_idx}: {label}")
    log.info(f"  entropy_coeff = {ent}  (≈ ε_final={ent:.3f} equiv.)")
    log.info("=" * 60)

    EpisodeMetricsStore.reset()   # clear any residual history from previous stage
    ray.init(ignore_reinit_error=True)
    tune.register_env("RelationalUAV", lambda cfg: RelationalUAVEnv(**cfg))

    config = build_config(grid_w, grid_h, n_sensors, ent)
    algo   = config.build_algo()

    # Warm-start from previous stage checkpoint
    prev_ckpt = CKPT_DIR / f"stage_{stage_idx - 1}" / "final"
    if stage_idx > 0 and prev_ckpt.exists():
        log.info(f"Restoring weights from {prev_ckpt}")
        algo.restore(str(prev_ckpt))

    stage_ckpt_dir = CKPT_DIR / f"stage_{stage_idx}"
    stage_ckpt_dir.mkdir(parents=True, exist_ok=True)

    gate = STAGE_GATES[stage_idx]

    advanced = False
    for iteration in range(MAX_ITERS_PER_STAGE):
        result  = algo.train()
        ep_ret  = result.get("env_runners", {}).get("episode_return_mean", float("nan"))
        ep_len  = result.get("env_runners", {}).get("episode_len_mean", 0)
        lifetime_steps = result.get("num_env_steps_sampled_lifetime", 0)
        steps_this_iter = lifetime_steps - (total_steps if total_steps > 0 else 0)
        total_steps = lifetime_steps  # track lifetime, not cumulative sum of lifetime

        m = EpisodeMetricsStore.rolling_means()
        log.info(
            f"[stage {stage_idx} | iter {iteration:3d}] "
            f"NDR={m['ndr']:.3f}  Jain={m['jains']:.3f}  "
            f"Eff={m['efficiency']:.1f}B/Wh  "
            f"(n={m['n_episodes']:2d} eps)  "
            f"ret={ep_ret:.0f}  steps={int(lifetime_steps):,}"
        )

        if (iteration + 1) % 10 == 0:
            ckpt_path = algo.save(str(stage_ckpt_dir))
            log.info(f"Checkpoint saved → {ckpt_path}")

        if EpisodeMetricsStore.ready() and gate(m):
            log.info(
                f"  ✓ Stage {stage_idx} complete — "
                f"NDR={m['ndr']:.3f}  Jain={m['jains']:.3f}  "
                f"Eff={m['efficiency']:.1f} B/Wh  (n={m['n_episodes']} eps)"
            )
            advanced = True
            break

    # Always save final checkpoint before tearing down Ray
    final_ckpt = stage_ckpt_dir / "final"
    final_ckpt.mkdir(parents=True, exist_ok=True)
    algo.save(str(final_ckpt))
    log.info(f"Stage {stage_idx} final checkpoint → {final_ckpt}")

    algo.stop()
    ray.shutdown()          # clean teardown before next stage spawns workers

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
