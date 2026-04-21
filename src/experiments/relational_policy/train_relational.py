"""
Training script: Optimal Relational Policy via Ray RLlib PPO.

Design choices
--------------
  Algorithm  : PPO (stable with custom RLModules and recurrent state;
               DQN's replay buffer is incompatible with GTrXL-style memory).
  Exploration: PPO explores via stochastic action sampling controlled by
               entropy_coeff.  The coefficient follows a two-phase schedule:
                 Phase 1 (small grids)  : entropy_coeff = 0.05  — broad search
                 Phase 2 (500×500)      : entropy_coeff = 0.001 — near-deterministic
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

Fixes vs. original remote-cluster version
------------------------------------------
  1. train_batch_size  : 32768 → 8192   (sparse signal preservation)
  2. minibatch_size    : 1024  → 256    (match proven local config)
  3. lr                : 3e-4  → 1e-4   (stable for transformer at this batch size)
  4. d_model           : 256   → 128    (match proven local architecture)
  5. n_heads           : 8     → 4      (match proven local)
  6. gru_hidden        : 512   → 256    (match proven local)
  7. num_env_runners   : 4     → 10     (use available CPUs for throughput)
  8. rollout_fragment_length: 128 → "auto"  (fixes RLlib batch-size validator)
  9. num_envs_per_env_runner: added = 2
  10. Entropy per stage: flat → linear decay within each stage

Usage
-----
  uv run python src/experiments/relational_policy/train_relational.py

  Override worker count via env var (default = 10):
    RLLIB_NUM_WORKERS=8 uv run python ...

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
for _p in (str(_HERE), str(_SRC), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

from env_wrapper       import RelationalUAVEnv, EpisodeMetricsStore, GAMMA, N_MAX  # noqa: E402
from relational_module import RelationalUAVModule                                   # noqa: E402

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
# (grid_W, grid_H, n_sensors, entropy_start, entropy_end, label)
# Two entropy values per stage: high at stage start → low at gate threshold.
# This gives exploration early in the stage and exploitation as NDR climbs,
# mirroring the epsilon-decay pattern from the DQN baseline.
CURRICULUM: list[tuple] = [
    (100, 100, 10,    0.05,  0.01,  "100x100 N=10"),
    (200, 200, 15,    0.04,  0.01,  "200x200 N=15"),
    (300, 300, 20,    0.02,  0.005, "300x300 N=20"),
    (400, 400, 35,    0.01,  0.003, "400x400 N=35"),
    (500, 500, N_MAX, 0.005, 0.001, "500x500 N=50"),
]

# ── Curriculum gating criteria ────────────────────────────────────────────────
# Each gate is a callable(metrics_dict) → bool.
# Metrics dict keys: "ndr", "jains", "efficiency", "n_episodes".
#
# Stage 0–1  NDR ≥ 95%                Basic collection mastery
# Stage 2–3  NDR ≥ 90% + Jain ≥ 0.80 Link-budget + fairness
# Stage 4    Efficiency ≥ 200 B/Wh    Optimal energy–throughput mix

def _gate_ndr95(m):      return m["ndr"]        >= 0.95
def _gate_ndr90_jain(m): return m["ndr"]        >= 0.90 and m["jains"] >= 0.80
def _gate_efficiency(m): return m["efficiency"] >= 200.0

STAGE_GATES = [
    _gate_ndr95,        # Stage 0: 100×100 N=10
    _gate_ndr95,        # Stage 1: 200×200 N=15
    _gate_ndr90_jain,   # Stage 2: 300×300 N=20  (first grid exceeding SF12 range)
    _gate_ndr90_jain,   # Stage 3: 400×400 N=35
    _gate_efficiency,   # Stage 4: 500×500 N=50  (dissertation target)
]

# Training budget
MAX_ITERS_PER_STAGE = 200
TOTAL_ENV_STEPS     = 10_000_000

# ── Model ─────────────────────────────────────────────────────────────────────
# Proven local config — do NOT increase d_model/n_heads until NDR > 0.5
# is confirmed on the remote cluster. Scale one variable at a time after that.
MODULE_CONFIG: dict[str, Any] = {
    "d_model":     128,   # proven locally — was 256 on failing remote run
    "n_heads":     4,     # must divide d_model evenly — was 8 on failing remote run
    "gru_hidden":  256,   # proven locally — was 512 on failing remote run
    "dropout":     0.1,
    "max_seq_len": 20,    # required by RLlib for stateful RLModules (BPTT window)
}

# ── Environment config ────────────────────────────────────────────────────────
BASE_ENV_CONFIG: dict[str, Any] = {
    "max_steps":   2100,    # ~7 min flight at 2 s/step
    "max_battery": 274.0,   # Wh (single-UAV feasibility ceiling at 500×500)
    "n_max":       N_MAX,
}

# ── Worker count ──────────────────────────────────────────────────────────────
# GPU utilisation comes from parallelism (many workers → full batches fast),
# NOT from larger batch sizes (which dilute sparse delivery rewards).
#
# Formula: leave 2 CPUs for the Ray driver + learner process.
#   13 available CPUs → 13 - 2 = 11, round down to 10 for safety.
#
# Override via env var:  RLLIB_NUM_WORKERS=8 uv run python ...
NUM_WORKERS = int(os.environ.get("RLLIB_NUM_WORKERS", "10"))


def build_config(
    grid_w: int,
    grid_h: int,
    n_sensors: int,
    entropy_coeff: float,
) -> PPOConfig:
    """
    Construct a PPOConfig for one curriculum stage iteration.

    entropy_coeff is passed in pre-computed by the training loop so that
    the caller controls the annealing schedule without rebuilding the full
    config object every iteration (only the coefficient is hot-updated).
    """
    env_config = {
        **BASE_ENV_CONFIG,
        "grid_size":   (grid_w, grid_h),
        "num_sensors": n_sensors,
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
            gamma=GAMMA,              # must match env_wrapper.GAMMA
            lr=1e-4,                  # reduced from 3e-4 — stable for transformer
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
        # ── Rollout workers ───────────────────────────────────────────────
        # GPU utilisation via parallelism: 10 workers × 2 envs = 20 parallel
        # environments continuously filling 8k-step batches.
        #
        # rollout_fragment_length="auto" lets RLlib compute the exact fragment
        # size needed (≈409 steps) so the batch-size validator passes.
        # Fixed fragment=128 with these worker counts does NOT divide into 8192
        # evenly and causes a ValueError at build_algo().
        .env_runners(
            num_env_runners=NUM_WORKERS,
            num_envs_per_env_runner=2,
            rollout_fragment_length="auto",   # fixes batch-size validator error
        )
        # ── Framework ─────────────────────────────────────────────────────
        .framework("torch")
        # ── GPU ───────────────────────────────────────────────────────────
        .learners(
            num_learners=1,
            num_gpus_per_learner=1,
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

    Each iteration hot-updates the entropy coefficient on the existing algo
    object rather than rebuilding the config, which avoids re-spawning workers.

    Returns (updated_total_steps, completed_successfully).
    Isolating each stage in its own Ray init/shutdown prevents the raylet
    access-violation that can occur when workers are killed between stages.
    """
    grid_w, grid_h, n_sensors, ent_start, ent_end, label = CURRICULUM[stage_idx]

    log.info("=" * 60)
    log.info(f"CURRICULUM STAGE {stage_idx}: {label}")
    log.info(f"  entropy: {ent_start:.4f} → {ent_end:.4f} over {MAX_ITERS_PER_STAGE} iters")
    log.info("=" * 60)

    EpisodeMetricsStore.reset()
    ray.init(ignore_reinit_error=True)
    tune.register_env("RelationalUAV", lambda cfg: RelationalUAVEnv(**cfg))

    # Build with starting entropy (iteration 0 = maximum exploration)
    config = build_config(grid_w, grid_h, n_sensors, ent_start)
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

        # Linear entropy annealing within the stage
        # iter 0 → ent_start,  iter MAX-1 → ent_end
        progress     = iteration / max(MAX_ITERS_PER_STAGE - 1, 1)
        entropy_now  = ent_start + progress * (ent_end - ent_start)
        algo.config.training(entropy_coeff=entropy_now)

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
    ray.shutdown()   # clean teardown before next stage spawns workers

    return total_steps, advanced


def train() -> None:
    total_steps = 0

    for stage_idx in range(len(CURRICULUM)):
        if total_steps >= TOTAL_ENV_STEPS:
            log.info(f"Total env step budget ({TOTAL_ENV_STEPS:,}) reached — stopping.")
            break
        total_steps, _ = _run_stage(stage_idx, total_steps)

    log.info("Training complete.")


if __name__ == "__main__":
    train()