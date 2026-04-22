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
# Must be set BEFORE ray is imported so that Ray Train workers inherit this
# via Windows CreateProcess (runtime_env alone doesn't reach _RayTrainWorker).
# torch 2.5.1+cu121 on Windows is built without libuv; setting this to "0"
# makes Ray fall back to a compatible gloo/nccl process-group backend.
os.environ.setdefault("RAY_TRAIN_ENABLE_LIBUV", "0")

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
from ray.rllib.callbacks.callbacks import RLlibCallback
from ray.rllib.core.rl_module.rl_module import RLModuleSpec

# Use package-style imports so Ray workers (separate processes without the
# local directory on sys.path) can resolve these via PYTHONPATH = _SRC.
from experiments.relational_policy.env_wrapper      import RelationalUAVEnv, EpisodeMetricsStore, GAMMA, N_MAX  # noqa: E402
from experiments.relational_policy.relational_module import RelationalUAVModule                                  # noqa: E402


# ── Metrics callback (new API stack) ──────────────────────────────────────────
class RelationalMetricsCallback(RLlibCallback):
    """
    Reads NDR / Jain's / Efficiency from the terminal info dict and logs them
    via metrics_logger so Ray aggregates them back to the main-process result
    dict under result["env_runners"]["ndr|jains|efficiency"].

    Required because EpisodeMetricsStore is process-local: with num_env_runners>0
    the envs run in separate worker processes and the main process store stays
    empty.  metrics_logger sends values over the Ray object store instead.
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

        metrics_logger.log_value("ndr",        float(ndr),                    window=20)
        metrics_logger.log_value("jains",       float(info.get("jains",  0.0)), window=20)
        metrics_logger.log_value("efficiency",  float(info.get("efficiency", 0.0)), window=20)

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
# Reverted to local-success values: the larger (256/8) model overfits the
# survival-reward local optimum under sparse delivery rewards and takes 4x
# more gradient steps to see its first useful signal.
MODULE_CONFIG: dict[str, Any] = {
    "d_model":     128,
    "n_heads":     4,     # must divide d_model evenly
    "gru_hidden":  256,
    "dropout":     0.1,
    "max_seq_len": 20,
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
        # train_batch_size=8192 matches local success. Sparse reward requires
        # fast iteration: ~4 episodes per update instead of ~16, giving 4x
        # more chances to amplify the first delivery signal.
        # GPU utilisation: compensate with num_epochs=20 (2x local) so the
        # GPU does more work per batch without hurting exploration.
        .training(
            gamma=GAMMA,
            lr=3e-4,
            train_batch_size=8192,
            num_epochs=20,          # 2x local — extract more signal per batch on GPU
            minibatch_size=256,
            clip_param=0.2,
            vf_clip_param=50_000,  # ~0.05× return scale (~1M); 10 was too small, 1e8 caused explosion
            vf_loss_coeff=0.5,
            entropy_coeff=entropy_coeff,
            lambda_=0.95,
            grad_clip=0.5,
        )
        # ── Rollout workers ───────────────────────────────────────────────
        # 8 workers × ~1024 steps each = 8192 batch, filled faster than 4
        # workers so the GPU learner stays busy.
        .env_runners(
            num_env_runners=8,
            rollout_fragment_length="auto",
        )
        # ── Framework ─────────────────────────────────────────────────────
        .framework("torch")
        # ── GPU ───────────────────────────────────────────────────────────
        .learners(
            num_learners=1,
            num_gpus_per_learner=1,
        )
        # ── Callbacks ─────────────────────────────────────────────────────
        .callbacks(RelationalMetricsCallback)
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
        num_gpus=1,
        _temp_dir="/workspace/ray_tmp",
        runtime_env={
            "env_vars": {
                "PYTHONPATH": str(_SRC),
                "UV_PROJECT_ENVIRONMENT": "/workspace/uav/.venv",
                "UV_NO_SYNC": "1",
            }
        },
    )
    tune.register_env("RelationalUAV", lambda cfg: RelationalUAVEnv(**cfg))

    config = build_config(grid_w, grid_h, n_sensors, ent)
    algo   = config.build_algo()   # build_algo() is the Ray 2.x new-API equivalent of build()

    # Warm-start from previous stage checkpoint — RLModule weights only.
    # Restoring the full optimizer state (algo.restore) hits a PyTorch bug:
    #   "beta1 as a Tensor is not supported for capturable=False and foreach=True"
    # Adam serialises betas as 0-d Tensors; the new optimiser with foreach=True
    # (PyTorch 2.x default) rejects them. A fresh optimiser per stage is also
    # the correct semantics: env and entropy_coeff change at every transition,
    # so the running Adam moments from the previous stage would be misleading.
    #
    # Ray 2.55 note: algo.restore_from_path(component="learner_group/learner/rl_module")
    # has a multi-segment path-descent bug that tries to open module_state.pkl at the
    # wrong directory level. Workaround: descend manually to the rl_module subdir and
    # call restore_from_path on the Learner's MultiRLModule, then sync to env runners.
    prev_ckpt = CKPT_DIR / f"stage_{stage_idx - 1}" / "final"
    rl_module_ckpt = prev_ckpt / "learner_group" / "learner" / "rl_module"
    if stage_idx > 0 and rl_module_ckpt.exists():
        log.info(f"Restoring RLModule weights from {rl_module_ckpt} (optimizer reset)")
        algo.learner_group.foreach_learner(
            func=lambda learner, _p=str(rl_module_ckpt): learner.module.restore_from_path(_p)
        )
        algo.env_runner_group.sync_weights(
            from_worker_or_learner_group=algo.learner_group
        )
        log.info("RLModule weights restored and synced to env runners")
    elif stage_idx > 0:
        log.warning(
            f"No RLModule checkpoint at {rl_module_ckpt}; stage {stage_idx} starts cold"
        )

    stage_ckpt_dir = CKPT_DIR / f"stage_{stage_idx}"
    stage_ckpt_dir.mkdir(parents=True, exist_ok=True)

    gate = STAGE_GATES[stage_idx]

    advanced = False
    for iteration in range(MAX_ITERS_PER_STAGE):
        result  = algo.train()
        env_r   = result.get("env_runners", {})
        ep_ret  = env_r.get("episode_return_mean", float("nan"))
        lifetime_steps = result.get("num_env_steps_sampled_lifetime", 0)
        total_steps = lifetime_steps

        # Metrics come from RelationalMetricsCallback via metrics_logger,
        # aggregated by Ray across all workers into result["env_runners"].
        n_eps = int(env_r.get("num_episodes", 0))
        m = {
            "ndr":        float(env_r.get("ndr",        0.0)),
            "jains":      float(env_r.get("jains",      0.0)),
            "efficiency": float(env_r.get("efficiency", 0.0)),
            "n_episodes": n_eps,
        }
        log.info(
            f"[stage {stage_idx} | iter {iteration:3d}] "
            f"NDR={m['ndr']:.3f}  Jain={m['jains']:.3f}  "
            f"Eff={m['efficiency']:.1f}B/Wh  "
            f"(n={m['n_episodes']:2d} eps)  "
            f"ret={ep_ret:.0f}  steps={int(lifetime_steps):,}"
        )

        if (iteration + 1) % 10 == 0:
            import shutil
            ckpt_path = algo.save(str(stage_ckpt_dir))
            # Ray creates a new checkpoint_XXXXXX subdir each save; keep only
            # the latest to prevent disk fill on long runs.
            subdirs = sorted(stage_ckpt_dir.glob("checkpoint_*"))
            for old in subdirs[:-1]:
                shutil.rmtree(old, ignore_errors=True)
            log.info(f"Checkpoint saved → {ckpt_path}")

        ready = m["n_episodes"] >= 5
        if ready and gate(m):
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
    algo.save_to_path(str(final_ckpt))
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
