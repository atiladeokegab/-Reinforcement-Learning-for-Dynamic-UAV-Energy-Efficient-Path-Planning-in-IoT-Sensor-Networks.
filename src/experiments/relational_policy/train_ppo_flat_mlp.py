"""
Training script: PPO + flat MLP (architecture ablation control).

Purpose
-------
This script is the controlled condition for the architecture ablation:
  - Same PPO algorithm as train_relational.py
  - Same environment (RelationalUAVEnv), same curriculum, same gates
  - Same PPO hyperparameters (lr, batch, clip, etc.)
  - Different network: flat-MLP trunk [512, 512, 256] instead of cross-attention

The only variable is the network inductive bias.  Comparing results
against train_relational.py separates the architecture contribution from
the algorithm contribution (PPO vs DQN) in the dissertation head-to-head.

Usage
-----
  uv run python src/experiments/relational_policy/train_ppo_flat_mlp.py

Checkpoints saved to:
  src/experiments/relational_policy/results_flat_mlp/checkpoints/
TensorBoard logs:
  src/experiments/relational_policy/results_flat_mlp/tb/
"""

from __future__ import annotations

import os
os.environ.setdefault("RAY_TRAIN_ENABLE_LIBUV", "0")

import sys
import logging
from pathlib import Path
from typing import Any

# ── Path bootstrap ─────────────────────────────────────────────────────────────
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

from experiments.relational_policy.env_wrapper    import RelationalUAVEnv, EpisodeMetricsStore, GAMMA, N_MAX  # noqa: E402
from experiments.relational_policy.flat_mlp_module import FlatMLPUAVModule                                    # noqa: E402


# ── Metrics callback (identical to train_relational.py) ───────────────────────
class FlatMLPMetricsCallback(RLlibCallback):
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

        metrics_logger.log_value("ndr",        float(ndr),                     window=20)
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
RESULTS_DIR = _HERE / "results_flat_mlp"
CKPT_DIR    = RESULTS_DIR / "checkpoints"
TB_DIR      = RESULTS_DIR / "tb"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
TB_DIR.mkdir(parents=True, exist_ok=True)

# ── Curriculum (identical to train_relational.py) ─────────────────────────────
CURRICULUM: list[tuple] = [
    (100, 100, 10,  0.05,  "100x100 N=10"),
    (200, 200, 15,  0.04,  "200x200 N=15"),
    (300, 300, 20,  0.02,  "300x300 N=20"),
    (400, 400, 35,  0.008, "400x400 N=35"),
    (500, 500, N_MAX, 0.003, "500x500 N=50"),
]

def _gate_ndr95(m):      return m["ndr"]     >= 0.95
def _gate_ndr90_jain(m): return m["ndr"]     >= 0.90 and m["jains"] >= 0.80
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

# ── Module config (flat MLP, matches DQN trunk width) ─────────────────────────
MODULE_CONFIG: dict[str, Any] = {
    "hidden_sizes": [512, 512, 256],
}

# ── Environment config (identical to train_relational.py) ─────────────────────
BASE_ENV_CONFIG: dict[str, Any] = {
    "max_steps":   2100,
    "max_battery": 274.0,
    "n_max":       N_MAX,
}


def build_config(
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
        .environment(
            env="FlatMLPUAV",
            env_config=env_config,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=FlatMLPUAVModule,
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
        .callbacks(FlatMLPMetricsCallback)
        .reporting(
            metrics_num_episodes_for_smoothing=20,
        )
        .checkpointing(
            export_native_model_files=True,
        )
    )


def _run_stage(stage_idx: int, total_steps: int) -> tuple[int, bool]:
    grid_w, grid_h, n_sensors, ent, label = CURRICULUM[stage_idx]

    log.info("=" * 60)
    log.info(f"[PPO+FlatMLP] CURRICULUM STAGE {stage_idx}: {label}")
    log.info(f"  entropy_coeff = {ent}")
    log.info("=" * 60)

    EpisodeMetricsStore.reset()
    os.environ["PYTHONPATH"] = str(_SRC)
    os.environ["RAY_TRAIN_ENABLE_LIBUV"] = "0"
    os.environ.pop("VIRTUAL_ENV", None)
    os.environ["UV_PROJECT_ENVIRONMENT"] = "/workspace/uav/.venv"
    os.environ["UV_NO_SYNC"] = "1"

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
    tune.register_env("FlatMLPUAV", lambda cfg: RelationalUAVEnv(**cfg))

    config = build_config(grid_w, grid_h, n_sensors, ent)
    algo   = config.build_algo()

    # Warm-start RLModule weights from previous stage (optimizer reset, same
    # rationale as train_relational.py).
    prev_ckpt = CKPT_DIR / f"stage_{stage_idx - 1}" / "final"
    rl_module_ckpt = prev_ckpt / "learner_group" / "learner" / "rl_module"
    if stage_idx > 0 and rl_module_ckpt.exists():
        log.info(f"Restoring RLModule weights from {rl_module_ckpt}")
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

    gate    = STAGE_GATES[stage_idx]
    advanced = False

    for iteration in range(MAX_ITERS_PER_STAGE):
        result  = algo.train()
        env_r   = result.get("env_runners", {})
        ep_ret  = env_r.get("episode_return_mean", float("nan"))
        lifetime_steps = result.get("num_env_steps_sampled_lifetime", 0)
        total_steps = lifetime_steps

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
            subdirs = sorted(stage_ckpt_dir.glob("checkpoint_*"))
            for old in subdirs[:-1]:
                shutil.rmtree(old, ignore_errors=True)
            log.info(f"Checkpoint saved → {ckpt_path}")

        ready = m["n_episodes"] >= 5
        if ready and gate(m):
            log.info(
                f"  ✓ Stage {stage_idx} complete — "
                f"NDR={m['ndr']:.3f}  Jain={m['jains']:.3f}  "
                f"Eff={m['efficiency']:.1f} B/Wh"
            )
            advanced = True
            break

    final_ckpt = stage_ckpt_dir / "final"
    final_ckpt.mkdir(parents=True, exist_ok=True)
    algo.save_to_path(str(final_ckpt))
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
