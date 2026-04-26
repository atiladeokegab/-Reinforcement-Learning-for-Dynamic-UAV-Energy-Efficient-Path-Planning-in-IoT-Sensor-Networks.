"""
transformer_model.py
====================
Registers ray.rllib.models.torch.attention_net.GTrXLNet under the name
"gtrxl_uav" in the RLlib ModelCatalog and exposes a single helper that
returns the ready-to-use model config dict.

Design choices
--------------
* attention_dim=256 / num_heads=8 / num_transformer_units=4
  Four GTrXL layers give sufficient depth to learn multi-hop routing
  heuristics (visit high-urgency sensor → collect → re-route) without
  over-parameterisation on the 253-dim flat observation.

* memory_training = memory_inference = 50
  The EMA-ADR update rule has effective lag τ ≈ 1/λ = 10 timesteps.
  A 50-step window covers 5× this horizon so the network can observe a
  complete ADR adaptation cycle plus the resulting buffer dynamics.

* use_n_prev_actions=5 / use_n_prev_rewards=5
  Appending the last 5 (action, reward) pairs as auxiliary inputs
  substantially reduces the partial-observability penalty in environments
  with sparse, delayed collection rewards.

* init_gru_gate_bias=2.0
  Initialising the GRU gate bias high (≈ sigmoid(2) ≈ 0.88 open) prevents
  vanishing gradients at the start of curriculum Stage 0, where the agent
  has no prior policy to fall back on.

Compatibility note
------------------
GTrXLNet is a TorchModelV2 (legacy model API).  It integrates cleanly with
the *old* RLlib learner stack.  For full New API Stack training (RLModule +
LearnerGroup), wrap this in a TorchStatefulEncoderRLModule.  The training
script uses `.api_stack(enable_rl_module_and_learner=False)` so that
`num_gpus=2` activates data-parallel multi-GPU training while keeping
GTrXLNet's recurrence management intact.
"""

from __future__ import annotations

from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.attention_net import GTrXLNet

MODEL_NAME = "gtrxl_uav"

# ---------------------------------------------------------------------------
# Observation layout (must stay in sync with env_wrapper.py)
# ---------------------------------------------------------------------------
UAV_FEATURES = 3        # [x_norm, y_norm, battery_norm]
SENSOR_FEATURES = 5     # [buffer, urgency, link_quality, dx, dy]
MAX_SENSORS = 50        # zero-padding ceiling
OBS_DIM = UAV_FEATURES + SENSOR_FEATURES * MAX_SENSORS   # 3 + 250 = 253

# ---------------------------------------------------------------------------
# GTrXL hyper-parameters
# ---------------------------------------------------------------------------
# RLlib reads these keys directly from model_config, hence the "attention_"
# prefix.  Values are passed either through PPOConfig's model={...} dict or
# through custom_model_config when the model is registered as a custom model.
GTRXL_CONFIG: dict = {
    # ── architecture ──────────────────────────────────────────────────────
    "attention_dim": 256,                    # d_model for all projection layers
    "attention_num_transformer_units": 4,    # stacked GTrXL layers
    "attention_num_heads": 8,               # multi-head attention heads
    "attention_head_dim": 32,               # per-head dim = 256 / 8
    "attention_position_wise_mlp_dim": 512, # feed-forward hidden size

    # ── memory ────────────────────────────────────────────────────────────
    "attention_memory_training": 50,         # segments cached during training
    "attention_memory_inference": 50,        # segments cached at inference

    # ── stabilisation ─────────────────────────────────────────────────────
    "attention_init_gru_gate_bias": 2.0,    # open gates at init

    # ── auxiliary inputs ──────────────────────────────────────────────────
    "attention_use_n_prev_actions": 5,
    "attention_use_n_prev_rewards": 5,
}

# max_seq_len must match the memory length so the recurrent state slices
# align correctly inside GTrXLNet's forward pass.
MAX_SEQ_LEN: int = GTRXL_CONFIG["attention_memory_training"]


def register_model() -> dict:
    """
    Register GTrXLNet in the ModelCatalog under MODEL_NAME and return the
    model config dict for use in AlgorithmConfig.training(model={...}).

        from transformer_model import register_model
        model_cfg = register_model()
        config = PPOConfig().training(model=model_cfg, ...)

    Why use_attention=True instead of custom_model
    -----------------------------------------------
    Passing ``custom_model=GTrXLNet`` bypasses the RLlib recurrent-network
    protocol: the TorchPolicy never calls ``get_initial_states()`` and the
    rollout workers crash with ``state_out_0 is not available``.

    Setting ``use_attention=True`` activates RLlib's built-in attention path,
    which:
      1. Calls ``policy.get_initial_states()`` to seed the XL memory segments.
      2. Stores / retrieves the attention state across rollout steps.
      3. Pads episode batches into fixed-length sequences before the BPTT pass.

    The ModelCatalog.register_custom_model() call is kept so that the model
    is named and discoverable (e.g. for checkpoint inspection), but the
    actual training uses the use_attention=True config key.

    Safe to call multiple times; subsequent registrations are no-ops.
    """
    ModelCatalog.register_custom_model(MODEL_NAME, GTrXLNet)

    return {
        # use_attention=True is the canonical trigger for GTrXLNet in the
        # old API stack; it activates full recurrent-state management.
        "use_attention": True,
        **GTRXL_CONFIG,
        # max_seq_len controls the BPTT window inside RLlib's sequence
        # padding; align it with the attention memory length.
        "max_seq_len": MAX_SEQ_LEN,
    }
