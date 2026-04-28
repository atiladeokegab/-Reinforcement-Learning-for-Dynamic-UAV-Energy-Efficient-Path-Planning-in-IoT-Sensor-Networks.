"""
Flat-MLP TorchRLModule for UAV IoT data collection (PPO control).

Architecture
------------
  Input  : Dict obs {"uav": (3,), "sensors": (N_MAX, 5), "mask": (N_MAX,)}
  Flatten: concatenate → (3 + N_MAX*5 + N_MAX,) = (303,) for N_MAX=50
  Trunk  : Linear(303→512) → ReLU → Linear(512→512) → ReLU → Linear(512→256) → ReLU
  Heads  : policy Linear(256→5), value Linear(256→1)

This is the controlled condition for the architecture ablation:
  - Same PPO algorithm as RelationalUAVModule
  - Same environment and curriculum
  - Same hyperparameters
  - No permutation invariance (flat concatenation, index-sensitive)

The MLP width [512, 512, 256] matches the DQN trunk exactly so that the
only variable is architecture inductive bias, not parameter count.
"""

from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn

from ray.rllib.core.columns import Columns
from ray.rllib.core.rl_module.apis.value_function_api import ValueFunctionAPI
from ray.rllib.core.rl_module.torch.torch_rl_module import TorchRLModule
from ray.rllib.utils.annotations import override

from experiments.relational_policy.env_wrapper import N_MAX  # noqa: E402

# Flattened observation dimension: uav(3) + sensors(N_MAX * 5) + mask(N_MAX)
OBS_DIM = 3 + N_MAX * 5 + N_MAX  # = 303 for N_MAX=50


class FlatMLPUAVModule(TorchRLModule, ValueFunctionAPI):
    """
    Flat-MLP RLModule for UAV IoT data collection.

    model_config keys
    -----------------
    hidden_sizes : list[int] — MLP layer widths (default [512, 512, 256]).
    """

    @override(TorchRLModule)
    def setup(self) -> None:
        cfg = self.model_config
        hidden_sizes = cfg.get("hidden_sizes", [512, 512, 256])

        layers: list[nn.Module] = []
        in_dim = OBS_DIM
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h

        self.trunk = nn.Sequential(*layers)
        self.policy_head = nn.Linear(in_dim, self.action_space.n)
        self.value_head  = nn.Linear(in_dim, 1)

    @override(ValueFunctionAPI)
    def compute_values(self, batch, embeddings=None):
        return self._forward_common(batch)[Columns.VF_PREDS]

    def _forward_common(self, batch: Dict[str, Any], **_) -> Dict[str, Any]:
        obs     = batch[Columns.OBS]
        uav     = obs["uav"].float()              # (B, 3)
        sensors = obs["sensors"].float()          # (B, N_MAX, 5)
        mask    = obs["mask"].float()             # (B, N_MAX)

        sensors_flat = sensors.reshape(sensors.size(0), -1)  # (B, N_MAX*5)
        x = torch.cat([uav, sensors_flat, mask], dim=-1)     # (B, 303)

        hidden = self.trunk(x)                               # (B, 256)
        logits = self.policy_head(hidden)                    # (B, 5)
        value  = self.value_head(hidden).squeeze(-1)         # (B,)

        return {
            Columns.ACTION_DIST_INPUTS: logits,
            Columns.VF_PREDS:           value,
        }

    @override(TorchRLModule)
    def _forward_inference(self, batch, **kwargs):
        return self._forward_common(batch, **kwargs)

    @override(TorchRLModule)
    def _forward_exploration(self, batch, **kwargs):
        return self._forward_common(batch, **kwargs)

    @override(TorchRLModule)
    def _forward_train(self, batch, **kwargs):
        return self._forward_common(batch, **kwargs)
