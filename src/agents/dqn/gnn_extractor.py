"""
GNN Feature Extractor for UAV IoT Path Planning (v5)
=====================================================
Architecture:
  1. Per-frame sensor self-attention  — sensors attend to each other (spatial graph reasoning)
  2. GRU across k=10 frames           — captures ADR convergence over ~10-step dwell window
  3. Current-frame UAV context        — appended at output

Observation layout after VecFrameStack(k=10) — 1530-dim flat vector:
    [frame_0 | frame_1 | ... | frame_9]
Each 153-dim frame:
    [uav_x, uav_y, battery  |  s0_buf, s0_urg, s0_link  | ... | s49_...]
     ^^--- UAV_FEATURES=3 ---^^  ^^--- 50 × SENSOR_FEATURES=3 ---^^

Author: ATILADE GABRIEL OKE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GNNExtractor(BaseFeaturesExtractor):
    """
    Relational feature extractor using sensor self-attention + GRU.

    Parameters
    ----------
    observation_space : gym.Space
        Flat Box space of shape (k * frame_dim,).
    features_dim : int
        Output feature dimension for SB3 policy head.
    k : int
        Frame stack depth (must match VecFrameStack n_stack).
    max_sensors : int
        Padded sensor count (MAX_SENSORS_LIMIT from training config).
    embed_dim : int
        Hidden dimension for sensor encoder + self-attention.
    n_heads : int
        Number of attention heads.
    gru_hidden : int
        GRU hidden dimension.
    """

    UAV_FEATURES    = 3
    SENSOR_FEATURES = 3   # buffer, urgency, link_quality

    def __init__(
        self,
        observation_space,
        features_dim: int = 256,
        k: int            = 10,
        max_sensors: int  = 50,
        embed_dim: int    = 64,
        n_heads: int      = 4,
        gru_hidden: int   = 128,
    ):
        super().__init__(observation_space, features_dim)
        self.k           = k
        self.max_sensors = max_sensors
        self.embed_dim   = embed_dim
        self.gru_hidden  = gru_hidden
        self.frame_dim   = self.UAV_FEATURES + max_sensors * self.SENSOR_FEATURES  # 253

        # Sensor node encoder: 5 → embed_dim
        self.sensor_proj = nn.Sequential(
            nn.Linear(self.SENSOR_FEATURES, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
        )

        # Self-attention over sensor nodes (complete-graph GNN)
        self.self_attn = nn.MultiheadAttention(
            embed_dim   = embed_dim,
            num_heads   = n_heads,
            batch_first = True,
            dropout     = 0.0,
        )
        self.attn_norm = nn.LayerNorm(embed_dim)

        # GRU: input = pooled sensor embed + UAV features per frame
        gru_input_dim = embed_dim + self.UAV_FEATURES  # 64 + 3 = 67
        self.gru = nn.GRU(
            input_size  = gru_input_dim,
            hidden_size = gru_hidden,
            num_layers  = 1,
            batch_first = True,
        )

        # Output: GRU hidden + current UAV features → features_dim
        self.output_proj = nn.Sequential(
            nn.Linear(gru_hidden + self.UAV_FEATURES, features_dim),
            nn.ReLU(),
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        B = obs.shape[0]

        # Reshape flat obs into frames: (B, k, frame_dim)
        frames = obs.view(B, self.k, self.frame_dim)

        # UAV states across frames: (B, k, 3)
        uav_states = frames[:, :, :self.UAV_FEATURES]

        # Sensor features per frame: (B, k, N, 5)
        sensor_feats = frames[:, :, self.UAV_FEATURES:].view(
            B, self.k, self.max_sensors, self.SENSOR_FEATURES
        )

        # --- Per-frame sensor self-attention (spatial GNN) ---
        # Merge batch and time dims for parallel attention
        sf_flat = sensor_feats.view(B * self.k, self.max_sensors, self.SENSOR_FEATURES)

        # Ghost sensor mask: all features == 0 (zero-padded slots)
        is_ghost = (sf_flat.abs().sum(dim=-1) < 1e-6)   # (B*k, N)
        all_masked = is_ghost.all(dim=1, keepdim=True)
        key_pad_mask = is_ghost & ~all_masked            # at least 1 visible per row

        # Encode nodes
        node_embed = self.sensor_proj(sf_flat)           # (B*k, N, embed_dim)

        # Self-attention among sensor nodes
        attn_out, _ = self.self_attn(
            node_embed, node_embed, node_embed,
            key_padding_mask=key_pad_mask,
        )                                                # (B*k, N, embed_dim)
        attn_out = self.attn_norm(attn_out)

        # Masked mean pool over real sensors → frame-level sensor embedding
        real_mask = (~is_ghost).float().unsqueeze(-1)    # (B*k, N, 1)
        pooled = (attn_out * real_mask).sum(dim=1) / real_mask.sum(dim=1).clamp(min=1)
        pooled = pooled.view(B, self.k, self.embed_dim)  # (B, k, embed_dim)

        # --- Temporal GRU over k frames ---
        gru_input = torch.cat([pooled, uav_states], dim=-1)  # (B, k, embed_dim+3)
        _, h_n = self.gru(gru_input)                          # h_n: (1, B, gru_hidden)
        h_n = h_n.squeeze(0)                                  # (B, gru_hidden)

        # Append current-frame UAV features for routing context
        current_uav = uav_states[:, -1, :]                    # (B, 3)
        combined = torch.cat([h_n, current_uav], dim=-1)      # (B, gru_hidden+3)

        return self.output_proj(combined)                     # (B, features_dim)
