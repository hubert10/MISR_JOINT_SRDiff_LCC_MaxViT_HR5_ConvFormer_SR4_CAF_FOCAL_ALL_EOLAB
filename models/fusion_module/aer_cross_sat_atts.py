"""
Link: https://arxiv.org/abs/2203.04838
FFM: Feature Fusion Module:

Feature Fusion Module (FFM) with two stages of (1) information exchange and (2) fusion

Inputs are: 
    (1): SITS feature maps:   512x16x16
    (2): Aerial feature maps: 512x16x16
Outpus are:
     Fused Features:   512x16x16

"""

import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import math


class FFCA(nn.Module):
    """
    Fusion with Cross-Attention + Attention-Based Gating
    - Uses multihead cross-attention between aerial & SITS features
    - Adds 2D positional encodings
    - Replaces scalar gates with feature-adaptive gating
    """

    def __init__(self, aer_channels_list, sits_channels_list, num_heads=8, pos_enc=True):
        super().__init__()

        self.levels = len(aer_channels_list)
        assert self.levels == len(sits_channels_list), "Feature levels must match"

        self.pos_enc = pos_enc

        # Project SITS features to match aerial feature channels
        self.sits_projs = nn.ModuleList(
            [nn.Conv2d(sits_ch, aer_ch, kernel_size=1)
             for aer_ch, sits_ch in zip(aer_channels_list, sits_channels_list)]
        )

        # Cross-attention layers
        self.attentions = nn.ModuleList(
            [nn.MultiheadAttention(embed_dim=aer_ch, num_heads=num_heads, batch_first=True)
             for aer_ch in aer_channels_list]
        )

        # Normalization
        self.norms = nn.ModuleList([nn.LayerNorm(aer_ch) for aer_ch in aer_channels_list])

        # Attention-based gating network (per level)
        self.gates = nn.ModuleList([
            nn.Sequential(
                nn.Linear(aer_ch, aer_ch // 2),
                nn.ReLU(inplace=True),
                nn.Linear(aer_ch // 2, aer_ch),
                nn.Sigmoid()
            )
            for aer_ch in aer_channels_list
        ])

    def _make_pos_enc(self, H, W, C, device):
        """2D sinusoidal positional encoding"""
        y, x = torch.meshgrid(
            torch.arange(H, device=device),
            torch.arange(W, device=device),
            indexing="ij",
        )
        pos_y = y.flatten().unsqueeze(1)
        pos_x = x.flatten().unsqueeze(1)
        div_term = torch.exp(torch.arange(0, C, 2, device=device) * -(math.log(10000.0) / C))
        pe = torch.zeros(H * W, C, device=device)
        pe[:, 0::2] = torch.sin(pos_x * div_term)
        pe[:, 1::2] = torch.cos(pos_y * div_term)
        return pe

    def forward(self, aer_feats, sits_feats):
        fused_feats = []

        for i in range(self.levels):
            B, C_aer, H, W = aer_feats[i].shape

            # Project SITS features
            sits_proj = self.sits_projs[i](sits_feats[i])

            # Flatten
            aer_flat = aer_feats[i].flatten(2).permute(0, 2, 1)  # (B, HW, C)
            sits_flat = sits_proj.flatten(2).permute(0, 2, 1)    # (B, HW, C)

            # Positional encoding
            if self.pos_enc:
                pos_aer = self._make_pos_enc(H, W, C_aer, aer_feats[i].device)
                pos_sits = self._make_pos_enc(H, W, C_aer, sits_proj.device)
                aer_flat = aer_flat + pos_aer.unsqueeze(0).expand(B, -1, -1)
                sits_flat = sits_flat + pos_sits.unsqueeze(0).expand(B, -1, -1)

            # Cross-attention (SITS conditions Aerial)
            fused, _ = self.attentions[i](aer_flat, sits_flat, sits_flat)

            # Attention-based gating
            # Compute adaptive gate per feature
            gate = self.gates[i](aer_flat)  # (B, HW, C) ∈ (0,1)

            # Weighted fusion
            fused = self.norms[i]((1 - gate) * aer_flat + gate * fused)

            # Reshape back
            fused = fused.permute(0, 2, 1).contiguous().view(B, C_aer, H, W)
            fused_feats.append(fused)

        return fused_feats



# LEARNABLE ATTENTION GATES for Feature Fusion

# The learnable gate implemented here is a attention-base gating per feature level.

# Looking at this block:

# # Learnable gates for adaptive fusion
# self.gates = nn.ParameterList(
#     [nn.Parameter(torch.zeros(1)) for _ in aer_channels_list]
# )

# For each level 𝑖, there is exactly one scalar parameter self.gates[i].

# alpha = torch.sigmoid(self.gates[i])
# fused = self.norms[i](aer_flat + alpha * fused)

# That alpha is a single scalar in (0,1), shared across all spatial positions and channels at that level.

# So the fusion is: output=aerial+α⋅fused_attention


# Interpretation

# Each level learns how much weight to give to the fused cross-attention signal vs. the raw aerial features, globally.

# For example:

# If α≈0: rely mostly on aerial features.

# If α≈1: rely strongly on fused (aerial+SITS) features.

# It’s simple and lightweight, but less expressive than channel-wise or spatial gates.