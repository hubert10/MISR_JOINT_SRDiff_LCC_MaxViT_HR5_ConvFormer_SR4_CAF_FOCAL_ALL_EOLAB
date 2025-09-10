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
    Fuses multi-scale features from aerial and SITS sources using cross-attention:
    - Aerial features serve as queries
    - SITS features (projected) serve as keys and values

    Conceptual:

        Aerial features are treated as dominant, and they query information from the SITS features.
        Attention learns what parts of the SITS representation are most informative for each aerial feature.
        The fusion is more targeted and modality-aware.

    Multiple Heads;

        Different heads can learn to focus on different aspects or patterns of the feature vectors.
        Improves the model's ability to capture diverse relationships between aerial and SITS inputs.

    """

    def __init__(self, aer_channels_list, sits_channels_list, num_heads=8):
        """
        Args:
            aer_channels_list (List[int]): Channels at each aerial feature level
            sits_channels_list (List[int]): Channels at each SITS feature level
        """
        super().__init__()

        self.levels = len(aer_channels_list)
        assert self.levels == len(sits_channels_list), "Feature levels must match"

        # Project SITS features to match aerial feature channels per level
        self.sits_projs = nn.ModuleList(
            [
                nn.Conv2d(sits_ch, aer_ch, kernel_size=1)
                for aer_ch, sits_ch in zip(aer_channels_list, sits_channels_list)
            ]
        )

        # Cross-attention: query = aerial, key/value = SITS
        self.attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    embed_dim=aer_ch, num_heads=num_heads, batch_first=True
                )
                for aer_ch in aer_channels_list
            ]
        )
        self.norms = nn.ModuleList(
            [nn.LayerNorm(aer_ch) for aer_ch in aer_channels_list]
        )

    def forward(self, aer_feats, sits_feats):
        """
        Args:
            aer_feats (List[Tensor]): Aerial feature maps of shape (B, C_aer, H, W)
            sits_feats (List[Tensor]): SITS feature maps of shape (B, C_sits, H, W)

        Returns:
            List[Tensor]: Fused features per level of shape (B, C_aer, H, W)
        """
        fused_feats = []

        for i in range(self.levels):
            B, C_aer, H, W = aer_feats[i].shape

            # Project SITS features to aerial dimension
            sits_proj = self.sits_projs[i](sits_feats[i])  # (B, C_aer, H, W)

            # Flatten spatial dimensions for attention input
            aer_flat = aer_feats[i].flatten(2).permute(0, 2, 1)  # (B, H*W, C_aer)
            sits_flat = sits_proj.flatten(2).permute(0, 2, 1)  # (B, H*W, C_aer)

            # Cross-attention: query=aerial, key=value=SITS
            fused, _ = self.attentions[i](aer_flat, sits_flat, sits_flat)  # (B, H*W, C_aer)

            fused = self.norms[i](aer_flat + fused)

            # Reshape back to (B, C_aer, H, W)
            fused = fused.permute(0, 2, 1).contiguous().view(B, C_aer, H, W)

            fused_feats.append(fused)

        return fused_feats
