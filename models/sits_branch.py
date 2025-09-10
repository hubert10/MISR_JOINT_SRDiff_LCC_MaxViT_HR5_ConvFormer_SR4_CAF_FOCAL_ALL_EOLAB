import torch
import os
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from models.decoders.uper_head import UPerHead
from models.encoders.conv_transformer import ConvFormerSits

# II. Swin Like Architecture (Encoder + Decoder)
# An encoder is implemented here;
# 1. ConvFormerSits(For timeseries) (Swintime)
# A decoder is implemented here;
# 1. UPerHead
# Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
# Description: It uses shifted window approach for computing self-attention
# Adapated from https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
# Paper associated to it https://ieeexplore.ieee.org/document/9710580


class SITSSegmenter(nn.Module):
    def __init__(
        self,
        config,
        img_size,
        in_chans,
        embed_dim,
        uper_head_dim,
        depths,
        num_heads,
        mlp_ratio,
        num_classes,
        merge_after_stage,
        nbts,  # number of inputs time series (default=12, one image per month across the year)
        pool_scales,
        spa_temp_att,
        conv_spa_att,
        dropout_ratio=0.1,
    ):
        super().__init__()
        self.backbone_dims = [embed_dim * 2**i for i in range(len(depths))]
        self.img_size = img_size
        self.num_classes = num_classes
        self.nbts = nbts
        self.pool_scales = pool_scales
        self.pool_scales = pool_scales
        self.spa_temp_att = spa_temp_att
        self.conv_spa_att = conv_spa_att
        self.config = config

        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

        self.backbone = ConvFormerSits(
            config=self.config,
            img_size=self.img_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            nbts=self.nbts,
            merge_after_stage=merge_after_stage,
            spa_temp_att=spa_temp_att,
            conv_spa_att=conv_spa_att,
            drop_rate=0.0,
            attn_drop_rate=0.0,
            drop_path_rate=0.2,
        )

        # define decoder network
        self.decode_head = UPerHead(
            self.backbone_dims,
            uper_head_dim,
            num_classes,
            pool_scales,
            dropout_ratio=0.1,
        )

    def forward(self, x, batch_positions=None):
        x_enc = self.backbone(x, batch_positions)
        # The output here of the swin encoder have different spatial resolution
        sits_feats, final_feat_map_cls, multi_lvl_cls = self.decode_head(x_enc)
        
        return sits_feats, final_feat_map_cls, multi_lvl_cls
        
