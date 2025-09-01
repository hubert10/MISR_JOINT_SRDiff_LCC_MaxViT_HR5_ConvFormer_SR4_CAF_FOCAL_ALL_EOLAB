import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.base_layers import ConvModule
from utils.utils_models import resize

########################## Decoders: UperNet Start #####################################
# An encoder is implemented here
# 1. UPerHeadTimeSteps
# Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
# Description: It uses shifted window approach for computing self-attention
# Adapated from https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
# Paper associated to it https://ieeexplore.ieee.org/document/9710580


class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet. From mmseg.

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
        align_corners (bool): align_corners argument of F.interpolate.

    ATTENTION: THIS HAS BEEN SIMPLIFIED FOR USAGE WITH SWIN TRANSFORMER! ORIGINAL
    CODE HAS SOME MORE ARGUMENTS TO SET!
    """

    def __init__(self, pool_scales, in_channels, channels, align_corners):
        super().__init__()
        self.pool_scales = tuple(pool_scales)
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels

        # print("pool_scales ppm:", self.pool_scales)
        for pool_scale in self.pool_scales:
            # print("pool_scale ppm:", pool_scale)
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    ConvModule(
                        in_channels=self.in_channels,
                        out_channels=self.channels,
                        kernel_size=1,
                    ),
                )
            )

    def forward(self, x):
        """Forward function."""

        ppm_outs = []
        for ppm in self:
            # print()
            # print("ppm:", ppm)
            # print("ppm_out in:", x.shape)
            ppm_out = ppm(x)
            # print("ppm_out out:", ppm_out.shape)
            # print()

            upsampled_ppm_out = resize(
                ppm_out,
                size=x.size()[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )
            ppm_outs.append(upsampled_ppm_out)

        # print("pool_scales:", self.pool_scales)
        # print("ppm_outs 0:", ppm_outs[0].shape)
        # print("ppm_outs 1:", ppm_outs[1].shape)
        # print("ppm_outs 2:", ppm_outs[2].shape)
        # print("ppm_outs 3:", ppm_outs[3].shape)

        return ppm_outs
