import math
import torch
import warnings
import torch.nn as nn
import torch.nn.functional as F
from abc import ABCMeta
from utils.base_layers import ConvModule
from utils.utils_models import resize
from models.decoders.ppm import PPM


class UPerHead(nn.Module, metaclass=ABCMeta):
    """Unified Perceptual Parsing for Scene Understanding.

    This head is the implementation of `UPerNet
    Paper: <https://arxiv.org/abs/1807.10221>`_.
    Code : <https://github.com/CSAILVision/unifiedparsing>

    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(
        self,
        in_channels,
        channels,
        num_classes,
        in_index=-1,
        input_transform=None,
        pool_scales=(1, 2, 3),
        dropout_ratio=0.1,
        align_corners=False,
        **kwargs,
    ):
        super().__init__()
        # PSP Module
        self.in_channels = in_channels
        self.channels = channels
        self.align_corners = align_corners
        self.pool_scales = pool_scales
        self.psp_modules = PPM(
            self.pool_scales,
            self.in_channels[-1],
            self.channels,
            align_corners=self.align_corners,
        )
        self.bottleneck = ConvModule(
            self.in_channels[-1] + len(pool_scales) * self.channels,
            self.channels,
            3,  # kernel size
            padding=1,
        )
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = ConvModule(in_channels, self.channels, 1, inplace=False)
            fpn_conv = ConvModule(
                self.channels, self.channels, 3, padding=1, inplace=False
            )
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = ConvModule(
            len(self.in_channels) * self.channels,
            self.channels,
            3,
            padding=1,
        )
        self.conv_seg = nn.Conv2d(channels, num_classes, kernel_size=1)
        if dropout_ratio > 0:
            self.dropout = nn.Dropout2d(dropout_ratio)
        else:
            self.dropout = None

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)
        return output

    def cls_seg(self, feat):
        """Classify each pixel."""
        if self.dropout is not None:
            feat = self.dropout(feat)
            output = self.conv_seg(feat)
            return output

    def forward(self, inputs):
        """Forward function."""
        laterals = [
            lateral_conv(inputs[i]) for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        laterals_out = laterals.copy()

        for i in range(used_backbone_levels - 1, -1, -1):
            if i == used_backbone_levels - 1:
                laterals_out[i] = laterals[i]
            else:
                prev_shape = laterals[i].shape[-2:]
                laterals_out[i] = laterals[i] + resize(
                    laterals[i + 1],
                    size=prev_shape,
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
        # Starting from
        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals_out[i]) for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals_out[-1])

        # The fusion block fuses P1, P2, P3, and P4 to generate the fused
        # features with 1/4 resolution and applies a convolution layer
        # (self.cls_seg) to map the dimensions to the category numbers
        # for the purpose to obtain the final segmentation map
        # From EfficientNet paper: Page 6

        for i in range(used_backbone_levels - 1, 0, -1):
            # print("fpn_outs[i]:", i)
            # print("fpn_outs[i]:", fpn_outs[i].shape)
            fpn_outs[i] = resize(
                fpn_outs[i],
                size=fpn_outs[0].shape[2:],
                mode="bilinear",
                align_corners=self.align_corners,
            )

        multi_levels_feature_maps = fpn_outs.copy()
        # CONCAT the multi_levels_feats_out using a conv layer: 2D + BN + Relu
        fpn_outs = torch.cat(fpn_outs, dim=1)
        # The combined multi_levels_feats_out maps at 10m GSD
        output = self.fpn_bottleneck(fpn_outs)
        classified_sits_feat_maps = self.cls_seg(output)

        # For Auxiliary Loss calculations at each level of the UperNet
        # Each feature map is classified at 10m resolution

        # STEP: UPSAMPLE ALL THE 4 FEATURES OF DIFFERENT RESOLUTIONS
        # FROM 40m to 10m WITH AN UPSAMPLING FACTOR OF 4X

        multi_levels_classifications = [
            self.cls_seg(feature) for feature in multi_levels_feature_maps
        ]
        # torch.Size([4, 512, 40, 40]), torch.Size([4, 13, 40, 40]),
        # torch.Size([4, 13, 40, 40], [4, 13, 40, 40], [4, 13, 40, 40],
        # [4, 13, 40, 40])
        # The encoder outputs are used for fusion with the aerial images
        # and the decoder outputs are used for the final segmentation map
        return (
            inputs,
            classified_sits_feat_maps,
            multi_levels_classifications,
        )
