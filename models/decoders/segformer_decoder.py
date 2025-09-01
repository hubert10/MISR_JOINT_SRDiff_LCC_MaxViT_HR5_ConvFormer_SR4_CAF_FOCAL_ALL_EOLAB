import math
import torch
import torch.nn.functional as F
from timm.models.layers import drop_path, trunc_normal_


class SegFormerDecoder(torch.nn.Module):
    """
    From:  https://github.com/macdonaldezra/MineSegSAT/blob/main/mine_seg_sat/models/segformer.py

    """

    def __init__(
        self,
        in_channels: tuple[int],
        num_classes: int,
        embed_dim: int,
        dropout_p: float = 0.0,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.dropout_p = dropout_p

        # Initialize linear layers which unify the channel dimension of the encoder blocks
        # to the same as the fixed embedding dimension
        self.layers = torch.nn.ModuleList(
            [
                torch.nn.Conv2d(channels, embed_dim, (1, 1))
                for channels in reversed(in_channels)
            ]
        )

        self.linear_fuse = torch.nn.Conv2d(
            embed_dim * len(self.in_channels), embed_dim, kernel_size=1, bias=False
        )
        self.batch_norm = torch.nn.BatchNorm2d(embed_dim, eps=1e-5)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=dropout_p)
        self.classifier = torch.nn.Conv2d(embed_dim, num_classes, kernel_size=1)
        self._init_weights()

    def _init_weights(self) -> None:
        torch.nn.init.kaiming_normal_(
            self.linear_fuse.weight, mode="fan_out", nonlinearity="relu"
        )
        torch.nn.init.constant_(self.batch_norm.weight, 1)
        torch.nn.init.constant_(self.batch_norm.bias, 0)

    def forward(self, x):
        feature_size = x[0].shape[2:]

        x = [layer(x_i) for layer, x_i in zip(self.layers, reversed(x))]
        x = [
            F.interpolate(x_i, size=feature_size, mode="bilinear", align_corners=False)
            for x_i in x[:-1]
        ] + [x[-1]]

        x = self.linear_fuse(torch.cat(x[::-1], dim=1))
        x = self.batch_norm(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.classifier(x)

        return x
