import math
import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple
from utils.utils_models import time_embedding
from utils.base_layers import LayerNormGeneral

# Video Animation
# 1. https://www.youtube.com/watch?v=SndHALawoag
#

########################## Encoders: ConvFormer for Satellite Image Time Series ##########################
#
# II.follows the Swin Transformer Architecture (Encoder + Decoder)
# One encoder is implemented here;
# 1. SwinTransformerTimeSteps modified from initial
# Swin Transformer: Hierarchical Vision Transformer using Shifted Windows
# Description: It uses shifted window approach for computing self-attention
# Adapated from https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer.py
# Paper associated to it https://ieeexplore.ieee.org/document/9710580


class Mlp(nn.Module):
    """Multilayer perceptron. Per Swin-Transformer Definition.
    Only change wrt. Mlp-Definition in BuildFormer is default for act_layer!"""

    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class ConvBlock(nn.Module):
    """
    CONV Block to replace the MHSA layer in the Swin Transformer Window Self-attention
    This ConvBlock replaces the self-attention mechanism with

    1. Separable Convolution: depthwise and pointwise
    2. Batch Normalization
    3. RELU activation

    convolutions, followed by batch normalization and a GELU activation.
    """

    def __init__(self, dim, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        # Depthwise Convolution is applied to each input channel
        # independently, which captures spatial information.
        self.conv = nn.Conv2d(
            dim,
            dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=dim,
        )
        # Pointwise Convolution (1x1) is then applied to combine information
        # across different channels, which captures cross-channel dependencies.

        self.pointwise_conv = nn.Conv2d(dim, dim, kernel_size=1)
        self.norm = nn.BatchNorm2d(dim)
        self.activation = nn.GELU()

    def forward(self, x):
        # Apply depthwise and pointwise convolutions
        x = self.conv(x)
        x = self.pointwise_conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class ImageEmbedConv(nn.Module):
    """Image Embedding with convolution + maxpool, we remove downsampling
    to process the image at their original resolution


    Args:
        red_factor (int): Equal to Patch token size.
        Output is reduced to H/red_factor, W/red_factor, Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.convblock1 = nn.Sequential(
            nn.Conv2d(in_chans, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )
        # self.down = nn.MaxPool2d(2)
        self.convblock2 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.BatchNorm2d(embed_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """Forward function."""
        # print("Image Embeddings inputs:", x.shape)

        x = self.convblock1(x)
        x = self.convblock2(x)
        # print("Image Embeddings outputs:", x.shape)
        return x


class ImageEmbedding(nn.Module):
    """
    Here, the Image or Feature Embedding block replaces the traditional patchification
    process (where an image is divided into patches) with a convolutional
    layer that reduces the spatial resolution using a stride.
    The embed_dim defines the number of output channels, similar to the token
    embedding dimension in the Swin Transformer

    Larger kernels reduce the spatial dimensions of the output more than smaller kernels.
    Smaller kernels preserve more of the input’s spatial dimensions, resulting in larger output feature maps.

    Stride = 1: The filter moves one pixel at a time (denser convolution).
    Stride > 1: The filter moves by larger steps, reducing the output size.
    """

    def __init__(self, in_channels=3, embed_dim=96, kernel_size=7, stride=4, padding=2):
        super(ImageEmbedding, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=kernel_size,  # Larger kernels reduce the spatial dimensions of the output more than smaller kernels.
            stride=stride,  #  A larger stride leads to a smaller output size, as the filter skips positions.
            padding=padding,
        )
        self.norm = nn.BatchNorm2d(embed_dim)

    def forward(self, x):
        # Apply convolution to reduce spatial resolution and embed patches
        # print("Image Embeddings inputs:", x.shape)
        x = self.conv(x)
        x = self.norm(x)
        # print("Image Embeddings outputs:", x.shape)
        return x


class ConvFormerBlock(nn.Module):
    """
    Define a Convolutional Block Replacement for Swin Transformer Block:
    1. Normalization
    2. Separable Convolution
    3. Normalization
    4. MLP

    """

    def __init__(self, dim, kernel_size=3, stride=1, padding=1):
        super(ConvFormerBlock, self).__init__()
        self.dim = dim
        self.conv_block = ConvBlock(dim, kernel_size, stride, padding)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, dim * 4, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim * 4, dim, kernel_size=1),
        )
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        # Reduce spatial resolution (stride=2)
        # self.downsample = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        # print("Spatial Conv block input:", x.shape)

        if x.dim() == 4:
            B, C, H, W = x.shape  # torch.Size([4, 100, 64])
        else:
            B, N, C = x.shape  # torch.Size([4, 100, 64])
            H = W = int(math.sqrt(N))
            x = x.view(B, H, W, C).permute(
                0, 3, 1, 2
            )  # B, C, H, W :: torch.Size([4, 64, 40, 40])

        # print("Spatial conv_block 1 input:", x.shape)

        shortcut = x
        # x = self.norm1(x)
        x = self.conv_block(x)
        x = shortcut + x  # Residual connection

        # print("Spatial conv_block 2 input:", x.shape)

        shortcut = x
        # x = self.norm2(x)
        x = self.mlp(x)
        x = shortcut + x  # Residual connection

        # print("Spatial Conv block output:", x.shape) #  4, 128, 5, 5
        x = x.reshape(B, H * W, C)
        # x = self.downsample(x)
        # print("Spatial Conv block viewed downsampled:", x.shape)
        return x


class ConvFormerTemporalBlock(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        nbts,
        qkv_bias=True,
        qk_scale=None,
        spa_temp_att=None,
        conv_spa_att=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.nbts = nbts
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        # @hubert from https://github.com/xianlin7/ConvFormer/blob/main/models/SETR.py#L73
        # self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, padding=1, bias=False)
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_spa_tmp = nn.Linear(2 * dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.proj_drop_spa_tmp = nn.Dropout(proj_drop)
        self.softmax = nn.Softmax(dim=-1)
        self.drop_path = drop_path
        self.drop = drop
        self.spa_temp_att = spa_temp_att  # Separiting spatial and temporal computation
        self.conv_spa_att = conv_spa_att  # use convolution instead of MHSA of spatial feature extraction

        if self.conv_spa_att:
            self.conv_block = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=3, padding=1)
            )

    def forward(self, x):
        """Forward function.

        Args:
            x: input features with shape of (B, T, N, C)
            mask: (0/-inf) mask with shape of (B, Wh*Ww, Wh*Ww) or None
        """
        # B, T, N, C = x.shape
        # print("Spatial-temporal Conv block input:", x.shape)

        if x.dim() == 5:
            B, T, C, H, W = x.shape  # torch.Size([4, 100, 64])
            N = H * W
            x = x.reshape(
                B, T, H * W, C
            )  # .permute(0, 3, 1, 2) # B, C, H, W :: torch.Size([4, 64, 40, 40])
        else:
            B, T, N, C = x.shape  # torch.Size([4, 6, 100, 64])

        # Full computation of Spatio-Temporal attention at the same time
        # This is ok for lower-resolution images as the dimension
        # of the Q,K are of dim: N * T which could be heavy for large images

        if self.spa_temp_att == "full-st":
            x = x.reshape(B, -1, C)  # torch.Size([4, 600, 64]
            # derive, query, keys and values from inputs embeddings
            qkv = (
                self.qkv(x)
                .reshape(B, N * T, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            # make torchscript happy (cannot use tensor as tuple)
            q, k, v = qkv[0], qkv[1], qkv[2]
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = self.softmax(attn)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N * T, C)
            x = self.proj(x)
            x = self.proj_drop(x)
            x = x.view(B, T, N, C)
            # x = x.reshape(B, H * W, C)
            return x

        elif self.spa_temp_att == "only-temp":
            # Only Temporal attention between all timesteps of one patch - in
            # parallel for each patches
            # NOTE: The vector dim of the Q, K is T and this is where @hubert
            # made an error, what happened was that I tested with the full images
            # resolution with different conditions, so this part of the code was
            # not executed (no need to check with Mareike) just make sure that
            # that the spa_temp_att is set to FALSE

            # Compute the sum of the values weighted by the corresponding attention mask as the
            # output for each time step

            x_t = x.permute(0, 2, 1, 3)  # from  B x T x N x C -->  B x N x T x C
            x_t = x_t.reshape(B * N, T, C)
            qkv_t = (
                self.qkv(x_t)
                .reshape(B * N, T, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            q_t, k_t, v_t = qkv_t[0], qkv_t[1], qkv_t[2]
            # print("q_t:", q_t.shape)
            q_t = q_t * self.scale
            attn_t = q_t @ k_t.transpose(-2, -1)
            attn_t = self.softmax(attn_t)
            attn_t = self.attn_drop(attn_t)

            x_t = (attn_t @ v_t).transpose(1, 2).reshape(B * N, T, C)
            x_t = x_t.view(B, N, T, C).permute(0, 2, 1, 3)
            x = self.proj_drop(x_t)  # size: B x T X N x C
            # print(
            #     "temporal attention between all timesteps of one patch:", x.shape
            # )  #  torch.Size([36, 6, 16, 64])
            return x

        else:
            # Separate the Spatial and Temporal self-attention computation
            # and concatenate the output in the channel dimension
            # Idea from https://www.mdpi.com/2072-4292/15/3/618
            # Two options here are possible

            # Option 1 (Prefered for Dubai paper): Spatial attention with Self-Attention (self.conv_spa_att == False)
            # Option 2: Spatial attention with Self-Attention self.conv_spa_att == True)

            if self.conv_spa_att == True:
                # Case 1: Replacing MSA with Convolutions computations
                # input: B, T, N, C
                # print("x for convs:", x.shape) #  torch.Size([36, 6, 16, 64])
                H = int(math.sqrt(N))
                x_sp = x.view(B, T, H, H, C).permute(
                    0, 1, 4, 2, 3
                )  # B, T, C, H, W :: torch.Size([36, 6, 16, 64])

                x_sp = [
                    self.conv_block(x_sp[:, i, :, :, :]) for i in range(T)
                ]  # input conv: B, C_in, H, W
                x_sp = torch.stack(x_sp, 1)  # expected output wieder: B,T,C_out,H,W
                x_sp = x_sp.view(B, T, C, N).permute(0, 1, 3, 2)  # B, T, N, C

            else:
                # The second Option 2 (Not considered):
                # Case 2: Standard MSA computations
                # spatial attention between all patches of one timesteps - in parallel for each timestep
                x_sp = x.view(B * T, N, C)

                qkv_sp = (
                    self.qkv(x_sp)
                    .reshape(B * T, N, 3, self.num_heads, C // self.num_heads)
                    .permute(2, 0, 3, 1, 4)
                )
                q_sp, k_sp, v_sp = qkv_sp[0], qkv_sp[1], qkv_sp[2]
                q_sp = q_sp * self.scale

                attn_sp = q_sp @ k_sp.transpose(-2, -1)
                attn_sp = self.softmax(attn_sp)
                attn_sp = self.attn_drop(attn_sp)

                x_sp = (attn_sp @ v_sp).transpose(1, 2).reshape(B * T, N, C)
                x_sp = x_sp.view(B, T, N, C)

            x_t = x.permute(0, 2, 1, 3)  # from  B x T x N x C -->  B x N x T x C
            x_t = x_t.reshape(B * N, T, C)
            qkv_t = (
                self.qkv(x_t)
                .reshape(B * N, T, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
            )
            q_t, k_t, v_t = qkv_t[0], qkv_t[1], qkv_t[2]

            # print("q_t:", q_t.shape)

            q_t = q_t * self.scale
            attn_t = q_t @ k_t.transpose(-2, -1)
            attn_t = self.softmax(attn_t)
            attn_t = self.attn_drop(attn_t)

            x_t = (attn_t @ v_t).transpose(1, 2).reshape(B * N, T, C)
            x_t = x_t.view(B, N, T, C).permute(0, 2, 1, 3)

            # Dimension Concat Layer for Spatial and Temporal features
            # are concatenated in the channel dimensions
            x = torch.cat((x_sp, x_t), dim=3)  # x should have size B x T x N x 2C
            x = self.proj_spa_tmp(x)  # size: B x T X N x C
            x = self.proj_drop_spa_tmp(x)  # size: B x T X N x C

            # print(
            #     "spatial and temporal attention between all timesteps of one patch:",
            #     x.shape,
            # )  #  torch.Size([36, 6, 16, 64])
            return x


class DownSampling(nn.Module):
    """Similar to Patch Merging Layer in the Original Swin Transformer
    ---> To produce a hierarchical representations, the number of
        input image is reduced to half with a downsampling factor of 2
        with a stride of 2 as the network gets deeper.

    Downsampling implemented by a layer of convolution.

    Args:
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(
        self,
        dim,
        norm_layer=nn.LayerNorm,
        post_norm=partial(LayerNormGeneral, bias=False, eps=1e-6),
    ):
        super().__init__()
        self.dim = dim
        # LayerNorm Across only the Channel Dimension (for each spatial location)
        self.norm = norm_layer(dim)
        self.post_norm = post_norm(dim * 2)
        self.downsample = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        """Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        # print('DownSampling inputs forward shape:', x.shape)

        if x.dim() == 4:
            B, C, H, W = x.shape  # torch.Size([4, 100, 64])
        else:
            B, N, C = x.shape  # torch.Size([4, 100, 64])
            H = W = int(math.sqrt(N))
            x = x.view(B, H, W, C).permute(
                0, 3, 1, 2
            )  # B, C, H, W :: torch.Size([4, 64, 40, 40])
        x = self.norm(x.reshape(B, -1, C))  # B H*W/ C reshaped before normalization

        x = self.downsample(x.view(B, C, H, W))  # reshaped back to the original shape
        x = self.post_norm(x.permute(0, 2, 3, 1))
        x = x.permute(0, 3, 1, 2)
        return x


class BasicConvLayer(nn.Module):
    """A basic Conv Layer per Swin Transformer layer for one stage.

    Args:
        dim (int): Number of feature channels
        depth (int): Depths of this stage.
        num_heads (int): Number of attention head.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        dim,
        depth,
        norm_layer=nn.LayerNorm,
        downsample=None,
    ):
        super().__init__()
        self.depth = depth
        self.norm_layer = norm_layer
        self.downsample = downsample
        self.blocks = nn.ModuleList(
            [
                nn.Sequential(
                    ConvFormerBlock(dim),
                    ConvFormerBlock(dim),
                    # nn.Conv2d(dim * 2 ** i, dim * 2 ** (i + 1) , kernel_size=3, stride=2, padding=1),
                )
                for i in range(depth)
            ]
        )
        # similar to patch merging layer but here s standard downsampling
        if downsample is not None:
            self.downsample = downsample(
                dim=dim
            )  # , norm_layer=norm_layer) #downsample  # , norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        if x.dim() == 4:
            B, C, H, W = x.shape  # torch.Size([4, 100, 64])
        else:
            B, N, C = x.shape  # torch.Size([4, 100, 64])
            H = W = int(math.sqrt(N))
            x = x.view(B, H, W, C).permute(
                0, 3, 1, 2
            )  # B, C, H, W :: torch.Size([4, 64, 40, 40])

        # H = int(math.sqrt(N))
        # x = x.view(B, H, H, C).permute(
        #     0, 3, 1, 2
        # )  # B, T, C, H, W :: torch.Size([36, 6, 16, 64])

        # print("x forward blk view:", x.shape)

        for blk in self.blocks:
            blk.H, blk.W = H, W
            # print("blk:", blk)
            # print("x blk before:", x.shape) # 4, 25, 128
            x = blk(x)
            # print("x blk after:", x.shape) # 4, 25, 128

        if self.downsample is not None:
            # x_down = x
            x_down = self.downsample(x)
            # print("x x_downsampled from spatial layers:", x_down.shape)

            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            # print("x output from spatial layers:", x.shape)
            return x, H, W, x_down, Wh, Ww
        else:
            # print("x output from spatial layers:", x.shape)
            return x, H, W, x, H, W


class BasicConvTemporalLayer(nn.Module):
    def __init__(
        self,
        dim,
        depth,
        num_heads,
        nbts,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        spa_temp_att=None,
        conv_spa_att=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
        downsample=None,
        use_time_embed=True,
    ):
        super().__init__()
        self.depth = depth
        self.use_time_embed = use_time_embed
        self.nbts = nbts
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio
        self.norm_layer = norm_layer
        self.blocks = nn.ModuleList()
        self.spa_temp_att = spa_temp_att
        self.conv_spa_att = conv_spa_att

        for i in range(depth):
            layer_spatial_2d = ConvFormerBlock(
                dim=dim,
            )
            layer_timestep = ConvFormerTemporalBlock(
                dim=dim,
                num_heads=num_heads,
                nbts=nbts,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                spa_temp_att=spa_temp_att,
                conv_spa_att=conv_spa_att,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path,
                proj_drop=proj_drop,
            )

            self.blocks.append(layer_spatial_2d)
            self.blocks.append(layer_timestep)

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(dim=dim)  # , norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x, H, W):
        """Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        # print('BasicTimestepLayer inputs forward', x.shape)

        self.nbts = x.size(1)  # To be removed
        # B, T, N, C = x.shape
        # H = W = int(math.sqrt(N))

        if x.dim() == 5:
            B, T, C, H, W = x.shape  # torch.Size([4, 100, 64])
        else:
            B, T, N, C = x.shape  # torch.Size([4, 100, 64])
            H = W = int(math.sqrt(N))

        x = x.view(B, T, H, W, C).permute(
            0, 1, 4, 2, 3
        )  # B, C, H, W :: torch.Size([36, 6, 16, 64])

        counter = 0
        for blk in self.blocks:
            blk.H, blk.W = H, W
            counter += 1
            if counter % 2 == 0:
                x = blk(x)
            else:
                blk_outs = []
                for i in range(self.nbts):
                    # print("x blk time before before:", x.shape)
                    # print("x blk time:", blk)

                    blk_outs.append(blk(x[:, i, :, :]))
                    # print("x blk time after after:", blk(x[:, i, :, :]).shape)

                # Fusing the outputs of all timesteps
                x = torch.stack(blk_outs, 1)  # torch.Size([4, 6, 64, 10, 10])
                # print("x stack from temporal layers:", x.shape)

        if self.downsample is not None:
            x_down_list = [self.downsample(x[:, i, :, :]) for i in range(self.nbts)]

            x_down = torch.stack(x_down_list, 1)
            # print("x_down_list stacked:", x_down.shape)

            Wh, Ww = (H + 1) // 2, (W + 1) // 2
            # print("x output from temporal layers:", x.shape)
            return x, H, W, x_down, Wh, Ww
        else:
            # print("x output from temporal layers:", x.shape)
            return x, H, W, x, H, W


class ConvFormerSits(nn.Module):

    """Comv Transformer backbone for handling Satellite Time Series
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        patch_size (int | tuple(int)): Patch size. Default: 4.
        in_chans (int): Number of input image channels. Default: 10.
        embed_dim or C (int): Number of linear projection output channels. Default: 96.
                    It is the capacity of the model because it determines the parameter size
                    or the amount of hidden units in the fully connected layers.
                    summary; C= capacity or size of our Transformer model, for BERT base model
                    has C=768 for vector representations

        depths (tuple[int]): Depths of each Swin Transformer stage.
        num_heads (tuple[int]): Number of attention head of each stage.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4.
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set.
        drop_rate (float): Dropout rate.
        attn_drop_rate (float): Attention dropout rate. Default: 0.
        drop_path_rate (float): Stochastic depth rate. Default: 0.2.
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        merge_after_stage: merging of ts each after stage (1: after stae 1, 2: after stage 2)
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False.
        patch_norm (bool): If True, add normalization after patch embedding. Default: True.
        out_indices (Sequence[int]): Output from which stages.
        frozen_stages (int): Stages to be frozen (stop grad and set eval mode).
            -1 means not freezing any parameters.
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(
        self,
        config,
        img_size,
        in_chans,
        embed_dim,
        depths,
        num_heads,
        mlp_ratio,
        nbts,
        merge_after_stage,  # merging of ts each after stage 1
        qkv_bias=True,
        qk_scale=None,
        spa_temp_att=None,
        conv_spa_att=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
        patch_norm=True,
        out_indices=(0, 1, 2),  # , 3),
    ):
        super().__init__()

        self.img_size = img_size
        self.in_chans = in_chans
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.merge_after_stage = merge_after_stage
        self.nbts = nbts
        # self.image_embed = ImageEmbedding(in_chans, embed_dim)
        self.image_embed = ImageEmbedConv(in_chans=in_chans, embed_dim=embed_dim)
        self.pos_drop = nn.Dropout(p=drop_rate)
        self.spa_temp_att = spa_temp_att
        self.conv_spa_att = conv_spa_att
        self.config = config
        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        # self.att_layers = nn.ModuleList()

        # self.downsample = nn.Conv2d(
        #     embed_dim, embed_dim * 2, kernel_size=3, stride=2, padding=1
        # )

        # build layer for merging temporal information
        for i_layer in range(self.merge_after_stage):
            layer = BasicConvTemporalLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                num_heads=num_heads[i_layer],
                nbts=nbts,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                spa_temp_att=spa_temp_att,
                conv_spa_att=conv_spa_att,
                drop=drop_rate,
                proj_drop=proj_drop,
                attn_drop=attn_drop_rate,
                drop_path=dpr[sum(depths[:i_layer]) : sum(depths[: i_layer + 1])],
                downsample=DownSampling if (i_layer < self.num_layers - 1) else None,
                norm_layer=norm_layer,
            )
            self.layers.append(layer)

        # build standard layers per Swin layers
        for i_layer in range(self.merge_after_stage, self.num_layers):
            layer = BasicConvLayer(
                dim=int(embed_dim * 2**i_layer),
                depth=depths[i_layer],
                downsample=DownSampling if (i_layer < self.num_layers - 1) else None,
            )
            self.layers.append(layer)

        num_features = [int(embed_dim * 2**i) for i in range(self.num_layers)]
        self.num_features = num_features

        # add a norm layer for each output
        for i_layer in out_indices:
            layer = norm_layer(num_features[i_layer])
            layer_name = f"norm{i_layer}"
            self.add_module(layer_name, layer)

    def forward(self, x, batch_positions):
        """Forward function."""

        self.nbts = x.size(1)  # To be removed
        dates = batch_positions

        # include time embedding
        time_embed = time_embedding(dates, self.nbts, self.embed_dim)

        # @hubert CHANGED from x[:, :, i, :, :]  to x[:, i, :, :, :]
        embedded = [self.image_embed(x[:, i, :, :, :]) for i in range(self.nbts)]

        x = torch.stack(embedded, 1)
        Wh, Ww = x.size(-2), x.size(-1)
        x = x.flatten(-2).transpose(-2, -1)

        time_embed = torch.stack(
            [
                time_embed[:, i, :].unsqueeze(1).repeat(1, Wh * Ww, 1)
                for i in range(self.nbts)
            ],
            1,
        )
        x = x.add_(time_embed.to(x.device))
        x = self.pos_drop(x)

        outs = []

        for i in range(self.num_layers):
            layer = self.layers[i]
            x_out, H, W, x, Wh, Ww = layer(
                x, Wh, Ww
            )  # x_out: # torch.Size([4, 3, 100, 64])

            if i in self.out_indices:
                norm_layer = getattr(self, f"norm{i}")
                x_out = norm_layer(x_out)  # torch.Size([4, 300, 64])

                # The Temporal dimension is only used in the first block of the CONV Block

                if i < self.merge_after_stage:
                    out = (
                        x_out.view(-1, self.nbts, H, W, self.num_features[i])
                        .permute(0, 1, 4, 2, 3)
                        .contiguous()
                    )
                else:
                    out = (
                        x_out.view(-1, H, W, self.num_features[i])
                        .permute(0, 3, 1, 2)
                        .contiguous()
                    )
                outs.append(out)

                if (
                    i == self.merge_after_stage - 1
                    and self.num_layers != self.merge_after_stage
                ):
                    if self.config["sits_temp_merging"] == "mean":
                        x = torch.mean(x, dim=(1))
                    else:
                        x = torch.sum(x, dim=(1))

        merged_time = []

        merge_after_stage = [i for i in range(self.merge_after_stage)]

        for i in range(len(outs)):
            if i in merge_after_stage:
                if outs[i].dim() > 4:
                    if self.config["sits_temp_merging"] == "mean":
                        x_out = torch.mean(outs[i], dim=(1))
                    else:
                        x_out = torch.sum(outs[i], dim=(1))

                    merged_time.append(x_out)
            else:
                merged_time.append(outs[i])

        return merged_time
