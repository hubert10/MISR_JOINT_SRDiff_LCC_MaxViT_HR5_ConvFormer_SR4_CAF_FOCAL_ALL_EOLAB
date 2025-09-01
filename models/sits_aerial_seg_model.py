import torch
import timm
from torch import nn
import torchvision.transforms as T
from models.sits_branch import SITSSegmenter
from models.fusion_module.cross_atts import FFCA
from utils.hparams import hparams
import torch.nn.functional as F
from timm.layers import create_conv2d, create_pool2d
from models.decoders.unet_former_decoder import UNetFormerDecoder

class SITSAerialSegmenter(nn.Module):
    def __init__(self, gaussian, config):
        super().__init__()
        self.gaussian = gaussian
        self.config = config
        self.embed_dim = config["embed_dim"]
        self.decoder_channels = config["decoder_channels"]
        self.num_classes = config["num_classes"]
        self.dropout = config["dropout"]
        self.window_size = config["window_size"]
        
        # 1. SITS Encoder
        self.sits_encoder = SITSSegmenter(
            img_size=config["img_size"],
            in_chans=config["num_channels_sat"],
            embed_dim=config["embed_dim"],
            uper_head_dim=config["uper_head_dim"],
            depths=config["depths"],
            num_heads=config["num_heads"],
            mlp_ratio=config["mlp_ratio"],
            num_classes=config["num_classes"],
            nbts=config["nbts"],
            merge_after_stage=config["merge_after_stage"],
            pool_scales=config["pool_scales"],
            spa_temp_att=config["spa_temp_att"],
            conv_spa_att=config["conv_spa_att"],
            # weights_path=config["weights_path"],
            config=config,
        )

        # # 2. Aerial Network
        self.aerial_net = timm.create_model(
            "maxvit_tiny_tf_512.in1k",
            pretrained=True,
            features_only=True,
            num_classes=config["num_classes"],
        )

        # Get first conv layer (usually called 'stem.conv' in MaxViT)
        conv1 = self.aerial_net.stem.conv1  # <-- sometimes it's model.stem.conv or model.conv_stem, check print(model)

        # Create new conv with 5 input channels instead of 3
        new_conv = create_conv2d(
            in_channels=config["num_channels_aer"],  # Use num_channels from config
            out_channels=conv1.out_channels,
            kernel_size=conv1.kernel_size,
            stride=conv1.stride,
            padding=1, # original padding was None, but we set it to 1 for compatibility
            bias=conv1.bias is not None
        )

        # Initialize the first 3 channels with pretrained weights
        with torch.no_grad():
            new_conv.weight[:, :3, :, :] = conv1.weight  # copy RGB weights
            # Initialize the extra channels randomly (e.g., Kaiming normal)
            nn.init.kaiming_normal_(new_conv.weight[:, 3:, :, :])

        # Replace the old conv with the new one
        self.aerial_net.stem.conv1 = new_conv

        encoder_channels = [
            self.embed_dim,
            self.embed_dim * 2,
            self.embed_dim * 4,
            self.embed_dim * 8,
        ]

        # 3. Decoder from U-Net Former paper 
        self.decoder = UNetFormerDecoder(
            encoder_channels, self.decoder_channels, self.dropout, self.window_size, self.num_classes
        )
        self.fusion_module = FFCA( aer_channels_list=[128, 256, 512], sits_channels_list=[64, 128, 256], num_heads=8)

    def norm_fused_feats(self, fused_feats_list):
        # Applies Layer Normalization to each feature map in the list.
        # across the channel dimension (C), for each pixel, to ensure
        # that each feature has a stable ans well-scaled value distribution.
        normalized_feats = []
        
        for x in fused_feats_list:
            B, C, H, W = x.shape
            # Permute to [B, H, W, C]
            x_ = x.permute(0, 2, 3, 1)
            # Apply LayerNorm over channel dimension
            layer_norm = nn.LayerNorm(C).to(x.device)  # or nn.LayerNorm([C])
            x_norm = layer_norm(x_)
            # Permute back to [B, C, H, W]
            x_norm = x_norm.permute(0, 3, 1, 2)
            normalized_feats.append(x_norm)
        return normalized_feats

    def forward(
        self,
        aerial: torch.FloatTensor,
        img_sr: torch.FloatTensor,
        labels: torch.FloatTensor,
        dates: torch.FloatTensor,
        config,
    ):
      
        h, w = aerial.size()[-2:]
        # Aerial branch
        res0, res1, res2, res3, res4 = self.aerial_net(aerial)
        # SITS branch
        output_sen, cls_sits, multi_lvls_outs = self.sits_encoder(img_sr, dates)

        # Fusion FFCA
        fused_sits_aer_feats = self.fusion_module([res2, res3, res4], output_sen)
        res2, res3, res4 = self.norm_fused_feats(fused_sits_aer_feats)
        
        # Decoder
        logits = self.decoder(res0, res1, res2, res3, res4, h, w)
        return cls_sits, multi_lvls_outs, logits
