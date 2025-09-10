import os.path
import json
import torch
import numpy as np
import torch.nn as nn
from trainer import Trainer
from utils.hparams import hparams
from utils.utils import load_ckpt
from models.diffsr_modules import Unet
from models.diffusion import GaussianDiffusion
from models.misr_module import HighResLtaeNet, RRDBLtaeNet
from models.sits_aerial_seg_model import SITSAerialSegmenter
from models.losses.focal_smooth import FocalLossWithSmoothing


class SRDiffTrainer(Trainer):
    def build_model(self):
        hidden_size = hparams["hidden_size"]
        dim_mults = hparams["unet_dim_mults"]
        dim_mults = [int(x) for x in dim_mults.split("|")]

        self.criterion_aer = FocalLossWithSmoothing(hparams["num_classes"], gamma=2, alpha=1, lb_smooth=0.2)
        self.criterion_sat = FocalLossWithSmoothing(hparams["num_classes"], gamma=2, alpha=1, lb_smooth=0.2)

        self.loss_aux_sat_weight = hparams["loss_aux_sat_weight"]
        self.loss_main_sat_weight = hparams["loss_main_sat_weight"]

        denoise_fn = Unet(
            hidden_size,
            out_dim=hparams["num_channels_sat"],
            cond_dim=hparams["rrdb_num_feat"],
            dim_mults=dim_mults,
        )

        # Define the diffusion model for training
        if not hparams["infer"]:
            if hparams["use_highresnet_ltae"]:
                with open("./tasks/config_hrnet.json", "r") as read_file:
                    self.config = json.load(read_file)
                cond_net = HighResLtaeNet(self.config)
                if hparams["cond_net_ckpt"] != "" and os.path.exists(
                    hparams["cond_net_ckpt"]
                ):
                    load_ckpt(cond_net, hparams["cond_net_ckpt"])
            elif hparams["use_rrdb_ltae"]:
                with open("./tasks/config_rrdb_misr.json", "r") as read_file:
                    self.config = json.load(read_file)
                cond_net = RRDBLtaeNet(
                    self.config,
                    3,
                    3,
                    hparams["rrdb_num_feat"],
                    hparams["rrdb_num_block"],
                    hparams["rrdb_num_feat"] // 2,
                )
                if hparams["cond_net_ckpt"] != "" and os.path.exists(
                    hparams["cond_net_ckpt"]
                ):
                    load_ckpt(cond_net, hparams["cond_net_ckpt"])
            else:
                cond_net = None

            gaussian = GaussianDiffusion(
                denoise_fn=denoise_fn,
                cond_net=cond_net,
                timesteps=hparams["timesteps"],
                loss_type=hparams["loss_type"],
            )

            self.model = SITSAerialSegmenter(gaussian=gaussian, config=hparams)
            # what is used for?
            self.global_step = 0

        else:
            # Load the diffusion model for inference
            if hparams["use_highresnet_ltae"]:
                with open("./tasks/config_hrnet.json", "r") as read_file:
                    self.config = json.load(read_file)
                cond_net = HighResLtaeNet(self.config)
            elif hparams["use_rrdb_ltae"]:
                with open("./tasks/config_rrdb_misr.json", "r") as read_file:
                    self.config = json.load(read_file)
                cond_net = RRDBLtaeNet(
                    self.config,
                    3,
                    3,
                    hparams["rrdb_num_feat"],
                    hparams["rrdb_num_block"],
                    hparams["rrdb_num_feat"] // 2,
                )
            else:
                cond_net = None

            gaussian = GaussianDiffusion(
                denoise_fn=denoise_fn,
                cond_net=cond_net,
                timesteps=hparams["timesteps"],
                loss_type=hparams["loss_type"],
            )
            self.model = SITSAerialSegmenter(gaussian=gaussian, config=hparams)

            self.global_step = 0
            if hparams["cond_net_ckpt"] != "" and os.path.exists(
                hparams["cond_net_ckpt"]
            ):
                load_ckpt(self.model, hparams["cond_net_ckpt"])
        return self.model

    def training_step(self, batch):
        img = batch["img"]  # torch.Size([4, 5, 512, 512])
        img_hr = batch["img_hr"]  # torch.Size([4, 5, 512, 512])
        img_lr = batch["img_lr"]  # torch.Size([4, 2, 3, 40, 40])
        img_lr_up = batch["img_lr_up"]  # torch.Size([4, 2, 3, 160, 160])
        labels = batch["labels"]  # torch.Size([4, 2, 3, 160, 160])
        labels_sr = batch["labels_sr"]  # torch.Size([4, 2, 3, 160, 160])
        dates = batch["dates_encoding"]
        closest_idx = batch["closest_idx"]  # torch.Size([4, 2, 3, 160, 160])
        sc_img_hr = img_hr[:, :4, :, :]

        if hparams["use_highresnet_ltae"]:
            # call gaussian diffusion model for SR-prediction this should also
            # return the SR-SITS images alongside the diffusion losses
            losses, _, _, img_sr = self.model.gaussian(
                sc_img_hr,
                img_lr,
                img_lr_up,
                labels_sr,
                dates=dates,
                closest_idx=closest_idx,
                config=self.config,
            )

            # for classification branches
            # cls_sits - final classified feat maps
            # multi_outputs - classified multi-level feats
            # aer_outputs - main decoder logits

            cls_sits, multi_outputs, aer_outputs = self.model(
                img, img_sr, labels, dates, self.config
            )

            # Auxiliary losses
            # The CE loss for the SITS classification branch is done at 6.25m GSD

            aux_loss1 = self.criterion_sat(multi_outputs[2], labels_sr)
            aux_loss2 = self.criterion_sat(multi_outputs[1], labels_sr)
            aux_loss3 = self.criterion_sat(multi_outputs[0], labels_sr)

            # loss for main SITS classification branch
            loss_main_sat = self.criterion_sat(cls_sits, labels_sr)

            # Total loss for SITS branch
            loss_sat = self.loss_main_sat_weight * loss_main_sat + (
                self.loss_aux_sat_weight * aux_loss1
                + self.loss_aux_sat_weight * aux_loss2
                + self.loss_aux_sat_weight * aux_loss3
            )

            # Loss for AER branch
            loss_aer = self.criterion_aer(aer_outputs, labels.long())

            # The CE loss for the SITS classification branch is done at 1.6m GSD
            # that combines the loss from the SR-diffusion model and the SITS 
            # segmentation branch
            
            losses["sr"] = hparams["loss_weights_aer_sat"][1] * (
                losses["sr"] + loss_sat
            )

            # The CE loss for the AER classification branch is done at 20cm GSD
            losses["aer"] = hparams["loss_weights_aer_sat"][0] * loss_aer

        else:
            losses, _, _ = self.model(img_hr, img_lr, img_lr_up)
        total_loss = sum(losses.values())
        return losses, total_loss

    def sample_and_test(self, sample):
        # Sample images and calculate evaluation metrics
        # Used for inference mode
        ret = {k: [] for k in self.metric_keys}
        ret["n_samples"] = 0
        img = sample["img"]
        img_hr = sample["img_hr"]
        img_lr = sample["img_lr"]
        img_lr_up = sample["img_lr_up"]
        labels = sample["labels"]
        dates = sample["dates_encoding"]
        closest_idx = sample["closest_idx"]  # torch.Size([4, 2, 3, 160, 160])
        sc_img_hr = img_hr[:, :4, :, :]

        if hparams["use_highresnet_ltae"]:
            img_sr, rrdb_out = self.model.gaussian.sample(
                img_lr,
                img_lr_up,
                sc_img_hr.shape,
                dates=dates,
                config=self.config,
            )
            # during sampling, only the aer branch is used
            _, _, aer_outputs = self.model(img, img_sr, labels, dates, self.config)
            proba = torch.softmax(aer_outputs, dim=1)
            preds = torch.argmax(proba, dim=1)

        else:
            img_sr, rrdb_out = self.model.sample(img_lr, img_lr_up, img_hr.shape)

        # Loop over batch
        for b in range(img_sr.shape[0]):
            s = self.measure.measure(
                img_sr[b][int(closest_idx[b].item()), :, :, :],  # SR image at t
                sc_img_hr[b],  # reference HR image
                img_lr[b][int(closest_idx[b].item()), :, :, :],  # LR input at t
                preds[b],
                labels[b],
            )
            ret["psnr"].append(s["psnr"])
            ret["ssim"].append(s["ssim"])
            ret["lpips"].append(s["lpips"])
            ret["mae"].append(s["mae"])
            ret["mse"].append(s["mse"])
            ret["shift_mae"].append(s["shift_mae"])
            ret["miou"].append(s["miou"])

            ret["n_samples"] += 1
        return img_sr, preds, rrdb_out, ret, ret

    def build_optimizer(self, model):
        params = list(model.named_parameters())

        # Filter out cond_net parameters that are not trainable
        if hparams["fix_cond_net_parms"]:
            params = [p[1] for p in params if "cond_net" not in p[0]]
        else:
            params = [p[1] for p in params]

        optimizer = torch.optim.AdamW(model.parameters(), lr=hparams["lr"])
        return optimizer

    def build_scheduler(self, optimizer):
        scheduler_param = {
            "milestones": [
                np.floor(hparams["decay_steps"] * 0.5),
                np.floor(hparams["decay_steps"] * 0.9),
            ],
            "gamma": 0.1,
        }
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer, **scheduler_param
        )
