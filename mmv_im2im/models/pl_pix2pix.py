import os
from pathlib import Path
from typing import Dict
from importlib import import_module
import numpy as np
from tifffile import imsave
from skimage.io import imsave as save_rgb
import pytorch_lightning as pl
import torch
from mmv_im2im.utils.misc import parse_config_func
from mmv_im2im.utils.model_utils import init_weights, state_dict_simplification
from mmv_im2im.models.nets.gans import define_generator, define_discriminator


class Model(pl.LightningModule):
    def __init__(self, model_info_xx: Dict, train: bool = True, verbose: bool = False):
        super(Model, self).__init__()

        self.verbose = verbose
        gen_init = None
        dis_init = None
        if "init_weight" in model_info_xx.net["generator"]:
            gen_init = model_info_xx.net["generator"].pop("init_weight")
        if "init_weight" in model_info_xx.net["discriminator"]:
            dis_init = model_info_xx.net["discriminator"].pop("init_weight")
        self.generator = define_generator(model_info_xx.net["generator"])
        self.discriminator = define_discriminator(model_info_xx.net["discriminator"])
        # self.sliding_window = model_info_xx["sliding_window_params"]
        if train:
            # initialize model weights
            if gen_init is not None:
                if Path(gen_init).is_file:
                    pre_train = torch.load(Path(gen_init))
                    try:
                        self.generator.load_state_dict(pre_train["state_dict"])
                    except Exception:
                        cleaned_state_dict = state_dict_simplification(
                            pre_train["state_dict"]
                        )
                        self.generator.load_state_dict(cleaned_state_dict)
                else:
                    init_weights(self.generator, init_type=gen_init)
            else:
                init_weights(self.generator, init_type="normal")

            if dis_init is not None:
                if Path(dis_init).is_file:
                    pre_train = torch.load(Path(dis_init))
                    self.discriminator.load_state_dict(pre_train["state_dict"])
                else:
                    init_weights(self.discriminator, init_type=dis_init)
            else:
                init_weights(self.discriminator, init_type="normal")

            # get loss functions
            loss_category = model_info_xx.criterion.pop("loss_type")
            loss_module = import_module("mmv_im2im.utils.gan_losses")
            loss_func = getattr(loss_module, loss_category)
            self.loss = loss_func(**model_info_xx.criterion)

            # get info of optimizer and scheduler
            self.optimizer_info = model_info_xx.optimizer
            self.scheduler_info = model_info_xx.scheduler

    def forward(self, x):
        return self.generator(x)

    def configure_optimizers(self):
        discriminator_optimizer_func = parse_config_func(
            self.optimizer_info["discriminator"]
        )
        discriminator_scheduler_func = parse_config_func(
            self.scheduler_info["discriminator"]
        )
        discriminator_optimizer = discriminator_optimizer_func(
            self.discriminator.parameters()
        )
        discriminator_scheduler = discriminator_scheduler_func(discriminator_optimizer)

        generator_optimizer_func = parse_config_func(self.optimizer_info["generator"])
        generator_scheduler_func = parse_config_func(self.scheduler_info["generator"])
        generator_optimizer = generator_optimizer_func(
            self.generator.parameters(),
        )
        generator_scheduler = generator_scheduler_func(generator_optimizer)

        return (
            [discriminator_optimizer, generator_optimizer],
            [
                discriminator_scheduler,
                generator_scheduler,
            ],
        )

    def save_pix2pix_output(self, image_A, image_B, fake_image, current_stage):

        # check if the log path exists, if not create one
        log_dir = Path(self.trainer.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        current_epoch = self.current_epoch

        if len(fake_image.shape) == 3 and fake_image.shape[0] == 3:
            out_fn = log_dir / f"{current_stage}_epoch_{current_epoch}_fake_B.png"
            save_rgb(out_fn, np.moveaxis(fake_image, 0, -1))

            out_fn = log_dir / f"{current_stage}_epoch_{current_epoch}_real_B.png"
            save_rgb(out_fn, np.moveaxis(image_B, 0, -1))

            out_fn = log_dir / f"{current_stage}_epoch_{current_epoch}_real_A.png"
            save_rgb(out_fn, np.moveaxis(image_A, 0, -1))
        else:
            out_fn = log_dir / f"{current_stage}_epoch_{current_epoch}_fake_B.tiff"
            imsave(out_fn, fake_image)

            out_fn = log_dir / f"{current_stage}_epoch_{current_epoch}_real_B.tiff"
            imsave(out_fn, image_B)

            out_fn = log_dir / f"{current_stage}_epoch_{current_epoch}_real_A.tiff"
            imsave(out_fn, image_A)

    def training_step(self, batch, batch_idx, optimizer_idx):

        # imageB : real image
        # imageA : condition image
        # conditional GAN refers generating a realistic image (image B as ground truth),
        # conditioned on image A
        image_A = batch["IM"]
        image_B = batch["GT"]

        self.generator.train()

        # calculate loss
        loss = None
        if optimizer_idx == 0:
            fake_B = self.generator(image_A).detach()
            loss = self.loss.discriminator_step(
                self.discriminator, image_A, image_B, fake_B
            )
            self.log("D Loss", loss)
        elif optimizer_idx == 1:
            fake_B = self.generator(image_A)
            loss = self.loss.generator_step(
                self.discriminator, image_A, image_B, fake_B
            )
            self.log("G Loss", loss)

        if self.verbose and batch_idx == 0 and optimizer_idx == 0:
            fake_images = fake_B.detach()
            fake_image = fake_images[0].cpu().numpy()

            image_A0 = image_A[0].detach().cpu().numpy()
            image_B0 = image_B[0].detach().cpu().numpy()

            self.save_pix2pix_output(image_A0, image_B0, fake_image, "train")

        return loss

    def training_epoch_end(self, outputs):
        G_mean_loss = torch.stack([x["loss"] for x in outputs[0]]).mean().item()
        D_mean_loss = torch.stack([x["loss"] for x in outputs[1]]).mean().item()
        self.log("generator_loss", G_mean_loss)
        self.log("discriminator_loss", D_mean_loss)

    def validation_step(self, batch, batch_idx):
        image_A = batch["IM"]
        image_B = batch["GT"]

        fake_B = self.generator(image_A).detach()
        D_loss = self.loss.discriminator_step(
            self.discriminator, image_A, image_B, fake_B
        )
        G_loss = self.loss.generator_step(self.discriminator, image_A, image_B, fake_B)

        if self.verbose and batch_idx == 0:
            # check if the log path exists, if not create one
            Path(self.trainer.log_dir).mkdir(parents=True, exist_ok=True)

            fake_images = fake_B.detach()
            fake_image = fake_images[0].cpu().numpy()

            image_A0 = image_A[0].detach().cpu().numpy()
            image_B0 = image_B[0].detach().cpu().numpy()

            self.save_pix2pix_output(image_A0, image_B0, fake_image, "val")

        return {"G_loss": G_loss, "D_loss": D_loss}

    def validation_epoch_end(self, val_step_outputs):
        val_gen_loss = (
            torch.stack([x["G_loss"] for x in val_step_outputs], dim=0).mean().item()
        )
        val_disc_loss = (
            torch.stack([x["D_loss"] for x in val_step_outputs], dim=0).mean().item()
        )
        self.log("val_loss_generator", val_gen_loss)
        self.log("val_loss_discriminator", val_disc_loss)
