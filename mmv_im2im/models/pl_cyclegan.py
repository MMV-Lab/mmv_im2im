"""
This module provides lighting module for cycleGAN
"""
import os
import torch
import pytorch_lightning as pl
from pathlib import Path
from mmv_im2im.utils.misc import parse_config, parse_config_func
from mmv_im2im.utils.gan_utils import ReplayBuffer
from mmv_im2im.models.nets.gans import define_generator, define_discriminator
from typing import Dict
from collections import OrderedDict
import itertools
from tifffile import imsave


class Model(pl.LightningModule):
    def __init__(self, model_info_xx: Dict, train: bool = True, verbose: bool = False):
        super(Model, self).__init__()
        self.verbose = verbose
        self.generator_model_image_AtoB = define_generator(
            model_info_xx.net["generator"]
        )
        self.generator_model_image_BtoA = define_generator(
            model_info_xx.net["generator"]
        )
        self.discriminator_model_image_A = define_discriminator(
            model_info_xx.net["discriminator"]
        )
        self.discriminator_model_image_B = define_discriminator(
            model_info_xx.net["discriminator"]
        )
        if train:
            # buffer for fake images
            if "fake_pool_size" in model_info_xx.criterion:
                max_fake_pool = model_info_xx.criterion["fake_pool_size"]
            else:
                max_fake_pool = 50
            self.fake_pool_A = ReplayBuffer(max_fake_pool)
            self.fake_pool_B = ReplayBuffer(max_fake_pool)

            # get loss functions
            self.gan_loss = parse_config(model_info_xx.criterion["gan_loss"])
            self.cycle_loss = parse_config(model_info_xx.criterion["cycle_loss"])
            self.identity_loss = parse_config(model_info_xx.criterion["identity_loss"])

            # get weights of different loss
            self.gan_w = model_info_xx.criterion["weights"]["gan_loss"]
            self.cycle_w = model_info_xx.criterion["weights"]["cycle_loss"]
            self.identity_w = model_info_xx.criterion["weights"]["identity_loss"]

            # get info of optimizer and scheduler
            self.optimizer_info = model_info_xx.optimizer
            self.scheduler_info = model_info_xx.scheduler
        else:
            self.direction = model_info_xx.model_extra["inference_direction"]

    def forward(self, x):
        if self.direction == "AtoB":
            return self.generator_model_image_AtoB(x)
        elif self.direction == "BtoA":
            return self.generator_model_image_BtoA(x)

    def configure_optimizers(self):
        generator_optimizer_func = parse_config_func(self.optimizer_info["generator"])
        generator_scheduler_func = parse_config_func(self.scheduler_info["generator"])
        generator_optimizer = generator_optimizer_func(
            itertools.chain(
                self.generator_model_image_AtoB.parameters(),
                self.generator_model_image_BtoA.parameters(),
            )
        )
        generator_scheduler = generator_scheduler_func(generator_optimizer)

        discriminator_optimizer_func = parse_config_func(
            self.optimizer_info["discriminator"]
        )
        discriminator_scheduler_func = parse_config_func(
            self.scheduler_info["discriminator"]
        )
        discriminator_optimizer = discriminator_optimizer_func(
            itertools.chain(
                self.discriminator_model_image_A.parameters(),
                self.discriminator_model_image_B.parameters(),
            )
        )
        discriminator_scheduler = discriminator_scheduler_func(discriminator_optimizer)

        return (
            [generator_optimizer, discriminator_optimizer],
            [
                generator_scheduler,
                discriminator_scheduler,
            ],
        )

    # only for test or validation
    def run_step(self, batch, batch_idx):
        # get the data
        image_A = batch["IM"]
        image_B = batch["GT"]

        # run generators and discriminators
        fake_B_from_A = self.generator_model_image_AtoB(image_A)
        fake_A_from_B = self.generator_model_image_BtoA(image_B)

        predFakeA = self.discriminator_model_image_A(fake_A_from_B)
        predFakeB = self.discriminator_model_image_B(fake_B_from_A)

        pred_real = torch.ones_like(predFakeA, requires_grad=False)
        pred_fake = torch.zeros_like(predFakeA, requires_grad=False)

        # GAN loss
        gan_loss_B = self.gan_loss(predFakeB, pred_real)
        gan_loss_A = self.gan_loss(predFakeA, pred_real)
        gan_loss = (gan_loss_A + gan_loss_B) / 2

        # cycle loss
        cycle_loss_A = self.cycle_loss(
            self.generator_model_image_BtoA(fake_B_from_A), image_A
        )
        cycle_loss_B = self.cycle_loss(
            self.generator_model_image_AtoB(fake_A_from_B), image_B
        )
        cycle_loss = (cycle_loss_A + cycle_loss_B) / 2

        # identity loss
        fake_A_from_A = self.generator_model_image_BtoA(image_A)
        fake_B_from_B = self.generator_model_image_BtoA(image_B)
        identity_loss_A = self.identity_loss(fake_A_from_A, image_A)
        identity_loss_B = self.identity_loss(fake_B_from_B, image_B)
        identity_loss = (identity_loss_A + identity_loss_B) / 2

        # weighted sum into generator loss
        G_loss = (
            gan_loss * self.gan_w
            + cycle_loss * self.cycle_w
            + identity_loss * self.identity_w
        )

        # discriminator on domain A
        predRealA = self.discriminator_model_image_A(image_A)
        dis_loss_A_real = self.gan_loss(predRealA, pred_real)
        dis_loss_A_fake = self.gan_loss(predFakeA, pred_fake)

        # discriminator on domain B
        predRealB = self.discriminator_model_image_B(image_B)
        dis_loss_B_real = self.gan_loss(predRealB, pred_real)
        dis_loss_B_fake = self.gan_loss(predFakeB, pred_fake)

        # average over all dis loss
        D_loss = 0.5 * (
            dis_loss_A_fake + dis_loss_A_real + dis_loss_B_fake + dis_loss_B_real
        )

        output = OrderedDict({"generator_loss": G_loss, "discriminator_loss": D_loss})
        return output

    def training_step(self, batch, batch_idx, optimizer_idx):
        # get the batch
        image_A = batch["IM"]
        image_B = batch["GT"]

        # generate fake images
        fake_A_from_B = self.generator_model_image_BtoA(image_B)
        fake_B_from_A = self.generator_model_image_AtoB(image_A)

        if optimizer_idx == 0:
            ############################
            # train generators
            ############################
            # GAN loss
            predFakeA = self.discriminator_model_image_A(fake_A_from_B)
            predFakeB = self.discriminator_model_image_B(fake_B_from_A)

            pred_real = torch.ones_like(predFakeA, requires_grad=False)

            gan_loss_B = self.gan_loss(predFakeB, pred_real)
            gan_loss_A = self.gan_loss(predFakeA, pred_real)
            gan_loss = (gan_loss_A + gan_loss_B) / 2

            # cycle loss
            cycle_loss_A = self.cycle_loss(
                self.generator_model_image_BtoA(fake_B_from_A), image_A
            )
            cycle_loss_B = self.cycle_loss(
                self.generator_model_image_AtoB(fake_A_from_B), image_B
            )
            cycle_loss = (cycle_loss_A + cycle_loss_B) / 2

            # identity loss
            fake_A_from_A = self.generator_model_image_BtoA(image_A)
            fake_B_from_B = self.generator_model_image_BtoA(image_B)
            identity_loss_A = self.identity_loss(fake_A_from_A, image_A)
            identity_loss_B = self.identity_loss(fake_B_from_B, image_B)
            identity_loss = (identity_loss_A + identity_loss_B) / 2

            # weighted sum into generator loss
            G_loss = (
                gan_loss * self.gan_w
                + cycle_loss * self.cycle_w
                + identity_loss * self.identity_w
            )
            tqdm_dict = {
                "g_loss": G_loss,
                "validity": gan_loss,
                "reconstr": cycle_loss,
                "identity": identity_loss,
            }
            output = OrderedDict(
                {"loss": G_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
            )
            self.log("Generator Loss", tqdm_dict["g_loss"])

            if self.verbose and batch_idx == 0:
                # check if the log path exists, if not create one
                Path(self.trainer.log_dir).mkdir(parents=True, exist_ok=True)

                fake_images_A = fake_A_from_B.detach()
                out_fn = (
                    self.trainer.log_dir
                    + os.sep
                    + str(self.current_epoch)
                    + "_fake_A.tiff"
                )
                imsave(out_fn, fake_images_A[0].detach().cpu().numpy())
                fake_images_B = fake_B_from_A.detach()
                out_fn = (
                    self.trainer.log_dir
                    + os.sep
                    + str(self.current_epoch)
                    + "_fake_B.tiff"
                )
                imsave(out_fn, fake_images_B[0].detach().cpu().numpy())
                out_fn = (
                    self.trainer.log_dir
                    + os.sep
                    + str(self.current_epoch)
                    + "_real_B.tiff"
                )
                imsave(out_fn, image_B[0].detach().cpu().numpy())
                out_fn = (
                    self.trainer.log_dir
                    + os.sep
                    + str(self.current_epoch)
                    + "_real_A.tiff"
                )
                imsave(out_fn, image_A[0].detach().cpu().numpy())

            return output

        elif optimizer_idx == 1:
            ############################
            # train discriminators
            ############################
            fakeA_sample = self.fake_pool_A.push_and_pop(fake_A_from_B)
            fakeB_sample = self.fake_pool_B.push_and_pop(fake_B_from_A)
            predFakeA = self.discriminator_model_image_A(fakeA_sample)
            predFakeB = self.discriminator_model_image_B(fakeB_sample)

            pred_real = torch.ones_like(predFakeA, requires_grad=False)
            pred_fake = torch.zeros_like(predFakeA, requires_grad=False)

            # discriminator on domain A
            predRealA = self.discriminator_model_image_A(image_A)
            dis_loss_A_real = self.gan_loss(predRealA, pred_real)
            dis_loss_A_fake = self.gan_loss(predFakeA, pred_fake)

            # discriminator on domain B
            predRealB = self.discriminator_model_image_B(image_B)
            dis_loss_B_real = self.gan_loss(predRealB, pred_real)
            dis_loss_B_fake = self.gan_loss(predFakeB, pred_fake)

            # average over all dis loss
            D_loss = 0.5 * (
                dis_loss_A_fake + dis_loss_A_real + dis_loss_B_fake + dis_loss_B_real
            )
            tqdm_dict = {"d_loss": D_loss}
            output = OrderedDict(
                {"loss": D_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
            )
            self.log(
                "Discriminator Loss", tqdm_dict["d_loss"]
            )  # on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return output

    def training_epoch_end(self, outputs):
        avg_loss = sum(
            [
                torch.stack([x["loss"] for x in outputs[i]]).mean().item() / 2
                for i in range(2)
            ]
        )
        G_mean_loss = torch.stack([x["loss"] for x in outputs[0]]).mean().item()
        D_mean_loss = torch.stack([x["loss"] for x in outputs[1]]).mean().item()
        self.log("Average Loss", avg_loss)
        self.log("Generator Loss", G_mean_loss)
        self.log("Discriminator Loss", D_mean_loss)
        return None

    def validation_step(self, batch, batch_idx):
        loss_dictionary = self.run_step(batch, batch_idx)
        return loss_dictionary

    def validation_epoch_end(self, validation_step_outputs):
        validation_generator_loss = (
            torch.stack([x["generator_loss"] for x in validation_step_outputs], dim=0)
            .mean()
            .item()
        )
        validation_discriminator_loss = (
            torch.stack(
                [x["discriminator_loss"] for x in validation_step_outputs], dim=0
            )
            .mean()
            .item()
        )
        self.log("val_loss_generator", validation_generator_loss)
        self.log("val_loss_discriminator", validation_discriminator_loss)
