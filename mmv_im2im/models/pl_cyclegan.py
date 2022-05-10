"""
This module provides lighting module for cycleGAN
# adapted from https://github.com/Adi-iitd/AI-Art/blob/master/src/CycleGAN/CycleGAN-PL.py
"""

import torchio as tio
import torch
import pytorch_lightning as pl
from mmv_im2im.utils.misc import parse_config, parse_config_func
from mmv_im2im.utils.gan_utils import ReplayBuffer
from typing import Dict
from collections import OrderedDict
import itertools


class Model(pl.LightningModule):
    def __init__(self, model_info_xx: Dict, train: bool = True):
        super(Model, self).__init__()
        self.generator_model_image_AtoB = parse_config(model_info_xx["generator"])
        self.generator_model_image_BtoA = parse_config(model_info_xx["generator"])
        self.discriminator_model_image_A = parse_config(model_info_xx["discriminator"])
        self.discriminator_model_image_B = parse_config(model_info_xx["discriminator"])
        if train:
            # buffer for fake images
            if "fake_pool_size" in model_info_xx["criterion"]:
                max_fake_pool = model_info_xx["criterion"]["fake_pool_size"]
            else:
                max_fake_pool = 50
            self.fake_pool_A = ReplayBuffer(max_fake_pool)
            self.fake_pool_B = ReplayBuffer(max_fake_pool)

            # get loss functions
            self.gan_loss = parse_config(model_info_xx["criterion"]["gan_loss"])
            self.cycle_loss = parse_config(model_info_xx["criterion"]["cycle_loss"])
            self.identity_loss = parse_config(
                model_info_xx["criterion"]["identity_loss"]
            )

            # get weights of different loss
            self.gan_w = model_info_xx["criterion"]["gan_loss"]
            self.cycle_w = model_info_xx["criterion"]["cycle_loss"]
            self.identity_w = model_info_xx["criterion"]["identity_loss"]

            # get info of optimizer and scheduler
            self.optimizer_info = model_info_xx["optimizer"]
            self.scheduler_info = model_info_xx["scheduler"]
        else:
            self.direction = model_info_xx["direction"]

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

        return [generator_optimizer, discriminator_optimizer], [
            generator_scheduler,
            discriminator_scheduler,
        ]

    # only for test or validation
    def run_step(self, batch, batch_idx):
        # get the data
        image_A = batch["source"][tio.DATA]
        image_B = batch["target"][tio.DATA]

        # generator A to B: A --> fake B
        fake_image_B = self.generator_model_image_AtoB(image_A)
        D_B_real = self.discriminator_model_image_B(image_B)
        D_B_fake = self.discriminator_model_image_B(fake_image_B.detach())
        D_B_real_loss = self.MSE_criterion(D_B_real, torch.ones_like(D_B_real))
        D_B_fake_loss = self.MSE_criterion(D_B_fake, torch.zeros_like(D_B_fake))
        D_loss = D_B_real_loss + D_B_fake_loss
        loss_G_B = self.MSE_criterion(D_B_fake, torch.ones_like(D_B_fake))
        validity_loss = loss_G_B

        # generator B to A: fake B --> reconstructed A
        cycle_A = self.generator_model_image_BtoA(fake_image_B)
        cycle_A_loss = self.L1_criterion(image_A, cycle_A)
        cycle_loss = cycle_A_loss

        # identity regularization;
        # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/issues/322
        identity_A = self.generator_model_image_BtoA(image_A)
        identity_B = self.generator_model_image_AtoB(image_B)
        identity_A_loss = self.L1_criterion(image_A, identity_A)
        identity_B_loss = self.L1_criterion(image_B, identity_B)
        identity_loss = (identity_A_loss + identity_B_loss) / 2

        # combine the loss
        G_loss = validity_loss + cycle_loss * self.cycle_w + identity_loss * self.id_w
        output = OrderedDict({"generator_loss": G_loss, "discriminator_loss": D_loss})
        return output

    def training_step(self, batch, batch_idx, optimizer_idx):
        # get the batch
        image_A = batch["source"][tio.DATA]
        image_B = batch["target"][tio.DATA]

        # generate fake images
        fake_A_from_B = self.generator_model_image_BtoA(image_B)
        fake_B_from_A = self.generator_model_image_AtoB(image_A)

        if optimizer_idx == 0:
            ############################
            # train generators
            ############################
            # GAN loss
            gan_valid = torch.ones_like(image_A, requires_grad=False)
            gan_loss_B = self.gan_loss(
                self.discriminator_model_image_B(fake_B_from_A), gan_valid
            )
            gan_loss_A = self.gan_loss(
                self.discriminator_model_image_A(fake_A_from_B), gan_valid
            )
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
            return output

        elif optimizer_idx == 1:
            ############################
            # train discriminators
            ############################

            pred_real = torch.ones_like(image_A, requires_grad=False)
            pred_fake = torch.zeros_like(image_A, requires_grad=False)
            fakeA_sample = self.fake_pool_A.push_and_pop(fake_A_from_B)
            fakeB_sample = self.fake_pool_B.push_and_pop(fake_B_from_A)

            # discriminator on domain A
            predRealA = self.discriminator_model_image_A(image_A)
            dis_loss_A_real = self.gan_loss(predRealA, pred_real)

            predFakeA = self.discriminator_model_image_A(fakeA_sample)
            dis_loss_A_fake = self.gan_loss(predFakeA, pred_fake)

            # discriminator on domain B
            predRealB = self.discriminator_model_image_B(image_B)
            dis_loss_B_real = self.gan_loss(predRealB, pred_real)

            predFakeB = self.discriminator_model_image_B(fakeB_sample)
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
