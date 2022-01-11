"""
This module provides lighting module for cycleGAN
"""

import torchio as tio
import torch
import pytorch_lightning as pl
from mmv_im2im.utils.misc import parse_config, parse_config_func
from mmv_im2im.utils.piecewise_inference import predict_piecewise
from typing import Dict
from collections import OrderedDict


class Model(pl.LightningModule):
    def __init__(self, model_info_xx: Dict, train: bool = True):
        super(Model, self).__init__()
        self.generator_model_image_A = parse_config(
                                            model_info_xx["generator_net"]
                                            )
        self.generator_model_image_B = parse_config(
                                            model_info_xx["generator_net"]
                                            )
        self.discriminator_model_image_A = parse_config(
                                            model_info_xx["discriminator_net"]
                                            )
        self.discriminator_model_image_B = parse_config(
                                            model_info_xx["discriminator_net"]
                                            )
        self.sliding_window = model_info_xx["sliding_window_params"]
        if train:
            self.L1_criterion = parse_config(model_info_xx["L1_criterion"])
            self.MSE_criterion = parse_config(model_info_xx["MSE_criterion"])
            self.optimizer_func = parse_config_func(model_info_xx["optimizer"])
            self.cycle_w = 10
            self.id_w = 2
            self.save_checkpoint_n_epochs = 100

    def forward(self, x):
        x = self.generator_model_image_B(x)
        return x

    def configure_optimizers(self):
        self.optim_gen = self.optimizer_func(
                        list(self.generator_model_image_A.parameters())
                        + list(self.generator_model_image_B.parameters())
                        )
        self.optim_disc = self.optimizer_func(
                        list(self.discriminator_model_image_A.parameters())
                        + list(self.discriminator_model_image_B.parameters())
                        )
        optimizers = [self.optim_gen, self.optim_disc]
        return optimizers

    def run_step(self, batch, batch_idx):
        image_A = batch["source"][tio.DATA]
        image_B = batch["target"][tio.DATA]
        fake_image_B = predict_piecewise(
            self,
            image_A[
                    0,
            ],
            **self.sliding_window
        )
        D_B_real = self.discriminator_model_image_B(image_B)
        D_B_fake = self.discriminator_model_image_B(fake_image_B.detach())
        D_B_real_loss = self.MSE_criterion(
                                D_B_real, torch.ones_like(D_B_real)
                                )
        D_B_fake_loss = self.MSE_criterion(
                                D_B_fake, torch.zeros_like(D_B_fake)
                                )
        D_loss = D_B_real_loss+D_B_fake_loss
        loss_G_B = self.MSE_criterion(D_B_fake, torch.ones_like(D_B_fake))
        validity_loss = loss_G_B
        cycle_A = self.generator_model_image_A(fake_image_B)
        cycle_A_loss = self.L1_criterion(image_A, cycle_A)
        cycle_loss = cycle_A_loss
        identity_A = self.generator_model_image_A(image_A)
        identity_B = self.generator_model_image_B(image_B)
        identity_A_loss = self.L1_criterion(image_A, identity_A)
        identity_B_loss = self.L1_criterion(image_B, identity_B)
        identity_loss = (identity_A_loss+identity_B_loss)/2
        G_loss = (
            validity_loss
            + cycle_loss*self.cycle_w
            + identity_loss*self.id_w
        )
        output = OrderedDict({"generator_loss": G_loss,
                              "discriminator_loss": D_loss})
        return output

    def training_step(self, batch, batch_idx, optimizer_idx):
        image_A = batch["source"][tio.DATA]
        image_B = batch["target"][tio.DATA]
        fake_image_A = self.generator_model_image_A(image_B)
        fake_image_B = self.generator_model_image_B(image_A)
        D_B_real = self.discriminator_model_image_B(image_B)
        D_B_fake = self.discriminator_model_image_B(fake_image_B.detach())
        D_A_real = self.discriminator_model_image_A(image_A)
        D_A_fake = self.discriminator_model_image_A(fake_image_A.detach())
        if self.current_epoch % (self.save_checkpoint_n_epochs) == 0:
            self.trainer.save_checkpoint(
                            f'cyclegan_model_{self.current_epoch}_state.ckpt')
        if optimizer_idx == 1:
            D_B_real_loss = self.MSE_criterion(
                                    D_B_real, torch.ones_like(D_B_real)
                                    )
            D_B_fake_loss = self.MSE_criterion(
                                    D_B_fake, torch.zeros_like(D_B_fake)
                                    )
            D_B_loss = D_B_real_loss+D_B_fake_loss
            D_A_real_loss = self.MSE_criterion(
                                    D_A_real, torch.ones_like(D_A_real)
                                    )
            D_A_fake_loss = self.MSE_criterion(
                                    D_A_fake, torch.zeros_like(D_A_fake)
                                    )
            D_A_loss = D_A_real_loss+D_A_fake_loss
            D_loss = (D_A_loss+D_B_loss)/2
            D_loss = (D_A_loss+D_B_loss)/2
            tqdm_dict = {"d_loss": D_loss}
            output = OrderedDict(
                                {"loss": D_loss, "progress_bar": tqdm_dict,
                                 "log": tqdm_dict})
            self.log("Discriminator Loss", tqdm_dict["d_loss"])
            return output
        elif optimizer_idx == 0:
            loss_G_B = self.MSE_criterion(D_B_fake, torch.ones_like(D_B_fake))
            loss_G_A = self.MSE_criterion(D_A_fake, torch.ones_like(D_A_fake))
            validity_loss = (loss_G_B+loss_G_A)/2
            cycle_A = self.generator_model_image_A(fake_image_B)
            cycle_B = self.generator_model_image_B(fake_image_A)
            cycle_A_loss = self.L1_criterion(image_A, cycle_A)
            cycle_B_loss = self.L1_criterion(image_B, cycle_B)
            cycle_loss = (cycle_A_loss + cycle_B_loss) / 2
            identity_A = self.generator_model_image_A(image_A)
            identity_B = self.generator_model_image_B(image_B)
            identity_A_loss = self.L1_criterion(image_A, identity_A)
            identity_B_loss = self.L1_criterion(image_B, identity_B)
            identity_loss = (identity_A_loss+identity_B_loss)/2
            G_loss = (
                validity_loss
                + cycle_loss*self.cycle_w
                + identity_loss*self.id_w
            )
            tqdm_dict = {"g_loss": G_loss,
                         'validity': validity_loss,
                         'reconstr': cycle_loss,
                         'identity': identity_loss}
            output = OrderedDict({"loss": G_loss,
                                  "progress_bar": tqdm_dict,
                                  "log": tqdm_dict})
            self.log("Generator Loss", tqdm_dict["g_loss"])
            return output

    def training_epoch_end(self, outputs):
        avg_loss = sum(
                        [
                            torch.stack(
                                [
                                    x['loss'] for x in outputs[i]
                                ]
                            ).mean().item() / 2 for i in range(2)
                        ]
                    )
        G_mean_loss = torch.stack(
                                    [
                                        x['loss'] for x in outputs[0]
                                    ]
                                ).mean().item()
        D_mean_loss = torch.stack(
                                    [
                                        x['loss'] for x in outputs[1]
                                    ]
                                ).mean().item()
        self.log("Average Loss", avg_loss)
        self.log("Generator Loss", G_mean_loss)
        self.log("Discriminator Loss", D_mean_loss)
        return None

    def validation_step(self, batch, batch_idx):
        loss_dictionary = self.run_step(batch, batch_idx)
        return loss_dictionary

    def validation_epoch_end(self, validation_step_outputs):
        validation_generator_loss = torch.stack(
                                    [
                                        x["generator_loss"]
                                        for x in validation_step_outputs
                                    ], dim=0).mean().item()
        validation_discriminator_loss = torch.stack(
                                    [
                                        x["discriminator_loss"]
                                        for x in validation_step_outputs
                                    ], dim=0).mean().item()
        self.log("val_loss_generator", validation_generator_loss)
        self.log("val?loss_discriminator", validation_discriminator_loss)
