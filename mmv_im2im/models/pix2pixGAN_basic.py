
import torch
from typing import Dict
import pytorch_lightning as pl
import torchio as tio
from mmv_im2im.utils.misc import parse_config, parse_config_func
from mmv_im2im.utils.piecewise_inference import predict_piecewise
from collections import OrderedDict


class Model(pl.LightningModule):
    def __init__(self, model_info_xx: Dict, train: bool = True):
        super(Model, self).__init__()
        self.generator_model = parse_config(
                                model_info_xx["generator_net"]
                                )
        self.discriminator_model = parse_config(
                                model_info_xx["discriminator_net"]
                                )
        self.sliding_window = model_info_xx["sliding_window_params"]
        if train:
            self.criterion_BCE = parse_config(model_info_xx["BCE_criterion"])
            self.criterion_L1 = parse_config(model_info_xx["L1_criterion"])
            self.optimizer_func = parse_config_func(model_info_xx["optimizer"])
            self.lambda_l1 = 100
            self.save_checkpoint_n_epochs = 1000

    def forward(self, x):
        x = self.generator_model(x)
        return x

    def configure_optimizers(self):
        self.optim_gen = self.optimizer_func(
                                self.generator_model.parameters()
                                )
        self.optim_disc = self.optimizer_func(
                                self.discriminator_model.parameters()
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
        pred_fake = self.discriminator_model(image_A, fake_image_B)
        G_loss_real = self.criterion_BCE(pred_fake, torch.ones_like(pred_fake))
        L1_loss = self.criterion_L1(fake_image_B, image_B)
        G_Loss = G_loss_real + self.lambda_l1 * L1_loss
        pred_real = self.discriminator_model(image_A, image_B)
        D_loss_real = self.criterion_BCE(
                                pred_real, torch.ones_like(pred_real)
                                )
        D_loss_fake = self.criterion_BCE(
                                pred_fake, torch.zeros_like(pred_fake)
                                )
        D_Loss = (D_loss_real+D_loss_fake)/2
        output = OrderedDict({"generator_loss": G_Loss,
                              "discriminator_loss": D_Loss})
        return output

    def training_step(self, batch, batch_idx, optimizer_idx):
        image_A = batch["source"][tio.DATA]
        image_B = batch["target"][tio.DATA]
        fake_image_B = self.generator_model(image_A)
        pred_fake = self.discriminator_model(image_A, fake_image_B)
        if self.current_epoch % (self.save_checkpoint_n_epochs) == 0:
            self.trainer.save_checkpoint(
                            f'pix2pix_model_{self.current_epoch}_state.ckpt'
                            )
        if optimizer_idx == 0:
            G_loss_real = self.criterion_BCE(
                            pred_fake, torch.ones_like(pred_fake)
                            )
            L1_loss = self.criterion_L1(fake_image_B, image_B)
            G_Loss = G_loss_real + self.lambda_l1 * L1_loss
            tqdm_dict = {"g_loss": G_Loss}
            output = OrderedDict({"loss": G_Loss, "progress_bar": tqdm_dict,
                                  "log": tqdm_dict})
            return output

        elif optimizer_idx == 1:
            pred_real = self.discriminator_model(image_A, image_B)
            D_loss_real = self.criterion_BCE(
                                pred_real, torch.ones_like(pred_real)
                                )
            D_loss_fake = self.criterion_BCE(
                                pred_fake, torch.zeros_like(pred_fake)
                                )
            D_Loss = (D_loss_real+D_loss_fake)/2
            d_real_acu = torch.ge(pred_real.squeeze(), 0.5).float()
            d_fake_acu = torch.le(pred_fake.squeeze(), 0.5).float()
            d_total_acu = torch.mean(torch.cat((d_real_acu, d_fake_acu), 0))
            tqdm_dict = {"d_loss": D_Loss}
            output = OrderedDict({"loss": D_Loss,
                                  "progress_bar": tqdm_dict,
                                  "log": tqdm_dict,
                                  "accuracy": d_total_acu})
            return output

    def training_epoch_end(self, outputs):
        G_mean_loss = torch.stack(
                            [x['loss'] for x in outputs[0]]
                            ).mean().item()
        D_mean_loss = torch.stack(
                            [x['loss'] for x in outputs[1]]
                            ).mean().item()
        D_mean_accuracy = torch.stack(
                            [x['accuracy'] for x in outputs[1]]
                            ).mean().item()
        self.log("Generator Loss", G_mean_loss)
        self.log("Discriminator Loss", D_mean_loss)
        self.log("Discriminator Accuracy", D_mean_accuracy)

    def validation_step(self, batch, batch_idx):
        loss_dictionary = self.run_step(batch, batch_idx)
        return loss_dictionary

    def validation_epoch_end(self, val_step_outputs):
        val_generator_loss = torch.stack(
                    [
                        x["generator_loss"] for x in val_step_outputs], dim=0
                    ).mean().item()
        val_discriminator_loss = torch.stack(
                    [x["discriminator_loss"] for x in val_step_outputs], dim=0
                    ).mean().item()
        self.log("val_loss_generator", val_generator_loss)
        self.log("val_loss_discriminator", val_discriminator_loss)
