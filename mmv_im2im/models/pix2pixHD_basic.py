import torch
from typing import Dict
import pytorch_lightning as pl
import torchio as tio
from mmv_im2im.utils.misc import parse_config, parse_config_func
from mmv_im2im.models.pix2pixHD_generator_discriminator_2D import _get_grid
from collections import OrderedDict


class Model(pl.LightningModule):
    def __init__(self, model_info_xx: Dict, train: bool = True):
        super(Model, self).__init__()
        self.generator_model = parse_config(model_info_xx["generator_net"])
        self.discriminator_model = parse_config(model_info_xx["discriminator_net"])
        if train:
            self.MSE_criterion = parse_config(model_info_xx["MSE_criterion"])
            self.FMcriterion = parse_config(model_info_xx["FM_criterion"])
            self.optimizer_func = parse_config_func(model_info_xx["optimizer"])
            self.lamda = 10
            self.save_checkpoint_n_epochs = 50
            self.n_D = model_info_xx["discriminator_net"]["params"]["n_D"]

    def forward(self, x):
        x = self.generator_model(x)
        return x

    def configure_optimizers(self):
        self.optim_gen = self.optimizer_func(self.generator_model.parameters())
        self.optim_disc = self.optimizer_func(self.discriminator_model.parameters())
        optimizers = [self.optim_gen, self.optim_disc]
        return optimizers

    def run_step(self, batch, batch_idx):
        image_A = batch["source"][tio.DATA]
        image_B = batch["target"][tio.DATA]
        ##########################################
        # check if the data is 2D or 3D
        ##########################################
        # torchio will add dummy dimension to 2D images to ensure 4D tensor
        # see: https://github.com/fepegar/torchio/blob/1c217d8716bf42051e91487ece82f0372b59d903/torchio/data/io.py#L402  # noqa E501
        # but in PyTorch, we usually follow the convention as C x D x H x W
        # or C x Z x Y x X, where the Z dimension of depth dimension is before
        # HW or YX. Padding the dummy dimmension at the end is okay and
        # actually makes data augmentation easier to implement. For example,
        # you have 2D image of Y x X, then becomes 1 x Y x X x 1. If you want
        # to apply a flip on the first dimension of your image, i.e. Y, you
        # can simply specify axes as [0], which is compatable
        # with the syntax for fliping along Y in 1 x Y x X x 1.
        # But, the FCN models do not like this. We just need to remove the
        # dummy dimension

        if image_A.size()[-1] == 1:
            image_A = torch.squeeze(image_A, dim=-1)
            image_B = torch.squeeze(image_B, dim=-1)
        loss_D = 0
        loss_G = 0
        loss_G_FM = 0
        fake = self.generator_model(image_A)
        real_features = self.discriminator_model(torch.cat((image_A, image_B), dim=1))
        fake_features = self.discriminator_model(
            torch.cat((image_A, fake.detach()), dim=1)
        )
        for i in range(self.n_D):
            real_grid = _get_grid(real_features[i][-1], is_real=True).to(
                self.device, self.dtype
            )
            fake_grid = _get_grid(fake_features[i][-1], is_real=False).to(
                self.device, self.dtype
            )
            loss_D += (
                self.MSE_criterion(real_features[i][-1], real_grid)
                + self.MSE_criterion(fake_features[i][-1], fake_grid)
            ) * 0.5
        fake_features = self.discriminator_model(torch.cat((image_A, fake), dim=1))

        for i in range(self.n_D):
            real_grid = _get_grid(fake_features[i][-1], is_real=True).to(
                self.device, self.dtype
            )
            loss_G += self.MSE_criterion(fake_features[i][-1], real_grid)
            for j in range(len(fake_features[0])):
                loss_G_FM += self.FMcriterion(
                    fake_features[i][j], real_features[i][j].detach()
                )
            loss_G += loss_G_FM * (1.0 / self.n_D) * self.lamda

        output = OrderedDict({"generator_loss": loss_G, "discriminator_loss": loss_D})
        return output

    def training_step(self, batch, batch_idx, optimizer_idx):
        image_A = batch["source"][tio.DATA]
        image_B = batch["target"][tio.DATA]
        if image_A.size()[-1] == 1:
            image_A = torch.squeeze(image_A, dim=-1)
            image_B = torch.squeeze(image_B, dim=-1)
        loss_D = 0
        loss_G = 0
        loss_G_FM = 0
        fake = self.generator_model(image_A)
        if optimizer_idx == 0:
            real_features = self.discriminator_model(
                torch.cat((image_A, image_B), dim=1)
            )
            fake_features = self.discriminator_model(torch.cat((image_A, fake), dim=1))
            for i in range(self.n_D):
                real_grid = _get_grid(fake_features[i][-1], is_real=True).to(
                    self.device, self.dtype
                )
                loss_G += self.MSE_criterion(fake_features[i][-1], real_grid)
                for j in range(len(fake_features[0])):
                    loss_G_FM += self.FMcriterion(
                        fake_features[i][j], real_features[i][j].detach()
                    )
                loss_G += loss_G_FM * (1.0 / self.n_D) * self.lamda
            tqdm_dict = {"g_loss": loss_G}
            output = OrderedDict(
                {"loss": loss_G, "progress_bar": tqdm_dict, "log": tqdm_dict}
            )
            return output

        elif optimizer_idx == 1:
            real_features = self.discriminator_model(
                torch.cat((image_A, image_B), dim=1)
            )
            fake_features = self.discriminator_model(
                torch.cat((image_A, fake.detach()), dim=1)
            )
            for i in range(self.n_D):
                real_grid = _get_grid(real_features[i][-1], is_real=True).to(
                    self.device, self.dtype
                )
                fake_grid = _get_grid(fake_features[i][-1], is_real=False).to(
                    self.device, self.dtype
                )
                loss_D += (
                    self.MSE_criterion(real_features[i][-1], real_grid)
                    + self.MSE_criterion(fake_features[i][-1], fake_grid)
                ) * 0.5
            d_real_acu = torch.ge(image_B.squeeze(), 0.5).float()
            d_fake_acu = torch.le(fake.squeeze(), 0.5).float()
            d_total_acu = torch.mean(torch.cat((d_real_acu, d_fake_acu), 0))
            tqdm_dict = {"d_loss": loss_D}
            output = OrderedDict(
                {
                    "loss": loss_D,
                    "progress_bar": tqdm_dict,
                    "log": tqdm_dict,
                    "accuracy": d_total_acu,
                }
            )
            return output

    def training_epoch_end(self, outputs):
        G_mean_loss = torch.stack([x["loss"] for x in outputs[0]]).mean().item()
        D_mean_loss = torch.stack([x["loss"] for x in outputs[1]]).mean().item()
        D_mean_accuracy = torch.stack([x["accuracy"] for x in outputs[1]]).mean().item()
        self.log("Generator Loss", G_mean_loss)
        self.log("Discriminator Loss", D_mean_loss)
        self.log("Discriminator Accuracy", D_mean_accuracy)

    def validation_step(self, batch, batch_idx):
        loss_dictionary = self.run_step(batch, batch_idx)
        if self.current_epoch % (self.save_checkpoint_n_epochs) == 0:
            self.trainer.save_checkpoint(
                f"pix2pixHD_model_2d_{self.current_epoch}_state.ckpt"
            )
        return loss_dictionary

    def validation_epoch_end(self, val_step_outputs):
        val_gen_loss = (
            torch.stack([x["generator_loss"] for x in val_step_outputs], dim=0)
            .mean()
            .item()
        )
        val_disc_loss = (
            torch.stack([x["discriminator_loss"] for x in val_step_outputs], dim=0)
            .mean()
            .item()
        )
        self.log("val_loss_generator", val_gen_loss)
        self.log("val_loss_discriminator", val_disc_loss)
