import torch
from typing import Dict
import pytorch_lightning as pl
import torchio as tio
from mmv_im2im.utils.misc import (
    parse_config,
    parse_config_func
)
from mmv_im2im.utils.model_utils import init_weights
from mmv_im2im.models.nets.gans import define_generator, define_discriminator
from collections import OrderedDict


class Model(pl.LightningModule):
    def __init__(self, model_info_xx: Dict, train: bool = True):
        super(Model, self).__init__()
        # print(model_info_xx)
        self.generator = define_generator(model_info_xx["generator"])
        self.discriminator = define_discriminator(model_info_xx["discriminator"])
        # self.sliding_window = model_info_xx["sliding_window_params"]
        if train:
            # initialize model weights
            init_weights(self.generator)
            init_weights(self.discriminator)

            # get loss functions
            self.gan_loss = parse_config(model_info_xx["criterion"]["gan_loss"])
            self.recon_loss = parse_config(model_info_xx["criterion"]["reconstruction_loss"])

            # get weights of recon loss
            self.lamda = model_info_xx["criterion"]["lamda"]

            # get info of optimizer and scheduler
            self.optimizer_info = model_info_xx["optimizer"]
            self.scheduler_info = model_info_xx["scheduler"]

    def forward(self, x):
        if x.size()[-1] == 1:
            x = torch.squeeze(x, dim=-1)
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

        return [discriminator_optimizer, generator_optimizer], [
            discriminator_scheduler,
            generator_scheduler
        ]

    def run_step(self, batch, batch_idx):
        # get the data
        image_A = batch["source"][tio.DATA]
        image_B = batch["target"][tio.DATA]

        if image_A.size()[-1] == 1:
            image_A = torch.squeeze(image_A, dim=-1)
            image_B = torch.squeeze(image_B, dim=-1)

        # run generators and discriminators
        fake_B = self.generator(image_A)

        # discriminator loss
        predFakeB = self.discriminator(torch.cat((fake_B, image_A), axis=1))
        predRealB = self.discriminator(torch.cat((image_B, image_A), axis=1))
        pred_real = torch.ones_like(predFakeB, requires_grad=False)
        pred_fake = torch.zeros_like(predFakeB, requires_grad=False)

        fake_loss = self.gan_loss(predFakeB, pred_fake)
        real_loss = self.gan_loss(predRealB, pred_real)
        D_loss = (real_loss + fake_loss) / 2

        # GAN loss
        gan_loss = self.gan_loss(predFakeB, pred_real)

        # reconstruction loss
        recon_loss = self.recon_loss(fake_B, image_B)

        G_loss = gan_loss + self.lamda * recon_loss

        output = OrderedDict({"generator_loss": G_loss, "discriminator_loss": D_loss})
        return output

    def training_step(self, batch, batch_idx, optimizer_idx):

        # imageB : real image
        # imageA : condition image
        # conditional GAN refers generating a realistic image (image B as ground truth),
        # conditioned on image A
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

        # generate fake image
        fake_B = self.generator(image_A)

        if optimizer_idx == 0:
            ############################
            # train discriminators
            ############################
            predFakeB = self.discriminator(torch.cat((fake_B, image_A), axis=1))
            predRealB = self.discriminator(torch.cat((image_B, image_A), axis=1))
            pred_real = torch.ones_like(predFakeB, requires_grad=False)
            pred_fake = torch.zeros_like(predFakeB, requires_grad=False)

            fake_loss = self.gan_loss(predFakeB, pred_fake)
            real_loss = self.gan_loss(predRealB, pred_real)
            D_loss = (real_loss + fake_loss) / 2

            tqdm_dict = {"D_loss": D_loss}
            output = OrderedDict(
                {"loss": D_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
            )
            self.log(
                "Discriminator Loss", tqdm_dict["D_loss"]
            )  # on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return output
        elif optimizer_idx == 1:
            ############################
            # train generators
            ############################
            # GAN loss
            predFakeB = self.discriminator(torch.cat((fake_B, image_A), axis=1))
            pred_real = torch.ones_like(predFakeB, requires_grad=False)
            gan_loss = self.gan_loss(predFakeB, pred_real)

            # reconstruction loss
            recon_loss = self.recon_loss(fake_B, image_B)

            G_loss = gan_loss + self.lamda * recon_loss

            tqdm_dict = {
                "g_loss": G_loss,
                "validity": gan_loss,
                "reconstr": recon_loss,
            }
            output = OrderedDict(
                {"loss": G_loss, "progress_bar": tqdm_dict, "log": tqdm_dict}
            )
            self.log("Generator Loss", tqdm_dict["g_loss"])
            return output

    def training_epoch_end(self, outputs):
        G_mean_loss = torch.stack([x["loss"] for x in outputs[0]]).mean().item()
        D_mean_loss = torch.stack([x["loss"] for x in outputs[1]]).mean().item()
        self.log("Generator Loss", G_mean_loss)
        self.log("Discriminator Loss", D_mean_loss)

    def validation_step(self, batch, batch_idx):
        loss_dictionary = self.run_step(batch, batch_idx)
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
