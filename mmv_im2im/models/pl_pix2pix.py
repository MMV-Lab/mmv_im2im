import os
import torch
from typing import Dict
import pytorch_lightning as pl
from mmv_im2im.utils.misc import parse_config_func
from mmv_im2im.utils.model_utils import init_weights
from mmv_im2im.models.nets.gans import define_generator, define_discriminator
from tifffile import imsave
from importlib import import_module


class Model(pl.LightningModule):
    def __init__(self, model_info_xx: Dict, train: bool = True, verbose: bool = False):
        super(Model, self).__init__()

        self.verbose = verbose
        self.generator = define_generator(model_info_xx["generator"])
        self.discriminator = define_discriminator(model_info_xx["discriminator"])
        # self.sliding_window = model_info_xx["sliding_window_params"]
        if train:
            # initialize model weights
            init_weights(self.generator, init_type="normal")
            init_weights(self.discriminator, init_type="normal")

            # get loss functions
            loss_category = model_info_xx["criterion"].pop("type")
            loss_module = import_module("mmv_im2im.utils.gan_losses")
            loss_func = getattr(loss_module, loss_category)
            self.loss = loss_func(**model_info_xx["criterion"])

            # get info of optimizer and scheduler
            self.optimizer_info = model_info_xx["optimizer"]
            self.scheduler_info = model_info_xx["scheduler"]

    def forward(self, x):
        # if x.size()[-1] == 1:
        #    x = torch.squeeze(x, dim=-1)
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
            generator_scheduler,
        ]

    def training_step(self, batch, batch_idx, optimizer_idx):

        # imageB : real image
        # imageA : condition image
        # conditional GAN refers generating a realistic image (image B as ground truth),
        # conditioned on image A
        image_A = batch["IM"]
        image_B = batch["GT"]

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
        # if image_A.size()[-1] == 1:
        #    image_A = torch.squeeze(image_A, dim=-1)
        #    image_B = torch.squeeze(image_B, dim=-1)

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
            if not os.path.exists(self.trainer.log_dir):
                os.mkdir(self.trainer.log_dir)
            fake_images = fake_B.detach()
            out_fn = (
                self.trainer.log_dir + os.sep + str(self.current_epoch) + "_fake_B.tiff"
            )
            imsave(out_fn, fake_images[0].detach().cpu().numpy())
            out_fn = (
                self.trainer.log_dir + os.sep + str(self.current_epoch) + "_real_B.tiff"
            )
            imsave(out_fn, image_B[0].detach().cpu().numpy())
            out_fn = (
                self.trainer.log_dir + os.sep + str(self.current_epoch) + "_real_A.tiff"
            )
            imsave(out_fn, image_A[0].detach().cpu().numpy())

        return loss

    def training_epoch_end(self, outputs):
        G_mean_loss = torch.stack([x["loss"] for x in outputs[0]]).mean().item()
        D_mean_loss = torch.stack([x["loss"] for x in outputs[1]]).mean().item()
        self.log("generator_loss", G_mean_loss)
        self.log("discriminator_loss", D_mean_loss)

    """
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
    """
