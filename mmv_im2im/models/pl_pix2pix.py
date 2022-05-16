import torch
from typing import Dict
import pytorch_lightning as pl
import torchio as tio
from torch import nn
from mmv_im2im.utils.misc import (
    parse_config,
    parse_config_func
)
from mmv_im2im.utils.model_utils import init_weights
from mmv_im2im.models.nets.gans import define_generator, define_discriminator
from collections import OrderedDict
from tifffile import imsave

class DownSampleConv(nn.Module):

    def __init__(self, in_channels, out_channels, kernel=4, strides=2, padding=1, activation=True, batchnorm=True):
        """
        Paper details:
        - C64-C128-C256-C512-C512-C512-C512-C512
        - All convolutions are 4×4 spatial filters applied with stride 2
        - Convolutions in the encoder downsample by a factor of 2
        """
        super().__init__()
        self.activation = activation
        self.batchnorm = batchnorm

        self.conv = nn.Conv2d(in_channels, out_channels, kernel, strides, padding)

        if batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)

        if activation:
            self.act = nn.LeakyReLU(0.2)

    def forward(self, x):
        x = self.conv(x)
        if self.batchnorm:
            x = self.bn(x)
        if self.activation:
            x = self.act(x)
        return x


class UpSampleConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel=4,
        strides=2,
        padding=1,
        activation=True,
        batchnorm=True,
        dropout=False
    ):
        super().__init__()
        self.activation = activation
        self.batchnorm = batchnorm
        self.dropout = dropout

        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel, strides, padding)

        if batchnorm:
            self.bn = nn.BatchNorm2d(out_channels)

        if activation:
            self.act = nn.ReLU(True)

        if dropout:
            self.drop = nn.Dropout2d(0.5)

    def forward(self, x):
        x = self.deconv(x)
        if self.batchnorm:
            x = self.bn(x)

        if self.dropout:
            x = self.drop(x)
        return x


class Generator(nn.Module):

    def __init__(self, in_channels, out_channels):
        """
        Paper details:
        - Encoder: C64-C128-C256-C512-C512-C512-C512-C512
        - All convolutions are 4×4 spatial filters applied with stride 2
        - Convolutions in the encoder downsample by a factor of 2
        - Decoder: CD512-CD1024-CD1024-C1024-C1024-C512 -C256-C128
        """
        super().__init__()

        # encoder/donwsample convs
        self.encoders = [
            DownSampleConv(in_channels, 64, batchnorm=False),  # bs x 64 x 128 x 128
            DownSampleConv(64, 128),  # bs x 128 x 64 x 64
            DownSampleConv(128, 256),  # bs x 256 x 32 x 32
            DownSampleConv(256, 512),  # bs x 512 x 16 x 16
            DownSampleConv(512, 512),  # bs x 512 x 8 x 8
            DownSampleConv(512, 512),  # bs x 512 x 4 x 4
            DownSampleConv(512, 512),  # bs x 512 x 2 x 2
            DownSampleConv(512, 512, batchnorm=False),  # bs x 512 x 1 x 1
        ]

        # decoder/upsample convs
        self.decoders = [
            UpSampleConv(512, 512, dropout=True),  # bs x 512 x 2 x 2
            UpSampleConv(1024, 512, dropout=True),  # bs x 512 x 4 x 4
            UpSampleConv(1024, 512, dropout=True),  # bs x 512 x 8 x 8
            UpSampleConv(1024, 512),  # bs x 512 x 16 x 16
            UpSampleConv(1024, 256),  # bs x 256 x 32 x 32
            UpSampleConv(512, 128),  # bs x 128 x 64 x 64
            UpSampleConv(256, 64),  # bs x 64 x 128 x 128
        ]
        self.decoder_channels = [512, 512, 512, 512, 256, 128, 64]
        self.final_conv = nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1)
        self.tanh = nn.Tanh()

        self.encoders = nn.ModuleList(self.encoders)
        self.decoders = nn.ModuleList(self.decoders)

    def forward(self, x):
        skips_cons = []
        for encoder in self.encoders:
            x = encoder(x)

            skips_cons.append(x)

        skips_cons = list(reversed(skips_cons[:-1]))
        decoders = self.decoders[:-1]

        for decoder, skip in zip(decoders, skips_cons):
            x = decoder(x)
            # print(x.shape, skip.shape)
            x = torch.cat((x, skip), axis=1)

        x = self.decoders[-1](x)
        # print(x.shape)
        x = self.final_conv(x)
        return self.tanh(x)


class PatchGAN(nn.Module):

    def __init__(self, input_channels):
        super().__init__()
        self.d1 = DownSampleConv(input_channels, 64, batchnorm=False)
        self.d2 = DownSampleConv(64, 128)
        self.d3 = DownSampleConv(128, 256)
        self.d4 = DownSampleConv(256, 512)
        self.final = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, x, y):
        x = torch.cat([x, y], axis=1)
        x0 = self.d1(x)
        x1 = self.d2(x0)
        x2 = self.d3(x1)
        x3 = self.d4(x2)
        xn = self.final(x3)
        return xn


class Model(pl.LightningModule):
    def __init__(self, model_info_xx: Dict, train: bool = True):
        super(Model, self).__init__()
        # print(model_info_xx)
        self.generator = Generator(1, 1)
        self.discriminator = PatchGAN(2)
        # self.generator = define_generator(model_info_xx["generator"])
        # self.discriminator = define_discriminator(model_info_xx["discriminator"])
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

    def set_requires_grad(self, nets, requires_grad=False):
        """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

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

    def _gen_step(self, real_images, conditioned_images):
        # Pix2Pix has adversarial and a reconstruction loss
        # First calculate the adversarial loss
        fake_images = self.generator(conditioned_images)
        disc_logits = self.discriminator(fake_images, conditioned_images)
        adversarial_loss = self.gan_loss(disc_logits, torch.ones_like(disc_logits))

        # calculate reconstruction loss
        recon_loss = self.recon_loss(fake_images, real_images)
        lambda_recon = self.lamda

        return adversarial_loss + lambda_recon * recon_loss

    def _disc_step(self, real_images, conditioned_images):
        fake_images = self.generator(conditioned_images).detach()
        fake_logits = self.discriminator(fake_images, conditioned_images)

        real_logits = self.discriminator(real_images, conditioned_images)

        fake_loss = self.gan_loss(fake_logits, torch.zeros_like(fake_logits))
        real_loss = self.gan_loss(real_logits, torch.ones_like(real_logits))
        return (real_loss + fake_loss) / 2

    """
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
    """

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

        real = image_B
        condition = image_A

        loss = None
        if optimizer_idx == 0:
            loss = self._disc_step(real, condition)
            self.log('PatchGAN Loss', loss)
        elif optimizer_idx == 1:
            loss = self._gen_step(real, condition)
            self.log('Generator Loss', loss)

        if batch_idx == 0 and optimizer_idx == 0:
            fake_images = self.generator(condition).detach()
            out_fn = "./tmp2/"+ str(self.current_epoch) +"_fake_B.tiff"
            imsave(out_fn, fake_images[0].detach().cpu().numpy())
            out_fn = "./tmp2/"+ str(self.current_epoch) +"_real_B.tiff"
            imsave(out_fn, real[0].detach().cpu().numpy())
            out_fn = "./tmp2/"+ str(self.current_epoch) +"_real_A.tiff"
            imsave(out_fn, condition[0].detach().cpu().numpy())

        return loss

        """

        if optimizer_idx == 0:
            self.set_requires_grad(self.discriminator, True)
            # generate fake image
            fake_B = self.generator(image_A)

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
                "discriminator_loss", tqdm_dict["D_loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return output
        elif optimizer_idx == 1:
            self.set_requires_grad(self.discriminator, False)
            # generate fake image
            fake_B = self.generator(image_A)

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
            self.log("generator_loss", tqdm_dict["g_loss"], on_step=True, on_epoch=True, prog_bar=True, logger=True)
            return output
        """

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
