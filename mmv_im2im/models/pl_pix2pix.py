import os
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
from functools import partial


def weights_init(module):
    if isinstance(module, nn.Conv2d):
        module.weight.detach().normal_(0.0, 0.02)

    elif isinstance(module, nn.BatchNorm2d):
        module.weight.detach().normal_(1.0, 0.02)
        module.bias.detach().fill_(0.0)

"""
class PatchDiscriminator(nn.Module):
    def __init__(self):
        super(PatchDiscriminator, self).__init__()

        act = nn.LeakyReLU(0.2, inplace=True)
        input_channel = 2  # opt.input_ch + opt.target_ch
        n_df = 64   # opt.n_df
        norm = nn.InstanceNorm2d

        blocks = []
        blocks += [[nn.Conv2d(input_channel, n_df, kernel_size=4, padding=1, stride=2), act]]
        blocks += [[nn.Conv2d(n_df, 2 * n_df, kernel_size=4, padding=1, stride=2), norm(2 * n_df), act]]
        blocks += [[nn.Conv2d(2 * n_df, 4 * n_df, kernel_size=4, padding=1, stride=2), norm(4 * n_df), act]]
        blocks += [[nn.Conv2d(4 * n_df, 8 * n_df, kernel_size=4, padding=1, stride=1), norm(8 * n_df), act]]
        blocks += [[nn.Conv2d(8 * n_df, 1, kernel_size=4, padding=1, stride=1)]]

        self.n_blocks = len(blocks)
        for i in range(self.n_blocks):
            setattr(self, 'block_{}'.format(i), nn.Sequential(*blocks[i]))

    def forward(self, x):
        result = [x]
        for i in range(self.n_blocks):
            block = getattr(self, 'block_{}'.format(i))
            result.append(block(result[-1]))

        return result[1:]  # except for the input


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        for i in range(2):  # opt.n_D
            setattr(self, 'Scale_{}'.format(str(i)), PatchDiscriminator())
        self.n_D = 2  # opt.n_D

        print(self)
        print("the number of D parameters", sum(p.numel() for p in self.parameters() if p.requires_grad))

    def forward(self, x):
        result = []
        for i in range(self.n_D):
            result.append(getattr(self, 'Scale_{}'.format(i))(x))
            if i != self.n_D - 1:
                x = nn.AvgPool2d(kernel_size=3, padding=1, stride=2, count_include_pad=False)(x)
        return result
"""
# [3] Objective (Loss) functions


class Loss(object):
    def __init__(self):
        # self.device = torch.device('cuda:0')
        # self.dtype = torch.float16 if opt.data_type == 16 else torch.float32

        self.criterion = nn.MSELoss()
        self.FMcriterion = nn.L1Loss()
        self.n_D = 2  # opt.n_D

    def __call__(self, D, G, input, target):
        loss_D = 0
        loss_G = 0
        loss_G_FM = 0

        fake = G(input)

        real_features = D(torch.cat((input, target), dim=1))
        fake_features = D(torch.cat((input, fake.detach()), dim=1))

        for i in range(self.n_D):
            real_grid = torch.ones_like(real_features[i][-1])
            # get_grid(real_features[i][-1], is_real=True).to(self.device, self.dtype)
            fake_grid = torch.zeros_like(real_features[i][-1])
            # get_grid(fake_features[i][-1], is_real=False).to(self.device, self.dtype)

            loss_D += (self.criterion(real_features[i][-1], real_grid)
                       + self.criterion(fake_features[i][-1], fake_grid)) * 0.5

        fake_features = D(torch.cat((input, fake), dim=1))

        for i in range(self.n_D):
            real_grid = torch.ones_like(fake_features[i][-1])
            # get_grid(fake_features[i][-1], is_real=True).to(self.device, self.dtype)
            loss_G += self.criterion(fake_features[i][-1], real_grid)

            for j in range(len(fake_features[0])):
                loss_G_FM += self.FMcriterion(fake_features[i][j], real_features[i][j].detach())

            loss_G += loss_G_FM * 0.5 * 10  # loss_G_FM * (1.0 / self.opt.n_D) * self.opt.lambda_FM

        return loss_D, loss_G, target, fake


class Model(pl.LightningModule):
    def __init__(self, model_info_xx: Dict, train: bool = True, verbose: bool = False):
        super(Model, self).__init__()

        self.verbose = verbose
        # print(model_info_xx)
        # self.generator = Generator().apply(weights_init)
        # self.discriminator = Discriminator().apply(weights_init)
        # self.generator = Generator(1, 1)
        # self.discriminator = PatchGAN(2)
        self.generator = define_generator(model_info_xx["generator"])
        self.discriminator = define_discriminator(model_info_xx["discriminator"])
        # self.sliding_window = model_info_xx["sliding_window_params"]
        if train:
            # initialize model weights
            init_weights(self.generator, init_type='normal')
            init_weights(self.discriminator, init_type='normal')

            # get loss functions
            self.loss_func = Loss()
            # self.gan_loss = parse_config(model_info_xx["criterion"]["gan_loss"])
            # self.recon_loss = parse_config(model_info_xx["criterion"]["reconstruction_loss"])

            # get weights of recon loss
            self.lamda = model_info_xx["criterion"]["lamda"]

            # get info of optimizer and scheduler
            self.optimizer_info = model_info_xx["optimizer"]
            self.scheduler_info = model_info_xx["scheduler"]

    """
    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad
    """

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

    """
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

        self.generator.train()
        D_loss, G_loss, target_tensor, generated_tensor = self.loss_func(self.discriminator, self.generator, image_A, image_B)
        self.log('Generator Loss', G_loss)
        self.log('Discriminator Loss', D_loss)
        """
        real = image_B
        condition = image_A

        loss = None
        if optimizer_idx == 0:
            loss = self._disc_step(real, condition)
            self.log('PatchGAN Loss', loss)
        elif optimizer_idx == 1:
            loss = self._gen_step(real, condition)
            self.log('Generator Loss', loss)
        """

        if self.verbose and batch_idx == 0 and optimizer_idx == 0:
            if not os.path.exists(self.trainer.log_dir):
                os.mkdir(self.trainer.log_dir)
            fake_images = generated_tensor.detach()
            out_fn = self.trainer.log_dir + os.sep + str(self.current_epoch) +"_fake_B.tiff"
            imsave(out_fn, fake_images[0].detach().cpu().numpy())
            out_fn = self.trainer.log_dir + os.sep + str(self.current_epoch) +"_real_B.tiff"
            imsave(out_fn, image_B[0].detach().cpu().numpy())
            out_fn = self.trainer.log_dir + os.sep + str(self.current_epoch) +"_real_A.tiff"
            imsave(out_fn, image_A[0].detach().cpu().numpy())

        # return loss

        if optimizer_idx == 0:
            return D_loss
        elif optimizer_idx == 1:
            return G_loss


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
