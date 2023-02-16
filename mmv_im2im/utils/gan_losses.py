# -*- coding: utf-8 -*-

import torch
from mmv_im2im.utils.misc import parse_config

###############################################################################
# pix2pix HD original
# Total loss = ( GAN loss ) + ( lambda_FM ) * ( Feature Matching loss )
###############################################################################


class pix2pix_HD_original:
    def __init__(self, gan_loss, fm_loss, weights, **kwargs):
        super().__init__()
        self.gan_loss = parse_config(gan_loss)
        self.fm_loss = parse_config(fm_loss)
        self.gan_weight = weights["gan_loss"]
        self.fm_weight = weights["fm_loss"]

    def multi_gan(self, real_features, fake_features, n_D):
        loss_gan = 0
        for i in range(n_D):
            real_grid = torch.ones_like(real_features[i][-1], requires_grad=False)
            loss_gan += self.gan_loss(fake_features[i][-1], real_grid)
        return loss_gan

    def feature_matching(self, real_features, fake_features, n_D):
        loss_fm = 0
        for i in range(n_D):
            for j in range(len(fake_features[0])):
                loss_fm += self.fm_loss(
                    fake_features[i][j], real_features[i][j].detach()
                )
        return loss_fm

    def generator_step(self, discriminator, image_A, image_B, fake_B):
        assert image_A.size() == image_B.size()

        n_D = discriminator.n_D
        real_features = discriminator(torch.cat((image_A, image_B), dim=1))
        fake_features = discriminator(torch.cat((image_A, fake_B), dim=1))

        # GAN loss
        loss_gan = self.multi_gan(real_features, fake_features, n_D)

        # Feature Matching Loss
        loss_fm = self.feature_matching(real_features, fake_features, n_D)

        return (1.0 / n_D) * (self.gan_weight * loss_gan + self.fm_weight * loss_fm)

    def discriminator_step(self, discriminator, image_A, image_B, fake_B):
        # fake_B has been detached!!!!
        # note: here, A and B could have different number of channels
        assert image_A.size()[2:] == image_B.size()[2:]
        loss_D = 0
        n_D = discriminator.n_D
        real_features = discriminator(torch.cat((image_A, image_B), dim=1))
        fake_features = discriminator(torch.cat((image_A, fake_B), dim=1))

        # Note: real_features[i][j] refers to the j-th output from the
        # i-th discriminator. When j = -1, it refers to the final prediction,
        # otherwise refers to intermediate features
        for i in range(n_D):
            real_grid = torch.ones_like(real_features[i][-1], requires_grad=False)
            fake_grid = torch.zeros_like(real_features[i][-1], requires_grad=False)

            loss_D += (
                self.gan_loss(real_features[i][-1], real_grid)
                + self.gan_loss(fake_features[i][-1], fake_grid)
            ) * 0.5

        return loss_D * (1.0 / n_D)


###############################################################################
# modified pix2pix HD
# Total loss = weighted sum of "GAN loss", "Feature Matching loss", and
# "reconstruction loss"
###############################################################################


class pix2pix_HD(pix2pix_HD_original):
    def __init__(self, gan_loss, fm_loss, weights, **kwargs):
        super().__init__(gan_loss, fm_loss, weights)
        self.recon_loss = parse_config(kwargs["reconstruction_loss"])
        self.recon_weight = weights["reconstruction_loss"]

    def generator_step(self, discriminator, image_A, image_B, fake_B):
        # note: A and B could have different numbers of channels
        assert image_A.size()[2:] == image_B.size()[2:]

        n_D = discriminator.n_D
        real_features = discriminator(torch.cat((image_A, image_B), dim=1))
        fake_features = discriminator(torch.cat((image_A, fake_B), dim=1))

        # GAN loss
        loss_gan = self.multi_gan(real_features, fake_features, n_D)

        # Feature Matching Loss
        loss_fm = self.feature_matching(real_features, fake_features, n_D)

        # reconstruction loss
        loss_recon = self.recon_loss(image_B, fake_B)

        return (1.0 / n_D) * (
            self.gan_weight * loss_gan + self.fm_weight * loss_fm
        ) + self.recon_weight * loss_recon


###############################################################################
# original pix2pix basic
# Total loss = weighted sum of "GAN loss", "Feature Matching loss", and
# "reconstruction loss"
###############################################################################


class pix2pix_basic:
    def __init__(self, gan_loss, reconstruction_loss, weights):
        super().__init__()
        self.gan_loss = parse_config(gan_loss)
        self.recon_loss = parse_config(reconstruction_loss)
        self.gan_weight = weights["gan_loss"]
        self.recon_weight = weights["reconstruction_loss"]

    def generator_step(self, discriminator, image_A, image_B, fake_B):
        assert image_A.size() == image_B.size()
        fake_features = discriminator(torch.cat((image_A, fake_B), dim=1))

        # GAN loss
        real_grid = torch.ones_like(fake_features, requires_grad=False)
        loss_gan = self.gan_loss(fake_features, real_grid)

        # reconstruction loss
        loss_recon = self.recon_loss(image_B, fake_B)

        return self.gan_weight * loss_gan + self.recon_weight * loss_recon

    def discriminator_step(self, discriminator, image_A, image_B, fake_B):
        # fake_B has been detached!!!!

        assert image_A.size() == image_B.size()
        real_features = discriminator(torch.cat((image_A, image_B), dim=1))
        fake_features = discriminator(torch.cat((image_A, fake_B), dim=1))

        real_grid = torch.ones_like(real_features, requires_grad=False)
        fake_grid = torch.zeros_like(real_features, requires_grad=False)

        loss_D = (
            self.gan_loss(real_features, real_grid)
            + self.gan_loss(fake_features, fake_grid)
        ) * 0.5

        return loss_D
