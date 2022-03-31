# -*- coding: utf-8 -*-

import torch
from mmv_im2im.utils.misc import parse_config
from mmv_im2im.models.pix2pixHD_generator_discriminator_2D import _get_grid

# Objective (Loss) functions


# The loss configuration of the objective functions

# Total loss = ( LSGAN loss ) + ( lambda_FM ) * ( Feature Matching loss )


class Pix2PixHD_loss:
    def __init__(self, model_info_xx):
        self.model_dict = model_info_xx
        self.sliding_window = self.model_dict["sliding_window_params"]
        self.Lambda = self.model_dict["Lambda"]
        self.d_accuracy = 0
        self.n_D = model_info_xx["discriminator_net"]["params"]["n_D"]
        self.device = torch.device("cuda:0")

    def _get_generator_loss(self, discriminator_model, image_A, image_B, fake_image):
        assert image_A.size() == image_B.size()
        self.dtype = image_A.dtype
        MSE_criterion = parse_config(self.model_dict["MSE_criterion"])
        FMcriterion = parse_config(self.model_dict["L1_criterion"])
        self.n_D = self.model_dict["discriminator_net"]["params"]["n_D"]
        loss_G = 0
        loss_G_FM = 0
        real_features = discriminator_model(torch.cat((image_A, image_B), dim=1))
        fake_features = discriminator_model(torch.cat((image_A, fake_image), dim=1))
        for i in range(self.n_D):
            real_grid = _get_grid(fake_features[i][-1], is_real=True).to(
                self.device, self.dtype
            )
        loss_G += MSE_criterion(fake_features[i][-1], real_grid)
        for j in range(len(fake_features[0])):
            loss_G_FM += FMcriterion(fake_features[i][j], real_features[i][j].detach())
        loss_G += loss_G_FM * (1.0 / self.n_D) * self.Lambda
        return loss_G

    def _get_discriminator_loss(
        self, discriminator_model, image_A, image_B, fake_image
    ):
        assert image_A.size() == image_B.size()
        self.dtype = image_A.dtype
        MSE_criterion = parse_config(self.model_dict["MSE_criterion"])
        loss_D = 0
        real_features = discriminator_model(torch.cat((image_A, image_B), dim=1))
        fake_features = discriminator_model(
            torch.cat((image_A, fake_image.detach()), dim=1)
        )
        for i in range(self.n_D):
            real_grid = _get_grid(real_features[i][-1], is_real=True).to(
                self.device, self.dtype
            )
            fake_grid = _get_grid(fake_features[i][-1], is_real=False).to(
                self.device, self.dtype
            )
            loss_D += (
                MSE_criterion(real_features[i][-1], real_grid)
                + MSE_criterion(fake_features[i][-1], fake_grid)
            ) * 0.5
        d_real_acu = torch.ge(image_B.squeeze(), 0.5).float()
        d_fake_acu = torch.le(fake_image.squeeze(), 0.5).float()
        self.d_accuracy = torch.mean(torch.cat((d_real_acu, d_fake_acu), 0))
        return loss_D

    def _get_discriminator_accuracy(self):
        return self.d_accuracy
