# -*- coding: utf-8 -*-
# Adapted from https://github.com/JeongHyunJin/Pix2PixHD/blob/main/Pix2PixHD_Networks.py
# References:
# 1) https://github.com/JeongHyunJin/Jeong2020_SolarFarsideMagnetograms
# 2) https://iopscience.iop.org/article/10.3847/2041-8213/abc255
# 3) https://iopscience.iop.org/article/10.3847/2041-8213/ab9085


from functools import partial
import torch
import torch.nn as nn

"""
Pix2PixHD is a popular deep learning methods that has proven to be effective
for translating high resolution images. It consists of two major networks.
They are generator network and discriminator network. The generator tries to
produce realistic output from input. The discriminator tries to distingush
the more realistic pair between the real pair and generated pair.
It consistes of multiscale discriminator architecture which have
identical structure but operate at different scales and different
image sizes. The real pair consists of real input and a real target.
The generated pair consists of real input and
generated output.
"""

# [1] True or False grid


def _get_grid(input, is_real=True):
    if is_real:
        grid = torch.FloatTensor(input.shape).fill_(1.0)

    elif not is_real:
        grid = torch.FloatTensor(input.shape).fill_(0.0)

    return grid


# [2] Set the Normalization method for the input layer


def _get_norm_layer(type):
    if type == "BatchNorm2d":
        layer = partial(nn.BatchNorm2d, affine=True)

    elif type == "InstanceNorm2d":
        layer = partial(nn.InstanceNorm2d, affine=False)

    return layer


# [3] Set the Padding method for the input layer


def _get_pad_layer(type):
    if type == "reflection":
        layer = nn.ReflectionPad2d

    elif type == "replication":
        layer = nn.ReplicationPad2d

    elif type == "zero":
        layer = nn.ZeroPad2d

    else:
        raise NotImplementedError(
            "Padding type {} is not valid."
            " Please choose among ['reflection', 'replication', 'zero']".format(type)
        )

    return layer


"""
The generator consists of encoder, decoder along with residual block.
The encoder block extracts the required features from the input image,
downsamples the feature maps to half and increase the number of
weights to almost twice. The decoder aims at restoring the reduced
dimension to the original image. To ensure that there are
enough number of learnable parameters, residual blocks are used
between encoder and decoder.
Description of paramaters:

input_ch: Number of input channels
output_ch: Number of output channels
n_downsample: Number of down samples in the encoder block
n_residual: Number of residual blocks
n_gf: number of layers in the first layer of generator
"""


class Generator(nn.Module):
    def __init__(
        self,
        input_ch,
        output_ch,
        n_downsample=4,
        n_residual=9,
        n_gf=64,
        norm_type="InstanceNorm2d",
        padding_type="reflection",
    ):
        super(Generator, self).__init__()
        act = nn.ReLU(inplace=True)
        input_ch = input_ch
        n_gf = n_gf
        norm = _get_norm_layer(norm_type)
        output_ch = output_ch
        pad = _get_pad_layer(padding_type)

        model = []
        model += [
            pad(3),
            nn.Conv2d(input_ch, n_gf, kernel_size=7, padding=0),
            norm(n_gf),
            act,
        ]

        for _ in range(n_downsample):
            model += [
                nn.Conv2d(n_gf, 2 * n_gf, kernel_size=3, padding=1, stride=2),
                norm(2 * n_gf),
                act,
            ]
            n_gf *= 2

        for _ in range(n_residual):
            model += [ResidualBlock(n_gf, pad, norm, act)]

        for _ in range(n_downsample):
            model += [
                nn.ConvTranspose2d(
                    n_gf,
                    n_gf // 2,
                    kernel_size=3,
                    padding=1,
                    stride=2,
                    output_padding=1,
                ),
                norm(n_gf // 2),
                act,
            ]
            n_gf //= 2

        model += [pad(3), nn.Conv2d(n_gf, output_ch, kernel_size=7, padding=0)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class ResidualBlock(nn.Module):
    def __init__(self, n_channels, pad, norm, act):
        super(ResidualBlock, self).__init__()
        block = [
            pad(1),
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=0, stride=1),
            norm(n_channels),
            act,
        ]
        block += [
            pad(1),
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=0, stride=1),
            norm(n_channels),
        ]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return x + self.block(x)


# ------------------------------------------------------------------------------
# [2] Discriminative Network


class PatchDiscriminator(nn.Module):
    def __init__(self, input_ch, output_ch, n_df):
        super(PatchDiscriminator, self).__init__()

        act = nn.LeakyReLU(0.2, inplace=True)
        n_df = n_df
        norm = nn.InstanceNorm2d
        input_channel = input_ch + output_ch

        blocks = []
        blocks += [
            [nn.Conv2d(input_channel, n_df, kernel_size=4, padding=1, stride=2), act]
        ]
        blocks += [
            [
                nn.Conv2d(n_df, 2 * n_df, kernel_size=4, padding=1, stride=2),
                norm(2 * n_df),
                act,
            ]
        ]
        blocks += [
            [
                nn.Conv2d(2 * n_df, 4 * n_df, kernel_size=4, padding=1, stride=2),
                norm(4 * n_df),
                act,
            ]
        ]
        blocks += [
            [
                nn.Conv2d(4 * n_df, 8 * n_df, kernel_size=4, padding=1, stride=1),
                norm(8 * n_df),
                act,
            ]
        ]
        blocks += [[nn.Conv2d(8 * n_df, 1, kernel_size=4, padding=1, stride=1)]]

        self.n_blocks = len(blocks)
        for i in range(self.n_blocks):
            setattr(self, "block_{}".format(i), nn.Sequential(*blocks[i]))

    def forward(self, x):
        result = [x]
        for i in range(self.n_blocks):
            block = getattr(self, "block_{}".format(i))
            result.append(block(result[-1]))

        return result[1:]  # except for the input


"""
Discriminator consists of several convolutional layers and pix2pixHD uses more than 
one discriminator. In discriminator features are passed through convolutional layers
derived as a probability between 0 and 1.
Description of parameters:

n_D: number of discriminators
n_df: number of channels in the first layer of discriminator
"""


class Discriminator(nn.Module):
    def __init__(self, input_ch, output_ch, n_D=2, n_df=64):
        super(Discriminator, self).__init__()

        for i in range(n_D):
            setattr(
                self,
                "Scale_{}".format(str(i)),
                PatchDiscriminator(input_ch, output_ch, n_df),
            )
        self.n_D = n_D

    def forward(self, x):
        result = []
        for i in range(self.n_D):
            result.append(getattr(self, "Scale_{}".format(i))(x))
            if i != self.n_D - 1:
                x = nn.AvgPool2d(
                    kernel_size=3, padding=1, stride=2, count_include_pad=False
                )(x)
        return result
