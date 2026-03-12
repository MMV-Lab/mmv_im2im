import torch
import torch.nn as nn
import torch.nn.functional as F
from monai.networks.blocks.convolutions import Convolution
from monai.networks.layers.factories import Norm


class ConvBlock(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        strides=1,
        dropout=0.0,
    ):
        super().__init__()
        # padding=None in MONAI Convolution defaults to "same" padding
        layers = [
            Convolution(
                spatial_dims,
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                strides=strides,
                padding=None,
                adn_ordering="NDA",
                act="relu",
                norm=Norm.BATCH,
                dropout=dropout,
            ),
            Convolution(
                spatial_dims,
                out_channels,
                out_channels,
                kernel_size=kernel_size,
                strides=1,
                padding=None,
                adn_ordering="NDA",
                act="relu",
                norm=Norm.BATCH,
                dropout=dropout,
            ),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpConv(nn.Module):
    def __init__(
        self,
        spatial_dims: int,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        strides=2,
        dropout=0.0,
    ):
        super().__init__()
        self.up = Convolution(
            spatial_dims,
            in_channels,
            out_channels,
            strides=strides,
            kernel_size=kernel_size,
            act="relu",
            adn_ordering="NDA",
            norm=Norm.BATCH,
            dropout=dropout,
            is_transposed=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.up(x)


class Encoder(nn.Module):
    def __init__(
        self,
        spatial_dims,
        in_channels,
        channels,
        strides,
        kernel_size=3,
        dropout=0.0,
        latent_dim=6,
    ):
        super().__init__()
        self.blocks = nn.ModuleList()
        curr_c = in_channels
        for c, s in zip(channels, strides):
            self.blocks.append(
                ConvBlock(spatial_dims, curr_c, c, kernel_size, s, dropout)
            )
            curr_c = c
        self.gap = (
            nn.AdaptiveAvgPool2d(1) if spatial_dims == 2 else nn.AdaptiveAvgPool3d(1)
        )
        self.mu_layer = Convolution(
            spatial_dims, curr_c, latent_dim, 1, 1, 0, conv_only=True
        )
        self.logvar_layer = Convolution(
            spatial_dims, curr_c, latent_dim, 1, 1, 0, conv_only=True
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        x = self.gap(x)
        return self.mu_layer(x).flatten(1), self.logvar_layer(x).flatten(1)


class ProbabilisticUNet(nn.Module):
    def __init__(
        self,
        spatial_dims,
        in_channels,
        out_channels,
        channels,
        strides,
        kernel_size=3,
        up_kernel_size=3,
        latent_dim=6,
        dropout=0.0,
        task="segmentation",
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.out_channels = out_channels
        self.latent_dim = latent_dim
        self.task = task

        # Backbone Encoder
        self.unet_encoder = nn.ModuleList()
        curr_c = in_channels
        for c, s in zip(channels, strides):
            self.unet_encoder.append(
                ConvBlock(spatial_dims, curr_c, c, kernel_size, s, dropout)
            )
            curr_c = c

        # Backbone Decoder
        self.unet_decoder = nn.ModuleList()
        rev_c = channels[::-1]
        rev_s = strides[::-1]
        for i in range(len(channels) - 1):
            self.unet_decoder.append(
                UpConv(
                    spatial_dims,
                    rev_c[i],
                    rev_c[i + 1],
                    up_kernel_size,
                    rev_s[i],
                    dropout,
                )
            )
            self.unet_decoder.append(
                ConvBlock(spatial_dims, rev_c[i], rev_c[i + 1], kernel_size, 1, dropout)
            )

        self.prior_net = Encoder(
            spatial_dims,
            in_channels,
            channels,
            strides,
            kernel_size,
            dropout,
            latent_dim,
        )

        # on regression out_channels spaciol concatyenation
        self.posterior_net = Encoder(
            spatial_dims,
            in_channels + out_channels,
            channels,
            strides,
            kernel_size,
            dropout,
            latent_dim,
        )

        self.f_comb = nn.Sequential(
            Convolution(
                spatial_dims,
                channels[0] + latent_dim,
                channels[0],
                1,
                1,
                0,
                conv_only=True,
                act="relu",
            ),
            Convolution(
                spatial_dims,
                channels[0],
                channels[0],
                1,
                1,
                0,
                conv_only=True,
                act="relu",
            ),
            Convolution(
                spatial_dims, channels[0], out_channels, 1, 1, 0, conv_only=True
            ),
        )

    def reparameterize(self, mu, logvar):
        return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)

    def forward(self, x, seg=None, train_posterior=True):
        skips = []
        unet_x = x
        for block in self.unet_encoder:
            unet_x = block(unet_x)
            skips.append(unet_x)

        unet_x = skips.pop()  # Bottleneck

        for i in range(0, len(self.unet_decoder), 2):
            up_x = self.unet_decoder[i](unet_x)
            skip_x = skips.pop()
            unet_x = torch.cat([up_x, skip_x], dim=1)
            unet_x = self.unet_decoder[i + 1](unet_x)

        mu_prior, logvar_prior = self.prior_net(x)
        mu_post, logvar_post = None, None

        if train_posterior and seg is not None:
            if self.task == "regression":
                # seg -> [B, out_channels]
                # expand
                dims_to_add = len(x.shape) - 2
                seg_spatial = seg.view(seg.shape[0], seg.shape[1], *([1] * dims_to_add))
                seg_input = seg_spatial.expand(-1, -1, *x.shape[2:]).float()
            else:
                # regular segmentation
                if seg.shape[1] != self.out_channels:
                    seg_temp = seg
                    if seg_temp.shape[1] == 1:
                        seg_temp = seg_temp.squeeze(1)

                    seg_one_hot = F.one_hot(
                        seg_temp.long(), num_classes=self.out_channels
                    )

                    dims = list(range(seg_one_hot.ndim))
                    seg_input = (
                        seg_one_hot.permute(0, dims[-1], *dims[1:-1])
                        .contiguous()
                        .float()
                    )
                else:
                    seg_input = seg.float()

            cat_input = torch.cat([x, seg_input], dim=1)
            mu_post, logvar_post = self.posterior_net(cat_input)
            z_sample = self.reparameterize(mu_post, logvar_post)
        else:
            z_sample = self.reparameterize(mu_prior, logvar_prior)

        # Broadcast z and combine
        z_b = z_sample.view(
            z_sample.shape[0], self.latent_dim, *([1] * self.spatial_dims)
        ).expand(-1, -1, *unet_x.shape[2:])
        reconstruction = self.f_comb(torch.cat([unet_x, z_b], dim=1))

        #  (GAP)
        if self.task == "regression":
            reconstruction = reconstruction.view(
                reconstruction.size(0), reconstruction.size(1), -1
            ).mean(dim=-1)

        return {
            "pred": reconstruction,
            "mu_post": mu_post,
            "logvar_post": logvar_post,
            "prior_mu": mu_prior,
            "prior_logvar": logvar_prior,
        }
