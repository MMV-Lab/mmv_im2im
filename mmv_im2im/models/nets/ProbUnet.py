import torch
import torch.nn as nn
import torch.nn.functional as F


def get_valid_num_groups(channels):
    """Returns a valid number of groups for GroupNorm."""
    for g in [8, 4, 2, 1]:
        if channels % g == 0:
            return g
    return 1


class ConvBlock(nn.Module):
    """Standard 2D/3D Convolutional Block."""

    def __init__(self, in_channels, out_channels, Conv, GroupNorm):
        super().__init__()
        gn_groups1 = get_valid_num_groups(out_channels)
        gn_groups2 = get_valid_num_groups(out_channels)
        self.conv1 = Conv(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn1 = GroupNorm(gn_groups1, out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = Conv(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = GroupNorm(gn_groups2, out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.gn1(self.conv1(x)))
        x = self.relu2(self.gn2(self.conv2(x)))
        return x


class Down(nn.Module):
    """Downsampling block (MaxPool + ConvBlock)."""

    def __init__(self, in_channels, out_channels, MaxPool, ConvBlock, Conv, GroupNorm):
        super().__init__()
        self.pool = MaxPool(2)
        self.conv_block = ConvBlock(in_channels, out_channels, Conv, GroupNorm)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv_block(x)
        return x


class Up(nn.Module):
    """Upsampling block (ConvTranspose + Concat + ConvBlock).

    Args:
        in_channels_x1_before_upsample (int): Number of channels of the feature map (x1)
                                                before being upsampled by ConvTranspose.
        in_channels_x2_skip_connection (int): Number of channels of the skip connection (x2).
        out_channels (int): Number of output channels for the final ConvBlock in this Up stage.
    """

    def __init__(
        self,
        in_channels_x1_before_upsample,
        in_channels_x2_skip_connection,
        out_channels,
        ConvTranspose,
        ConvBlock,
        Conv,
        GroupNorm,
        interpolation_mode,
    ):
        super().__init__()
        self.interpolation_mode = interpolation_mode
        self.up = ConvTranspose(
            in_channels_x1_before_upsample,
            in_channels_x1_before_upsample // 2,
            kernel_size=2,
            stride=2,
        )

        channels_for_conv_block = (
            in_channels_x1_before_upsample // 2
        ) + in_channels_x2_skip_connection
        self.conv_block = ConvBlock(
            channels_for_conv_block, out_channels, Conv, GroupNorm
        )

    def forward(self, x1, x2):
        x1 = self.up(x1)

        # Get the spatial size of the skip connection tensor
        spatial_size_x2 = x2.size()[2:]

        if x1.size()[2:] != spatial_size_x2:
            x1 = F.interpolate(
                x1,
                size=spatial_size_x2,
                mode=self.interpolation_mode,
                align_corners=False,
            )

        x = torch.cat([x2, x1], dim=1)
        return self.conv_block(x)


class PriorNet(nn.Module):
    """Network to predict prior distribution (mu, logvar)."""

    def __init__(self, in_channels, latent_dim, Conv):
        super().__init__()
        self.conv = Conv(in_channels, 2 * latent_dim, kernel_size=1)

    def forward(self, x):
        mu_logvar = self.conv(x)
        mu = mu_logvar[:, : self.conv.out_channels // 2, ...]
        logvar = mu_logvar[:, self.conv.out_channels // 2 :, ...]
        return mu, logvar


class PosteriorNet(nn.Module):
    """Network to predict posterior distribution (mu, logvar)."""

    def __init__(self, in_channels, latent_dim, Conv):
        super().__init__()
        self.conv = Conv(in_channels, 2 * latent_dim, kernel_size=1)

    def forward(self, x):
        mu_logvar = self.conv(x)
        mu = mu_logvar[:, : self.conv.out_channels // 2, ...]
        logvar = mu_logvar[:, self.conv.out_channels // 2 :, ...]
        return mu, logvar


class ProbabilisticUNet(nn.Module):
    """Probabilistic UNet model.

    This model can operate in 2D or 3D based on the 'model_type' parameter.
    """

    def __init__(
        self,
        in_channels,
        n_classes,
        latent_dim=6,
        task="segment",
        model_type="2D",
        **kwargs,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        self.task = task
        self.model_type = model_type

        # Select the appropriate layers based on model_type
        if model_type == "2D":
            self.Conv = nn.Conv2d
            self.MaxPool = nn.MaxPool2d
            self.ConvTranspose = nn.ConvTranspose2d
            self.GroupNorm = nn.GroupNorm
            self.interpolation_mode = "bilinear"
        elif model_type == "3D":
            self.Conv = nn.Conv3d
            self.MaxPool = nn.MaxPool3d
            self.ConvTranspose = nn.ConvTranspose3d
            self.GroupNorm = nn.GroupNorm
            self.interpolation_mode = "trilinear"
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        # Encoder path (U-Net)
        self.inc = ConvBlock(in_channels, 32, self.Conv, self.GroupNorm)
        self.down1 = Down(32, 64, self.MaxPool, ConvBlock, self.Conv, self.GroupNorm)
        self.down2 = Down(64, 128, self.MaxPool, ConvBlock, self.Conv, self.GroupNorm)
        self.down3 = Down(128, 256, self.MaxPool, ConvBlock, self.Conv, self.GroupNorm)
        self.down4 = Down(256, 512, self.MaxPool, ConvBlock, self.Conv, self.GroupNorm)

        # Prior and Posterior Networks
        self.prior_net = PriorNet(512, latent_dim, self.Conv)
        # PosteriorNet input channels: 512 (features) + n_classes (one-hot y or float y)
        self.posterior_net = PosteriorNet(512 + n_classes, latent_dim, self.Conv)

        # Decoder Path (U-Net upsampling path)
        self.up1 = Up(
            in_channels_x1_before_upsample=512 + latent_dim,
            in_channels_x2_skip_connection=256,
            out_channels=256,
            ConvTranspose=self.ConvTranspose,
            ConvBlock=ConvBlock,
            Conv=self.Conv,
            GroupNorm=self.GroupNorm,
            interpolation_mode=self.interpolation_mode,
        )

        self.up2 = Up(
            in_channels_x1_before_upsample=256,
            in_channels_x2_skip_connection=128,
            out_channels=128,
            ConvTranspose=self.ConvTranspose,
            ConvBlock=ConvBlock,
            Conv=self.Conv,
            GroupNorm=self.GroupNorm,
            interpolation_mode=self.interpolation_mode,
        )

        self.up3 = Up(
            in_channels_x1_before_upsample=128,
            in_channels_x2_skip_connection=64,
            out_channels=64,
            ConvTranspose=self.ConvTranspose,
            ConvBlock=ConvBlock,
            Conv=self.Conv,
            GroupNorm=self.GroupNorm,
            interpolation_mode=self.interpolation_mode,
        )

        self.up4 = Up(
            in_channels_x1_before_upsample=64,
            in_channels_x2_skip_connection=32,
            out_channels=32,
            ConvTranspose=self.ConvTranspose,
            ConvBlock=ConvBlock,
            Conv=self.Conv,
            GroupNorm=self.GroupNorm,
            interpolation_mode=self.interpolation_mode,
        )

        self.outc = self.Conv(32, n_classes, kernel_size=1)

    def forward(self, x, y=None):
        """
        Forward pass of the Probabilistic UNet.

        Args:
            x (torch.Tensor): Input image tensor (B, C, H, W) for 2D or (B, C, D, H, W) for 3D.
            y (torch.Tensor, optional): Ground truth segmentation mask used for training to
                                         calculate posterior. Defaults to None (for inference).

        Returns:
            tuple: A tuple containing:
                - logits (torch.Tensor): Output logits of the UNet.
                - prior_mu (torch.Tensor): Mean of the prior distribution.
                - prior_logvar (torch.Tensor): Log-variance of the prior distribution.
                - post_mu (torch.Tensor or None): Mean of the posterior distribution (None if y is None).
                - post_logvar (torch.Tensor or None): Log-variance of the posterior distribution (None if y is None).
        """
        # Encoder (U-Net)
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        features = self.down4(x4)  # Bottleneck

        # Prior distribution
        prior_mu, prior_logvar = self.prior_net(features)

        # Posterior calculation and latent variable sampling
        post_mu, post_logvar = None, None
        if y is not None:
            if self.task == "segment":
                # Ensure y is one-hot encoded and downsampled
                y_one_hot = F.one_hot(y.long().squeeze(1), num_classes=self.n_classes)
                y_one_hot = y_one_hot.permute(
                    0, -1, *range(1, y_one_hot.dim() - 1)
                ).float()

                y_downsampled = F.interpolate(
                    y_one_hot, size=features.shape[2:], mode="nearest"
                )
            elif self.task == "regression":
                # For regression, y is already a float tensor, just downsample
                y_downsampled = F.interpolate(
                    y,
                    size=features.shape[2:],
                    mode=self.interpolation_mode,
                    align_corners=False,
                )
            else:
                raise ValueError(f"Unknown task type: {self.task}")

            # Concatenate features and downsampled y for posterior network
            post_mu, post_logvar = self.posterior_net(
                torch.cat([features, y_downsampled], dim=1)
            )

            # Sample 'z' from the posterior distribution
            std_post = torch.exp(0.5 * post_logvar)
            eps = torch.randn_like(std_post)
            z = post_mu + eps * std_post
        else:
            # If 'y' is not provided (inference), sample 'z' from the prior.
            std_prior = torch.exp(0.5 * prior_logvar)
            eps = torch.randn_like(std_prior)
            z = prior_mu + eps * std_prior

        # Expand 'z' to spatial dimensions for concatenation
        spatial_dims = features.size()[2:]
        if z.dim() == 2:  # [B, latent_dim]
            # Expands latent vector to spatial dimensions
            z_expanded = z.unsqueeze(-1)
            for _ in range(len(spatial_dims) - 1):
                z_expanded = z_expanded.unsqueeze(-1)
            z_expanded = z_expanded.repeat(1, 1, *spatial_dims)
        elif z.dim() == 2 + len(spatial_dims):
            if z.size()[2:] != spatial_dims:
                z_expanded = F.interpolate(z, size=spatial_dims, mode="nearest")
            else:
                z_expanded = z
        else:
            raise ValueError(f"Unexpected latent vector z dimension: {z.dim()}")

        # Concatenate bottleneck features with latent vector
        concat_bottleneck = torch.cat([features, z_expanded], dim=1)

        # Decoder (U-Net upsampling path)
        x_up = self.up1(concat_bottleneck, x4)
        x_up = self.up2(x_up, x3)
        x_up = self.up3(x_up, x2)
        x_up = self.up4(x_up, x1)
        output = self.outc(x_up)

        # Return all necessary components for ELBO calculation
        return output, prior_mu, prior_logvar, post_mu, post_logvar
