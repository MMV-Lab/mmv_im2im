# Save this as ProbUnet.py (or mmv_im2im/models/ProbUnet.py if that's its actual path)
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
    """Standard 2D Convolutional Block."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        gn_groups1 = get_valid_num_groups(out_channels)
        gn_groups2 = get_valid_num_groups(out_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(gn_groups1, out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(gn_groups2, out_channels)
        self.relu2 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu1(self.gn1(self.conv1(x)))
        x = self.relu2(self.gn2(self.conv2(x)))
        return x


class Down(nn.Module):
    """Downsampling block (MaxPool + ConvBlock)."""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool2d(2)
        self.conv_block = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv_block(x)
        return x


class Up(nn.Module):
    """Upsampling block (ConvTranspose + Concat + ConvBlock).

    Args:
        in_channels_x1_before_upsample (int): Number of channels of the feature map (x1)
                                                before being upsampled by ConvTranspose2d.
        in_channels_x2_skip_connection (int): Number of channels of the skip connection (x2).
        out_channels (int): Number of output channels for the final ConvBlock in this Up stage.
    """

    def __init__(
        self,
        in_channels_x1_before_upsample,
        in_channels_x2_skip_connection,
        out_channels,
    ):
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels_x1_before_upsample,
            in_channels_x1_before_upsample // 2,
            kernel_size=2,
            stride=2,
        )

        channels_for_conv_block = (
            in_channels_x1_before_upsample // 2
        ) + in_channels_x2_skip_connection
        self.conv_block = ConvBlock(channels_for_conv_block, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Adjust dimensions if there's a mismatch due to padding or odd sizes
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv_block(x)


class PriorNet(nn.Module):
    """Network to predict prior distribution (mu, logvar)."""

    def __init__(self, in_channels, latent_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 2 * latent_dim, kernel_size=1)

    def forward(self, x):
        mu_logvar = self.conv(x)
        mu = mu_logvar[:, : self.conv.out_channels // 2, :, :]
        logvar = mu_logvar[:, self.conv.out_channels // 2 :, :, :]
        return mu, logvar


class PosteriorNet(nn.Module):
    """Network to predict posterior distribution (mu, logvar)."""

    def __init__(self, in_channels, latent_dim):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 2 * latent_dim, kernel_size=1)

    def forward(self, x):
        mu_logvar = self.conv(x)
        mu = mu_logvar[:, : self.conv.out_channels // 2, :, :]
        logvar = mu_logvar[:, self.conv.out_channels // 2 :, :, :]
        return mu, logvar


class ProbabilisticUNet(nn.Module):
    """Probabilistic UNet model."""

    def __init__(
        self, in_channels, n_classes, latent_dim=6, **kwargs
    ):  # Added **kwargs to capture extra params
        super().__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.latent_dim = latent_dim
        # self.beta is no longer needed here as it's handled by the loss function

        # Encoder path (U-Net)
        self.inc = ConvBlock(in_channels, 32)
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)
        self.down4 = Down(256, 512)  # Bottleneck features

        # Prior and Posterior Networks
        self.prior_net = PriorNet(512, latent_dim)
        # PosteriorNet input channels: 512 (features) + n_classes (one-hot y)
        self.posterior_net = PosteriorNet(512 + n_classes, latent_dim)

        # Decoder Path (U-Net upsampling path)
        # Input channels for Up blocks adjusted to include latent_dim
        self.up1 = Up(
            in_channels_x1_before_upsample=512 + latent_dim,
            in_channels_x2_skip_connection=256,
            out_channels=256,
        )

        self.up2 = Up(
            in_channels_x1_before_upsample=256,
            in_channels_x2_skip_connection=128,
            out_channels=128,
        )

        self.up3 = Up(
            in_channels_x1_before_upsample=128,
            in_channels_x2_skip_connection=64,
            out_channels=64,
        )

        self.up4 = Up(
            in_channels_x1_before_upsample=64,
            in_channels_x2_skip_connection=32,
            out_channels=32,
        )

        self.outc = nn.Conv2d(32, n_classes, kernel_size=1)

    def forward(self, x, y=None):
        """
        Forward pass of the Probabilistic UNet.

        Args:
            x (torch.Tensor): Input image tensor (B, C, H, W).
            y (torch.Tensor, optional): Ground truth segmentation mask (B, 1, H, W or B, H, W)
                                         used for training to calculate posterior.
                                         Defaults to None (for inference).

        Returns:
            tuple: A tuple containing:
                - logits (torch.Tensor): Output logits of the UNet (B, n_classes, H, W).
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
            # Ensure y is one-hot encoded and downsampled to match features spatial dimensions.
            # y typically comes as [B, 1, H, W] with integer class labels.
            # Convert to [B, n_classes, H, W] for one-hot, then permute for channel dim.
            y_one_hot = (
                F.one_hot(y.long().squeeze(1), num_classes=self.n_classes)
                .permute(0, 3, 1, 2)
                .float()
            )

            # Downsample y_one_hot to match features' spatial dimensions
            y_downsampled = F.interpolate(
                y_one_hot, size=features.shape[2:], mode="nearest"
            )

            # Concatenate features and downsampled one-hot y for posterior network
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
        if z.dim() == 2:  # [B, latent_dim]
            z_expanded = (
                z.unsqueeze(-1)
                .unsqueeze(-1)
                .repeat(1, 1, features.size(2), features.size(3))
            )
        elif z.dim() == 4:  # [B, latent_dim, H, W]
            if z.size(2) != features.size(2) or z.size(3) != features.size(3):
                z_expanded = F.interpolate(
                    z, size=(features.size(2), features.size(3)), mode="nearest"
                )
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
