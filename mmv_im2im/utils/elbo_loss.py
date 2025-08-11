import torch
import torch.nn as nn
import torch.nn.functional as F
from mmv_im2im.utils.fractal_layers import Slice_windows, FractalDimension
from mmv_im2im.utils.topological_loss import TI_Loss
from mmv_im2im.utils.connectivity_loss import ConnectivityCoherenceLoss
from monai.losses import GeneralizedDiceFocalLoss


class KLDivergence(nn.Module):
    """Calculates KL Divergence between two diagonal Gaussians."""

    def __init__(self):
        super().__init__()

    def forward(self, mu_q, logvar_q, mu_p, logvar_p, kl_clamp=None):
        """
        Calculates the KL Divergence between two diagonal Gaussian distributions.

        Args:
            mu_q (torch.Tensor): Mean of the approximate posterior distribution.
            logvar_q (torch.Tensor): Log-variance of the approximate posterior
                                     distribution.
            mu_p (torch.Tensor): Mean of the prior distribution.
            logvar_p (torch.Tensor): Log-variance of the prior distribution.
            clamp (float): Value to clamp logvar_q, logvar_p in case of gradient explotion.

        Returns:
            torch.Tensor: The mean KL divergence over the batch.
        """
        # Clamp log-variances to prevent numerical instability
        # This limits exp(logvar) to a stable range, e.g., [2.06e-9, 4.85e8]
        if kl_clamp is not None:
            logvar_q = torch.clamp(logvar_q, min=-kl_clamp, max=kl_clamp)
            logvar_p = torch.clamp(logvar_p, min=-kl_clamp, max=kl_clamp)

        kl_batch_sum = 0.5 * torch.sum(
            logvar_p
            - logvar_q
            + (torch.exp(logvar_q) + (mu_q - mu_p) ** 2) / torch.exp(logvar_p)
            - 1,
            dim=[1, 2, 3],  # Sum over latent channels, H, W
        )
        return torch.mean(kl_batch_sum)


class ELBOLoss(nn.Module):
    """
    Calculates the Evidence Lower Bound (ELBO) loss for Probabilistic UNet,
    with optional fractal dimension, topological, and connectivity regularization.

    Args:
        beta (float): Weighting factor for the KL divergence term.
        n_classes (int): Number of classes in the segmentation task.
        kl_clamp (float): Value to clamp logvar_q, logvar_p in case of gradient explotion for kl.
        use_fractal_regularization (bool): If True, includes the fractal dimension regularization term.
        fractal_weight (float): Weighting factor for the fractal dimension loss term (only if use_fractal_regularization is True).
        fractal_num_kernels (int): Number of kernels for FractalDimension (only if use_fractal_regularization is True).
        fractal_mode (str): Mode for FractalDimension ("classic" or "entropy") (only if use_fractal_regularization is True).
        fractal_to_binary (bool): Whether to binarize input for FractalDimension (only if use_fractal_regularization is True).
        use_topological_regularization (bool): If True, includes the topological regularization term.
        topological_weight (float): Weighting factor for the topological loss term (only if use_topological_regularization is True).
        topological_dim (int): Dimension for TI_Loss (2 for 2D, 3 for 3D) (only if use_topological_regularization is True).
        topological_connectivity (int): Connectivity for TI_Loss (4 or 8 for 2D; 6 or 26 for 3D) (only if use_topological_regularization is True).
        topological_inclusion (list): List of [A,B] class pairs for inclusion in TI_Loss (only if use_topological_regularization is True).
        topological_exclusion (list): List of [A,C] class pairs for exclusion in TI_Loss (only if use_topological_regularization is True).
        topological_min_thick (int): Minimum thickness for TI_Loss (only if use_topological_regularization is True and connectivity is 8 or 26).
        use_connectivity_regularization (bool): If True, includes the new connectivity coherence regularization term.
        connectivity_weight (float): Weighting factor for the connectivity coherence loss term.
        connectivity_kernel_size (int): Kernel size for connectivity coherence loss (e.g., 3).
        connectivity_ignore_background (bool): If True, ignore background for connectivity loss.
        elbo_class_weights (list or torch.Tensor, optional): Weights for each class in the cross-entropy loss.
        use_gdl_focal_regularization (bool): If True, Includes Generalized Dice Focal (GDF) regularization.
        gdl_focal_weight (float): Weighting factor for GDF.
        gdl_class_weights (list):  Weights for each class.
    """

    def __init__(
        self,
        beta: float = 1.0,
        n_classes: int = 2,
        kl_clamp: float = None,
        use_fractal_regularization: bool = False,
        fractal_weight: float = 0.1,
        fractal_num_kernels: int = 5,
        fractal_mode: str = "classic",
        fractal_to_binary: bool = True,
        use_topological_regularization: bool = False,
        topological_weight: float = 0.1,
        topological_dim: int = 2,
        topological_connectivity: int = 4,
        topological_inclusion: list = None,
        topological_exclusion: list = None,
        topological_min_thick: int = 1,
        use_connectivity_regularization: bool = False,
        connectivity_weight: float = 0.1,
        connectivity_kernel_size: int = 3,
        connectivity_ignore_background: bool = True,
        use_gdl_focal_regularization: bool = False,
        gdl_focal_weight: float = 1.0,
        elbo_class_weights: list = None,
        gdl_class_weights: list = None,
    ):
        super().__init__()
        self.beta = beta
        self.n_classes = n_classes
        self.kl_clamp = kl_clamp
        self.kl_divergence_calculator = KLDivergence()

        self.use_fractal_regularization = use_fractal_regularization
        if self.use_fractal_regularization:
            self.fractal_weight = fractal_weight
            self.fractal_dimension_calculator = FractalDimension(
                num_kernels=fractal_num_kernels,
                mode=fractal_mode,
                to_binary=fractal_to_binary,
            )
        else:
            self.fractal_weight = 0.0

        self.use_topological_regularization = use_topological_regularization
        if self.use_topological_regularization:
            self.topological_weight = topological_weight
            if topological_inclusion is None:
                topological_inclusion = []
            if topological_exclusion is None:
                topological_exclusion = []
            self.topological_loss_calculator = TI_Loss(
                dim=topological_dim,
                connectivity=topological_connectivity,
                inclusion=topological_inclusion,
                exclusion=topological_exclusion,
                min_thick=topological_min_thick,
            )
        else:
            self.topological_weight = 0.0

        # New Connectivity Regularization
        self.use_connectivity_regularization = use_connectivity_regularization
        if self.use_connectivity_regularization:
            self.connectivity_weight = connectivity_weight
            self.connectivity_coherence_calculator = ConnectivityCoherenceLoss(
                kernel_size=connectivity_kernel_size,
                ignore_background=connectivity_ignore_background,
                num_classes=n_classes,
            )
        else:
            self.connectivity_weight = 0.0

        self.use_gdl_focal_regularization = use_gdl_focal_regularization
        if self.use_gdl_focal_regularization:
            self.gdl_focal_weight = gdl_focal_weight
            monai_focal_weights = None
            if gdl_class_weights is not None:
                monai_focal_weights = torch.tensor(
                    gdl_class_weights, dtype=torch.float32
                )
            self.gdl_focal_loss_calculator = GeneralizedDiceFocalLoss(
                softmax=True, to_onehot_y=True, weight=monai_focal_weights
            )
        else:
            self.gdl_focal_weight = 0.0

        # Convert class_weights list to a torch.Tensor
        if elbo_class_weights is not None:
            self.elbo_class_weights = torch.tensor(
                elbo_class_weights, dtype=torch.float32
            )
        else:
            self.elbo_class_weights = None

    def forward(self, logits, y_true, prior_mu, prior_logvar, post_mu, post_logvar):
        """
        Computes the ELBO loss, with optional fractal dimension and topological regularization terms.

        Args:
            logits (torch.Tensor): Output logits from the Probabilistic UNet
                                   (B, C, H, W).
            y_true (torch.Tensor): Ground truth segmentation mask (B, 1, H, W
                                   or B, H, W).
            prior_mu (torch.Tensor): Mean of the prior distribution.
            prior_logvar (torch.Tensor): Log-variance of the prior distribution.
            post_mu (torch.Tensor): Mean of the approximate posterior distribution.
            post_logvar (torch.Tensor): Log-variance of the approximate posterior
                                        distribution.

        Returns:
            torch.Tensor: The calculated ELBO loss.
        """
        # Ensure y_true has correct dimensions (e.g., [B, H, W]) for cross_entropy
        if y_true.ndim == 4 and y_true.shape[1] == 1:
            y_true_squeezed = y_true.squeeze(1)  # Squeeze channel dim to [B, H, W]
        else:
            y_true_squeezed = y_true

        # Negative Cross-Entropy (Log-Likelihood)
        if (
            self.elbo_class_weights is not None
            and self.elbo_class_weights.device != logits.device
        ):
            elbo_class_weights_on_device = self.elbo_class_weights.to(logits.device)
        else:
            elbo_class_weights_on_device = self.elbo_class_weights

        log_likelihood = -F.cross_entropy(
            logits,
            y_true_squeezed.long(),
            reduction="mean",
            weight=elbo_class_weights_on_device,
        )

        # KL-Divergence
        kl_div = self.kl_divergence_calculator(
            post_mu, post_logvar, prior_mu, prior_logvar, self.kl_clamp
        )

        elbo_loss = -(log_likelihood - self.beta * kl_div)

        total_loss = elbo_loss

        if self.use_fractal_regularization:
            y_pred_mask = F.softmax(logits, dim=1).argmax(dim=1, keepdim=True).float()

            if y_true_squeezed.ndim == 3:
                y_true_for_fractal = y_true_squeezed.unsqueeze(1).float()
            else:
                y_true_for_fractal = y_true.float()

            fd_true = self.fractal_dimension_calculator(y_true_for_fractal)
            fd_pred = self.fractal_dimension_calculator(y_pred_mask)

            fractal_loss = torch.mean(torch.abs(fd_true - fd_pred))
            total_loss += self.fractal_weight * fractal_loss

        if self.use_topological_regularization:
            # y_true needs to be B, C, H, W or B, C, H, W, D for TI_Loss, where C=1
            # If y_true is B, H, W, unsqueeze to B, 1, H, W
            if y_true_squeezed.ndim == 3:
                y_true_for_topological = y_true_squeezed.unsqueeze(1).float()
            else:
                y_true_for_topological = (
                    y_true.float()
                )  # This should already be B, 1, H, W

            # logits are B, C, H, W (or B, C, H, W, D), which is what TI_Loss expects for x
            topological_loss = self.topological_loss_calculator(
                logits, y_true_for_topological
            )
            total_loss += self.topological_weight * topological_loss

        if self.use_connectivity_regularization:
            # y_pred_softmax: (B, C, H, W)
            y_pred_softmax = F.softmax(logits, dim=1)

            # y_true_one_hot: Need to convert y_true_squeezed (B, H, W) to one-hot (B, C, H, W)
            # Ensure the number of classes matches n_classes used in ELBOLoss
            y_true_one_hot = (
                F.one_hot(y_true_squeezed.long(), num_classes=self.n_classes)
                .permute(0, 3, 1, 2)
                .float()
            )

            connectivity_loss = self.connectivity_coherence_calculator(
                y_pred_softmax, y_true_one_hot
            )
            total_loss += self.connectivity_weight * connectivity_loss

        if self.use_gdl_focal_regularization:
            # logits: (B, C, H, W)
            # y_true: (B, H, W) o (B, 1, H, W)
            # GeneralizedDiceFocalLoss de MONAI puede manejar esto directamente
            y_true_for_gdl_focal = y_true_squeezed.unsqueeze(1).long()
            gdl_focal_loss = self.gdl_focal_loss_calculator(
                logits, y_true_for_gdl_focal
            )
            total_loss += self.gdl_focal_weight * gdl_focal_loss

        return total_loss
