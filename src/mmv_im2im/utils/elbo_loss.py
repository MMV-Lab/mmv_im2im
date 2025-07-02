import torch
import torch.nn as nn
import torch.nn.functional as F


class KLDivergence(nn.Module):
    """Calculates KL Divergence between two diagonal Gaussians."""

    def __init__(self):
        super().__init__()

    def forward(self, mu_q, logvar_q, mu_p, logvar_p):
        """
        Calculates the KL Divergence between two diagonal Gaussian distributions.

        Args:
            mu_q (torch.Tensor): Mean of the approximate posterior distribution.
            logvar_q (torch.Tensor): Log-variance of the approximate posterior distribution.
            mu_p (torch.Tensor): Mean of the prior distribution.
            logvar_p (torch.Tensor): Log-variance of the prior distribution.

        Returns:
            torch.Tensor: The mean KL divergence over the batch.
        """
        kl_batch_sum = 0.5 * torch.sum(
            logvar_p
            - logvar_q
            + (torch.exp(logvar_q) + (mu_q - mu_p) ** 2) / torch.exp(logvar_p)
            - 1,
            dim=[1, 2, 3],  # Sum over latent channels, H, W
        )
        return torch.mean(kl_batch_sum)  # Average over batch


class ELBOLoss(nn.Module):
    """
    Calculates the Evidence Lower Bound (ELBO) loss for Probabilistic UNet.

    Args:
        beta (float): Weighting factor for the KL divergence term.
        n_classes (int): Number of classes in the segmentation task.
    """

    def __init__(self, beta: float = 1.0, n_classes: int = 2):
        super().__init__()
        self.beta = beta
        self.n_classes = n_classes
        self.kl_divergence_calculator = KLDivergence()

    def forward(self, logits, y_true, prior_mu, prior_logvar, post_mu, post_logvar):
        """
        Computes the ELBO loss.

        Args:
            logits (torch.Tensor): Output logits from the Probabilistic UNet (B, C, H, W).
            y_true (torch.Tensor): Ground truth segmentation mask (B, 1, H, W or B, H, W).
            prior_mu (torch.Tensor): Mean of the prior distribution.
            prior_logvar (torch.Tensor): Log-variance of the prior distribution.
            post_mu (torch.Tensor): Mean of the approximate posterior distribution.
            post_logvar (torch.Tensor): Log-variance of the approximate posterior distribution.

        Returns:
            torch.Tensor: The calculated ELBO loss.
        """
        # Ensure y_true has correct dimensions (e.g., [B, H, W]) for cross_entropy
        if y_true.ndim == 4 and y_true.shape[1] == 1:
            y_true = y_true.squeeze(1)  # Squeeze channel dim to [B, H, W]

        # Negative Cross-Entropy (Log-Likelihood)
        # Using reduction='mean' to get a scalar loss per batch
        log_likelihood = -F.cross_entropy(logits, y_true.long(), reduction="mean")

        # KL-Divergence
        kl_div = self.kl_divergence_calculator(
            post_mu, post_logvar, prior_mu, prior_logvar
        )

        # ELBO = Log-Likelihood - beta * KL_Divergence
        # We minimize the negative ELBO to maximize the ELBO
        elbo_loss = -(log_likelihood - self.beta * kl_div)

        return elbo_loss
