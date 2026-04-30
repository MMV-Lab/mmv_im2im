import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union


class KLDivergence(nn.Module):
    """
    Calculates the KL Divergence between two diagonal Gaussian distributions.
    Used for the Probabilistic U-Net latent space regularization.

    CHANGE: Numerically stable KL computation.
    The original formula computed exp(logvar_q) and exp(logvar_p) independently,
    which can overflow (→ inf) when logvar values are large and the clamp is
    disabled. The reformulated version uses exp(logvar_q - logvar_p) and
    exp(-logvar_p), keeping the exponent as a difference which is bounded even
    when individual logvars are large:

      Original: (exp(lq) + (mu_q - mu_p)^2) / exp(lp)
      Stable:    exp(lq - lp)  +  (mu_q - mu_p)^2 * exp(-lp)

    These are mathematically identical but the stable form avoids intermediate
    overflow. The improvement is most relevant in early training when the
    posterior and prior are far apart.
    """

    def __init__(self):
        super().__init__()

    def forward(self, mu_q, logvar_q, mu_p, logvar_p, kl_clamp=None):
        if kl_clamp is not None:
            logvar_q = torch.clamp(logvar_q, min=-kl_clamp, max=kl_clamp)
            logvar_p = torch.clamp(logvar_p, min=-kl_clamp, max=kl_clamp)

        # CHANGE: Stable formulation using log-domain subtraction.
        # exp(logvar_q) / exp(logvar_p) = exp(logvar_q - logvar_p)
        # (mu_q - mu_p)^2 / exp(logvar_p) = (mu_q - mu_p)^2 * exp(-logvar_p)
        kl_batch_sum = 0.5 * torch.sum(
            logvar_p
            - logvar_q
            + torch.exp(logvar_q - logvar_p)
            + (mu_q - mu_p) ** 2 * torch.exp(-logvar_p)
            - 1,
            dim=1,
        )
        return torch.mean(kl_batch_sum)


class ELBOLoss(nn.Module):
    def __init__(
        self,
        beta: float = 1.0,
        n_classes: int = 2,
        spatial_dims: int = 2,
        kl_clamp: float = None,
        elbo_class_weights: list = None,
        task: str = "segmentation",
        regression_loss_type: str = "mse",
        # --- Fractal Regularization ---
        use_fractal_regularization: bool = False,
        fractal_weight: float = 0.1,
        fractal_warmup_epochs: int = 10,
        fractal_num_kernels: int = 5,
        fractal_mode: str = "classic",
        fractal_to_binary: bool = True,
        # --- Topological Regularization (TI Loss) ---
        use_topological_regularization: bool = False,
        topological_weight: float = 0.1,
        topological_warmup_epochs: int = 10,
        topological_connectivity: int = 4,
        topological_inclusion: list = None,
        topological_exclusion: list = None,
        topological_min_thick: int = 1,
        # --- Connectivity Regularization ---
        use_connectivity_regularization: bool = False,
        connectivity_weight: float = 0.1,
        connectivity_warmup_epochs: int = 10,
        connectivity_kernel_size: int = 3,
        connectivity_mode: str = "single",
        kernel_shape: str = "square",
        connectivity_ignore_background: bool = True,
        lambda_density: float = 1.0,
        lambda_gradient: float = 0.2,
        connectivity_metric_density: str = "huber",
        connectivity_metric_gradient: str = "cosine",
        # --- GDL Focal Regularization (MONAI) ---
        use_gdl_focal_regularization: bool = False,
        gdl_focal_weight: float = 1.0,
        gdl_warmup_epochs: int = 10,
        gdl_class_weights: list = None,
        # --- Hausdorff Regularization ---
        use_hausdorff_regularization: bool = False,
        hausdorff_weight: float = 0.1,
        hausdorff_downsample_scale: float = 0.5,
        hausdorff_dt_iterations: Union[int, str] = "auto",
        hausdorff_warmup_epochs: int = 10,
        hausdorff_include_background: bool = False,
        hausdorff_distance_mode: str = "l2",
        hausdorff_normalize_weights: bool = True,
        # --- Homology Regularization (Persistence Image) ---
        use_homology_regularization: bool = False,
        homology_weight: float = 0.1,
        homology_warmup_epochs: int = 10,
        homology_interval: int = 1,
        homology_downsample_scale: float = 0.5,
        homology_class_context: str = "general",
        homology_metric: str = "smooth_l1",
        homology_features: str = "all",
        homology_sigma: float = 0.05,
        homology_resolution: tuple = (30, 30),
        homology_filtering: bool = True,
        homology_threshold: float = 0.01,
        homology_k_top: int = 500,
        chunks: int = 2000,
        weighting_power: float = 2.0,
        composite_flag: bool = True,
        homology_adaptive_sigma: bool = True,
        # --- Topological Complexity Regularization ---
        use_topological_complexity: bool = False,
        topological_complexity_weight: float = 0.1,
        complexity_warmup_epochs: int = 10,
        complexity_interval: int = 1,
        complexity_downsample_scale: float = 0.5,
        complexity_features: str = "all",
        complexity_class_context: str = "general",
        complexity_metric: str = "mse",
        complexity_threshold: float = 0.001,
        complexity_k_top: int = 2000,
        complexity_temperature: float = 0.01,
        complexity_auto_balance: bool = True,
        complexity_normalize_lifetimes: bool = True,
        # --- Warmup schedule ---
        warmup_schedule: str = "linear",  # "linear" or "cosine"
    ):
        super().__init__()
        self.beta = beta
        self.n_classes = n_classes
        self.spatial_dims = spatial_dims
        self.kl_clamp = kl_clamp
        self.task = task
        self.regression_loss_type = regression_loss_type.lower()
        self.warmup_schedule = warmup_schedule
        self.kl_divergence_calculator = KLDivergence()

        if elbo_class_weights is not None:
            self.elbo_class_weights = torch.tensor(
                elbo_class_weights, dtype=torch.float32
            )
        else:
            self.elbo_class_weights = None

        if self.task == "segmentation":
            reg_used = []

            # 1. Fractal
            self.use_fractal_regularization = use_fractal_regularization
            if self.use_fractal_regularization:
                self.fractal_weight = fractal_weight
                self.fractal_warmup_epochs = fractal_warmup_epochs
                reg_used.append(f"Fractal-Dimension {fractal_mode}")
                from mmv_im2im.utils.fractal_layers import FractalDimension

                self.fractal_dimension_calculator = FractalDimension(
                    num_kernels=fractal_num_kernels,
                    mode=fractal_mode,
                    to_binary=fractal_to_binary,
                    spatial_dims=self.spatial_dims,
                )

            # 2. Topological (TI Loss)
            self.use_topological_regularization = use_topological_regularization
            if self.use_topological_regularization:
                self.topological_weight = topological_weight
                self.topological_warmup_epochs = topological_warmup_epochs
                reg_used.append("Topological-Restrictions (TI Loss)")
                from mmv_im2im.utils.topological_loss import TI_Loss

                self.topological_loss_calculator = TI_Loss(
                    dim=self.spatial_dims,
                    connectivity=topological_connectivity,
                    inclusion=topological_inclusion if topological_inclusion else [],
                    exclusion=topological_exclusion if topological_exclusion else [],
                    min_thick=topological_min_thick,
                )

            # 3. Connectivity
            self.use_connectivity_regularization = use_connectivity_regularization
            if self.use_connectivity_regularization:
                self.connectivity_weight = connectivity_weight
                self.connectivity_warmup_epochs = connectivity_warmup_epochs
                reg_used.append(f"Connectivity Coherence {connectivity_mode}")
                from mmv_im2im.utils.connectivity_loss import ConnectivityCoherenceLoss

                self.connectivity_coherence_calculator = ConnectivityCoherenceLoss(
                    spatial_dims=self.spatial_dims,
                    connectivity_mode=connectivity_mode,
                    kernel_shape=kernel_shape,
                    connectivity_kernel_size=connectivity_kernel_size,
                    ignore_background=connectivity_ignore_background,
                    num_classes=n_classes,
                    lambda_density=lambda_density,
                    lambda_gradient=lambda_gradient,
                    metric_density=connectivity_metric_density,
                    metric_gradient=connectivity_metric_gradient,
                )

            # 4. GDL Focal
            self.use_gdl_focal_regularization = use_gdl_focal_regularization
            if self.use_gdl_focal_regularization:
                self.gdl_focal_weight = gdl_focal_weight
                self.gdl_warmup_epochs = gdl_warmup_epochs
                reg_used.append("Generalized Dice Focal")
                from monai.losses import GeneralizedDiceFocalLoss

                monai_focal_weights = (
                    torch.tensor(gdl_class_weights, dtype=torch.float32)
                    if gdl_class_weights
                    else None
                )
                self.gdl_focal_loss_calculator = GeneralizedDiceFocalLoss(
                    softmax=True, to_onehot_y=True, weight=monai_focal_weights
                )

            # 5. Hausdorff
            self.use_hausdorff_regularization = use_hausdorff_regularization
            if self.use_hausdorff_regularization:
                self.hausdorff_weight = hausdorff_weight
                self.hausdorff_warmup_epochs = hausdorff_warmup_epochs
                self.hausdorff_downsample_scale = hausdorff_downsample_scale
                self.hausdorff_dt_iterations = hausdorff_dt_iterations
                self.hausdorff_include_background = hausdorff_include_background
                reg_used.append("Hausdorff")
                from mmv_im2im.utils.hausdorff_loss import HausdorffLoss

                self.hausdorff_loss_calculator = HausdorffLoss(
                    spatial_dims=self.spatial_dims,
                    dt_iterations=self.hausdorff_dt_iterations,
                    include_background=self.hausdorff_include_background,
                    distance_mode=hausdorff_distance_mode,
                    normalize_weights=hausdorff_normalize_weights,
                )

            # 6. Homology (Persistence Image)
            self.use_homology_regularization = use_homology_regularization
            if self.use_homology_regularization:
                self.homology_interval = max(1, homology_interval)
                self.homology_weight = homology_weight
                self.homology_warmup_epochs = homology_warmup_epochs
                self.homology_downsample_scale = homology_downsample_scale
                reg_used.append("Persistence Image (Homology)")
                from mmv_im2im.utils.homology_loss import HomologyLoss

                self.homology_calculator = HomologyLoss(
                    spatial_dims=self.spatial_dims,
                    resolution=homology_resolution,
                    sigma=homology_sigma,
                    features=homology_features,
                    class_context=homology_class_context,
                    metric=homology_metric,
                    chunks=chunks,
                    filtering=homology_filtering,
                    threshold=homology_threshold,
                    k_top=homology_k_top,
                    weighting_power=weighting_power,
                    composite_flag=composite_flag,
                    adaptive_sigma=homology_adaptive_sigma,
                )

            # 7. Topological Complexity
            self.use_topological_complexity = use_topological_complexity
            if self.use_topological_complexity:
                self.complexity_interval = max(1, complexity_interval)
                self.topological_complexity_weight = topological_complexity_weight
                self.complexity_warmup_epochs = complexity_warmup_epochs
                self.complexity_downsample_scale = complexity_downsample_scale
                if complexity_metric == "wasserstein":
                    reg_used.append("Persistence Complexity (Diagrams)")
                else:
                    reg_used.append("Persistence Complexity (Entropy-Betti)")
                from mmv_im2im.utils.topological_complexity_loss import (
                    TopologicalComplexityLoss,
                )

                self.topological_complexity_calculator = TopologicalComplexityLoss(
                    spatial_dims=self.spatial_dims,
                    features=complexity_features,
                    class_context=complexity_class_context,
                    metric=complexity_metric,
                    threshold=complexity_threshold,
                    k_top=complexity_k_top,
                    temperature=complexity_temperature,
                    auto_balance=complexity_auto_balance,
                    normalize_lifetimes=complexity_normalize_lifetimes,
                )

            if len(reg_used) > 0:
                print(f"Active Regularizers: {reg_used}")

    def _get_warmup_factor(self, current_epoch, warmup_epochs):
        """
        CHANGE: Added cosine warmup schedule as an alternative to linear.
        Rationale: Linear warmup introduces the regularizer with a step-like
        gradient increase that can cause loss spikes. Cosine warmup starts
        very gently (near 0), accelerates in the middle, and smoothly
        approaches 1.0. This produces more stable early-training dynamics,
        especially for the heavy topological regularizers.
        """
        if torch.is_tensor(current_epoch):
            current_epoch = current_epoch.detach().cpu().item()
        if torch.is_tensor(warmup_epochs):
            warmup_epochs = warmup_epochs.detach().cpu().item()

        current_epoch = float(current_epoch)
        warmup_epochs = float(max(1, warmup_epochs))

        if current_epoch >= warmup_epochs:
            return 1.0

        t = current_epoch / warmup_epochs  # in [0, 1)

        if self.warmup_schedule == "cosine":
            # Cosine warmup: starts at 0, ends at 1
            import math

            return 0.5 * (1.0 - math.cos(math.pi * t))
        else:
            # Default: linear warmup
            return t

    def _downsample_inputs(self, logits, y_true, scale_factor):
        """Downsamples inputs for computationally expensive topological losses."""
        if scale_factor >= 1.0:
            return logits, y_true

        if self.spatial_dims == 3:
            mode = "trilinear"
        else:
            mode = "bilinear"

        logits_small = F.interpolate(
            logits, scale_factor=scale_factor, mode=mode, align_corners=False
        )

        if y_true.ndim == logits.ndim - 1:
            y_true_float = y_true.unsqueeze(1).float()
        else:
            y_true_float = y_true.float()

        y_true_small = F.interpolate(
            y_true_float, scale_factor=scale_factor, mode="nearest"
        )

        if y_true.ndim == logits.ndim - 1:
            y_true_small = y_true_small.squeeze(1)

        return logits_small, y_true_small.long()

    def forward(
        self, logits, y_true, prior_mu, prior_logvar, post_mu, post_logvar, epoch
    ):
        """
        Computes the ELBO loss + Regularizers.
        Args:
            logits: (B, C, H, W) or (B, C, D, H, W)
            y_true: Integer ground truth labels
        """

        # --- 1. Input Standardization ---
        if y_true.ndim == logits.ndim:
            y_true_ch = y_true
            y_true_flat = y_true.squeeze(1)
        elif y_true.ndim == logits.ndim - 1:
            y_true_ch = y_true.unsqueeze(1)
            y_true_flat = y_true
        else:
            raise ValueError(
                f"y_true shape {y_true.shape} incompatible with logits {logits.shape}"
            )

        if (
            self.elbo_class_weights is not None
            and self.elbo_class_weights.device != logits.device
        ):
            self.elbo_class_weights = self.elbo_class_weights.to(logits.device)

        # --- 2. Base Reconstruction Loss ---
        if self.task == "segmentation":
            reconstruction_loss = F.cross_entropy(
                logits,
                y_true_flat.long(),
                reduction="mean",
                weight=self.elbo_class_weights,
            )
        else:
            target = (
                y_true_ch.float() if y_true_ch.shape == logits.shape else y_true.float()
            )
            if self.regression_loss_type == "mse":
                reconstruction_loss = F.mse_loss(logits, target, reduction="mean")
            elif self.regression_loss_type == "l1":
                reconstruction_loss = F.l1_loss(logits, target, reduction="mean")
            elif self.regression_loss_type == "huber":
                reconstruction_loss = F.huber_loss(logits, target, reduction="mean")
            else:
                raise ValueError(
                    f"Regression loss should be mse/l1/huber but : {self.regression_loss_type} was given"
                )

        # --- 3. KL Divergence ---
        kl_div = self.kl_divergence_calculator(
            post_mu, post_logvar, prior_mu, prior_logvar, self.kl_clamp
        )

        total_loss = reconstruction_loss + (self.beta * kl_div)

        # --- 4. Regularizers ---
        if self.task == "segmentation":

            probs = None
            needs_probs = (
                self.use_fractal_regularization
                or self.use_connectivity_regularization
                or self.use_homology_regularization
                or self.use_topological_complexity
            )
            if needs_probs:
                probs = F.softmax(logits, dim=1)

            # A. Fractal Dimension
            if self.use_fractal_regularization:
                fractal_factor = self._get_warmup_factor(
                    epoch, self.fractal_warmup_epochs
                )
                if fractal_factor > 0:
                    if self.n_classes > 1:
                        fg_probs = 1.0 - probs[:, 0:1, ...]
                    else:
                        fg_probs = probs

                    y_fractal = (y_true_ch > 0).float()
                    with torch.no_grad():
                        fd_true = self.fractal_dimension_calculator(y_fractal)
                    fd_pred = self.fractal_dimension_calculator(fg_probs)
                    fractal_loss = F.mse_loss(fd_pred, fd_true)
                    total_loss += (self.fractal_weight * fractal_factor) * fractal_loss

            # B. TI Loss
            if self.use_topological_regularization:
                topological_factor = self._get_warmup_factor(
                    epoch, self.topological_warmup_epochs
                )
                if topological_factor > 0:
                    topo_loss = self.topological_loss_calculator(logits, y_true_ch)
                    total_loss += (
                        self.topological_weight * topological_factor
                    ) * topo_loss

            # C. Connectivity Coherence
            if self.use_connectivity_regularization:
                connectivity_factor = self._get_warmup_factor(
                    epoch, self.connectivity_warmup_epochs
                )
                if connectivity_factor > 0:
                    y_true_onehot = F.one_hot(
                        y_true_flat.long(), num_classes=self.n_classes
                    )
                    permute_dims = (0, logits.ndim - 1) + tuple(
                        range(1, logits.ndim - 1)
                    )
                    y_true_onehot = y_true_onehot.permute(*permute_dims).float()

                    conn_loss = self.connectivity_coherence_calculator(
                        probs, y_true_onehot
                    )
                    total_loss += (
                        self.connectivity_weight * connectivity_factor
                    ) * conn_loss

            # D. GDL Focal
            if self.use_gdl_focal_regularization:
                gdl_factor = self._get_warmup_factor(epoch, self.gdl_warmup_epochs)
                if gdl_factor > 0:
                    gdl_loss = self.gdl_focal_loss_calculator(logits, y_true_ch.long())
                    total_loss += (self.gdl_focal_weight * gdl_factor) * gdl_loss

            # E. Hausdorff
            if self.use_hausdorff_regularization:
                hausdorff_factor = self._get_warmup_factor(
                    epoch, self.hausdorff_warmup_epochs
                )
                if hausdorff_factor > 0:
                    logits_hs, y_true_hs = self._downsample_inputs(
                        logits, y_true_ch, self.hausdorff_downsample_scale
                    )
                    if torch.sum(y_true_hs > 0) > 0:
                        try:
                            h_loss = self.hausdorff_loss_calculator(
                                logits_hs, y_true_hs
                            )
                            if not torch.isfinite(h_loss):
                                h_loss = torch.tensor(0.0, device=logits.device)
                        except Exception:
                            h_loss = torch.tensor(0.0, device=logits.device)
                    else:
                        h_loss = torch.tensor(0.0, device=logits.device)
                    total_loss += (self.hausdorff_weight * hausdorff_factor) * h_loss

            # F. Homology
            if self.use_homology_regularization:
                homology_factor = self._get_warmup_factor(
                    epoch, self.homology_warmup_epochs
                )
                if epoch % self.homology_interval == 0 and homology_factor > 0:
                    if self.homology_downsample_scale >= 1.0:
                        probs_h = probs
                        y_true_h = y_true_flat
                    else:
                        logits_h, y_true_h = self._downsample_inputs(
                            logits, y_true_flat, self.homology_downsample_scale
                        )
                        probs_h = F.softmax(logits_h, dim=1)

                    h_loss = self.homology_calculator(probs_h, y_true_h)
                    total_loss += (self.homology_weight * homology_factor) * h_loss

            # G. Topological Complexity
            if self.use_topological_complexity:
                complexity_factor = self._get_warmup_factor(
                    epoch, self.complexity_warmup_epochs
                )
                if epoch % self.complexity_interval == 0 and complexity_factor > 0:
                    if self.complexity_downsample_scale >= 1.0:
                        probs_c = probs
                        y_true_c = y_true_flat
                    else:
                        logits_c, y_true_c = self._downsample_inputs(
                            logits, y_true_flat, self.complexity_downsample_scale
                        )
                        probs_c = F.softmax(logits_c, dim=1)

                    c_loss = self.topological_complexity_calculator(probs_c, y_true_c)
                    total_loss += (
                        self.topological_complexity_weight * complexity_factor
                    ) * c_loss

        return total_loss
