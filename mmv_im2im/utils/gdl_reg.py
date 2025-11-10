import torch
import torch.nn as nn
import torch.nn.functional as F
from mmv_im2im.utils.fractal_layers import FractalDimension
from mmv_im2im.utils.topological_loss import TI_Loss
from mmv_im2im.utils.connectivity_loss import ConnectivityCoherenceLoss
from monai.losses import GeneralizedDiceFocalLoss
from monai.metrics import HausdorffDistanceMetric


class SegmentationRegularizedLoss(nn.Module):
    """
    Segmentation loss that combines a primary loss (Generalized Dice Focal Loss)
    with structural regularizers (Fractal, Topological, Connectivity, Hausdorff).
    Designed to replace ELBOLoss for DETERMINISTIC UNet models.
    """

    def __init__(
        self,
        n_classes: int = 3,
        # --- Main Loss Parameters (GeneralizedDiceFocalLoss) ---
        gdl_focal_weight: float = 1.0,  # Overall weight for the GDL/Focal term
        gdl_class_weights: list = None,  # Class weights for GDL/Focal
        # --- Fractal Regularization ---
        use_fractal_regularization: bool = False,
        fractal_weight: float = 0.1,
        fractal_num_kernels: int = 5,
        fractal_mode: str = "classic",
        fractal_to_binary: bool = True,
        # --- Topological Regularization (TI_Loss) ---
        use_topological_regularization: bool = False,
        topological_weight: float = 0.1,
        topological_dim: int = 2,
        topological_connectivity: int = 4,
        topological_inclusion: list = None,
        topological_exclusion: list = None,
        topological_min_thick: int = 1,
        # --- Connectivity Regularization ---
        use_connectivity_regularization: bool = False,
        connectivity_weight: float = 0.1,
        connectivity_kernel_size: int = 3,
        connectivity_ignore_background: bool = True,
        # --- Hausdorff Regularization ---
        use_hausdorff_regularization: bool = False,
        hausdorff_weight: float = 0.1,
        hausdorff_ignore_background: bool = True,
        **kwargs,  # Catch-all for extra params
    ):
        super().__init__()
        self.n_classes = n_classes

        # 1. Main Segmentation Loss (GeneralizedDiceFocalLoss)
        self.gdl_focal_weight = gdl_focal_weight
        monai_focal_weights = None
        if gdl_class_weights is not None:
            # MONAI GeneralizedDiceFocalLoss expects a tensor for weights
            monai_focal_weights = torch.tensor(gdl_class_weights, dtype=torch.float32)

        self.main_seg_loss_calculator = GeneralizedDiceFocalLoss(
            softmax=True, to_onehot_y=True, weight=monai_focal_weights
        )

        # 2. Fractal Regularization Setup (from ELBOLoss.py)
        self.use_fractal_regularization = use_fractal_regularization
        self.fractal_weight = fractal_weight
        if self.use_fractal_regularization:
            self.fractal_dimension_calculator = FractalDimension(
                num_kernels=fractal_num_kernels,
                mode=fractal_mode,
                to_binary=fractal_to_binary,
            )

        # 3. Topological Regularization Setup (from ELBOLoss.py)
        self.use_topological_regularization = use_topological_regularization
        self.topological_weight = topological_weight
        if self.use_topological_regularization:
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

        # 4. Connectivity Regularization Setup (from ELBOLoss.py)
        self.use_connectivity_regularization = use_connectivity_regularization
        self.connectivity_weight = connectivity_weight
        if self.use_connectivity_regularization:
            self.connectivity_coherence_calculator = ConnectivityCoherenceLoss(
                kernel_size=connectivity_kernel_size,
                ignore_background=connectivity_ignore_background,
                num_classes=n_classes,
            )

        # 5. Hausdorff Regularization Setup (from ELBOLoss.py)
        self.use_hausdorff_regularization = use_hausdorff_regularization
        self.hausdorff_weight = hausdorff_weight
        if self.use_hausdorff_regularization:
            self.hausdorff_distance_calculator = HausdorffDistanceMetric(
                include_background=not hausdorff_ignore_background,
                reduction="mean",
            )

    # La UNet determinística solo devuelve logits, así que esta es la firma
    def forward(self, logits, y_true):
        """
        Computes the combined segmentation loss with structural regularizers.

        Args:
            logits (torch.Tensor): Output logits from the UNet (B, C, H, W).
            y_true (torch.Tensor): Ground truth segmentation mask (B, 1, H, W).

        Returns:
            torch.Tensor: The calculated total loss.
        """

        # Squeeze y_true to (B, H, W) if it's (B, 1, H, W)
        if y_true.shape[1] == 1:
            y_true_squeezed = y_true.squeeze(1)
        else:
            y_true_squeezed = y_true

        # 1. Primary Segmentation Loss (GDL + Focal)

        # The loss calculator handles logits and converts y_true internally
        y_true_for_gdl_focal = y_true_squeezed.unsqueeze(1).long()
        primary_loss = self.main_seg_loss_calculator(logits, y_true_for_gdl_focal)

        # Apply overall weight for the main term
        total_loss = self.gdl_focal_weight * primary_loss

        # Get softmax probabilities for regularizers that need them
        y_pred_proba = F.softmax(logits, dim=1)

        # --- REGULARIZATION TERMS ---

        # 2. Fractal Regularization (Uses argmax mask for prediction)
        if self.use_fractal_regularization and self.fractal_weight > 0.0:
            # y_pred_mask: (B, 1, H, W) with class indices
            y_pred_mask = y_pred_proba.argmax(dim=1, keepdim=True).float()

            # Prepare y_true for fractal (B, 1, H, W)
            y_true_for_fractal = y_true_squeezed.unsqueeze(1).float()

            fd_true = self.fractal_dimension_calculator(y_true_for_fractal)
            fd_pred = self.fractal_dimension_calculator(y_pred_mask)

            fractal_loss = torch.mean(torch.abs(fd_true - fd_pred))
            total_loss += self.fractal_weight * fractal_loss

        # 3. Topological Regularization (TI_Loss expects logits and y_true B, 1, H, W)
        if self.use_topological_regularization and self.topological_weight > 0.0:
            # TI_Loss expects logits for x and y_true (B, 1, H, W) for y
            y_true_for_topological = y_true_squeezed.unsqueeze(1).float()
            topological_loss = self.topological_loss_calculator(
                logits, y_true_for_topological
            )
            total_loss += self.topological_weight * topological_loss

        # 4. Connectivity Regularization (ConnectivityCoherenceLoss)
        if self.use_connectivity_regularization and self.connectivity_weight > 0.0:
            # y_pred_softmax: (B, C, H, W)
            # y_true_one_hot: Convert (B, H, W) to one-hot (B, C, H, W)
            y_true_one_hot = (
                F.one_hot(y_true_squeezed.long(), num_classes=self.n_classes)
                .permute(0, 3, 1, 2)
                .float()
            )
            connectivity_loss = self.connectivity_coherence_calculator(
                y_pred_proba, y_true_one_hot
            )
            total_loss += self.connectivity_weight * connectivity_loss

        # 5. Hausdorff Regularization
        if self.use_hausdorff_regularization and self.hausdorff_weight > 0.0:
            try:
                # Convert ground truth to one-hot format (B, C, H, W)
                y_true_one_hot = F.one_hot(
                    y_true_squeezed.long(), num_classes=self.n_classes
                ).permute(0, 3, 1, 2)

                # Get the one-hot encoded prediction from logits
                y_pred_one_hot = F.one_hot(
                    logits.argmax(dim=1), num_classes=self.n_classes
                ).permute(0, 3, 1, 2)

                # Calculate the Hausdorff distance
                hausdorff_loss = self.hausdorff_distance_calculator(
                    y_pred=y_pred_one_hot, y_true=y_true_one_hot
                ).mean()

                total_loss += self.hausdorff_weight * hausdorff_loss
            except Exception:
                pass

        return total_loss
