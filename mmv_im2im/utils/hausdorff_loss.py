import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import script


@script
def chamfer_distance_transform_gpu(
    input_mask: torch.Tensor, iterations: int = 30, spatial_dims: int = 2
) -> torch.Tensor:
    """
    Computes a GPU-native Distance Transform using iterative Min-Pooling.
    Operates on the union of classes to ensure global geometric consistency.
    """
    # Initialize: 0 inside the object, large value outside
    # We use a large enough constant to act as infinity
    dist_map = torch.where(
        input_mask > 0.5,
        torch.tensor(0.0, device=input_mask.device, dtype=input_mask.dtype),
        torch.tensor(200.0, device=input_mask.device, dtype=input_mask.dtype),
    )

    # Optimization: Use MaxPool on negative values to simulate Min-Pooling
    # This stays 100% on GPU and is fully JIT-compatible
    for _ in range(iterations):
        neg_dist = -dist_map
        if spatial_dims == 2:
            pooled = F.max_pool2d(neg_dist, kernel_size=3, stride=1, padding=1)
        else:
            pooled = F.max_pool3d(neg_dist, kernel_size=3, stride=1, padding=1)

        # Manhattan-like propagation: dist = min(current, neighbors + 1)
        dist_new = -pooled + 1.0
        dist_map = torch.min(dist_map, dist_new)

        # Enforce 0 inside the binary mask
        dist_map = torch.where(input_mask > 0.5, 0.0, dist_map)

    return dist_map


class HausdorffLoss(nn.Module):
    """
    Optimized Hausdorff Loss that treats all foreground classes as a single
    global structure. This prevents inter-class geometric conflicts.
    """

    def __init__(
        self,
        spatial_dims: int = 2,
        dt_iterations: int = 30,
        include_background: bool = False,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.dt_iterations = dt_iterations
        self.include_background = include_background

    def forward(self, logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: (B, C, ...) raw network outputs.
            y_true: (B, 1, ...) label indices.
        """

        # Convert to Probabilities and Get Global Foreground
        probs = F.softmax(logits, dim=1)

        # Union of all foreground classes:
        # We take all channels except index 0 (background) and sum them or take max.
        # Max is more stable for Hausdorff.
        if not self.include_background and probs.shape[1] > 1:
            global_probs = torch.max(probs[:, 1:, ...], dim=1, keepdim=True)[0]

            # Global Ground Truth (any label > 0 is foreground)
            global_gt = (y_true > 0).float()
        else:
            # If we include background or it's a single channel, use everything
            global_probs = probs
            global_gt = y_true.float()

        # Compute Distance Transforms on GPU (Union based)
        # DT to nearest background pixel
        gt_dist_map = chamfer_distance_transform_gpu(
            global_gt, iterations=self.dt_iterations, spatial_dims=self.spatial_dims
        )

        # DT to nearest foreground pixel (Inverted DT)
        bg_dist_map = chamfer_distance_transform_gpu(
            1.0 - global_gt,
            iterations=self.dt_iterations,
            spatial_dims=self.spatial_dims,
        )

        # Weighted Loss Computation
        # We penalize based on (probs - gt)^2 * (distance^2)
        # This informs the optimizer to move the union of boundaries to the GT boundaries
        weight_map = gt_dist_map**2 + bg_dist_map**2

        loss = torch.mean(((global_probs - global_gt) ** 2) * weight_map)

        return loss
