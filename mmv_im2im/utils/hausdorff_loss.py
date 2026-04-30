import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.jit import script
from typing import Union


@script
def chamfer_distance_transform_gpu(
    input_mask: torch.Tensor, iterations: int, spatial_dims: int = 2
) -> torch.Tensor:
    H = input_mask.shape[-2]
    W = input_mask.shape[-1]
    inf_val = float(H + W)

    dist_map = torch.where(
        input_mask > 0.5,
        torch.zeros(1, device=input_mask.device, dtype=input_mask.dtype),
        torch.full((1,), inf_val, device=input_mask.device, dtype=input_mask.dtype),
    )

    for _ in range(iterations):
        neg_dist = -dist_map
        if spatial_dims == 2:
            pooled = F.max_pool2d(neg_dist, kernel_size=3, stride=1, padding=1)
        else:
            pooled = F.max_pool3d(neg_dist, kernel_size=3, stride=1, padding=1)

        dist_new = -pooled + 1.0
        dist_map = torch.min(dist_map, dist_new)
        dist_map = torch.where(input_mask > 0.5, 0.0, dist_map)

    return dist_map


class HausdorffLoss(nn.Module):
    """
    Hausdorff Loss with automatic dt_iterations estimation.
    """

    def __init__(
        self,
        spatial_dims: int = 2,
        dt_iterations: Union[int, str] = "auto",
        include_background: bool = False,
        distance_mode: str = "l2",
        normalize_weights: bool = True,
        coverage_fraction: float = 0.25,
        dt_iterations_min: int = 10,
        dt_iterations_max: int = 200,
    ):
        super().__init__()
        if not (isinstance(dt_iterations, int) or dt_iterations == "auto"):
            raise ValueError(
                f"dt_iterations must be a positive integer or 'auto', got: {dt_iterations!r}"
            )
        self.spatial_dims = spatial_dims
        self.dt_iterations = dt_iterations
        self.include_background = include_background
        self.distance_mode = distance_mode
        self.normalize_weights = normalize_weights
        self.coverage_fraction = coverage_fraction
        self.dt_iterations_min = dt_iterations_min
        self.dt_iterations_max = dt_iterations_max

    @staticmethod
    def estimate_iterations(
        spatial_shape: torch.Size,
        coverage_fraction: float = 0.25,
        min_iters: int = 10,
        max_iters: int = 200,
    ) -> int:
        min_dim = min(spatial_shape)
        estimated = int(min_dim * coverage_fraction)
        return max(min_iters, min(estimated, max_iters))

    def _resolve_iterations(self, spatial_shape: torch.Size) -> int:
        if self.dt_iterations != "auto":
            return int(self.dt_iterations)
        return self.estimate_iterations(
            spatial_shape,
            self.coverage_fraction,
            self.dt_iterations_min,
            self.dt_iterations_max,
        )

    def forward(self, logits: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(logits, dim=1)

        if not self.include_background and probs.shape[1] > 1:
            global_probs = torch.max(probs[:, 1:, ...], dim=1, keepdim=True)[0]
            global_gt = (y_true > 0).float()
        else:
            global_probs = probs
            global_gt = y_true.float()

        iters = self._resolve_iterations(global_gt.shape[2:])

        gt_dist_map = chamfer_distance_transform_gpu(
            global_gt, iterations=iters, spatial_dims=self.spatial_dims
        )
        bg_dist_map = chamfer_distance_transform_gpu(
            1.0 - global_gt,
            iterations=iters,
            spatial_dims=self.spatial_dims,
        )

        if self.distance_mode == "l2":
            weight_map = gt_dist_map**2 + bg_dist_map**2
        else:
            weight_map = gt_dist_map + bg_dist_map

        if self.normalize_weights:
            norm_factor = weight_map.detach().quantile(0.99).clamp(min=1.0)
            weight_map = weight_map / norm_factor

        return torch.mean(((global_probs - global_gt) ** 2) * weight_map)
