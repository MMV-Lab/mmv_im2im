import torch
import torch.nn as nn
import torch.nn.functional as F


class Slice_windows_differentiable(nn.Module):
    """
    Computes fractal occupancy/entropy for multiple scales.
    Optimized for numerical stability and cleaner execution.
    Supports both 2D (B, C, H, W) and 3D (B, C, D, H, W) inputs.
    """

    def __init__(self, num_kernels: int, mode: str = "classic", spatial_dims=2):
        super().__init__()
        self.num_kernels = num_kernels
        # Kernel sizes are powers of 2
        self.kernel_sizes = [2**i for i in range(1, num_kernels + 1)]
        self.mode = mode
        self.spatial_dims = spatial_dims

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.spatial_dims not in [2, 3]:
            raise ValueError("Input must be 4D (2D Image) or 5D (3D Volume)")

        is_3d = self.spatial_dims == 3

        if is_3d:
            batch_size, channels, D, H, W = x.shape
        else:
            batch_size, channels, H, W = x.shape

        results = []

        # Sequential processing per scale
        for k in self.kernel_sizes:
            # Check dimensions against kernel size
            if is_3d:
                if k > D or k > H or k > W:
                    results.append(x.new_zeros(batch_size))
                    continue
                # 3D Pooling
                kernel_vol = k * k * k
                window_avg = F.avg_pool3d(x, kernel_size=k, stride=k)
            else:
                if k > H or k > W:
                    results.append(x.new_zeros(batch_size))
                    continue
                # 2D Pooling
                kernel_vol = k * k
                window_avg = F.avg_pool2d(x, kernel_size=k, stride=k)

            if self.mode == "classic":
                # Using sigmoid approximation for "box counting"
                # window_avg is [0, 1]. We want close to 0 -> 0, >0 -> 1.
                # Shifted Sigmoid: sigmoid(10 * (x - 0.05)) centers transition near 0.05
                soft_occupied = torch.sigmoid(10.0 * (window_avg - 0.05))

                # Sum over all spatial locations and channels
                # Flatten spatial dims and sum
                count = soft_occupied.reshape(batch_size, -1).sum(dim=1)
                results.append(count)

            elif self.mode == "entropy":
                eps = 1e-6
                window_sum = window_avg * kernel_vol * channels
                total_elems = kernel_vol * channels

                # Probability of occupancy
                p1 = window_sum / (total_elems + 1e-8)
                p1 = torch.clamp(p1, eps, 1.0 - eps)
                p0 = 1.0 - p1

                # Binary Entropy
                entropy = -(p0 * torch.log(p0) + p1 * torch.log(p1)) / 0.69314718
                avg_entropy = entropy.reshape(batch_size, -1).mean(dim=1)
                results.append(avg_entropy)

        # Stack results: (B, Num_Kernels)
        return torch.stack(results, dim=1)


class FractalDimension(nn.Module):
    """
    Calculates the Fractal Dimension via differentiable linear regression
    on the log-log plot of scale vs count/entropy.
    """

    def __init__(
        self,
        num_kernels: int,
        mode: str = "classic",
        to_binary: bool = False,
        spatial_dims=2,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.count_layer = Slice_windows_differentiable(
            num_kernels=num_kernels, mode=mode, spatial_dims=self.spatial_dims
        )
        self.kernel_sizes = self.count_layer.kernel_sizes
        self.mode = mode

        # Pre-compute X axis (scales)
        # We perform regression of Log(Count) vs Log(1/Scale)
        # Log(1/k) = -Log(k)
        inverse_k = torch.tensor(
            [1.0 / k for k in self.kernel_sizes], dtype=torch.float32
        )
        self.register_buffer("log_inv_k", torch.log(inverse_k))

    def differentiable_linregress(self, x, y):
        """
        Differentiable slope calculation of linear regression y = mx + c.
        x: (Num_Kernels) or (B, Num_Kernels)
        y: (B, Num_Kernels)
        """
        # Ensure x is broadcastable
        if x.dim() == 1:
            x = x.unsqueeze(0)  # (1, K)

        # Detach X because we only want gradients flowing through Y (counts/model output)
        x = x.detach()

        # Centering
        x_mean = x.mean(dim=1, keepdim=True)
        y_mean = y.mean(dim=1, keepdim=True)

        x_centered = x - x_mean
        y_centered = y - y_mean

        # Slope formula: sum((x-x_bar)(y-y_bar)) / sum((x-x_bar)^2)
        numerator = (x_centered * y_centered).sum(dim=1)
        denominator = (x_centered**2).sum(dim=1)

        slope = numerator / (denominator + 1e-8)

        if torch.isnan(slope).any() or torch.isinf(slope).any():
            slope = torch.nan_to_num(slope, nan=0.0, posinf=0.0, neginf=0.0)
        return slope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get counts/entropy for each scale
        y_values = self.count_layer(x)  # (B, Num_Kernels)

        eps = 1e-8
        if self.mode == "classic":
            # Log-Log plot for Box Counting
            log_y = torch.log(y_values + eps)
        else:
            # Entropy is already a 'dimension-like' measure
            log_y = y_values

        # Retrieve registered buffer for X axis
        log_x = self.log_inv_k

        # Calculate Slope (Fractal Dimension)
        fractal_dims = self.differentiable_linregress(log_x, log_y)

        return fractal_dims
