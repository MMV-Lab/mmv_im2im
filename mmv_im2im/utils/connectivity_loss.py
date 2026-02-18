import torch
import torch.nn as nn
import torch.nn.functional as F


class ConnectivityCoherenceLoss(nn.Module):
    """
    Calculates a connectivity coherence loss to penalize fragmentation.
    Supports both 2D and 3D data via spatial_dims argument.
    """

    def __init__(
        self,
        spatial_dims: int = 2,
        connectivity_mode: str = "single",
        kernel_shape: str = "square",
        connectivity_kernel_size: int = 3,
        ignore_background: bool = True,
        num_classes: int = 2,
        lambda_density: float = 1.0,
        lambda_gradient: float = 0.4,
        metric_density: str = "huber",
        metric_gradient: str = "cosine",
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.connectivity_mode = connectivity_mode
        self.connectivity_kernel_size = connectivity_kernel_size
        self.kernel_shape = kernel_shape
        self.ignore_background = ignore_background
        self.num_classes = num_classes
        self.lambda_density = lambda_density
        self.lambda_gradient = lambda_gradient

        if self.spatial_dims not in [2, 3]:
            raise ValueError("spatial_dims must be 2 or 3.")

        # Select metric functions
        self.density_loss_fn = self._get_metric_function(metric_density)
        self.gradient_loss_fn = self._get_metric_function(metric_gradient)

        valid_modes = [
            "single",
            "multiscale-linear",
            "multiscale-exp",
            "learneable-single",
            "learneable-linear",
            "learneable-exp",
        ]
        if self.connectivity_mode not in valid_modes:
            raise ValueError(f"connectivity_mode should be one of {valid_modes}")

        if (
            self.kernel_shape not in ["square", "gaussian"]
            and "learneable" not in self.connectivity_mode
        ):
            raise ValueError(
                f"Kernel shape should be square or gaussian. {self.kernel_shape} given."
            )

        # --- Prepare Sobel Kernels for Vectorized Gradient Alignment ---
        self._init_sobel_kernels()

        # --- Initialize Kernel Sizes ---
        if self.connectivity_mode in ["single", "learneable-single"]:
            k = self.connectivity_kernel_size
            k = k if k % 2 != 0 else k + 1
            self.kernel_sizes = [k]
        elif self.connectivity_mode in ["multiscale-linear", "learneable-linear"]:
            self.kernel_sizes = [
                2 * i + 1 for i in range(1, self.connectivity_kernel_size + 1)
            ]
        elif self.connectivity_mode in ["multiscale-exp", "learneable-exp"]:
            self.kernel_sizes = [
                2**i + 1 for i in range(1, self.connectivity_kernel_size + 1)
            ]

        # --- Initialize Filters ---
        self.kernels_are_learnable = "learneable" in self.connectivity_mode

        if self.kernels_are_learnable:
            self.learnable_filters = nn.ParameterList()
            for k in self.kernel_sizes:
                # Shape depends on dims:
                # 2D: (num_classes, 1, k, k)
                # 3D: (num_classes, 1, k, k, k)
                shape = (self.num_classes, 1) + (k,) * self.spatial_dims
                w_init = torch.empty(shape)
                nn.init.normal_(w_init, mean=0.0, std=0.01)
                self.learnable_filters.append(nn.Parameter(w_init))
        else:
            self._init_fixed_kernels()

    def _init_sobel_kernels(self):
        if self.spatial_dims == 2:
            # Shape: (1, 1, 3, 3)
            sobel_x = torch.tensor(
                [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
            ).view(1, 1, 3, 3)
            sobel_y = torch.tensor(
                [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
            ).view(1, 1, 3, 3)
            self.register_buffer("sobel_x", sobel_x)
            self.register_buffer("sobel_y", sobel_y)

        elif self.spatial_dims == 3:
            # 3D Sobel Kernels construction (3x3x3)
            # Smooth (1D): [1, 2, 1]
            # Diff (1D):   [-1, 0, 1]
            smooth = torch.tensor([1, 2, 1], dtype=torch.float32)
            diff = torch.tensor([-1, 0, 1], dtype=torch.float32)

            # Helper to create outer products
            def outer3(v1, v2, v3):
                return torch.einsum("i,j,k->ijk", v1, v2, v3).view(1, 1, 3, 3, 3)

            # Sobel X: Diff(x) * Smooth(y) * Smooth(z) -> indices k(z), j(y), i(x)
            # PyTorch Conv3d order is (Depth, Height, Width) i.e. (z, y, x)
            sobel_x = outer3(smooth, smooth, diff)
            sobel_y = outer3(smooth, diff, smooth)
            sobel_z = outer3(diff, smooth, smooth)

            self.register_buffer("sobel_x", sobel_x)
            self.register_buffer("sobel_y", sobel_y)
            self.register_buffer("sobel_z", sobel_z)

    def _init_fixed_kernels(self):
        self.fixed_kernel_names = []
        for k in self.kernel_sizes:
            center = k // 2
            shape = (k,) * self.spatial_dims

            if self.kernel_shape == "single":
                kernel = torch.ones((1, 1) + shape) / (k**self.spatial_dims - 1)
                # Set center to 0
                if self.spatial_dims == 2:
                    kernel[0, 0, center, center] = 0.0
                else:
                    kernel[0, 0, center, center, center] = 0.0
            else:
                # Gaussian
                sigma = max(0.5, 0.2 * float(k))
                coords = torch.arange(0, k) - center
                if self.spatial_dims == 2:
                    y, x = torch.meshgrid(coords, coords, indexing="ij")
                    dist_sq = x**2 + y**2
                else:
                    z, y, x = torch.meshgrid(coords, coords, coords, indexing="ij")
                    dist_sq = x**2 + y**2 + z**2

                kernel = torch.exp(-dist_sq / (2 * sigma**2))
                if self.spatial_dims == 2:
                    kernel[center, center] = 0.0
                else:
                    kernel[center, center, center] = 0.0

                kernel_sum = kernel.sum()
                if kernel_sum > 0:
                    kernel = kernel / kernel_sum

                kernel = kernel.view((1, 1) + shape)

            # Expand to (Num_Classes, 1, K, K, [K])
            repeat_shape = (self.num_classes, 1) + (1,) * self.spatial_dims
            kernel_expanded = kernel.repeat(repeat_shape)

            name = f"fixed_kernel_{k}"
            self.register_buffer(name, kernel_expanded)
            self.fixed_kernel_names.append(name)

    def _get_metric_function(self, metric_name: str):
        metric_name = metric_name.lower()
        if metric_name == "l1":
            return F.l1_loss
        elif metric_name == "mse":
            return F.mse_loss
        elif metric_name == "huber":
            return F.huber_loss
        elif metric_name == "charbonnier":
            return self._charbonnier_loss
        elif metric_name == "cosine":
            return self._cosine_loss
        else:
            raise ValueError(f"Unknown metric: {metric_name}")

    def _charbonnier_loss(self, pred, target, eps=1e-6):
        return torch.mean(torch.sqrt((pred - target) ** 2 + eps**2))

    def _cosine_loss(self, pred, target):
        # Flatten: (B, C, ...) -> (B, C, Total_Pixels)
        target_flat = target.reshape(target.shape[0], target.shape[1], -1)
        pred_flat = pred.reshape(pred.shape[0], pred.shape[1], -1)
        # Cosine Similarity across the spatial dimension (dim=2 in flat)
        return 1.0 - F.cosine_similarity(pred_flat, target_flat, dim=2).mean()

    def _compute_gradient_loss_vectorized(self, pred, target):
        """
        Vectorized gradient calculation using grouped convolutions.
        Adapts to 2D or 3D.
        """
        start_idx = 1 if self.ignore_background else 0
        p_slice = pred[:, start_idx:].contiguous()
        t_slice = target[:, start_idx:].contiguous()

        n_channels_eff = p_slice.shape[1]
        if n_channels_eff == 0:
            return torch.tensor(0.0, device=pred.device)

        # Convolution op and expansion shape
        if self.spatial_dims == 2:
            conv_op = F.conv2d
            expand_shape = (n_channels_eff, 1, 1, 1)
        else:
            conv_op = F.conv3d
            expand_shape = (n_channels_eff, 1, 1, 1, 1)

        # Expand Sobel kernels
        sx = self.sobel_x.repeat(expand_shape)
        sy = self.sobel_y.repeat(expand_shape)

        # Compute gradients
        g_x_pred = conv_op(p_slice, sx, padding=1, groups=n_channels_eff)
        g_y_pred = conv_op(p_slice, sy, padding=1, groups=n_channels_eff)
        g_x_true = conv_op(t_slice, sx, padding=1, groups=n_channels_eff)
        g_y_true = conv_op(t_slice, sy, padding=1, groups=n_channels_eff)

        loss = self.gradient_loss_fn(g_x_pred, g_x_true) + self.gradient_loss_fn(
            g_y_pred, g_y_true
        )

        if self.spatial_dims == 3:
            sz = self.sobel_z.repeat(expand_shape)
            g_z_pred = conv_op(p_slice, sz, padding=1, groups=n_channels_eff)
            g_z_true = conv_op(t_slice, sz, padding=1, groups=n_channels_eff)
            loss = loss + self.gradient_loss_fn(g_z_pred, g_z_true)

        return loss

    def forward(self, y_pred_softmax, y_true_one_hot):
        # Ensure inputs are contiguous
        y_pred_softmax = y_pred_softmax.contiguous()
        y_true_one_hot = y_true_one_hot.float().contiguous()

        device = y_pred_softmax.device
        start_idx = 1 if self.ignore_background else 0

        total_loss = torch.tensor(0.0, device=device)

        # --- Gradient Alignment Term ---
        if self.lambda_gradient > 0:
            grad_loss = self._compute_gradient_loss_vectorized(
                y_pred_softmax, y_true_one_hot
            )
            total_loss = total_loss + (self.lambda_gradient * grad_loss)

        # --- Density / Connectivity Term ---
        if self.lambda_density <= 0:
            return total_loss

        density_loss_accum = []

        # Prepare list of kernels
        if self.kernels_are_learnable:
            kernels_to_use = self.learnable_filters
        else:
            kernels_to_use = [getattr(self, name) for name in self.fixed_kernel_names]

        conv_op = F.conv2d if self.spatial_dims == 2 else F.conv3d

        for i, weight in enumerate(kernels_to_use):
            k = self.kernel_sizes[i]
            padding = k // 2
            weight = weight.contiguous()

            # --- Vectorized Convolution ---
            pred_neighbor_avg = conv_op(
                y_pred_softmax, weight, padding=padding, groups=self.num_classes
            )
            true_neighbor_avg = conv_op(
                y_true_one_hot, weight, padding=padding, groups=self.num_classes
            )

            # --- Slice Relevant Channels ---
            p_neigh_rel = pred_neighbor_avg[:, start_idx:]
            t_neigh_rel = true_neighbor_avg[:, start_idx:]
            p_pixel_rel = y_pred_softmax[:, start_idx:]
            t_pixel_rel = y_true_one_hot[:, start_idx:]

            # --- Full-Consistency Logic ---
            loss_a = self.density_loss_fn(p_neigh_rel, t_pixel_rel)
            loss_b = self.density_loss_fn(p_pixel_rel, t_neigh_rel)
            loss_c = self.density_loss_fn(p_neigh_rel, t_neigh_rel)

            density_loss_accum.append(loss_a + loss_b + loss_c)

        if density_loss_accum:
            density_term = torch.stack(density_loss_accum).mean()
            total_loss = total_loss + (self.lambda_density * density_term)

        return total_loss
