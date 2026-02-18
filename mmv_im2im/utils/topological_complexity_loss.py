import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_topological.nn import CubicalComplex
import math


class TopologicalComplexityLoss(nn.Module):
    """
    Calculates Topological Complexity Loss using torch_topological backends.
    Optimized version: Parallelized Wasserstein and Stats calculation.
    Includes numerical stability fixes for LogCosh and Softmax temperatures.
    Supports 2D and 3D data.
    """

    def __init__(
        self,
        spatial_dims=2,
        features="all",
        class_context="general",
        metric="wasserstein",
        threshold=0.001,
        k_top=2000,
        temperature=0.01,
        auto_balance=True,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.features = features
        self.class_context = class_context
        self.metric = metric
        self.threshold = threshold
        self.k_top = k_top
        self.temperature = max(temperature, 1e-4)
        self.auto_balance = auto_balance

        # Initialize the Cubical Complex calculator.
        self.cubical_complex = CubicalComplex(dim=self.spatial_dims)

        # Internal state to track device during forward pass
        self.current_device = None

    def _stable_log_cosh(self, pred, target):
        """
        Computes log(cosh(pred - target)) in a numerically stable way.
        Formula:
            log(cosh(x)) = log( (e^x + e^-x) / 2 )
                         = log(e^x + e^-x) - log(2)
            For large |x|, this approximates to |x| - log(2).
            We use softplus(2|x|) approach or direct approximation for stability.
        """
        x = pred - target
        abs_x = torch.abs(x)

        loss = torch.where(
            abs_x > 50.0,
            abs_x - math.log(2.0),
            torch.log(
                torch.cosh(x) + 1e-12
            ),  # 1e-12 prevents log(0) theoretically impossible but good practice
        )
        return torch.mean(loss)

    def _extract_lifetimes_batch(self, batch_info):
        """
        Robustly extracts lifetimes from the batch result.
        """
        lts_0_batch = []
        lts_1_batch = []

        for info in batch_info:
            lts_dict = {0: [], 1: []}
            stack = [info] if not isinstance(info, list) else info

            while stack:
                item = stack.pop()
                if isinstance(item, list):
                    stack.extend(item)
                elif hasattr(item, "dimension"):
                    dim = item.dimension
                    if isinstance(dim, torch.Tensor):
                        dim = int(dim.detach().cpu().numpy())

                    if dim in [0, 1]:
                        if hasattr(item, "diagram"):
                            p = item.diagram
                        elif hasattr(item, "persistence"):
                            p = item.persistence
                        else:
                            p = None

                        if p is not None:
                            if not isinstance(p, torch.Tensor):
                                p = torch.as_tensor(p, device=self.current_device)
                            else:
                                p = p.to(self.current_device)

                            if p.numel() > 0:
                                # Calculate persistence (death - birth)
                                pers = p[:, 1] - p[:, 0]
                                # Filter out infinite or NaN persistence just in case
                                valid_mask = torch.isfinite(pers)
                                if valid_mask.any():
                                    lts_dict[dim].append(pers[valid_mask])

            res = []
            for d in [0, 1]:
                if lts_dict[d]:
                    res.append(torch.cat(lts_dict[d]))
                else:
                    res.append(torch.tensor([], device=self.current_device))

            lts_0_batch.append(res[0])
            lts_1_batch.append(res[1])

        return lts_0_batch, lts_1_batch

    def _prepare_padded_lifetimes(self, lts_list):
        if len(lts_list) > 0 and isinstance(lts_list[0], torch.Tensor):
            device = lts_list[0].device
        else:
            device = self.current_device

        batch_size = len(lts_list)
        padded = torch.zeros((batch_size, self.k_top), device=device)

        for i, lt in enumerate(lts_list):
            if lt.numel() > 0:
                # Filter small persistence to reduce noise and computation on irrelevant features
                v = lt[lt > self.threshold]
                if v.numel() > 0:
                    v_sorted, _ = torch.sort(v, descending=True)
                    n = min(v_sorted.numel(), self.k_top)
                    padded[i, :n] = v_sorted[:n]
        return padded

    def _compute_vectorized_soft_stats(self, padded_lts):
        """
        Computes differentiable statistics (mean, count, max) using soft approximations.
        Includes clamping to prevent NaNs in Softmax/Exp.
        """
        # 1. Stability Clamp: Prevent massive values from exploding in exp()
        # If persistence is > 100 (unlikely in normalized images, but possible), clamp it.
        padded_clamped = torch.clamp(padded_lts, max=50.0)

        # 2. Weighted Mean (Softmax based)
        # If padded_lts are all zeros, softmax is uniform.
        weights = F.softmax(padded_clamped / self.temperature, dim=1)
        mean_top = torch.sum(padded_lts * weights, dim=1)

        # 3. Soft Count (Sigmoid based)
        # Calculate distance from threshold, scale by temperature
        # Using padded_clamped helps stability, but we use original for logic accuracy
        # unless it's huge.
        sigmoid_in = (padded_lts - self.threshold) / self.temperature
        # Clamp input to sigmoid to avoid extremely large negative/positive values (though sigmoid handles them well, gradients can vanish)
        count_top = torch.sigmoid(torch.clamp(sigmoid_in, min=-50, max=50)).sum(dim=1)

        # 4. Max Value
        max_val = torch.max(padded_lts, dim=1)[0]

        return torch.stack([mean_top, count_top, max_val], dim=1)

    def _compute_metric(self, pred_stats, target_stats):
        """
        Route to the correct metric calculation.
        """
        if self.metric == "wasserstein":
            # Note: For Wasserstein matching, stats are just raw padded vectors usually
            # But based on the code flow, this block is handled directly in forward for padded vectors
            # This method handles the 'stats' metrics.
            return F.mse_loss(pred_stats, target_stats)

        elif self.metric == "mse":
            return F.mse_loss(pred_stats, target_stats)

        elif self.metric == "log_cosh":
            return self._stable_log_cosh(pred_stats, target_stats)

        elif self.metric == "l1":
            return F.l1_loss(pred_stats, target_stats)

        else:
            # Default to MSE
            return F.mse_loss(pred_stats, target_stats)

    def forward(self, y_pred_softmax, y_true):
        device = y_pred_softmax.device
        self.current_device = device

        # Generic shape unpacking
        shape = y_pred_softmax.shape
        n_channels = shape[1]
        spatial_shape = shape[2:]

        if self.class_context == "general":
            # Skip background (assuming index 0 is background)
            if n_channels > 1:
                p_relevant = y_pred_softmax[:, 1:, ...]
                y_true_oh = F.one_hot(y_true.long(), num_classes=n_channels)
                # Permute OH to (B, C, Spatial...)
                permute_dims = (0, len(shape) - 1) + tuple(range(1, len(shape) - 1))
                y_true_oh = y_true_oh.permute(*permute_dims).float()
                g_relevant = y_true_oh[:, 1:, ...]
            else:
                # Binary case handling
                p_relevant = y_pred_softmax
                g_relevant = y_true.unsqueeze(1).float()
        else:
            # Custom context logic can be added here, currently defaulting to full pass
            p_relevant = y_pred_softmax
            y_true_oh = F.one_hot(y_true.long(), num_classes=n_channels)
            permute_dims = (0, len(shape) - 1) + tuple(range(1, len(shape) - 1))
            y_true_oh = y_true_oh.permute(*permute_dims).float()
            g_relevant = y_true_oh

        # Flatten batch and channels to treat them as independent maps
        # This allows computing topology for all images/channels in one batched call
        p_flat = p_relevant.reshape(-1, *spatial_shape)
        g_flat = g_relevant.reshape(-1, *spatial_shape)

        # Extract diagrams
        # Invert image (-1.0 *) for sublevel filtration equivalent to superlevel set filtration
        p_input = -1.0 * p_flat.unsqueeze(1)
        p_info_batch = self.cubical_complex(p_input)
        lts_p0, lts_p1 = self._extract_lifetimes_batch(p_info_batch)

        # Ground Truth Diagrams (No Grad needed)
        with torch.no_grad():
            g_input = -1.0 * g_flat.unsqueeze(1)
            g_info_batch = self.cubical_complex(g_input)
            lts_g0, lts_g1 = self._extract_lifetimes_batch(g_info_batch)

        total_loss = torch.tensor(0.0, device=device)

        # --- Dim 0 (Components) ---
        if self.features in ["all", "cc"]:
            vp0 = self._prepare_padded_lifetimes(lts_p0)
            vg0 = self._prepare_padded_lifetimes(lts_g0)

            if self.metric == "wasserstein":
                # Wasserstein on 1D slices approximates to MSE on sorted vectors (Sliced Wasserstein)
                # We use MSE on the padded sorted lifetimes directly.
                loss_cc = F.mse_loss(vp0, vg0)
            else:
                stats_p0 = self._compute_vectorized_soft_stats(vp0)
                stats_g0 = self._compute_vectorized_soft_stats(vg0)
                loss_cc = self._compute_metric(stats_p0, stats_g0)

            total_loss += loss_cc

        # --- Dim 1 (Holes/Tunnels) ---
        if self.features in ["all", "holes"]:
            vp1 = self._prepare_padded_lifetimes(lts_p1)
            vg1 = self._prepare_padded_lifetimes(lts_g1)

            if self.metric == "wasserstein":
                loss_holes = F.mse_loss(vp1, vg1)
            else:
                stats_p1 = self._compute_vectorized_soft_stats(vp1)
                stats_g1 = self._compute_vectorized_soft_stats(vg1)
                loss_holes = self._compute_metric(stats_p1, stats_g1)

            if self.auto_balance and self.features == "all":
                total_loss = (total_loss + loss_holes) * 0.5
            else:
                total_loss += loss_holes

        if not torch.isfinite(total_loss):
            return torch.tensor(0.0, device=device, requires_grad=True)

        return total_loss
