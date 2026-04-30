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
        normalize_lifetimes=True,
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
        self.normalize_lifetimes = normalize_lifetimes

        self.cubical_complex = CubicalComplex(dim=self.spatial_dims)
        self.current_device = None

    def _stable_log_cosh(self, pred, target):
        """
        Numerically stable log(cosh(pred - target)).
        For large |x|, log(cosh(x)) ≈ |x| - log(2), avoiding overflow.
        """
        x = pred - target
        abs_x = torch.abs(x)
        loss = torch.where(
            abs_x > 50.0,
            abs_x - math.log(2.0),
            torch.log(torch.cosh(x) + 1e-12),
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
            device = self.current_device

            worklist = info if isinstance(info, list) else [info]
            worklist = list(worklist)
            while worklist:
                item = worklist.pop()
                if isinstance(item, list):
                    worklist.extend(item)
                elif hasattr(item, "dimension"):
                    dim = item.dimension
                    if isinstance(dim, torch.Tensor):
                        dim = int(dim.item())

                    if dim in [0, 1]:
                        if hasattr(item, "diagram"):
                            p = item.diagram
                        elif hasattr(item, "persistence"):
                            p = item.persistence
                        else:
                            p = None

                        if p is not None:
                            if not isinstance(p, torch.Tensor):
                                p = torch.as_tensor(p, device=device)
                            else:
                                if device is None:
                                    device = p.device

                            if p.numel() > 0:
                                pers = p[:, 1] - p[:, 0]
                                valid_mask = torch.isfinite(pers)
                                if valid_mask.any():
                                    lts_dict[dim].append(pers[valid_mask])

            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            lts_0_batch.append(
                torch.cat(lts_dict[0])
                if lts_dict[0]
                else torch.tensor([], device=device)
            )
            lts_1_batch.append(
                torch.cat(lts_dict[1])
                if lts_dict[1]
                else torch.tensor([], device=device)
            )

        return lts_0_batch, lts_1_batch

    def _normalize_lifetimes(self, lts_list):
        non_empty = [lt for lt in lts_list if lt.numel() > 0]
        if not non_empty:
            return lts_list
        global_max = torch.cat(non_empty).max().clamp(min=1e-8)
        return [lt / global_max for lt in lts_list]

    def _prepare_padded_lifetimes(self, lts_list):
        device = next(
            (lt.device for lt in lts_list if lt.numel() > 0),
            self.current_device or torch.device("cpu"),
        )
        result = torch.zeros(len(lts_list), self.k_top, device=device)

        for i, lt in enumerate(lts_list):
            if lt.numel() == 0:
                continue
            v = lt[lt > self.threshold]
            if v.numel() == 0:
                continue
            v_sorted, _ = torch.sort(v, descending=True)
            n = min(v_sorted.numel(), self.k_top)
            result[i, :n] = v_sorted[:n]

        return result

    def _compute_vectorized_soft_stats(self, padded_lts):

        padded_clamped = torch.clamp(padded_lts, max=50.0)

        weights = F.softmax(padded_clamped / self.temperature, dim=1)
        mean_top = torch.sum(padded_lts * weights, dim=1)

        sigmoid_in = (padded_lts - self.threshold) / self.temperature

        count_top = (
            torch.sigmoid(torch.clamp(sigmoid_in, min=-50, max=50)).sum(dim=1)
            / self.k_top
        )

        max_val = torch.max(padded_lts, dim=1)[0]

        return torch.stack([mean_top, count_top, max_val], dim=1)

    def _compute_metric(self, pred_stats, target_stats):
        if self.metric == "wasserstein":
            return F.mse_loss(pred_stats, target_stats)
        elif self.metric == "mse":
            return F.mse_loss(pred_stats, target_stats)
        elif self.metric == "log_cosh":
            return self._stable_log_cosh(pred_stats, target_stats)
        elif self.metric == "l1":
            return F.l1_loss(pred_stats, target_stats)
        else:
            return F.mse_loss(pred_stats, target_stats)

    def _harmonic_balance(self, loss_a, loss_b):
        denom = (loss_a + loss_b).clamp(min=1e-8)
        return (2.0 * loss_a * loss_b) / denom

    def forward(self, y_pred_softmax, y_true):
        device = y_pred_softmax.device
        self.current_device = device

        shape = y_pred_softmax.shape
        n_channels = shape[1]
        spatial_shape = shape[2:]

        if self.class_context == "general":
            if n_channels > 1:
                p_relevant = y_pred_softmax[:, 1:, ...]
                y_true_oh = F.one_hot(y_true.long(), num_classes=n_channels)
                permute_dims = (0, len(shape) - 1) + tuple(range(1, len(shape) - 1))
                y_true_oh = y_true_oh.permute(*permute_dims).float()
                g_relevant = y_true_oh[:, 1:, ...]
            else:
                p_relevant = y_pred_softmax
                g_relevant = y_true.unsqueeze(1).float()
        else:
            p_relevant = y_pred_softmax
            y_true_oh = F.one_hot(y_true.long(), num_classes=n_channels)
            permute_dims = (0, len(shape) - 1) + tuple(range(1, len(shape) - 1))
            y_true_oh = y_true_oh.permute(*permute_dims).float()
            g_relevant = y_true_oh

        p_flat = p_relevant.reshape(-1, *spatial_shape)
        g_flat = g_relevant.reshape(-1, *spatial_shape)

        p_input = -1.0 * p_flat.unsqueeze(1)
        p_info_batch = self.cubical_complex(p_input)
        lts_p0, lts_p1 = self._extract_lifetimes_batch(p_info_batch)

        with torch.no_grad():
            g_input = -1.0 * g_flat.unsqueeze(1)
            g_info_batch = self.cubical_complex(g_input)
            lts_g0, lts_g1 = self._extract_lifetimes_batch(g_info_batch)

        if self.normalize_lifetimes:
            lts_p0 = self._normalize_lifetimes(lts_p0)
            lts_g0 = self._normalize_lifetimes(lts_g0)
            lts_p1 = self._normalize_lifetimes(lts_p1)
            lts_g1 = self._normalize_lifetimes(lts_g1)

        total_loss = torch.tensor(0.0, device=device)
        loss_cc = torch.tensor(0.0, device=device)
        loss_holes = torch.tensor(0.0, device=device)

        if self.features in ["all", "cc"]:
            vp0 = self._prepare_padded_lifetimes(lts_p0)
            vg0 = self._prepare_padded_lifetimes(lts_g0)

            if vp0.detach().sum() == 0 and vg0.detach().sum() == 0:
                loss_cc = torch.tensor(0.0, device=device)
            elif self.metric == "wasserstein":
                loss_cc = F.mse_loss(vp0, vg0)
            else:
                stats_p0 = self._compute_vectorized_soft_stats(vp0)
                stats_g0 = self._compute_vectorized_soft_stats(vg0)
                loss_cc = self._compute_metric(stats_p0, stats_g0)

            total_loss = total_loss + loss_cc

        if self.features in ["all", "holes"]:
            vp1 = self._prepare_padded_lifetimes(lts_p1)
            vg1 = self._prepare_padded_lifetimes(lts_g1)

            if vp1.detach().sum() == 0 and vg1.detach().sum() == 0:
                loss_holes = torch.tensor(0.0, device=device)
            elif self.metric == "wasserstein":
                loss_holes = F.mse_loss(vp1, vg1)
            else:
                stats_p1 = self._compute_vectorized_soft_stats(vp1)
                stats_g1 = self._compute_vectorized_soft_stats(vg1)
                loss_holes = self._compute_metric(stats_p1, stats_g1)

            if self.auto_balance and self.features == "all":
                total_loss = self._harmonic_balance(loss_cc, loss_holes)
            else:
                total_loss = total_loss + loss_holes

        if not torch.isfinite(total_loss):
            return torch.tensor(0.0, device=device, requires_grad=True)

        return total_loss
