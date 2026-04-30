import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional.image import structural_similarity_index_measure
from torch_topological.nn import CubicalComplex


class DifferentiablePersistenceImage(nn.Module):
    """
    Differentiable implementation of Persistence Images (PI) using PyTorch.
    Calculates the sum of Gaussians centered at (birth, death) pairs.
    """

    def __init__(
        self,
        resolution=(50, 50),
        range_vals=(0, 1),
        sigma=0.05,
        chunks=2000,
        weighting_power=2.0,
        adaptive_sigma: bool = False,
    ):
        super().__init__()
        if isinstance(resolution, str):
            resolution = tuple(int(x) for x in resolution.strip("()[]").split(","))
        self.resolution = resolution
        self.range_vals = range_vals
        self.sigma = max(sigma, 1e-4)
        self.chunks = chunks
        self.weighting_power = weighting_power
        self.adaptive_sigma = adaptive_sigma

        self._cached_grid_key = None
        self._cached_gx = None
        self._cached_gy = None

    def _get_grid(self, device, dtype):
        key = (str(device), str(dtype))
        if self._cached_grid_key != key:
            x = torch.linspace(
                self.range_vals[0],
                self.range_vals[1],
                self.resolution[0],
                device=device,
                dtype=dtype,
            )
            y = torch.linspace(
                self.range_vals[0],
                self.range_vals[1],
                self.resolution[1],
                device=device,
                dtype=dtype,
            )
            gx, gy = torch.meshgrid(x, y, indexing="ij")
            self._cached_gx = gx.unsqueeze(0).unsqueeze(0)
            self._cached_gy = gy.unsqueeze(0).unsqueeze(0)
            self._cached_grid_key = key
        return self._cached_gx, self._cached_gy

    def forward(self, diagrams):
        """
        Args:
            diagrams (list of torch.Tensor): Each tensor is (N_points, 2).
        Returns:
            torch.Tensor: Stacked Persistence Images, shape (B, H, W).
        """
        if len(diagrams) == 0:
            return torch.zeros(
                (1, *self.resolution),
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )

        device = None
        dtype = None
        for d in diagrams:
            if d.shape[0] > 0:
                device = d.device
                dtype = d.dtype
                break
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            dtype = torch.float32

        B = len(diagrams)
        H, W = self.resolution

        gx, gy = self._get_grid(device, dtype)

        sizes = [d.shape[0] for d in diagrams]
        max_pts = max(sizes) if sizes else 0

        if max_pts == 0:
            return torch.zeros((B, H, W), device=device, dtype=dtype)

        padded_b = torch.zeros(B, max_pts, device=device, dtype=dtype)
        padded_d = torch.zeros(B, max_pts, device=device, dtype=dtype)
        padded_w = torch.zeros(B, max_pts, device=device, dtype=dtype)
        sigmas = torch.full((B,), self.sigma, device=device, dtype=dtype)

        for i, diag in enumerate(diagrams):
            n = diag.shape[0]
            if n == 0:
                continue
            persistence = torch.abs(diag[:, 1] - diag[:, 0]).clamp(max=10.0)
            padded_b[i, :n] = diag[:, 0]
            padded_d[i, :n] = diag[:, 1]
            padded_w[i, :n] = torch.pow(persistence, self.weighting_power)

            if self.adaptive_sigma and n > 4:
                q75 = torch.quantile(persistence, 0.75)
                q25 = torch.quantile(persistence, 0.25)
                iqr = (q75 - q25).clamp(min=1e-4)
                sigma_eff = (0.9 * iqr * (n ** (-0.2))).detach().clamp(min=1e-4)
                sigmas[i] = sigma_eff

        norm_factor = (1.0 / (2.0 * sigmas.pow(2) + 1e-8)).view(B, 1, 1, 1)

        pi_batch = torch.zeros(B, H, W, device=device, dtype=dtype)
        chunk_size = max(1, self.chunks)

        for start in range(0, max_pts, chunk_size):
            end = min(start + chunk_size, max_pts)
            cx = padded_b[:, start:end].unsqueeze(-1).unsqueeze(-1)
            cy = padded_d[:, start:end].unsqueeze(-1).unsqueeze(-1)
            w = padded_w[:, start:end].unsqueeze(-1).unsqueeze(-1)
            dist_sq = (gx - cx) ** 2 + (gy - cy) ** 2
            gauss = torch.exp(-dist_sq * norm_factor)
            pi_batch.add_((w * gauss).sum(dim=1))

        return pi_batch


class HomologyLoss(nn.Module):
    """
    Differentiable Homology Loss using Persistence Images.
    Supports 2D and 3D data.
    """

    def __init__(
        self,
        spatial_dims=2,
        resolution=(50, 50),
        sigma=0.05,
        features="all",
        class_context=None,
        metric="smooth_l1",
        chunks=2000,
        filtering=True,
        threshold=0.01,
        treshold=None,
        k_top=500,
        weighting_power=2.0,
        composite_flag=True,
        adaptive_sigma=True,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        resolved_threshold = treshold if treshold is not None else threshold
        self.pi_generator = DifferentiablePersistenceImage(
            resolution=resolution,
            sigma=sigma,
            chunks=chunks,
            weighting_power=weighting_power,
            adaptive_sigma=adaptive_sigma,
        )
        self.features = features
        self.class_context = class_context
        self.metric = metric
        self.filtering = filtering
        self.filter_thresh = resolved_threshold
        self.k_top = k_top
        self.composite_flag = composite_flag

        self.cubical_complex = CubicalComplex(dim=self.spatial_dims)

        if metric == "ssim":
            self.ssim_func = structural_similarity_index_measure

    def _extract_persistence_diagrams_from_batch_result(self, batch_info):
        """
        Parses the output of CubicalComplex for a batch.
        Returns two lists: dim_0 diagrams and dim_1 diagrams.
        """
        batch_diag_0 = []
        batch_diag_1 = []

        for info in batch_info:
            d0_parts = []
            d1_parts = []
            device = None

            worklist = info if isinstance(info, list) else [info]
            worklist = list(worklist)
            while worklist:
                item = worklist.pop()
                if isinstance(item, list):
                    worklist.extend(item)
                elif hasattr(item, "diagram"):
                    diag_tensor = item.diagram
                    if device is None:
                        device = diag_tensor.device
                    d = item.dimension
                    if isinstance(d, torch.Tensor):
                        d = int(d.item())
                    if d == 0:
                        d0_parts.append(diag_tensor)
                    elif d == 1:
                        d1_parts.append(diag_tensor)

            if device is None:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            d0 = torch.cat(d0_parts) if d0_parts else torch.empty((0, 2), device=device)
            d1 = torch.cat(d1_parts) if d1_parts else torch.empty((0, 2), device=device)

            batch_diag_0.append(self._filter_and_topk(-1.0 * d0))
            batch_diag_1.append(self._filter_and_topk(-1.0 * d1))

        return batch_diag_0, batch_diag_1

    def _filter_and_topk(self, diagram):
        if diagram.shape[0] == 0:
            return diagram

        persistence = torch.abs(diagram[:, 1] - diagram[:, 0])
        mask = torch.isfinite(persistence)
        if self.filtering:
            mask = mask & (persistence > self.filter_thresh)

        if not mask.all():
            diagram = diagram[mask]
            persistence = persistence[mask]

        if diagram.shape[0] > self.k_top:
            _, idx = torch.topk(persistence, k=self.k_top)
            diagram = diagram[idx]

        return diagram

    def _compute_image_dist_batch(self, pi_p_batch, pi_g_batch):
        if self.metric == "mse":
            return F.mse_loss(pi_p_batch, pi_g_batch, reduction="none").mean(dim=[1, 2])
        if self.metric == "l1":
            return F.l1_loss(pi_p_batch, pi_g_batch, reduction="none").mean(dim=[1, 2])
        if self.metric == "smooth_l1":
            return F.smooth_l1_loss(
                pi_p_batch, pi_g_batch, beta=0.5, reduction="none"
            ).mean(dim=[1, 2])
        if self.metric == "ssim":
            return 1.0 - self.ssim_func(
                pi_p_batch.unsqueeze(1), pi_g_batch.unsqueeze(1), data_range=1.0
            )
        return torch.zeros(pi_p_batch.shape[0], device=pi_p_batch.device)

    def forward(self, y_pred_softmax, y_true):
        """
        y_pred_softmax: (B, C, spatial...)
        y_true: (B, spatial...) or (B, 1, spatial...)
        """
        shape = y_pred_softmax.shape
        batch_size = shape[0]
        n_channels = shape[1]
        spatial_shape = shape[2:]
        device = y_pred_softmax.device

        if y_true.dim() == len(shape) - 1:
            y_true = y_true.unsqueeze(1)

        y_true_oh = torch.zeros_like(y_pred_softmax).scatter_(1, y_true.long(), 1)

        num_relevant = n_channels - 1

        if num_relevant == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        p_flat = y_pred_softmax[:, 1:, ...].reshape(-1, *spatial_shape)
        g_flat = y_true_oh[:, 1:, ...].reshape(-1, *spatial_shape)

        if not g_flat.detach().any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        p_input = -1.0 * p_flat.unsqueeze(1)
        p_info_batch = self.cubical_complex(p_input)
        diag_p0_list, diag_p1_list = (
            self._extract_persistence_diagrams_from_batch_result(p_info_batch)
        )

        with torch.no_grad():
            g_input = -1.0 * g_flat.unsqueeze(1)
            g_info_batch = self.cubical_complex(g_input)
            diag_g0_list, diag_g1_list = (
                self._extract_persistence_diagrams_from_batch_result(g_info_batch)
            )

        loss_vector = None

        if self.features in ["all", "cc"]:
            pi_p0 = self.pi_generator(diag_p0_list)
            pi_g0 = self.pi_generator(diag_g0_list)
            cc_loss = self._compute_image_dist_batch(pi_p0, pi_g0)
            loss_vector = cc_loss

        if self.features in ["all", "holes"]:
            pi_p1 = self.pi_generator(diag_p1_list)
            pi_g1 = self.pi_generator(diag_g1_list)
            holes_loss = self._compute_image_dist_batch(pi_p1, pi_g1)
            loss_vector = (
                holes_loss if loss_vector is None else loss_vector + holes_loss
            )

        if loss_vector is None:
            return torch.tensor(0.0, device=device, requires_grad=True)

        loss_matrix = loss_vector.view(batch_size, num_relevant)

        mean_losses = loss_matrix.mean(dim=0)
        max_losses = loss_matrix.max(dim=0)[0]
        upper_bounds = 5.0 * mean_losses.detach() + 1e-8
        max_losses_clamped = torch.min(max_losses, upper_bounds)
        return (0.7 * mean_losses + 0.3 * max_losses_clamped).mean()
