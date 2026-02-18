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
    ):
        super().__init__()
        # Parse resolution if passed as a string from YAML
        if isinstance(resolution, str):
            resolution = tuple(int(x) for x in resolution.strip("()[]").split(","))
        self.resolution = resolution
        self.range_vals = range_vals
        self.sigma = max(sigma, 1e-4)
        self.chunks = chunks
        self.weighting_power = weighting_power

    def forward(self, diagrams):
        """
        Args:
            diagrams (list of torch.Tensor): List of persistence diagrams.
                                             Each tensor is (N_points, 2).
        Returns:
            torch.Tensor: Stacked Persistence Images.
        """
        if len(diagrams) == 0:
            return torch.zeros(
                (1, *self.resolution),
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )

        device = diagrams[0].device
        dtype = diagrams[0].dtype

        # Pre-compute grid
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
        grid_x, grid_y = torch.meshgrid(x, y, indexing="ij")

        # (1, H, W) for broadcasting
        grid_x_exp = grid_x.unsqueeze(0)
        grid_y_exp = grid_y.unsqueeze(0)

        norm_factor = 1.0 / (2 * (self.sigma**2) + 1e-8)
        pi_list = []

        # Iterate over the batch of diagrams
        for diag in diagrams:
            if diag.shape[0] == 0:
                pi_list.append(torch.zeros(self.resolution, device=device, dtype=dtype))
                continue

            b_vals, d_vals = diag[:, 0], diag[:, 1]
            persistence = torch.abs(d_vals - b_vals)
            persistence = torch.clamp(persistence, max=10.0)

            weights_all = torch.pow(persistence, self.weighting_power).view(-1, 1, 1)
            cx_all = b_vals.view(-1, 1, 1)
            cy_all = d_vals.view(-1, 1, 1)

            # Chunking to prevent OOM on high point counts
            if (
                isinstance(self.chunks, int)
                and self.chunks > 0
                and diag.shape[0] > self.chunks
            ):
                pi_accum = torch.zeros(self.resolution, device=device, dtype=dtype)
                for i in range(0, diag.shape[0], self.chunks):
                    end = i + self.chunks
                    w_c, cx_c, cy_c = weights_all[i:end], cx_all[i:end], cy_all[i:end]
                    dist_sq = (grid_x_exp - cx_c) ** 2 + (grid_y_exp - cy_c) ** 2
                    gauss = torch.exp(-dist_sq * norm_factor)
                    pi_accum += (w_c * gauss).sum(dim=0)
                pi_list.append(pi_accum)
            else:
                dist_sq = (grid_x_exp - cx_all) ** 2 + (grid_y_exp - cy_all) ** 2
                gauss = torch.exp(-dist_sq * norm_factor)
                pi_list.append((weights_all * gauss).sum(dim=0))

        return torch.stack(pi_list)


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
        treshold=0.01,
        k_top=500,
        weighting_power=2.0,
        composite_flag=True,
    ):
        super().__init__()
        self.spatial_dims = spatial_dims
        self.pi_generator = DifferentiablePersistenceImage(
            resolution=resolution,
            sigma=sigma,
            chunks=chunks,
            weighting_power=weighting_power,
        )
        self.features = features
        self.class_context = class_context
        self.metric = metric
        self.filtering = filtering
        self.filter_thresh = treshold
        self.k_top = k_top
        self.composite_flag = composite_flag

        # CubicalComplex handles the heavy lifting of TDA
        # Set dimension based on spatial_dims (2 for 2D images, 3 for 3D volumes)
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
            extracted = {0: [], 1: []}
            stack = [info] if not isinstance(info, list) else info
            while stack:
                item = stack.pop()
                if isinstance(item, list):
                    stack.extend(item)
                elif hasattr(item, "diagram"):
                    d = item.dimension
                    if isinstance(d, torch.Tensor):
                        d = int(d.detach().cpu().numpy())

                    # We currently only extract dim 0 (components) and dim 1 (loops/tunnels)
                    if d in [0, 1]:
                        extracted[d].append(item.diagram)

            device = info.device if hasattr(info, "device") else item.diagram.device
            d0 = (
                torch.cat(extracted[0], dim=0)
                if extracted[0]
                else torch.empty((0, 2), device=device if extracted[0] else None)
            )
            d1 = (
                torch.cat(extracted[1], dim=0)
                if extracted[1]
                else torch.empty((0, 2), device=device if extracted[1] else None)
            )

            batch_diag_0.append(self._filter_and_topk(-1.0 * d0))
            batch_diag_1.append(self._filter_and_topk(-1.0 * d1))

        return batch_diag_0, batch_diag_1

    def _filter_and_topk(self, diagram):
        if diagram.shape[0] == 0:
            return diagram

        persistence = torch.abs(diagram[:, 1] - diagram[:, 0])
        mask = torch.isfinite(persistence)
        diagram, persistence = diagram[mask], persistence[mask]

        if self.filtering:
            mask = persistence > self.filter_thresh
            diagram, persistence = diagram[mask], persistence[mask]

        if diagram.shape[0] > self.k_top:
            _, idx = torch.topk(persistence, k=self.k_top)
            diagram = diagram[idx]

        return diagram

    def _compute_image_dist_batch(self, pi_p_batch, pi_g_batch):
        """
        Computes distance between batches of Persistence Images.
        """
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
        Vectorized forward pass.
        y_pred_softmax: (B, C, spatial...)
        y_true: (B, spatial...) or (B, 1, spatial...)
        """
        # Determine shapes
        shape = y_pred_softmax.shape
        batch_size = shape[0]
        n_channels = shape[1]
        spatial_shape = shape[2:]

        device = y_pred_softmax.device

        # Ensure y_true has channel dim
        if y_true.dim() == len(shape) - 1:
            y_true = y_true.unsqueeze(1)

        # Convert y_true to One-Hot: (B, C, spatial...)
        y_true_oh = torch.zeros_like(y_pred_softmax).scatter_(1, y_true.long(), 1)

        relevant_channels = range(1, n_channels)
        num_relevant = len(relevant_channels)

        if num_relevant == 0:
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Flatten spatial dims to generalize 2D and 3D
        # We need independent maps for Cubical Complex
        # Reshape to (B * (C-1), spatial...)
        p_flat = y_pred_softmax[:, 1:, ...].reshape(-1, *spatial_shape)
        g_flat = y_true_oh[:, 1:, ...].reshape(-1, *spatial_shape)

        total_items = p_flat.shape[0]

        # Prediction Diagrams
        # CubicalComplex expects (Batch, C=1, Spatial...) or just (Batch, Spatial...) depending on version
        # We add a channel dim 1 for the TDA engine input
        p_input = -1.0 * p_flat.unsqueeze(1)
        p_info_batch = self.cubical_complex(p_input)
        diag_p0_list, diag_p1_list = (
            self._extract_persistence_diagrams_from_batch_result(p_info_batch)
        )

        # Ground Truth Diagrams (No Grad)
        with torch.no_grad():
            g_input = -1.0 * g_flat.unsqueeze(1)
            g_info_batch = self.cubical_complex(g_input)
            diag_g0_list, diag_g1_list = (
                self._extract_persistence_diagrams_from_batch_result(g_info_batch)
            )

        # Generate PIs
        loss_vector = torch.zeros(total_items, device=device)

        if self.features in ["all", "cc"]:
            pi_p0 = self.pi_generator(diag_p0_list)
            pi_g0 = self.pi_generator(diag_g0_list)
            loss_vector += self._compute_image_dist_batch(pi_p0, pi_g0)

        if self.features in ["all", "holes"]:
            pi_p1 = self.pi_generator(diag_p1_list)
            pi_g1 = self.pi_generator(diag_g1_list)
            loss_vector += self._compute_image_dist_batch(pi_p1, pi_g1)

        # Reshape loss back to (Batch, Num_Relevant_Channels)
        loss_matrix = loss_vector.view(batch_size, num_relevant)

        channel_losses = []
        for c_idx in range(num_relevant):
            losses_for_channel = loss_matrix[:, c_idx]
            c_loss = 0.7 * losses_for_channel.mean() + 0.3 * losses_for_channel.max()
            channel_losses.append(c_loss)

        if len(channel_losses) > 0:
            return torch.stack(channel_losses).mean()

        return torch.tensor(0.0, device=device, requires_grad=True)
