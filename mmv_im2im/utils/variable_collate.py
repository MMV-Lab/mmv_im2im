"""
variable_collate.py
-------------------
Custom PyTorch collate function for variable-size 3-D (or 2-D) images
paired with spherical-harmonic GT regression vectors.

------------------
PyTorch's default collate calls torch.stack() which requires all tensors
in a batch to have identical shapes.  Our images have variable spatial
sizes so stacking fails.

--------
1. Find the maximum size along each spatial dimension across the batch.
2. Round those maxima UP to the nearest multiple of k (default 16) so
   the padded tensors are compatible with the network's downsampling.
3. Pad every sample to that common target size SYMMETRICALLY (half before,
   half after), adjusting x0 in the GT vector accordingly.
4. Stack the now-uniform tensors.

The GT adjustment follows the same logic as DivisiblePadWithGTAdjustd:
    x0_new[i] = x0_old[i] + pad_before[i]
The SH coefficients are left untouched.

Usage
-----
Pass `collate_fn=variable_size_collate_fn` (or the factory variant
`make_collate_fn(k=16)`) to your DataLoader.  See variable_datamodule.py
for how this is injected automatically.
"""

from functools import partial
from typing import Dict, List
import math
import torch


from mmv_im2im.utils.custom_transforms import apply_pad_to_tensor

# ---------------------------------------------------------------------------
# Core collate logic
# ---------------------------------------------------------------------------


def variable_size_collate_fn(
    batch: List[Dict[str, torch.Tensor]],
    k: int = 16,
    mode: str = "constant",
    constant_value: float = 0.0,
    n_coord_dims: int = 3,
) -> Dict[str, torch.Tensor]:
    """
    Collate a list of sample dicts with variable-size images into a batch.

    Expected keys in each sample dict
    ----------------------------------
    "IM" : torch.Tensor  [C, *spatial]   the image (any spatial size)
    "GT" : torch.Tensor  [3 + n_coeffs]  the GT regression vector

    Additional keys (e.g. "CM" costmaps) are stacked with torch.stack if
    they already have the same shape, or padded the same way as "IM" if
    their spatial dims match the image.

    Parameters
    ----------
    batch           : list of sample dicts produced by the Dataset/transforms
    k               : divisibility target (16 for AttentionUnet 4× downsampling)
    mode            : padding mode for torch.nn.functional.pad
    constant_value  : fill value when mode="constant"
    n_coord_dims    : how many leading GT elements represent spatial coords
    """
    if len(batch) == 0:
        return {}

    # ── Determine target spatial shape ────────────────────────────────
    sample0 = batch[0]
    spatial_ndim = len(sample0["IM"].shape) - 1  # exclude channel dim
    max_spatial = [0] * spatial_ndim

    for sample in batch:
        for dim_i, s in enumerate(sample["IM"].shape[1:]):
            if s > max_spatial[dim_i]:
                max_spatial[dim_i] = s

    # Round up each max dim to the nearest multiple of k
    target_dims = [math.ceil(d / k) * k for d in max_spatial]

    # ── Pad images and adjust GT ───────────────────────────────────────
    padded_ims: List[torch.Tensor] = []
    adjusted_gts: List[torch.Tensor] = []

    # Track pad_before for each sample (needed for optional CM padding)
    all_pad_before: List[List[int]] = []

    for sample in batch:
        img = sample["IM"]
        gt = sample["GT"]

        if not isinstance(img, torch.Tensor):
            img = torch.tensor(img)
        if not isinstance(gt, torch.Tensor):
            gt = torch.tensor(gt)

        current_spatial = list(img.shape[1:])

        # Compute symmetric padding to reach target_dims
        pad_before = [(t - c) // 2 for t, c in zip(target_dims, current_spatial)]
        pad_after = [
            t - c - pb for t, c, pb in zip(target_dims, current_spatial, pad_before)
        ]

        img_padded = apply_pad_to_tensor(
            img.float(), pad_before, pad_after, mode=mode, value=constant_value
        )

        # Adjust the coordinate elements of GT
        gt_adjusted = gt.clone().float()
        n_adjust = min(n_coord_dims, len(pad_before))
        for i in range(n_adjust):
            gt_adjusted[i] = gt[i] + pad_before[i]

        padded_ims.append(img_padded)
        adjusted_gts.append(gt_adjusted)
        all_pad_before.append(pad_before)

    result: Dict[str, torch.Tensor] = {
        "IM": torch.stack(padded_ims, dim=0),
        "GT": torch.stack(adjusted_gts, dim=0),
    }

    # ── Handle any extra keys (e.g., "CM" costmaps) ───────────────────
    extra_keys = [k_ for k_ in sample0.keys() if k_ not in ("IM", "GT")]
    for key in extra_keys:
        samples_key = [s[key] for s in batch if key in s]
        if len(samples_key) != len(batch):
            continue  # skip if not all samples have this key

        # Try to stack without padding first
        try:
            result[key] = torch.stack(
                [
                    (
                        t.float()
                        if isinstance(t, torch.Tensor)
                        else torch.tensor(t).float()
                    )
                    for t in samples_key
                ],
                dim=0,
            )
        except RuntimeError:
            # Shape mismatch → pad the same way as IM
            padded_key = []
            for sample, pad_before in zip(batch, all_pad_before):
                if key not in sample:
                    continue
                t = sample[key]
                if not isinstance(t, torch.Tensor):
                    t = torch.tensor(t)
                current_spatial = list(t.shape[1:])
                pad_after = [
                    td - c - pb
                    for td, c, pb in zip(target_dims, current_spatial, pad_before)
                ]
                t_padded = apply_pad_to_tensor(
                    t.float(), pad_before, pad_after, mode=mode, value=constant_value
                )
                padded_key.append(t_padded)
            if padded_key:
                result[key] = torch.stack(padded_key, dim=0)

    return result


# ---------------------------------------------------------------------------
# Factory for easy partial application
# ---------------------------------------------------------------------------


def make_collate_fn(
    k: int = 16,
    mode: str = "constant",
    constant_value: float = 0.0,
    n_coord_dims: int = 3,
):
    """
    Returns a collate_fn with the given configuration.

    Example
    -------
    collate_fn = make_collate_fn(k=16)
    loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
    """
    return partial(
        variable_size_collate_fn,
        k=k,
        mode=mode,
        constant_value=constant_value,
        n_coord_dims=n_coord_dims,
    )
