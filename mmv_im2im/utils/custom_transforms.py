"""
--------------------
MONAI-compatible transform that solves the variable-size input problem
for spherical harmonic regression training.

Problem context
---------------
The GT vector has the format:  [x0_z, x0_y, x0_x,  c_1, c_2, ..., c_n]
   - x0 (first 3 elements): center-of-mass of the cell in the CROPPED
     volume's coordinate space (voxel units).
   - c_i (remaining elements): spherical harmonic coefficients encoding
     shape as radii from the center → they are SCALE-DEPENDENT but NOT
     position-dependent. Padding does NOT alter them.

When we pad an image by `pad_before = [pz, py, px]` pixels at the
start of each axis, the cell content shifts by exactly that amount, so:
    x0_new[i] = x0_old[i] + pad_before[i]    (i = z, y, x)

The SH coefficients are unchanged.

This transform:
  1. Pads the image (IM key) so every spatial dimension is divisible by `k`
     (default k=16 for Unet Models with 4 downsampling stages).
  2. Applies the padding SYMMETRICALLY (half before, half after) so the
     cell stays centred in the padded volume, which helps the network
     generalise positional predictions.
  3. Updates x0 in the GT vector to reflect the new coordinates.

Usage in YAML (preprocess section, after NormalizeIntensityd):
  - module_name: custom_transforms
    func_name: DivisiblePadWithGTAdjustd
    params:
      image_key: "IM"
      gt_key: "GT"
      k: 16
      mode: "constant"    # zero-padding is safest after normalisation
"""

from typing import Dict, Hashable, Mapping, Union
import numpy as np
import torch
import torch.nn.functional as F
from monai.transforms import MapTransform
from monai.transforms import Transform

# ---------------------------------------------------------------------------
# Core helper
# ---------------------------------------------------------------------------


def compute_symmetric_pad(spatial_shape, k: int):
    """
    For each spatial dimension of size `s`, compute (pad_before, pad_after)
    so that s + pad_before + pad_after is the smallest multiple of k >= s.

    Returns
    -------
    pad_before : list[int]  – one entry per spatial dim (Z, Y, X order)
    pad_after  : list[int]  – idem
    """
    pad_before, pad_after = [], []
    for s in spatial_shape:
        remainder = s % k
        total_pad = 0 if remainder == 0 else k - remainder
        pb = total_pad // 2
        pa = total_pad - pb
        pad_before.append(pb)
        pad_after.append(pa)
    return pad_before, pad_after


def apply_pad_to_tensor(
    img: torch.Tensor, pad_before, pad_after, mode: str = "constant", value: float = 0.0
) -> torch.Tensor:
    """
    Pad a tensor of shape [C, *spatial] using torch.nn.functional.pad.

    torch.nn.functional.pad expects padding in REVERSED dimension order
    (last dim first) and WITHOUT the channel dimension:
        (pad_last_front, pad_last_back, ..., pad_first_spatial_front, pad_first_spatial_back)

    Parameters
    ----------
    img        : [C, Z, Y, X] or [C, Y, X]
    pad_before : list[int] in spatial order (Z→Y→X or Y→X)
    pad_after  : list[int] in spatial order
    """
    # Build pad tuple: reversed spatial dims, NO channel padding
    pad_args = []
    for pb, pa in reversed(list(zip(pad_before, pad_after))):
        pad_args.extend([pb, pa])
    # Channel dim – no padding
    pad_args.extend([0, 0])

    if mode == "constant":
        return F.pad(img.float(), pad_args, mode="constant", value=value)
    elif mode == "reflect":
        # reflect requires pad < dim size; fall back to constant if unsafe
        try:
            return F.pad(img.float(), pad_args, mode="reflect")
        except RuntimeError:
            return F.pad(img.float(), pad_args, mode="constant", value=value)
    else:
        return F.pad(img.float(), pad_args, mode="constant", value=value)


# ---------------------------------------------------------------------------
# MONAI MapTransform
# ---------------------------------------------------------------------------


class DivisiblePadWithGTAdjustd(MapTransform):
    """
    Pads the image key to be spatially divisible by ``k`` and adjusts the
    spatial coordinates stored in the first n_coord_dims elements of the GT key.

    Parameters
    ----------
    keys : list[str] | None
        [image_key, gt_key].  keys[0] is the image, keys[1] is the GT vector.
        When used via parse_monai_ops (mmv_im2im training pipeline), this
        argument is popped by the parser before the constructor is called and
        never arrives here.  In that case the transform falls back to the
        explicit image_key / gt_key arguments, or to the defaults "IM" / "GT".
    k : int
        Each spatial dimension will be padded to the nearest multiple of k.
        For AttentionUnet with strides [1,2,2,2,2] → k=16.
    mode : str
        Padding mode for torch.nn.functional.pad.
        "constant" (zero-padding) is recommended after intensity normalisation.
    constant_value : float
        Fill value when mode="constant".  Default 0.0.
    n_coord_dims : int
        Number of leading GT elements that are spatial coordinates and must
        be adjusted.  Default 3 → (z, y, x).
    image_key : str | None
        Explicit image key override used when keys=None.  Default "IM".
    gt_key : str | None
        Explicit GT key override used when keys=None.  Default "GT".

    Three valid ways to instantiate
    --------------------------------
    # 1. Via YAML / parse_monai_ops  (keys is popped before __init__, defaults used)
    - module_name: mmv_im2im.utils.custom_transforms
      func_name: DivisiblePadWithGTAdjustd
      params:
        keys: ["IM", "GT"]   # consumed by parser; sets image_key="IM", gt_key="GT"
        k: 16

    # 2. Direct Python call with keys list
    t = DivisiblePadWithGTAdjustd(keys=["IM", "GT"], k=16)

    # 3. Direct Python call with custom key names
    t = DivisiblePadWithGTAdjustd(image_key="image", gt_key="label", k=16)
    """

    def __init__(
        self,
        keys=None,  # parse_monai_ops pops 'keys' from func_params before
        # calling the constructor of custom (non-monai) transforms,
        # so this argument may never actually arrive here.
        # keys[0] → image key, keys[1] → GT key.
        # Falls back to ["IM", "GT"] if None or not provided.
        k: int = 16,
        mode: str = "constant",
        constant_value: float = 0.0,
        n_coord_dims: int = 3,
        image_key: str = None,  # optional explicit override; ignored if keys provided
        gt_key: str = None,  # optional explicit override; ignored if keys provided
    ):
        # Resolve image_key / gt_key from keys or explicit overrides or defaults
        if keys is not None:
            if len(keys) != 2:
                raise ValueError(
                    f"DivisiblePadWithGTAdjustd expects exactly 2 keys "
                    f"[image_key, gt_key], got {keys}"
                )
            _image_key = keys[0]
            _gt_key = keys[1]

        super().__init__(keys=[_image_key, _gt_key])
        self.image_key = _image_key
        self.gt_key = _gt_key
        self.k = k
        self.mode = mode
        self.constant_value = constant_value
        self.n_coord_dims = n_coord_dims

    # ------------------------------------------------------------------
    def __call__(self, data: Mapping[Hashable, object]) -> Dict[Hashable, object]:

        d = dict(data)

        img = d[self.image_key]
        gt = d[self.gt_key]

        # ── Normalise to torch.Tensor ──────────────────────────────────
        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img.copy())
        else:
            img = img.as_tensor() if hasattr(img, "as_tensor") else img.clone()

        if isinstance(gt, np.ndarray):
            gt = torch.from_numpy(gt.copy()).float()
        else:
            gt = (
                gt.as_tensor().float()
                if hasattr(gt, "as_tensor")
                else gt.clone().float()
            )

        # ── Spatial shape: img is [C, *spatial] ───────────────────────
        spatial_shape = img.shape[1:]  # (Z, Y, X) for 3D; (Y, X) for 2D

        pad_before, pad_after = compute_symmetric_pad(spatial_shape, self.k)

        # ── Apply padding to image ─────────────────────────────────────
        img_padded = apply_pad_to_tensor(
            img, pad_before, pad_after, mode=self.mode, value=self.constant_value
        )

        # ── Adjust spatial coordinates in GT ──────────────────────────
        # GT layout: [x0_z, x0_y, x0_x, coeff_0, coeff_1, ...]
        # Padding shifts the cell by pad_before[i] voxels on each axis.
        gt_adjusted = gt.clone()
        n_adjust = min(self.n_coord_dims, len(pad_before))
        for i in range(n_adjust):
            gt_adjusted[i] = gt[i] + pad_before[i]

        d[self.image_key] = img_padded
        d[self.gt_key] = gt_adjusted

        return d


"""
---------------------
INFERENCE transforms that invert the effect of padding applied
by DivisiblePadWithGTAdjustd during training.

Issue with variable sizes during inference
--------------------------------------------
During training, each sample passes through DivisiblePadWithGTAdjustd which:
  - Pads the image to a multiple of k
  - Adds pad_before[i] to the x0 coordinates of the GT

In inference, the network predicts in the PADDED space. To recover
the coordinates in the ORIGINAL space, pad_before[i] must be subtracted.

    pad_before[i] = ((ceil(s_i / k) * k) - s_i) // 2

The only piece of data needed to calculate this is the original shape of the
image BEFORE padding. Since images have variable sizes, that shape changes 
for each image.

Solution: PadStateBuffer
------------------------
A lightweight object that acts as shared memory between:
  - RecordShapeAndPad    → writes the original shape and applies the padding
  - RemovePadFromPrediction → reads the original shape and corrects the prediction

Both transforms receive the SAME instance of PadStateBuffer.
As long as one image is processed at a time (standard sequential inference),
the state remains consistent.

Full inference flow
-----------------------------
    state = PadStateBuffer()

    preproc  = RecordShapeAndPad(state, k=16)
    postproc = RemovePadFromPrediction(state, k=16, n_coord_dims=3)

    for img in images:
        img_padded    = preproc(img)           # saves shape, pads
        pred_padded   = model(img_padded)      # inference in padded space
        pred_original = postproc(pred_padded)  # corrects coordinates
"""


# ---------------------------------------------------------------------------
# Shared state buffer
# ---------------------------------------------------------------------------


class PadStateBuffer:
    """
    Shared state object between RecordShapeAndPad and
    RemovePadFromPrediction.

    Attributes
    ----------
    original_spatial_shape : tuple[int, ...] | None
        Spatial shape (without channel) of the image BEFORE padding.
        Written in RecordShapeAndPad and read in RemovePadFromPrediction.
    pad_before : list[int] | None
        Padding applied at the start of each spatial axis.
        Calculated and saved in RecordShapeAndPad.
    """

    def __init__(self):
        self.original_spatial_shape: Union[tuple, None] = None
        self.pad_before: Union[list, None] = None

    def reset(self):
        self.original_spatial_shape = None
        self.pad_before = None

    def __repr__(self):
        return (
            f"PadStateBuffer("
            f"original_spatial_shape={self.original_spatial_shape}, "
            f"pad_before={self.pad_before})"
        )


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------


def _apply_divisible_pad(
    img: torch.Tensor,
    k: int,
    mode: str = "constant",
    value: float = 0.0,
) -> tuple:
    """
    Pads img=[C, *spatial] to the next multiple of k in each spatial axis.

    Returns
    -------
    img_padded : torch.Tensor  [C, *spatial_padded]
    pad_before : list[int]     padding applied at the start of each axis
    """
    spatial = img.shape[1:]
    pad_before, pad_after = [], []
    for s in spatial:
        remainder = s % k
        total = 0 if remainder == 0 else k - remainder
        pb = total // 2
        pad_before.append(pb)
        pad_after.append(total - pb)

    # torch.nn.functional.pad: inverse axis order, no channel dim
    pad_args = []
    for pb, pa in reversed(list(zip(pad_before, pad_after))):
        pad_args.extend([pb, pa])
    pad_args.extend([0, 0])

    try:
        img_padded = F.pad(img.float(), pad_args, mode=mode, value=value)
    except (RuntimeError, NotImplementedError):
        img_padded = F.pad(img.float(), pad_args, mode="constant", value=value)

    return img_padded, pad_before


# ---------------------------------------------------------------------------
# Transform 1/2 — RecordShapeAndPad  (preprocessing)
# ---------------------------------------------------------------------------


class RecordShapeAndPad(Transform):
    """
    PREPROCESSING transform that:
      1. Registers the original spatial shape in the PadStateBuffer.
      2. Applies divisible padding to the image.

    Replaces MONAI's DivisiblePad in the inference pipeline
    when padding needs to be undone in the prediction later.

    Parameters
    ----------
    state : PadStateBuffer
        Shared object with RemovePadFromPrediction.
    k : int
        All spatial dimensions are padded to the next multiple of k.
    mode : str
        Padding mode. "constant" recommended after NormalizeIntensity.
    constant_value : float
        Fill value when mode="constant".

    Input / Output
    --------------
    Input  : tensor or numpy  [C, *spatial]
    Output : tensor          [C, *spatial_padded]
    """

    def __init__(
        self,
        state: PadStateBuffer,
        k: int = 16,
        mode: str = "constant",
        constant_value: float = 0.0,
    ):
        self.state = state
        self.k = k
        self.mode = mode
        self.constant_value = constant_value

    def __call__(
        self,
        img: Union[np.ndarray, torch.Tensor],
    ) -> torch.Tensor:

        if isinstance(img, np.ndarray):
            img = torch.from_numpy(img.copy()).float()
        else:
            img = img.float()

        # 1. Save shape BEFORE padding
        self.state.original_spatial_shape = tuple(img.shape[1:])

        # 2. Pad and save pad_before in the shared state
        img_padded, pad_before = _apply_divisible_pad(
            img, self.k, self.mode, self.constant_value
        )
        self.state.pad_before = pad_before

        return img_padded


# ---------------------------------------------------------------------------
# Transform 2/2 — RemovePadFromPrediction  (postprocessing)
# ---------------------------------------------------------------------------


class RemovePadFromPrediction(Transform):
    """
    POSTPROCESSING transform that corrects the x0 coordinates of the
    prediction by subtracting the offset introduced by padding.

    Reads pad_before from the PadStateBuffer shared with RecordShapeAndPad,
    which was updated when processing the corresponding image.

    Parameters
    ----------
    state : PadStateBuffer
        Same object passed to RecordShapeAndPad.
    k : int
        Same k used in RecordShapeAndPad. Used as fallback to
        recalculate pad_before if it is not in the state for some reason.
    n_coord_dims : int
        Number of initial elements of the prediction vector that
        represent spatial coordinates and must be corrected.
        Default 3 → (z, y, x) of the center of mass.

    Note
    ----
    SH coefficients (indices from n_coord_dims onwards) are
    translation-invariant and are NOT modified.

    Input / Output
    --------------
    Input  : tensor or numpy  (N,) or (B, N) — prediction in padded space
    Output : same type with the first n_coord_dims elements corrected
    """

    def __init__(
        self,
        state: PadStateBuffer,
        k: int = 16,
        n_coord_dims: int = 3,
    ):
        self.state = state
        self.k = k
        self.n_coord_dims = n_coord_dims

    def __call__(
        self,
        pred_vector: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:

        # -- Recover pad_before --------------------------------------
        if self.state.pad_before is not None:
            pad_before = self.state.pad_before

        elif self.state.original_spatial_shape is not None:
            # Fallback: recalculate if pad_before was not saved
            pad_before = []
            for s in self.state.original_spatial_shape:
                remainder = s % self.k
                total = 0 if remainder == 0 else self.k - remainder
                pad_before.append(total // 2)
        else:
            raise RuntimeError(
                "RemovePadFromPrediction: PadStateBuffer is empty. "
                "Ensure RecordShapeAndPad processed the image "
                "BEFORE calling this transform."
            )

        # -- Type conversion ------------------------------------------
        return_numpy = isinstance(pred_vector, np.ndarray)
        if return_numpy:
            out = torch.from_numpy(pred_vector.copy()).float()
        else:
            out = pred_vector.clone().float()

        # -- Coordinate correction -----------------------------------
        batched = out.dim() == 2  # True if shape [B, N]
        n_adjust = min(self.n_coord_dims, len(pad_before))

        for i in range(n_adjust):
            if batched:
                out[:, i] = out[:, i] - pad_before[i]
            else:
                out[i] = out[i] - pad_before[i]

        return out.numpy() if return_numpy else out
