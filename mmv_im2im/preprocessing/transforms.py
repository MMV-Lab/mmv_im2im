# This file was adapted from https://github.com/AllenCellModeling/pytorch_fnet/

from typing import Optional, Tuple, List, Union
import numpy as np
from torch.nn import functional as F


def to_float(img):
    return img.float()


def dummy_to_ones(img):
    img[:] = 1
    return img


def norm_around_center(img, z_center: Optional[int] = None, min_z: Optional[int] = 32):
    """Returns normalized version of input img.
    img will be normalized with respect to the mean, std pixel intensity
    of the sub-array of length 32 in the z-dimension centered around the
    img's "z_center".
    Parameters
    ----------
    img
        Input 4D torch.Tensor to be normalized.
    z_center
        Z-index of cell centers.
    Returns
    -------
    4D torch.Tensor
       Normalized img
    """
    img_array = img.numpy()
    img_array = np.squeeze(img_array, axis=0).astype(np.float32)
    if img_array.shape[0] < min_z:
        raise ValueError("Input array must be at least length 32 in first dimension")
    if z_center is None:
        z_center = img_array.shape[0] // 2
    chunk_zlen = 32
    z_start = z_center - chunk_zlen // 2
    if z_start < 0:
        z_start = 0
        print(f"Warning: z_start set to {z_start}")
    if (z_start + chunk_zlen) > img_array.shape[0]:
        z_start = img_array.shape[0] - chunk_zlen
        print(f"Warning: z_start set to {z_start}")
    chunk = img_array[z_start : z_start + chunk_zlen, :, :]
    img = img - chunk.mean()
    img = img / chunk.std()

    return img


def pad_to_multiple(
    img,
    multiple_base: Union[int, Tuple[int], List[int]] = 8,
    pad_value: Union[str, float, int] = 0,
):
    """Returns padded image.
    img will be padded to the size of multiple of "multiple_base".

    Parameters
    ----------
    img:
        Input 4D torch.Tensor to be padded, C x X x Y x Z
    multiple_base:
        int or a sequence of int

    Returns
    -------
    4D torch.Tensor
       padded img
    """

    img_array = img.numpy()
    if isinstance(multiple_base, int):
        if img_array.shape[-1] == 1:
            # Z dim = 1, only pad XY
            pad_base = [multiple_base, multiple_base]
        else:
            # 3D data, pad XYZ
            pad_base = [multiple_base, multiple_base, multiple_base]
    else:
        if img_array.shape[-1] == 1:
            # Z dim = 1, only pad XY
            assert len(multiple_base) == 2, "multiple_base and image shape do not match"
            pad_base = [multiple_base[0], multiple_base[1]]
        else:
            # 3D data, pad XYZ
            assert len(multiple_base) == 3, "multiple_base and image shape do not match"
            pad_base = [multiple_base[0], multiple_base[1], multiple_base[2]]

    multiple_x = img_array.shape[1] // pad_base[0]
    if img_array.shape[1] % pad_base[0] != 0:
        diff_x = pad_base[0] * (multiple_x + 1) - img_array.shape[1]
    else:
        diff_x = 0

    multiple_y = img_array.shape[2] // pad_base[1]
    if img_array.shape[2] % pad_base[1] != 0:
        diff_y = pad_base[1] * (multiple_y + 1) - img_array.shape[2]
    else:
        diff_y = 0

    if len(pad_base) == 3:
        multiple_z = img_array.shape[3] // pad_base[2]
        if img_array.shape[3] % pad_base[2] != 0:
            diff_z = pad_base[2] * (multiple_z + 1) - img_array.shape[3]
        else:
            diff_z = 0
        pad_shape = (
            diff_z // 2,
            diff_z - diff_z // 2,
            diff_y // 2,
            diff_y - diff_y // 2,
            diff_x // 2,
            diff_x - diff_x // 2
        )
    else:
        pad_shape = (
            0,
            0,
            diff_y // 2,
            diff_y - diff_y // 2,
            diff_x // 2,
            diff_x - diff_x // 2,
        )

    print(pad_value)
    if isinstance(pad_value, str) and pad_value == "mean":
        avg_bg = img_array.mean()
        return F.pad(img, pad_shape, "constant", avg_bg)
    else:
        return F.pad(img, pad_shape, pad_value)
