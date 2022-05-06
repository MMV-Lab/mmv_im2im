# This file was adapted from https://github.com/AllenCellModeling/pytorch_fnet/

from typing import Optional
import numpy as np


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
