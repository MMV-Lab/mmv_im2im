# This file was adapted from https://github.com/AllenCellModeling/pytorch_fnet/

from typing import Optional
import logging
import torch
import numpy as np

logger = logging.getLogger(__name__)  # ?


def norm_around_center(img, z_center: Optional[int] = None):
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

    img = img.numpy()
    img = np.squeeze(img, axis=0)
    img = np.asarray(img, dtype="float32")

    if img.shape[0] < 32:
        raise ValueError("Input array must be at least length 32 in first dimension")
    if z_center is None:
        z_center = img.shape[0] // 2
    chunk_zlen = 32
    z_start = z_center - chunk_zlen // 2
    if z_start < 0:
        z_start = 0
        logger.warn(f"Warning: z_start set to {z_start}")
    if (z_start + chunk_zlen) > img.shape[0]:
        z_start = img.shape[0] - chunk_zlen
        logger.warn(f"Warning: z_start set to {z_start}")
    chunk = img[z_start : z_start + chunk_zlen, :, :]
    img = img - chunk.mean()
    img = img / chunk.std()

    img = np.expand_dims(img, axis=0)
    img = torch.from_numpy(img)

    return img
