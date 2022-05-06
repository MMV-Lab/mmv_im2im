import numpy as np
from typing import Union
import torch

from mmv_im2im.utils.embedseg_utils import Cluster_2d, Cluster_3d


def generate_instance_clusters(
    pred: Union[np.ndarray, torch.Tensor],
    grid_x: int = 1024,
    grid_y: int = 1024,
    pixel_x: int = 1,
    pixel_y: int = 1,
    n_sigma: int = 2,
    seed_thresh: float = 0.5,
    min_mask_sum: int = 100,
    min_unclustered_sum: int = 100,
    min_object_size: int = 100,
    grid_z: int = 32,
    pixel_z: int = 1
):
    if not torch.is_tensor(pred):
        pred = torch.from_numpy(pred)

    if len(pred.shape) == 4:  # B x C x W x H
        cluster = Cluster_2d(grid_y, grid_x, pixel_y, pixel_x)
    elif len(pred.shape) == 5:  # B x C x Z x Y x X
        cluster = Cluster_3d(grid_z, grid_y, grid_x, pixel_z, pixel_y, pixel_x)

    instance_map, _ = cluster.cluster(
        pred[0],
        n_sigma=n_sigma,
        seed_thresh=seed_thresh,
        min_mask_sum=min_mask_sum,
        min_unclustered_sum=min_unclustered_sum,
        min_object_size=min_object_size,
    )

    from aicsimageio.writers import OmeTiffWriter
    OmeTiffWriter.save("test_pred.tiff", pred)

    return instance_map.cpu().numpy()
