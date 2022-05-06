import numpy as np
from typing import Union
import torch

from mmv_im2im.utils.embedseg_utils import Cluster, Cluster_3d


def generate_instance_clusters(
    pred: Union[np.ndarray, torch.Tensor],
    grid_x: int = 1024,
    grid_y: int = 1024,
    pixel_x: int = 1,
    pixel_y: int = 1,
    avg_bg: int = 0,
    n_sigma: int = 2,
    seed_thresh: float = 0.5,
    min_mask_sum: int = 100,
    min_unclustered_sum: int = 100,
    min_object_size: int = 100,
    grid_z: int = 32,
    pixel_z: int = 1
):
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
