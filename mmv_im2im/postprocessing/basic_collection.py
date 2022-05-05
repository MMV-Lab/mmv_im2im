from typing import Union
from importlib import import_module
import numpy as np
import torch


def extract_segmentation(
    im: Union[np.ndarray, torch.Tensor], channel: int, cutoff: Union[float, str] = None
) -> np.ndarray:
    """extract segmentation from a prediction

    Parameters:
    -------------
    im: ndarray or torch.Tensor
        the multi-class prediction (1, C, W, H) or (1, C, Z, Y, X)
    channel: int
        which channel to select
    cutoff: float or str
        either a fixed cutoff value or a segmentation method from skimage, default is
        None (do not apply any cutoff)
    """

    # convert tensor to numpy
    if torch.is_tensor(im):
        im = im.cpu().numpy()
    assert len(im.shape) == 4 or len(im.shape) == 5, "extract seg only accepts 4D/5D"
    assert im.shape[0] == 1, "extract seg requires first dim to be 1"

    prob = im[0, channel, :]
    if cutoff is None:
        return prob
    else:
        if isinstance(cutoff, str):
            th_module = import_module("skimage.filters")
            try:
                th_func = getattr(th_module, "threshold_" + cutoff.lower())
            except Exception:
                raise ValueError("unsupported threhsold method")
            seg = th_func(prob)
        elif isinstance(cutoff, float):
            seg = prob > cutoff
        else:
            raise NotImplementedError("cutoff method only str or float")

        seg = seg.astype(np.uint8)
        seg[seg > 0] = 255

        return seg
