from typing import Union
from importlib import import_module
import numpy as np
import torch


def extract_segmentation(
    im: Union[np.ndarray, torch.Tensor],
    channel: int,
    cutoff: Union[float, str] = None,
    batch_dim: bool = True,
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
    batch_dim: bool
        whether there is a batch dimension (default is True)
    """

    # convert tensor to numpy
    if torch.is_tensor(im):
        im = im.cpu().numpy()
    if batch_dim:
        assert (
            len(im.shape) == 4 or len(im.shape) == 5
        ), "extract seg with batch_dim only accepts 4D/5D"
        assert im.shape[0] == 1, "extract seg with batch_dim requires first dim to be 1"
        prob = im[0, channel, :]
    else:
        assert (
            len(im.shape) == 3 or len(im.shape) == 4
        ), "extract seg without batch_dim only accepts 3D/4D"
        prob = im[channel, :]

    if cutoff is None:
        return prob
    else:
        if isinstance(cutoff, str):
            th_module = import_module("skimage.filters")
            try:
                th_func = getattr(th_module, "threshold_" + cutoff.lower())
            except Exception:
                raise ValueError("unsupported threhsold method")
            cutoff = th_func(prob)
        elif isinstance(cutoff, float):
            # just to confirm the type
            pass
        else:
            raise NotImplementedError("cutoff method only str or float")

        seg = prob > cutoff
        seg = seg.astype(np.uint8)
        seg[seg > 0] = 255

        return seg


def generate_classmap(im: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """generate the segmentation classmap from model prediction

    Parameters:
    -------------
    im: ndarray or torch.Tensor
        the multi-class prediction (1, C, W, H) or (1, C, Z, Y, X)
    """

    # convert tensor to numpy
    if torch.is_tensor(im):
        im = im.cpu().numpy()
    assert len(im.shape) == 4 or len(im.shape) == 5, "extract seg only accepts 4D/5D"
    assert im.shape[0] == 1, "extract seg requires first dim to be 1"

    classmap = np.argmax(im, axis=1).astype(np.uint8)
    return classmap
