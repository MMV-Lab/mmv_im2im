#####################################################################
# This script was adapted from aics-ml-segmentation and pytorch-fnet
#####################################################################
from scipy.signal import triang
from typing import List
import numpy as np
import torch
from typing import Sequence, Tuple


def _get_weights(shape: Sequence[int]) -> Tuple[np.ndarray, Tuple[int]]:
    """Get triangular weights

    Parameters
    ----------
    shape: numpy arraay of shape CZYX

    Returns
    -----------
    np.ndarray
        numpy array of shape 1ZYX representing the weights of each pixel
    """
    shape_in = shape
    shape = shape[1:]
    weights = 1
    for idx_d in range(len(shape)):
        slicey = [np.newaxis] * len(shape)
        slicey[idx_d] = slice(None)
        size = shape[idx_d]
        weights = weights * triang(size)[tuple(slicey)]
    return weights, shape_in


def _predict_piecewise_recurse(
    predictor,
    ar_in: torch.tensor,
    dims_max: List[int],
    overlaps: List[int],
    mode: str = "fast",
    **predict_kwargs,
):
    """Performs piecewise prediction recursively. (See `predict_piecewise`)"""
    if tuple(ar_in.shape[1:]) == tuple(dims_max[1:]):
        ar_in = torch.unsqueeze(ar_in, dim=0)
        ar_out = predictor.forward(ar_in, **predict_kwargs)
        if isinstance(ar_out, list):
            ar_out = ar_out[0]
        ar_out = torch.squeeze(
            ar_out, dim=0
        )  # remove N dimension so that multichannel outputs can be used
        if mode != "fast":
            ar_out = ar_out.detach().cpu()
        weights, shape_in = _get_weights(ar_out.shape)
        weights = torch.as_tensor(
            weights, dtype=ar_out.dtype, device=ar_out.device
        )  # noqa E501
        ar_weight = torch.broadcast_to(weights, shape_in)
        return ar_out * ar_weight, weights
    dim = None
    # Find first dim where input > max
    for idx_d in range(1, ar_in.ndim):
        if ar_in.shape[idx_d] > dims_max[idx_d]:
            dim = idx_d
            break
    # Size of channel dim is unknown until after first prediction
    shape_out = [None] + list(ar_in.shape[1:])
    ar_out = None
    ar_weight = None
    offset = 0
    done = False
    while not done:
        slices = [slice(None)] * len(ar_in.shape)
        end = offset + dims_max[dim]
        slices[dim] = slice(offset, end)
        slices = tuple(slices)
        ar_in_sub = ar_in[slices]
        pred_sub, pred_weight_sub = _predict_piecewise_recurse(
            predictor, ar_in_sub, dims_max, overlaps, mode=mode, **predict_kwargs
        )
        if ar_out is None or ar_weight is None:
            shape_out[0] = pred_sub.shape[0]  # Set channel dim for output
            ar_out = torch.zeros(
                shape_out, dtype=pred_sub.dtype, device=pred_sub.device
            )
            ar_weight = torch.zeros(
                shape_out[1:], dtype=pred_weight_sub.dtype, device=pred_sub.device
            )
        ar_out[slices] += pred_sub
        ar_weight[slices[1:]] += pred_weight_sub
        offset += dims_max[dim] - overlaps[dim]
        if end == ar_in.shape[dim]:
            done = True
        elif offset + dims_max[dim] > ar_in.shape[dim]:
            offset = ar_in.shape[dim] - dims_max[dim]
    return ar_out, ar_weight


def predict_piecewise(
    predictor,
    tensor_in: torch.Tensor,
    dims_max: List[int] = 64,
    overlaps: List[int] = 0,
    mode: str = "fast",
    **predict_kwargs,
) -> torch.Tensor:
    """Performs piecewise prediction and combines results.
    Parameters
    ----------
    predictor: Callable Object
        An object with a forward() method.
    tensor_in: torch.Tensor
        Tensor to be input into predictor piecewise. Should be 3d or 4d
        with the first dimension representing channel dimension. For 2D
        images, the tensor_in should be of shape C x Y x X, and for 3D
        images, the tensor_in should be of shape C x Z x Y x X.
    dims_max: List[int]
        Specifies dimensions of each sub prediction. No need to include
        channel dimension. For 3D images, the dims_max should be of shape
        [Z, Y, X], while for 2D images, the dims_max should be of shape
        [Y, X]
    overlaps: List[int]
        Specifies overlap along each dimension for sub predictions. No need
        to include channel dimension. For 3D images, the dims_max should be
        of shape [Z, Y, X], while for 2D images, the dims_max should be of
        shape [Y, X]
    mode: strss
        "fast" or "efficient". "fast" mode will use more RAM
    **predict_kwargs
        Kwargs to pass to predict method.
    Returns
    -------
    torch.Tensor
         Prediction with size tensor_in.size().
    """
    assert isinstance(tensor_in, torch.Tensor)

    # the input tensor needs to be 3D or 4D
    assert len(tensor_in.size()) > 2
    shape_in = tuple(tensor_in.size())
    n_dim = len(shape_in)
    assert len(dims_max) == len(overlaps) == n_dim

    # if the size of certain dimension of input tensor is smaller
    # than the size of that dimension in sub prediction, then
    # reduce the size of sub prediction
    for idx_d in range(1, n_dim):
        if dims_max[idx_d] > shape_in[idx_d]:
            dims_max[idx_d] = shape_in[idx_d]

    # Remove restrictions on channel dimension.
    dims_max[0] = None
    overlaps[0] = None
    ar_out, ar_weight = _predict_piecewise_recurse(
        predictor,
        tensor_in,
        dims_max=dims_max,
        overlaps=overlaps,
        mode=mode,
        **predict_kwargs,
    )

    weight_corrected = torch.unsqueeze(ar_out / ar_weight, dim=0)
    if mode != "fast":
        weight_corrected = weight_corrected.float()
    return weight_corrected
