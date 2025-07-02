# This file was adapted from https://github.com/AllenCellModeling/pytorch_fnet/

from typing import Optional, Tuple, List, Union
import numpy as np
from torch.nn import functional as F
import torch


def norm_around_center(img, z_center: Optional[int] = None, min_z: Optional[int] = 32):
    """Returns normalized version of input img.
    img will be normalized with respect to the mean, std pixel intensity
    of the sub-array of length 32 in the z-dimension centered around the
    img's "z_center".
    Parameters
    ----------
    img
        Input 4D torch.Tensor or numpy array to be normalized.
    z_center
        Z-index of cell centers.
    Returns
    -------
    4D numpy array or tensor
       Normalized image
    """

    # #TODO: Currently, I am not 100 percent sure about how F.pad deals with
    # tensor and ndarray. From the documentation, it seems like only tensors
    # are supported. But, it also works on some kind of ndarray (not all),
    # very strange. Need to follow up to have a better understanding

    if not isinstance(img, np.ndarray):
        img_array = img.numpy()
    else:
        img_array = img.copy()
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
        Input 4D torch.Tensor or numpy array to be padded, C x X x Y x Z
    multiple_base:
        int or a sequence of int

    Returns
    -------
    4D numpy array or tensor
        padded image
    """

    # #TODO: Currently, I am not 100 percent sure about how F.pad deals with
    # tensor and ndarray. From the documentation, it seems like only tensors
    # are supported. But, it also works on some kind of ndarray (not all),
    # very strange. Need to follow up to have a better understanding

    if not isinstance(img, np.ndarray):
        img_array = img.numpy()
    else:
        img_array = img.copy()

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
            diff_x - diff_x // 2,
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

    if isinstance(pad_value, str) and pad_value == "mean":
        avg_bg = img_array.mean()
        return F.pad(img, pad_shape, "constant", avg_bg)
    else:
        return F.pad(img, pad_shape, pad_value)


def pad_z(img, target_size: int = 64, pad_value: Union[str, float, int] = 0):
    """Returns padded image.
    img will be padded along z to the size of "target_size".

    Parameters
    ----------
    img:
        Input 4D torch.Tensor or numpy array to be padded, C x Z x Y x X
    target_size:
        int

    Returns
    -------
    4D numpy array or tensor
       padded image
    """

    # #TODO: Currently, I am not 100 percent sure about how F.pad deals with
    # tensor and ndarray. From the documentation, it seems like only tensors
    # are supported. But, it also works on some kind of ndarray (not all),
    # very strange. Need to follow up to have a better understanding

    if not isinstance(img, np.ndarray):
        img_array = img.numpy()
    else:
        img_array = img.copy()

    if img_array.shape[1] < target_size:
        diff_z = target_size - img_array.shape[1]
        pad_shape = (
            0,
            0,
            0,
            0,
            diff_z // 2,
            diff_z - diff_z // 2,
        )
        if isinstance(pad_value, str):
            if pad_value == "mean":
                avg_bg = img_array.mean()
                return F.pad(img, pad_shape, "constant", avg_bg)
            else:
                return F.pad(img, pad_shape, pad_value)
        else:
            print(type(img))
            return F.pad(img, pad_shape, "constant", pad_value)
    else:
        return img


def normalize_staining(
    img, Io=240, alpha=1, beta=0.15, return_unmix_results: bool = False
):
    """Normalize staining appearence of H&E stained images

    Input:
        I: RGB input image (we assume channel order CYX), numpy array or torch.tensor
        Io: (optional) transmitted light intensity
        alpha: parameter from paper (see Reference)
        beta: parameter from paper (see Reference)
        return_unmix_results: whether to return unmixed H and E channel for debugging

    Output:
        Inorm: normalized image
        H: (optional, only when return_unmix_results=True) hematoxylin image
        E: (optional, only when return_unmix_results=True) eosin image

    Reference:
        A method for normalizing histology slides for quantitative analysis. M.
        Macenko et al., ISBI 2009
    """

    HERef = np.array([[0.5626, 0.2159], [0.7201, 0.8012], [0.4062, 0.5581]])

    maxCRef = np.array([1.9705, 1.0308])

    back_to_tensor = False
    original_data_type = None
    if torch.is_tensor(img):
        back_to_tensor = True
        img = img.cpu().numpy()
        original_data_type = img.dtype

    # define height and width of image and move the color dimenstion to the last
    c, h, w = img.shape
    img = np.moveaxis(img, 0, -1)

    assert c == 3, "the first dimension of the image must be 3 (RGB)"

    # reshape image
    img = img.reshape((-1, 3))

    # calculate optical density
    OD = -np.log((img.astype(float) + 1) / Io)

    # remove transparent pixels
    ODhat = OD[~np.any(OD < beta, axis=1)]

    # compute eigenvectors
    eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))

    # eigvecs *= -1

    # project on the plane spanned by the eigenvectors corresponding to the two
    # largest eigenvalues
    That = ODhat.dot(eigvecs[:, 1:3])

    phi = np.arctan2(That[:, 1], That[:, 0])

    minPhi = np.percentile(phi, alpha)
    maxPhi = np.percentile(phi, 100 - alpha)

    vMin = eigvecs[:, 1:3].dot(np.array([(np.cos(minPhi), np.sin(minPhi))]).T)
    vMax = eigvecs[:, 1:3].dot(np.array([(np.cos(maxPhi), np.sin(maxPhi))]).T)

    # a heuristic to make the vector corresponding to hematoxylin first and the
    # one corresponding to eosin second
    if vMin[0] > vMax[0]:
        HE = np.array((vMin[:, 0], vMax[:, 0])).T
    else:
        HE = np.array((vMax[:, 0], vMin[:, 0])).T

    # rows correspond to channels (RGB), columns to OD values
    Y = np.reshape(OD, (-1, 3)).T

    # determine concentrations of the individual stains
    C = np.linalg.lstsq(HE, Y, rcond=None)[0]

    # normalize stain concentrations
    maxC = np.array([np.percentile(C[0, :], 99), np.percentile(C[1, :], 99)])
    tmp = np.divide(maxC, maxCRef)
    C2 = np.divide(C, tmp[:, np.newaxis])

    # recreate the image using reference mixing matrix
    Inorm = np.multiply(Io, np.exp(-HERef.dot(C2)))
    Inorm[Inorm > 255] = 254
    Inorm = np.reshape(Inorm.T, (h, w, 3)).astype(np.uint8)

    # unmix hematoxylin and eosin
    H = np.multiply(
        Io,
        np.exp(
            np.expand_dims(-HERef[:, 0], axis=1).dot(np.expand_dims(C2[0, :], axis=0))
        ),
    )
    H[H > 255] = 254
    H = np.reshape(H.T, (h, w, 3)).astype(np.uint8)

    E = np.multiply(
        Io,
        np.exp(
            np.expand_dims(-HERef[:, 1], axis=1).dot(np.expand_dims(C2[1, :], axis=0))
        ),
    )
    E[E > 255] = 254
    E = np.reshape(E.T, (h, w, 3)).astype(np.uint8)

    # move the color dimension to the first to be consistent with others transforms
    Inorm = np.moveaxis(Inorm, -1, 0)

    if back_to_tensor:
        Inorm = torch.tensor(Inorm.astype(original_data_type))

    if return_unmix_results:
        return Inorm, H, E
    else:
        return Inorm
