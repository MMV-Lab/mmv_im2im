import numpy as np
from typing import Union
from numba import jit
from scipy.ndimage.measurements import find_objects
from scipy.ndimage.morphology import binary_fill_holes
from bioio.writers import OmeTiffWriter
from bioio import BioImage
from tqdm import tqdm
from pathlib import Path
import warnings
from torch import from_numpy
from monai.data.meta_tensor import MetaTensor

from mmv_im2im.utils.misc import generate_dataset_dict


@jit(nopython=True)
def pairwise_python(X):
    M = X.shape[0]
    N = X.shape[1]
    D = np.empty((M, M), dtype=np.float32)
    for i in range(M):
        for j in range(M):
            d = 0.0
            for k in range(N):
                tmp = X[i, k] - X[j, k]
                d += tmp * tmp
            D[i, j] = np.sqrt(d)
    return D


def fill_label_holes(lbl_img, **kwargs):
    """Fill small holes in label image."""

    def grow(sl, interior):
        return tuple(
            slice(s.start - int(w[0]), s.stop + int(w[1])) for s, w in zip(sl, interior)
        )

    def shrink(interior):
        return tuple(slice(int(w[0]), (-1 if w[1] else None)) for w in interior)

    objects = find_objects(lbl_img)
    lbl_img_filled = np.zeros_like(lbl_img)
    for i, sl in enumerate(objects, 1):
        if sl is None:
            continue
        interior = [(s.start > 0, s.stop < sz) for s, sz in zip(sl, lbl_img.shape)]
        shrink_slice = shrink(interior)
        grown_mask = lbl_img[grow(sl, interior)] == i
        mask_filled = binary_fill_holes(grown_mask, **kwargs)[shrink_slice]
        lbl_img_filled[sl][mask_filled] = i
    return lbl_img_filled


def generate_center_image_2d(instance, center, ids):
    """
    Generates a `center_image` which is one (True) for all center locations and
    zero (False) otherwise.

    Parameters
    ----------
    instance: numpy array
        `instance` image containing unique `ids` for each object (YX)
         or present in a one-hot encoded style where each object is one in it
         own slice and zero elsewhere.
    center: string
        One of 'centroid', 'approximate-medoid' or 'medoid'.
    ids: list
        Unique ids corresponding to the objects present in the instance image.
    one_hot: boolean
        True (in this case, `instance` has shape DYX) or False (in this case,
        `instance` has shape YX).
    """

    center_image = np.zeros(instance.shape, dtype=bool)
    for j, id in enumerate(ids):
        y, x = np.where(instance == id)
        if len(y) != 0 and len(x) != 0:
            if center == "centroid":
                ym, xm = np.mean(y), np.mean(x)
            elif center == "approximate-medoid":
                ym_temp, xm_temp = np.median(y), np.median(x)
                imin = np.argmin((x - xm_temp) ** 2 + (y - ym_temp) ** 2)
                ym, xm = y[imin], x[imin]
            elif center == "medoid":
                dist_matrix = pairwise_python(np.vstack((x, y)).transpose())
                imin = np.argmin(np.sum(dist_matrix, axis=0))
                ym, xm = y[imin], x[imin]
            else:
                raise NotImplementedError("error in center method")
            center_image[int(np.round(ym)), int(np.round(xm))] = True
    return center_image


def generate_center_image_3d(instance, center, ids, anisotropy_factor, speed_up):
    center_image = np.zeros(instance.shape, dtype=bool)
    instance_downsampled = instance[
        :, :: int(speed_up), :: int(speed_up)
    ]  # down sample in x and y
    for j, id in enumerate(ids):
        z, y, x = np.where(instance_downsampled == id)
        if len(y) != 0 and len(x) != 0:
            if center == "centroid":
                zm, ym, xm = np.mean(z), np.mean(y), np.mean(x)
            elif center == "approximate-medoid":
                zm_temp, ym_temp, xm_temp = np.median(z), np.median(y), np.median(x)
                imin = np.argmin(
                    (x - xm_temp) ** 2
                    + (y - ym_temp) ** 2
                    + (anisotropy_factor * (z - zm_temp)) ** 2
                )
                zm, ym, xm = z[imin], y[imin], x[imin]
            elif center == "medoid":
                dist_matrix = pairwise_python(
                    np.vstack(
                        (speed_up * x, speed_up * y, anisotropy_factor * z)
                    ).transpose()
                )
                imin = np.argmin(np.sum(dist_matrix, axis=0))
                zm, ym, xm = z[imin], y[imin], x[imin]
            center_image[
                int(np.round(zm)),
                int(np.round(speed_up * ym)),
                int(np.round(speed_up * xm)),
            ] = True
    return center_image


def generate_center_image(instance, center, ids, anisotropy_factor=1, speed_up=1):
    if len(instance.shape) == 3:
        return generate_center_image_3d(
            instance, center, ids, anisotropy_factor, speed_up
        )
    elif len(instance.shape) == 2:
        return generate_center_image_2d(
            instance,
            center,
            ids,
        )
    else:
        raise ValueError("instance image must be either 2D or 3D")


def prepare_embedseg_cache(
    data_path: Union[str, Path], cache_path: Union[str, Path], data_cfg
):
    data_path = Path(data_path)
    cache_path = Path(cache_path)
    dataset_list = generate_dataset_dict(data_path)
    patch_size = None

    # parse the method for centroid computation
    if data_cfg.extra is not None:
        if "center_method" in data_cfg.extra:
            center_method = data_cfg.extra["center_method"]
        if "patch_size" in data_cfg.extra:
            patch_size = data_cfg.extra["patch_size"]

    # get some basic statistics from the data and do some basic validation
    min_xy = 65535
    min_z = 65535
    spatial_dim = 2
    for ds in dataset_list:
        fn = ds["source_fn"]
        reader = BioImage(fn)
        this_minXY = min(reader.dims.X, reader.dims.Y)
        min_xy = min((this_minXY, min_xy))
        if reader.dims.Z > 1 and spatial_dim == 2:
            spatial_dim = 3
        min_z = min((reader.dims.Z, min_z))
        assert this_minXY >= 128, "{fn}: XY dimension smaller than 128, not good"
        if spatial_dim == 3:
            assert reader.dims.Z >= 16, "{fn} has less than 16 Z slices, not good"

    if patch_size is None:
        crop_size = 128 * (min_xy // 128)
        if spatial_dim == 3:
            crop_size_z = min((32, min_z))
            new_patch_size = [crop_size_z, crop_size, crop_size]
        else:
            new_patch_size = [crop_size, crop_size]
        warnings.warn(
            UserWarning(
                f"A patch_size is determined from data. MAKE SURE to set data.patch_size as {new_patch_size}."  # noqa E501
            )
        )

    else:
        spatial_dim = len(patch_size)
        if spatial_dim == 3:
            crop_size_z = patch_size[0]
            crop_size = min(patch_size[1:])
        else:
            crop_size = min(patch_size)

    if spatial_dim == 3:
        reader_params = {"dimension_order_out": "ZYX", "C": 0, "T": 0}
        raw_reader_params = {"dimension_order_out": "CZYX", "T": 0}
    else:
        reader_params = {"dimension_order_out": "YX", "C": 0, "T": 0, "Z": 0}
        raw_reader_params = {"dimension_order_out": "CYX", "T": 0, "Z": 0}

    # loop through the dataset
    for ds in tqdm(dataset_list):
        # get instance segmentation labels
        instance_reader = BioImage(ds["target_fn"])
        instance = instance_reader.get_image_data(**reader_params)

        # get raw image
        image_reader = BioImage(ds["source_fn"])
        image = image_reader.get_image_data(**raw_reader_params)

        # check if costmap exists
        fn_base = ds["source_fn"].stem[:-3]
        cm_fn = ds["source_fn"].parent / f"{fn_base}_CM.tiff"
        costmap_flag = False
        if cm_fn.is_file():
            costmap_flag = True
            cm_reader = BioImage(cm_fn)
            costmap = cm_reader.get_image_data(**reader_params)

        # parse filename
        fn_base = Path(ds["source_fn"]).stem[:-3]  # get rid of "_IM"

        # get all obejcts
        instance_np = np.array(instance, copy=False)
        object_mask = instance_np > 0

        ids = np.unique(instance_np[object_mask])
        ids = ids[ids != 0]

        # loop over instances
        for j, id in enumerate(ids):
            if spatial_dim == 2:
                h, w = instance.shape
                y, x = np.where(instance_np == id)
                ym, xm = np.mean(y), np.mean(x)

                jj = int(np.clip(ym - crop_size / 2, 0, h - crop_size))
                ii = int(np.clip(xm - crop_size / 2, 0, w - crop_size))

                # only crop patches away from image borders
                if instance[jj : jj + crop_size, ii : ii + crop_size].shape == (
                    crop_size,
                    crop_size,
                ):
                    im_crop = image[:, jj : jj + crop_size, ii : ii + crop_size]
                    instance_crop = instance[jj : jj + crop_size, ii : ii + crop_size]
                    center_image_crop = generate_center_image(
                        instance_crop, center_method, ids
                    )
                    class_image_crop = object_mask[
                        jj : jj + crop_size, ii : ii + crop_size
                    ]
                    if costmap_flag:
                        costmap_crop = costmap[jj : jj + crop_size, ii : ii + crop_size]
                    dim_order = "YX"
                else:
                    continue

            elif spatial_dim == 3:
                d, h, w = instance.shape
                z, y, x = np.where(instance_np == id)
                zm, ym, xm = np.mean(z), np.mean(y), np.mean(x)
                kk = int(np.clip(zm - crop_size_z / 2, 0, d - crop_size_z))
                jj = int(np.clip(ym - crop_size / 2, 0, h - crop_size))
                ii = int(np.clip(xm - crop_size / 2, 0, w - crop_size))

                if instance[
                    kk : kk + crop_size_z, jj : jj + crop_size, ii : ii + crop_size
                ].shape == (crop_size_z, crop_size, crop_size):
                    im_crop = image[
                        :,
                        kk : kk + crop_size_z,
                        jj : jj + crop_size,
                        ii : ii + crop_size,
                    ]
                    instance_crop = instance[
                        kk : kk + crop_size_z,
                        jj : jj + crop_size,
                        ii : ii + crop_size,
                    ]
                    center_image_crop = generate_center_image(
                        instance_crop,
                        center_method,
                        ids,
                        anisotropy_factor=1,
                        speed_up=1,
                    )
                    class_image_crop = object_mask[
                        kk : kk + crop_size_z,
                        jj : jj + crop_size,
                        ii : ii + crop_size,
                    ]
                    if costmap_flag:
                        costmap_crop = costmap[
                            kk : kk + crop_size_z,
                            jj : jj + crop_size,
                            ii : ii + crop_size,
                        ]
                    dim_order = "ZYX"
                else:
                    continue

            else:
                raise ValueError(
                    "error in spatial dimension when preparing embedseg dataset"
                )

            if im_crop.shape[0] == 1:
                OmeTiffWriter.save(
                    im_crop[0,],
                    cache_path / f"{fn_base}_{j:04d}_IM.tiff",
                    dim_order=dim_order,
                )
            else:
                OmeTiffWriter.save(
                    im_crop,
                    cache_path / f"{fn_base}_{j:04d}_IM.tiff",
                    dim_order="C" + dim_order,
                )
            OmeTiffWriter.save(
                instance_crop.astype(np.uint16),
                cache_path / f"{fn_base}_{j:04d}_GT.tiff",
                dim_order=dim_order,
            )
            OmeTiffWriter.save(
                center_image_crop.astype(np.uint8),
                cache_path / f"{fn_base}_{j:04d}_CE.tiff",
                dim_order=dim_order,
            )
            OmeTiffWriter.save(
                class_image_crop.astype(np.uint8),
                cache_path / f"{fn_base}_{j:04d}_CL.tiff",
                dim_order=dim_order,
            )
            if costmap_flag:
                OmeTiffWriter.save(
                    costmap_crop.astype(float),
                    cache_path / f"{fn_base}_{j:04d}_CM.tiff",
                    dim_order=dim_order,
                )


def prepare_embedseg_tensor(
    instance_batch: MetaTensor,
    spatial_dim: int,
    center_method: str = "centroid",
):
    """
    Parameters:
    ------------
        instance: instance segmentation masks of shape BYX or BZYX
        spatial_dim: 2 or 3
        crop_size: values from cropping of shape YX or ZYX

    Return:
    ----------
        class_labels: MetaTensor
        center_images: MetaTensor
    """

    assert instance_batch.is_batch, "only batch is supported currently"
    num_samples = instance_batch.size()[0]

    center_images_list = []
    class_labels_list = []
    for sample_idx in range(num_samples):
        instance = instance_batch[sample_idx].detach().cpu().numpy()

        assert instance.shape[0] == 1, "ground truth has more than 1 channel"
        instance = np.squeeze(instance, axis=0)
        instance_np = np.array(instance, copy=False)

        class_image = instance_np > 0
        ids = np.unique(instance_np[class_image])
        ids = ids[ids != 0]
        center_image = generate_center_image(instance, center_method, ids)

        class_image = np.expand_dims(class_image, axis=0)
        class_labels_list.append(class_image.astype(np.ubyte))
        center_image = np.expand_dims(center_image, axis=0)
        center_images_list.append(center_image.astype(bool))

    # stack into batch and covert to MetaTensor
    center_images = from_numpy(np.stack(center_images_list, axis=0))
    class_labels = from_numpy(np.stack(class_labels_list, axis=0))

    center_images = MetaTensor(center_images)
    class_labels = MetaTensor(class_labels)

    # move to proer device
    if instance_batch.is_cuda:
        current_device = instance_batch.get_device()
        center_images = center_images.to(current_device)
        class_labels = class_labels.to(current_device)

    return class_labels, center_images
