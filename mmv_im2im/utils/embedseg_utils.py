import os
import numpy as np
from numba import jit
from scipy.ndimage.measurements import find_objects
from scipy.ndimage.morphology import binary_fill_holes
from aicsimageio.writers import OmeTiffWriter
from aicsimageio import AICSImage
from tqdm import tqdm

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


def prepare_embedseg_cache(data_path, cache_path, data_cfg):

    dataset_list = generate_dataset_dict(data_path)

    min_xy = 65535
    min_z = 65535
    for ds in dataset_list:
        fn = ds["source_fn"]
        reader = AICSImage(fn)
        this_minXY = min(reader.dims.X, reader.dims.Y)
        min_xy = min((this_minXY, min_xy))
        min_z = min((reader.dims.Z, min_z))
        assert this_minXY >= 128, "{fn}: XY dimension smaller than 128, not good"
    crop_size = 128 * (min_xy // 128)

    if data_cfg.spatial_dim == 3:
        assert min_z >= 16, "some 3D data has less than 16 Z slices, not good"
        crop_size_z = min((32, min_z))

    for ds in tqdm(dataset_list):
        instance, _ = data_cfg.target_reader(ds["target_fn"])
        image, _ = data_cfg.source_reader(ds["source_fn"])
        fn_base = str(os.path.basename(ds["target_fn"]))
        if fn_base.endswith("GT.tiff"):
            fn_base = fn_base[:-7]
        else:
            fn_base = os.path.splitext(fn_base)[0]

        instance_np = np.array(instance, copy=False)
        object_mask = instance_np > 0

        ids = np.unique(instance_np[object_mask])
        ids = ids[ids != 0]

        # loop over instances
        for j, id in enumerate(ids):
            if data_cfg.spatial_dim == 2:
                h, w = image.shape
                y, x = np.where(instance_np == id)
                ym, xm = np.mean(y), np.mean(x)

                jj = int(np.clip(ym - crop_size / 2, 0, h - crop_size))
                ii = int(np.clip(xm - crop_size / 2, 0, w - crop_size))

                if image[jj : jj + crop_size, ii : ii + crop_size].shape == (
                    crop_size,
                    crop_size,
                ):
                    im_crop = image[jj : jj + crop_size, ii : ii + crop_size]
                    instance_crop = instance[
                        jj : jj + crop_size, ii : ii + crop_size
                    ]
                    center_image_crop = generate_center_image(
                        instance_crop, "centroid", ids
                    )
                    class_image_crop = object_mask[
                        jj : jj + crop_size, ii : ii + crop_size
                    ]
                    dim_order = "YX"

            elif data_cfg.spatial_dim == 3:
                d, h, w = image.shape
                z, y, x = np.where(instance_np == id)
                zm, ym, xm = np.mean(z), np.mean(y), np.mean(x)
                kk = int(np.clip(zm - crop_size_z / 2, 0, d - crop_size_z))
                jj = int(np.clip(ym - crop_size / 2, 0, h - crop_size))
                ii = int(np.clip(xm - crop_size / 2, 0, w - crop_size))

                if image[
                    kk : kk + crop_size_z, jj : jj + crop_size, ii : ii + crop_size
                ].shape == (crop_size_z, crop_size, crop_size):
                    im_crop = image[
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
                        "centroid",
                        ids,
                        anisotropy_factor=1,
                        speed_up=1,
                    )
                    class_image_crop = object_mask[
                        kk : kk + crop_size_z,
                        jj : jj + crop_size,
                        ii : ii + crop_size,
                    ]
                    dim_order = "ZYX"

            OmeTiffWriter.save(
                im_crop,
                cache_path + os.sep + fn_base + f"_{j:04d}_IM.tiff",
                dim_order=dim_order,
            )
            OmeTiffWriter.save(
                instance_crop.astype(np.uint16),
                cache_path + os.sep + fn_base + f"_{j:04d}_GT.tiff",
                dim_order=dim_order,
            )
            OmeTiffWriter.save(
                center_image_crop.astype(np.uint8),
                cache_path + os.sep + fn_base + f"_{j:04d}_CE.tiff",
                dim_order=dim_order,
            )
            OmeTiffWriter.save(
                class_image_crop.astype(np.uint8),
                cache_path + os.sep + fn_base + f"_{j:04d}_CL.tiff",
                dim_order=dim_order,
            )
