import numpy as np
from numba import jit
import torch
from scipy.ndimage.measurements import find_objects
from scipy.ndimage.morphology import binary_fill_holes


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


def degrid(meter, grid_size, pixel_size):
    return int(meter * (grid_size - 1) / pixel_size + 1)


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
                ### option - 3 (`numba`)
                dist_matrix = pairwise_python(np.vstack((x, y)).transpose())
                imin = np.argmin(np.sum(dist_matrix, axis=0))
                ym, xm = y[imin], x[imin]
            else:
                raise NotImplemented("error in center method")
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


class Cluster_2d:
    def __init__(self, grid_y, grid_x, pixel_y, pixel_x):

        xm = torch.linspace(0, pixel_x, grid_x).view(1, 1, -1).expand(1, grid_y, grid_x)
        ym = torch.linspace(0, pixel_y, grid_y).view(1, -1, 1).expand(1, grid_y, grid_x)
        xym = torch.cat((xm, ym), 0)

        self.xym = xym.cuda()
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.pixel_x = pixel_x
        self.pixel_y = pixel_y

    def cluster_with_gt(
        self,
        prediction,
        instance,
        n_sigma=1,
    ):
        height, width = prediction.size(1), prediction.size(2)

        xym_s = self.xym[:, 0:height, 0:width]  # 2 x h x w

        spatial_emb = torch.tanh(prediction[0:2]) + xym_s  # 2 x h x w
        sigma = prediction[2 : 2 + n_sigma]  # n_sigma x h x w

        instance_map = torch.zeros(height, width).short().cuda()

        unique_instances = instance.unique()
        unique_instances = unique_instances[unique_instances != 0]

        for id in unique_instances:
            mask = instance.eq(id).view(1, height, width)

            center = (
                spatial_emb[mask.expand_as(spatial_emb)]
                .view(2, -1)
                .mean(1)
                .view(2, 1, 1)
            )  # 2 x 1 x 1

            s = (
                sigma[mask.expand_as(sigma)]
                .view(n_sigma, -1)
                .mean(1)
                .view(n_sigma, 1, 1)
            )

            s = torch.exp(s * 10)  # n_sigma x 1 x 1 #
            dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb - center, 2) * s, 0))
            proposal = dist > 0.5
            instance_map[proposal] = id.item()  # TODO

        return instance_map

    def cluster(
        self,
        prediction,
        n_sigma=2,
        seed_thresh=0.5,
        min_mask_sum=128,
        min_unclustered_sum=0,
        min_object_size=36,
    ):

        height, width = prediction.size(1), prediction.size(2)
        xym_s = self.xym[:, 0:height, 0:width]

        spatial_emb = torch.tanh(prediction[0:2]) + xym_s  # 2 x h x w

        sigma = prediction[2 : 2 + n_sigma]  # n_sigma x h x w
        seed_map = torch.sigmoid(prediction[2 + n_sigma : 2 + n_sigma + 1])  # 1 x h x w

        instance_map = torch.zeros(height, width).short()
        instances = []  # list

        count = 1
        mask = seed_map > 0.5

        if mask.sum() > min_mask_sum:

            spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(
                n_sigma, -1
            )
            sigma_masked = sigma[mask.expand_as(sigma)].view(n_sigma, -1)
            seed_map_masked = seed_map[mask].view(1, -1)

            unclustered = torch.ones(mask.sum()).short().cuda()
            instance_map_masked = torch.zeros(mask.sum()).short().cuda()

            while unclustered.sum() > min_unclustered_sum:
                seed = (seed_map_masked * unclustered.float()).argmax().item()
                seed_score = (seed_map_masked * unclustered.float()).max().item()
                if seed_score < seed_thresh:
                    break
                center = spatial_emb_masked[:, seed : seed + 1]
                unclustered[seed] = 0

                s = torch.exp(sigma_masked[:, seed : seed + 1] * 10)
                dist = torch.exp(
                    -1 * torch.sum(torch.pow(spatial_emb_masked - center, 2) * s, 0)
                )

                proposal = (dist > 0.5).squeeze()
                if proposal.sum() > min_object_size:
                    if (
                        unclustered[proposal].sum().float() / proposal.sum().float()
                        > 0.5
                    ):
                        instance_map_masked[proposal.squeeze()] = count
                        instance_mask = torch.zeros(height, width).short()
                        instance_mask[
                            mask.squeeze().cpu()
                        ] = proposal.short().cpu()  # TODO
                        center_image = torch.zeros(height, width).short()

                        center[0] = int(
                            degrid(
                                center[0].cpu().detach().numpy(),
                                self.grid_x,
                                self.pixel_x,
                            )
                        )
                        center[1] = int(
                            degrid(
                                center[1].cpu().detach().numpy(),
                                self.grid_y,
                                self.pixel_y,
                            )
                        )
                        center_image[
                            np.clip(int(center[1].item()), 0, height - 1),
                            np.clip(int(center[0].item()), 0, width - 1),
                        ] = True
                        instances.append(
                            {
                                "mask": instance_mask.squeeze() * 255,
                                "score": seed_score,
                                "center-image": center_image,
                            }
                        )
                        count += 1

                unclustered[proposal] = 0

            instance_map[mask.squeeze().cpu()] = instance_map_masked.cpu()

        return instance_map, instances


class Cluster_3d:
    def __init__(
        self, grid_z, grid_y, grid_x, pixel_z, pixel_y, pixel_x, one_hot=False
    ):

        xm = (
            torch.linspace(0, pixel_x, grid_x)
            .view(1, 1, 1, -1)
            .expand(1, grid_z, grid_y, grid_x)
        )
        ym = (
            torch.linspace(0, pixel_y, grid_y)
            .view(1, 1, -1, 1)
            .expand(1, grid_z, grid_y, grid_x)
        )
        zm = (
            torch.linspace(0, pixel_z, grid_z)
            .view(1, -1, 1, 1)
            .expand(1, grid_z, grid_y, grid_x)
        )
        xyzm = torch.cat((xm, ym, zm), 0)

        self.xyzm = xyzm.cuda()
        self.one_hot = one_hot
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_z = grid_z
        self.pixel_x = pixel_x
        self.pixel_y = pixel_y
        self.pixel_z = pixel_z

    def cluster_with_gt(
        self,
        prediction,
        instance,
        n_sigma=1,
    ):

        depth, height, width = (
            prediction.size(1),
            prediction.size(2),
            prediction.size(3),
        )
        xyzm_s = self.xyzm[:, 0:depth, 0:height, 0:width]  # 3 x d x h x w
        spatial_emb = torch.tanh(prediction[0:3]) + xyzm_s  # 3 x d x h x w
        sigma = prediction[3 : 3 + n_sigma]  # n_sigma x h x w

        instance_map = torch.zeros(depth, height, width).short().cuda()
        unique_instances = instance.unique()
        unique_instances = unique_instances[unique_instances != 0]

        for id in unique_instances:
            mask = instance.eq(id).view(1, depth, height, width)
            center = (
                spatial_emb[mask.expand_as(spatial_emb)]
                .view(3, -1)
                .mean(1)
                .view(3, 1, 1, 1)
            )  # 3 x 1 x 1 x 1
            s = (
                sigma[mask.expand_as(sigma)]
                .view(n_sigma, -1)
                .mean(1)
                .view(n_sigma, 1, 1, 1)
            )

            s = torch.exp(s * 10)  # n_sigma x 1 x 1
            dist = torch.exp(-1 * torch.sum(torch.pow(spatial_emb - center, 2) * s, 0))
            proposal = dist > 0.5
            instance_map[proposal] = id.item()  # TODO

        return instance_map

    def cluster(
        self,
        prediction,
        n_sigma=3,
        seed_thresh=0.5,
        min_mask_sum=128,
        min_unclustered_sum=0,
        min_object_size=36,
    ):

        depth, height, width = (
            prediction.size(1),
            prediction.size(2),
            prediction.size(3),
        )
        xyzm_s = self.xyzm[:, 0:depth, 0:height, 0:width]

        spatial_emb = torch.tanh(prediction[0:3]) + xyzm_s  # 3 x d x h x w

        sigma = prediction[3 : 3 + n_sigma]  # n_sigma x d x h x w
        seed_map = torch.sigmoid(
            prediction[3 + n_sigma : 3 + n_sigma + 1]
        )  # 1 x d x h x w
        instance_map = torch.zeros(depth, height, width).short()
        instances = []  # list

        count = 1
        mask = seed_map > 0.5
        if (
            mask.sum() > min_mask_sum
        ):  # top level decision: only start creating instances, if there are atleast 128 pixels in foreground!

            spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(
                n_sigma, -1
            )
            sigma_masked = sigma[mask.expand_as(sigma)].view(n_sigma, -1)
            seed_map_masked = seed_map[mask].view(1, -1)

            unclustered = torch.ones(mask.sum()).short().cuda()
            instance_map_masked = torch.zeros(mask.sum()).short().cuda()

            while (
                unclustered.sum() > min_unclustered_sum
            ):  # stop when the seed candidates are less than 128
                seed = (seed_map_masked * unclustered.float()).argmax().item()
                seed_score = (seed_map_masked * unclustered.float()).max().item()
                if seed_score < seed_thresh:
                    break
                center = spatial_emb_masked[:, seed : seed + 1]
                unclustered[seed] = 0

                s = torch.exp(sigma_masked[:, seed : seed + 1] * 10)
                dist = torch.exp(
                    -1 * torch.sum(torch.pow(spatial_emb_masked - center, 2) * s, 0)
                )

                proposal = (dist > 0.5).squeeze()
                if proposal.sum() > min_object_size:
                    if (
                        unclustered[proposal].sum().float() / proposal.sum().float()
                        > 0.5
                    ):
                        instance_map_masked[proposal.squeeze()] = count
                        instance_mask = torch.zeros(depth, height, width).short()
                        instance_mask[mask.squeeze().cpu()] = proposal.short().cpu()
                        count += 1
                unclustered[proposal] = 0

            instance_map[mask.squeeze().cpu()] = instance_map_masked.cpu()

        return instance_map, instances
