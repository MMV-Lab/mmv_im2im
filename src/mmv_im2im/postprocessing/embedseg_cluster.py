import numpy as np
from typing import Union
import torch


def degrid(meter, grid_size, pixel_size):
    return int(meter * (grid_size - 1) / pixel_size + 1)


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
        min_mask_sum=5,
        min_unclustered_sum=0,
        min_object_size=5,
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
                        instance_mask[mask.squeeze().cpu()] = (
                            proposal.short().cpu()
                        )  # TODO
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
        min_mask_sum=64,
        min_unclustered_sum=0,
        min_object_size=3,
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
        if mask.sum() > min_mask_sum:
            # top level decision: only start creating instances, if there are at least
            # min_mask_sum pixels in foreground!

            spatial_emb_masked = spatial_emb[mask.expand_as(spatial_emb)].view(
                n_sigma, -1
            )
            sigma_masked = sigma[mask.expand_as(sigma)].view(n_sigma, -1)
            seed_map_masked = seed_map[mask].view(1, -1)

            unclustered = torch.ones(mask.sum()).short().cuda()
            instance_map_masked = torch.zeros(mask.sum()).short().cuda()

            while (
                unclustered.sum() > min_unclustered_sum
            ):  # stop when the seed candidates are less than min_clustered_sum
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


def generate_instance_clusters(
    pred: Union[np.ndarray, torch.Tensor],
    grid_x: int = 768,
    grid_y: int = 768,
    pixel_x: int = 1,
    pixel_y: int = 1,
    n_sigma: int = 2,
    seed_thresh: float = 0.5,
    min_mask_sum: int = 10,
    min_unclustered_sum: int = 10,
    min_object_size: int = 10,
    grid_z: int = 32,
    pixel_z: int = 1,
):
    ##########################################################
    # Warninging: Even passing in a mini-batch, only the first
    # one will be used.
    ##########################################################
    if not torch.is_tensor(pred):
        pred = torch.from_numpy(pred)

    if len(pred.shape) == 4:  # B x C x W x H
        cluster = Cluster_2d(grid_y, grid_x, pixel_y, pixel_x)
    elif len(pred.shape) == 5:  # B x C x Z x Y x X
        cluster = Cluster_3d(grid_z, grid_y, grid_x, pixel_z, pixel_y, pixel_x)
    else:
        raise ValueError("prediction needs to be 4D or 5D in cluster")

    instance_map, _ = cluster.cluster(
        pred[0],  # get rid of batch dimension
        n_sigma=n_sigma,
        seed_thresh=seed_thresh,
        min_mask_sum=min_mask_sum,
        min_unclustered_sum=min_unclustered_sum,
        min_object_size=min_object_size,
    )

    return instance_map.cpu().numpy()
