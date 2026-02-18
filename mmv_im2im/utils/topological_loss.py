"""
This code was a direct implemetation for the work describes in the paper

@inproceedings{gupta2022learning,
  title={Learning Topological Interactions for Multi-Class Medical Image Segmentation},
  author={Gupta, Saumya and Hu, Xiaoling and Kaan, James and Jin, Michael and Mpoy, Mutshipay and Chung, Katherine and Singh, Gagandeep and Saltz, Mary and Kurc, Tahsin and Saltz, Joel and others},
  booktitle={Computer Vision--ECCV 2022: 17th European Conference, Tel Aviv, Israel, October 23--27, 2022, Proceedings, Part XXIX},
  pages={701--718},
  year={2022},
  organization={Springer}
}

https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136890691.pdf

and provided by the github repo:  https://github.com/TopoXLab/TopoInteraction
"""

import numpy as np
import torch
import torch.nn.functional as F


class TI_Loss(torch.nn.Module):
    def __init__(self, dim, connectivity, inclusion, exclusion, min_thick=1):
        """
        :param dim: 2 if 2D; 3 if 3D
        :param connectivity: 4 or 8 for 2D; 6 or 26 for 3D
        """
        super(TI_Loss, self).__init__()

        self.dim = dim
        self.connectivity = connectivity
        self.min_thick = min_thick
        self.interaction_list = []

        # Define operations based on dimension
        if self.dim == 2:
            self.sum_dim_list = [1, 2, 3]  # (B, C, H, W) -> sum over C, H, W
            self.conv_op = F.conv2d
        elif self.dim == 3:
            self.sum_dim_list = [1, 2, 3, 4]  # (B, C, D, H, W) -> sum over C, D, H, W
            self.conv_op = F.conv3d
        else:
            raise ValueError(f"Dimension {dim} not supported. Use 2 or 3.")

        self.apply_nonlin = lambda x: F.softmax(x, 1)
        self.ce_loss_func = torch.nn.CrossEntropyLoss(reduction="none")

        self.set_kernel()

        for inc in inclusion:
            temp_pair = []
            temp_pair.append(True)  # type inclusion
            temp_pair.append(inc[0])
            temp_pair.append(inc[1])
            self.interaction_list.append(temp_pair)

        for exc in exclusion:
            temp_pair = []
            temp_pair.append(False)  # type exclusion
            temp_pair.append(exc[0])
            temp_pair.append(exc[1])
            self.interaction_list.append(temp_pair)

    def set_kernel(self):
        k = 2 * self.min_thick + 1
        np_kernel = None

        if self.dim == 2:
            if self.connectivity == 4:
                np_kernel = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])
            elif self.connectivity == 8:
                np_kernel = np.ones((k, k))
            # Shape for Conv2d: (Out=1, In=1, H, W)
            if np_kernel is not None:
                self.kernel = torch.from_numpy(np_kernel).unsqueeze(0).unsqueeze(0)

        elif self.dim == 3:
            if self.connectivity == 6:
                # 6-connectivity structure
                np_kernel = np.zeros((3, 3, 3))
                # Center
                np_kernel[1, 1, 1] = 1
                # Neighbors
                np_kernel[0, 1, 1] = 1
                np_kernel[2, 1, 1] = 1  # Z neighbors
                np_kernel[1, 0, 1] = 1
                np_kernel[1, 2, 1] = 1  # Y neighbors
                np_kernel[1, 1, 0] = 1
                np_kernel[1, 1, 2] = 1  # X neighbors
            elif self.connectivity == 26:
                np_kernel = np.ones((k, k, k))

            # Shape for Conv3d: (Out=1, In=1, D, H, W)
            if np_kernel is not None:
                self.kernel = torch.from_numpy(np_kernel).unsqueeze(0).unsqueeze(0)

        if np_kernel is None:
            raise ValueError(
                f"Invalid connectivity {self.connectivity} for dim {self.dim}"
            )

    def topological_interaction_module(self, P):
        """
        P: Discrete segmentation map (B, 1, H, W) or (B, 1, D, H, W)
        """
        # Ensure kernel is on the same device/dtype as input P
        if self.kernel.device != P.device:
            self.kernel = self.kernel.to(P.device)

        critical_voxels_map = torch.zeros_like(P, dtype=torch.double)

        for ind, interaction in enumerate(self.interaction_list):
            interaction_type = interaction[0]
            label_A = interaction[1]
            label_C = interaction[2]

            mask_A = torch.where(P == label_A, 1.0, 0.0).double()
            if interaction_type:
                mask_C = torch.where(P == label_C, 1.0, 0.0).double()
                mask_C = torch.logical_or(mask_C, mask_A).double()
                mask_C = torch.logical_not(mask_C).double()
            else:
                mask_C = torch.where(P == label_C, 1.0, 0.0).double()

            # Apply Convolution (2D or 3D handled by self.conv_op)
            neighbourhood_C = self.conv_op(mask_C, self.kernel.double(), padding="same")
            neighbourhood_C = torch.where(neighbourhood_C >= 1.0, 1.0, 0.0)

            neighbourhood_A = self.conv_op(mask_A, self.kernel.double(), padding="same")
            neighbourhood_A = torch.where(neighbourhood_A >= 1.0, 1.0, 0.0)

            violating_A = neighbourhood_C * mask_A
            violating_C = neighbourhood_A * mask_C
            violating = violating_A + violating_C
            violating = torch.where(violating >= 1.0, 1.0, 0.0)

            critical_voxels_map = torch.logical_or(
                critical_voxels_map, violating
            ).double()

        return critical_voxels_map

    def forward(self, x, y):
        """
        x: Logits (B, C, H, W) or (B, C, D, H, W)
        y: GT (B, 1, H, W) or (B, 1, D, H, W)
        """
        if x.device.type == "cuda":
            # Ensure kernel follows device even if initialized on CPU
            if self.kernel.device != x.device:
                self.kernel = self.kernel.to(x.device)

        # Obtain discrete segmentation map
        x_softmax = self.apply_nonlin(x)
        P = torch.argmax(x_softmax, dim=1)
        # P is now (B, H, W) or (B, D, H, W) -> unsqueeze to restore channel
        P = torch.unsqueeze(P.double(), dim=1)
        del x_softmax

        critical_voxels_map = self.topological_interaction_module(P)

        # Compute TI loss
        # CE Loss returns (B, spatial...), we unsqueeze to (B, 1, spatial...) to match map
        ce_tensor = torch.unsqueeze(
            self.ce_loss_func(x.double(), y[:, 0].long()), dim=1
        )

        # Mask the CE loss with critical voxels
        ce_tensor = ce_tensor * critical_voxels_map

        # Sum over spatial dims and mean over batch
        ce_loss_value = ce_tensor.sum(dim=self.sum_dim_list).mean()

        return ce_loss_value
