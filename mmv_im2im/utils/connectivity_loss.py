import torch
import torch.nn as nn
import torch.nn.functional as F


class ConnectivityCoherenceLoss(nn.Module):
    """
    Calculates a connectivity coherence loss to penalize fragmentation or
    undesired isolated components within predicted regions.

    This loss encourages predicted regions to be spatially coherent and continuous
    by comparing local neighborhoods in predictions with ground truth.
    It penalizes:
    1. Discontinuities within what should be a single, connected region (e.g., breaking a vein).
    2. Isolated 'islands' of one class within another class.

    Args:
        kernel_size (int): Size of the convolutional kernel for neighborhood analysis (e.g., 3 for 3x3).
        ignore_background (bool): If True, the loss focuses primarily on non-background classes.
                                  Useful if background fragmentation is less critical.
        num_classes (int): Total number of classes, including background.
    """

    def __init__(
        self, kernel_size: int = 3, ignore_background: bool = True, num_classes: int = 2
    ):
        super().__init__()
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be an odd number.")
        self.kernel_size = kernel_size
        self.ignore_background = ignore_background
        self.num_classes = num_classes
        self.average_kernel = torch.ones(1, 1, kernel_size, kernel_size) / (
            kernel_size**2 - 1
        )
        self.center_offset = (kernel_size // 2, kernel_size // 2)

    def forward(self, y_pred_softmax, y_true_one_hot):
        """
        Args:
            y_pred_softmax (torch.Tensor): Softmax probabilities from the model (B, C, H, W).
            y_true_one_hot (torch.Tensor): Ground truth as one-hot encoded tensor (B, C, H, W).
                                           Should be float.

        Returns:
            torch.Tensor: The calculated connectivity coherence loss.
        """
        current_average_kernel = self.average_kernel.to(y_pred_softmax.device)

        y_true_one_hot = y_true_one_hot.float()

        loss_per_class = []

        for c in range(self.num_classes):
            if self.ignore_background and c == 0:
                continue

            true_mask_c = y_true_one_hot[:, c : c + 1, :, :]
            pred_prob_c = y_pred_softmax[:, c : c + 1, :, :]

            padded_true_mask_c = F.pad(
                true_mask_c,
                (
                    self.center_offset[1],
                    self.center_offset[1],
                    self.center_offset[0],
                    self.center_offset[0],
                ),
                mode="replicate",
            )

            neighbor_sum_true = F.conv2d(
                padded_true_mask_c, current_average_kernel, padding=0, groups=1
            ) * (self.kernel_size**2)

            padded_pred_prob_c = F.pad(
                pred_prob_c,
                (
                    self.center_offset[1],
                    self.center_offset[1],
                    self.center_offset[0],
                    self.center_offset[0],
                ),
                mode="replicate",
            )

            pred_neighbor_avg = F.conv2d(
                padded_pred_prob_c, current_average_kernel, padding=0, groups=1
            )

            true_neighbor_avg = neighbor_sum_true / (self.kernel_size**2 - 1)

            loss_b = F.mse_loss(pred_neighbor_avg, true_mask_c, reduction="none")
            loss_c = F.mse_loss(pred_prob_c, true_neighbor_avg, reduction="none")

            class_coherence_loss = torch.mean(loss_b + loss_c)
            loss_per_class.append(class_coherence_loss)

        if not loss_per_class:
            return torch.tensor(0.0, device=y_pred_softmax.device)

        return torch.sum(torch.stack(loss_per_class))
