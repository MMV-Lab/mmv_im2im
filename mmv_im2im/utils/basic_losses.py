########################################################################
# the code was adapted from https://github.com/wolny/pytorch-3dunet
# with MIT license
########################################################################
from typing import List
import torch
from torch import nn


def expand_as_one_hot(input, C, ignore_index=None):
    """
    Converts NxSPATIAL label image to NxCxSPATIAL, where each label gets converted to
    its corresponding one-hot vector. It is assumed that the batch dimension is present.
    Args:
        input (torch.Tensor): 3D/4D input image
        C (int): number of channels/labels
        ignore_index (int): ignore index to be kept during the expansion
    Returns:
        4D/5D output torch.Tensor (NxCxSPATIAL)
    """
    assert (
        input.dim() == 4 or input.dim() == 3
    ), "only support 3D/4D (i.e., 2D/3D images)"

    # expand the input tensor to Nx1xSPATIAL before scattering
    input = input.unsqueeze(1)
    # create output tensor shape (NxCxSPATIAL)
    shape = list(input.size())
    shape[1] = C

    if ignore_index is not None:
        # create ignore_index mask for the result
        mask = input.expand(shape) == ignore_index
        # clone the src tensor and zero out ignore_index in the input
        input = input.clone()
        input[input == ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input.device).scatter_(1, input, 1)
        # bring back the ignore_index in the result
        result[mask] = ignore_index
        return result
    else:
        # scatter to get the one-hot tensor
        return torch.zeros(shape).to(input.device).scatter_(1, input, 1)


class PixelWiseCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights: List = None, ignore_index=None, one_hot_gt=False):
        super(PixelWiseCrossEntropyLoss, self).__init__()
        if class_weights is None:
            self.class_weights = None
        else:
            self.class_weights = torch.FloatTensor(class_weights)
        self.ignore_index = ignore_index
        self.log_softmax = nn.LogSoftmax(dim=1)
        self.one_hot = one_hot_gt

    def forward(self, input, target, weights=None):
        # input: N x C x Spatial
        # target: N x 1 x Spatial (not one hot) or N x C x Spatial (one_hot)
        # weights: N x 1 x Spatial

        if weights is None:
            weights = torch.ones_like(target)

        # normalize the input
        log_probabilities = self.log_softmax(input)

        # convert to one hot
        if not self.one_hot:
            # need to remove the dummy C dimension before converting to one hot
            target = target.squeeze(1)
            target = expand_as_one_hot(
                target, C=input.size()[1], ignore_index=self.ignore_index
            )
        # expand weights
        weights = weights.expand_as(input)

        # create default class_weights if None
        if self.class_weights is None:
            class_weights = torch.ones(input.size()[1]).float().to(input.device)
        else:
            class_weights = self.class_weights.to(input.device)

        # resize class_weights to be broadcastable into the weights
        class_weights = class_weights.view(1, -1, 1, 1, 1)

        # multiply weights tensor by class weights
        weights = class_weights * weights

        # compute the losses
        result = -weights * target * log_probabilities
        # average the losses
        return result.mean()
