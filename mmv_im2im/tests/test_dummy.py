#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A simple example of a test file using a function.
NOTE: All test file names must have one of the two forms.
- `test_<XYY>.py`
- '<XYZ>_test.py'

Docs: https://docs.pytest.org/en/latest/
      https://docs.pytest.org/en/latest/goodpractices.html#conventions-for-python-test-discovery
"""

import pytest
import torch
import torch.nn.functional as F
import numpy as np
from mmv_im2im.utils.connectivity_loss import ConnectivityCoherenceLoss
from mmv_im2im.utils.elbo_loss import ELBOLoss
from mmv_im2im.utils.fractal_layers import FractalDimension
from mmv_im2im.utils.gdl_regularized import RegularizedGeneralizedDiceFocalLoss
from mmv_im2im.utils.hausdorff_loss import HausdorffLoss
from mmv_im2im.utils.homology_loss import HomologyLoss
from mmv_im2im.utils.topological_complexity_loss import TopologicalComplexityLoss
from mmv_im2im.utils.topological_loss import TI_Loss
from mmv_im2im.utils.urcentainity_extractor import perturb_image


# --- Test Data Factories ---
def get_data(spatial_dims=2, batch_size=1, n_classes=2, size=16):
    shape = (batch_size, n_classes) + (size,) * spatial_dims
    target_shape = (batch_size,) + (size,) * spatial_dims

    logits = torch.randn(shape, requires_grad=True)
    target_indices = torch.randint(0, n_classes, target_shape).long()
    return logits, target_indices


# --- Tests ---


@pytest.mark.parametrize("dims", [2, 3])
def test_connectivity_loss(dims):
    n_classes = 2
    loss_fn = ConnectivityCoherenceLoss(
        spatial_dims=dims, num_classes=n_classes, ignore_background=False
    )
    logits, target_indices = get_data(spatial_dims=dims, n_classes=n_classes)
    pred_softmax = F.softmax(logits, dim=1)
    target_one_hot = F.one_hot(target_indices, num_classes=n_classes).float()
    if dims == 2:
        target_one_hot = target_one_hot.permute(0, 3, 1, 2)
    else:
        target_one_hot = target_one_hot.permute(0, 4, 1, 2, 3)

    loss = loss_fn(pred_softmax, target_one_hot)
    assert not torch.isnan(loss)
    loss.backward()


def test_elbo_full_regularization():
    n_classes = 2
    latent_dim = 10
    loss_fn = ELBOLoss(
        n_classes=n_classes,
        spatial_dims=2,
        use_fractal_regularization=True,
        use_connectivity_regularization=False,
        use_hausdorff_regularization=True,
        use_homology_regularization=True,
        use_topological_complexity=True,
    )

    logits, target = get_data(spatial_dims=2, n_classes=n_classes, batch_size=1)
    mu = torch.randn(1, latent_dim, requires_grad=True)
    logvar = torch.randn(1, latent_dim, requires_grad=True)

    loss = loss_fn(
        logits=logits,
        y_true=target,
        epoch=100,
        prior_mu=mu,
        prior_logvar=logvar,
        post_mu=mu,
        post_logvar=logvar,
    )

    if loss.numel() > 1:
        loss = loss.mean()
    assert loss.item() >= 0
    loss.backward()


def test_ti_loss_forward():
    loss_fn = TI_Loss(dim=2, connectivity=4, inclusion=[[1, 0]], exclusion=[])
    logits = torch.randn(1, 2, 16, 16, requires_grad=True).double()
    target = torch.randint(0, 2, (1, 1, 16, 16)).double()
    loss = loss_fn(logits, target)
    assert loss.item() >= 0
    loss.backward()


def test_fractal_layers_2d():
    fractal_fn = FractalDimension(num_kernels=2, spatial_dims=2)
    x = torch.rand((1, 1, 16, 16), requires_grad=True)
    fd = fractal_fn(x)
    assert fd.numel() >= 1
    fd.mean().backward()


def test_hausdorff_loss():
    loss_fn = HausdorffLoss(spatial_dims=2)
    logits, target = get_data(spatial_dims=2)
    loss = loss_fn(logits, target)
    assert loss >= 0


def test_homology_loss():
    loss_fn = HomologyLoss(spatial_dims=2, resolution=(4, 4))
    logits, target = get_data(spatial_dims=2, size=8)
    loss = loss_fn(logits, target)
    assert loss >= 0


def test_topological_complexity():
    loss_fn = TopologicalComplexityLoss(spatial_dims=2, metric="mse", k_top=5)
    logits, target = get_data(spatial_dims=2, size=8)
    loss = loss_fn(logits, target)
    assert loss >= 0


def test_perturbations():
    img_2d = np.random.rand(1, 16, 16).astype(np.float32)
    perturbed = perturb_image(img_2d, opts="all_defaults")
    assert perturbed.shape == img_2d.shape


def test_regularized_gdl():
    loss_fn = RegularizedGeneralizedDiceFocalLoss(
        n_classes=2, spatial_dims=2, use_fractal_regularization=True
    )
    logits, target = get_data(spatial_dims=2)
    loss = loss_fn(logits, target, epoch=50)
    assert loss.item() > 0
