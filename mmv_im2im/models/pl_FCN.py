from typing import Dict

import lightning as pl
import torch
import torch.nn as nn
from mmv_im2im.utils.gdl_regularized import (
    RegularizedGeneralizedDiceFocalLoss as regularized,
)
from mmv_im2im.utils.misc import (
    parse_config,
    parse_config_func,
    parse_config_func_without_params,
)
from mmv_im2im.utils.model_utils import init_weights


class Model(pl.LightningModule):
    def __init__(self, model_info_xx: Dict, train: bool = True, verbose: bool = False):
        super().__init__()

        if isinstance(model_info_xx.net["params"], dict):
            self.task = model_info_xx.net["params"].pop("task", "segmentation")

        if self.task not in ("regression", "segmentation"):
            raise ValueError(
                f"Task should be 'regression' or 'segmentation'; got '{self.task}'"
            )

        self.net = parse_config(model_info_xx.net)
        init_weights(self.net, init_type="kaiming")

        self.model_info = model_info_xx
        self.verbose = verbose
        self.weighted_loss = False
        self.seg_flag = False

        # ── Regression: adaptive global average pooling ────────────────
        # Works for ANY spatial size (2-D or 3-D) and is more robust than
        # the manual .view(...).mean(-1) idiom.
        # AdaptiveAvgPool reduces every spatial dimension to size 1, after
        # which squeeze() gives a flat [B, C] tensor.
        self._gap_2d = nn.AdaptiveAvgPool2d(1)
        self._gap_3d = nn.AdaptiveAvgPool3d(1)

        # _logged_shapes  → verbose print fires exactly once (any task)
        # _shape_checked  → regression shape validation fires exactly once
        self._logged_shapes = False
        self._shape_checked = False

        if train:
            if "use_costmap" in model_info_xx.criterion[
                "params"
            ] and model_info_xx.criterion["params"].pop("use_costmap"):
                self.weighted_loss = True
            self.criterion = parse_config(model_info_xx.criterion)
            self.optimizer_func = parse_config_func(model_info_xx.optimizer)

            if (
                model_info_xx.model_extra is not None
                and "debug_segmentation" in model_info_xx.model_extra
                and model_info_xx.model_extra["debug_segmentation"]
            ):
                self.seg_flag = True

    # ------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = self.optimizer_func(self.parameters())
        if self.model_info.scheduler is None:
            return optimizer
        scheduler_func = parse_config_func_without_params(self.model_info.scheduler)
        lr_scheduler = scheduler_func(optimizer, **self.model_info.scheduler["params"])
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
                "strict": True,
            },
        }

    # ------------------------------------------------------------------
    def forward(self, x):
        return self.net(x)

    # ------------------------------------------------------------------
    def _global_average_pool(self, y_hat: torch.Tensor) -> torch.Tensor:
        """
        Reduce [B, C, *spatial] → [B, C] via global average pooling.

        Handles both 2-D [B, C, H, W] and 3-D [B, C, D, H, W] tensors
        of arbitrary spatial size.
        """
        ndim = y_hat.dim()
        if ndim == 4:  # 2-D input
            return self._gap_2d(y_hat).squeeze(-1).squeeze(-1)
        elif ndim == 5:  # 3-D input
            return self._gap_3d(y_hat).squeeze(-1).squeeze(-1).squeeze(-1)
        else:
            # Fallback: manual mean over all spatial dims
            return y_hat.view(y_hat.size(0), y_hat.size(1), -1).mean(dim=-1)

    # ------------------------------------------------------------------
    def run_step(self, batch, validation_stage: bool):
        x = batch["IM"]  # [B, 1, Z, Y, X] – spatial dims vary per batch
        y = batch["GT"]  # [B, 3 + n_coeffs]

        if "CM" in batch:
            assert (
                self.weighted_loss
            ), "Costmap detected in batch but use_costmap=False in criterion config."
            cm = batch["CM"]

        # ── Remove accidental trailing singleton (badly formatted data) ──
        if x.dim() > 4 and x.size(-1) == 1:
            x = x.squeeze(-1)
        if y.dim() > 2 and y.size(-1) == 1:
            y = y.squeeze(-1)

        # ── Forward pass ──────────────────────────────────────────────
        y_hat = self(x)  # [B, C, Z', Y', X']  (spatial may still vary)

        # Verbose: imprime shapes una sola vez al inicio, independiente del task
        if self.verbose and not self._logged_shapes:
            print(
                f"[pl_FCN] x.shape={tuple(x.shape)}  "
                f"y_hat.shape (pre-GAP)={tuple(y_hat.shape)}"
            )
            self._logged_shapes = True

        # ── Task-specific output processing ────────────────────────────
        if self.task == "regression":
            # Global Average Pool: [B, C, *spatial] → [B, C]
            y_hat = self._global_average_pool(y_hat)

            # GT: ensure [B, N]
            y = y.view(y.size(0), -1).float()

            # One-time shape sanity check
            if not self._shape_checked:
                C_pred = y_hat.shape[1]
                N_gt = y.shape[1]
                if C_pred != N_gt:
                    raise ValueError(
                        f"[pl_FCN] Shape mismatch: network predicts {C_pred} "
                        f"values (out_channels={C_pred}) but GT vector has "
                        f"{N_gt} elements.  "
                        f"Set out_channels: {N_gt} in your YAML."
                    )
                if self.verbose:
                    print(
                        f"[pl_FCN] Regression shapes OK  "
                        f"y_hat={tuple(y_hat.shape)}  y={tuple(y.shape)}"
                    )
                self._shape_checked = True

        else:  # segmentation
            if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
                y = y.squeeze(dim=1)

        # ── Loss ─────────────────────────────────────────────────────
        if isinstance(self.criterion, regularized):
            loss = self.criterion(y_hat, y, epoch=self.current_epoch)
        elif self.weighted_loss:
            loss = self.criterion(y_hat, y, cm)
        else:
            loss = self.criterion(y_hat, y)

        return loss

    # ------------------------------------------------------------------
    def on_train_epoch_end(self):
        torch.cuda.synchronize()

    def on_validation_epoch_end(self):
        torch.cuda.synchronize()

    def training_step(self, batch, batch_idx):
        loss = self.run_step(batch, validation_stage=False)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.run_step(batch, validation_stage=True)
        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return loss
