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

        # ── Adaptive global average pooling for regression ─────────────
        # Handles ANY spatial size, both 2-D and 3-D.
        self._gap_2d = nn.AdaptiveAvgPool2d(1)
        self._gap_3d = nn.AdaptiveAvgPool3d(1)
        self._logged_shapes = False  # verbose print: once, any task
        self._shape_checked = False  # shape validation: once, regression only

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

    def prepare_batch(self, batch):
        return

    def forward(self, x):
        return self.net(x)

    # ------------------------------------------------------------------
    def _global_average_pool(self, y_hat: torch.Tensor) -> torch.Tensor:
        """[B, C, *spatial] → [B, C] via adaptive global average pooling."""
        ndim = y_hat.dim()
        if ndim == 4:
            return self._gap_2d(y_hat).squeeze(-1).squeeze(-1)
        elif ndim == 5:
            return self._gap_3d(y_hat).squeeze(-1).squeeze(-1).squeeze(-1)
        else:
            return y_hat.view(y_hat.size(0), y_hat.size(1), -1).mean(dim=-1)

    # ------------------------------------------------------------------
    def run_step(self, batch, validation_stage):
        x = batch["IM"]
        y = batch["GT"]

        # FIX: initialise cm=None so it is always defined in scope
        cm = None
        if self.weighted_loss:
            assert "CM" in batch, (
                "weighted_loss=True but 'CM' key not found in batch. "
                "Check use_costmap setting in criterion config."
            )
            cm = batch["CM"]

        # Remove accidental trailing singleton dimension
        # (guard: only squeeze if the tensor really has an extra dim)
        spatial_dims = getattr(self.net, "spatial_dims", x.ndim - 2)
        if x.ndim > (spatial_dims + 2) and x.size(-1) == 1:
            x = torch.squeeze(x, dim=-1)
            y = torch.squeeze(y, dim=-1)
            # FIX: only squeeze cm if it was actually loaded
            if cm is not None:
                cm = torch.squeeze(cm, dim=-1)

        # ── Forward ───────────────────────────────────────────────────
        y_hat = self(x)

        # Verbose: imprime shapes una sola vez, independiente del task
        if self.verbose and not self._logged_shapes:
            print(
                f"[pl_nnUnet] x.shape={tuple(x.shape)}  "
                f"y_hat.shape (pre-GAP)={tuple(y_hat.shape)}"
            )
            self._logged_shapes = True

        # Handle DynUNet deep supervision: stacks intermediate outputs
        # along a new leading dim → shape [B, n_outputs, C, *spatial]
        if torch.is_tensor(y_hat) and y_hat.ndim == x.ndim + 1:
            y_hat = y_hat[:, 0, ...]  # take full-resolution head
        elif isinstance(y_hat, (list, tuple)):
            y_hat = y_hat[0]

        # ── Task-specific post-processing ─────────────────────────────
        if self.task == "regression":
            # AdaptiveAvgPool: [B, C, *spatial] → [B, C]  (any spatial size)
            y_hat = self._global_average_pool(y_hat)
            y = y.view(y.size(0), -1).float()

            # One-time shape sanity check
            if not self._shape_checked:
                C_pred, N_gt = y_hat.shape[1], y.shape[1]
                if C_pred != N_gt:
                    raise ValueError(
                        f"[pl_nnUnet] Shape mismatch: network predicts {C_pred} "
                        f"values but GT vector has {N_gt} elements. "
                        f"Set out_channels: {N_gt} in your YAML."
                    )
                if self.verbose:
                    print(
                        f"[pl_nnUnet] Regression shapes OK  "
                        f"y_hat={tuple(y_hat.shape)}  y={tuple(y.shape)}"
                    )
                self._shape_checked = True

        else:  # segmentation
            if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
                y = torch.squeeze(y, dim=1)

        # ── Loss ──────────────────────────────────────────────────────
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
