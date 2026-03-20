import lightning as pl
import torch
import torch.nn as nn
from mmv_im2im.utils.misc import (
    parse_config,
    parse_config_func,
    parse_config_func_without_params,
)
from mmv_im2im.utils.model_utils import init_weights


class Model(pl.LightningModule):
    def __init__(self, model_info_xx, train=True, verbose=False):
        super().__init__()

        if isinstance(model_info_xx.net["params"], dict):
            self.task = model_info_xx.net["params"].get("task", "segmentation")

        if self.task not in ("regression", "segmentation"):
            raise ValueError(
                f"Task should be 'regression' or 'segmentation'; got '{self.task}'"
            )

        if "utils.elbo_loss" in model_info_xx.criterion["module_name"]:
            model_info_xx.criterion["params"]["task"] = self.task

        self.net = parse_config(model_info_xx.net)
        init_weights(self.net, init_type="kaiming")

        self.model_info = model_info_xx
        self.verbose = verbose

        # ── Adaptive global average pooling for regression ─────────────
        # ProbUnet's decoder output is [B, C, *spatial] regardless of
        # whether the task is segmentation or regression.  For regression
        # we need to collapse the spatial dims before computing the loss
        # against the flat GT vector.
        self._gap_2d = nn.AdaptiveAvgPool2d(1)
        self._gap_3d = nn.AdaptiveAvgPool3d(1)
        self._logged_shapes = False  # verbose print: once, any task
        self._shape_checked = False  # shape validation: once, regression only

        if train:
            self.criterion = parse_config(model_info_xx.criterion)
            self.optimizer_func = parse_config_func(model_info_xx.optimizer)

    # ------------------------------------------------------------------
    def configure_optimizers(self):
        optimizer = self.optimizer_func(self.parameters())
        if self.model_info.scheduler is None:
            return optimizer
        scheduler_func = parse_config_func_without_params(self.model_info.scheduler)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler_func(
                    optimizer, **self.model_info.scheduler["params"]
                ),
                "monitor": "val_loss",
            },
        }

    # ------------------------------------------------------------------
    def forward(self, x, seg=None, train_posterior=False):
        return self.net(x, seg, train_posterior)

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
    def run_step(self, batch):
        x, y = batch["IM"], batch["GT"]

        # ── Remove trailing singleton (badly formatted data) ───────────
        if x.ndim > 4 and x.shape[-1] == 1:
            x = x.squeeze(-1)

        # ── Prepare GT depending on task ──────────────────────────────
        if self.task == "regression":
            # GT is a flat vector: ensure [B, N]
            if y.ndim > 2 and y.shape[-1] == 1:
                y = y.squeeze(-1)
            y = y.view(y.size(0), -1).float()

        else:  # segmentation
            if y.ndim > 4 and y.shape[-1] == 1:
                y = y.squeeze(-1)
            # Ensure channel dim present: [B, Z, Y, X] → [B, 1, Z, Y, X]
            if y.ndim == x.ndim - 1:
                y = y.unsqueeze(1)

        # ── Forward (Train Posterior) ─────────────────────────────────
        output = self(x, seg=y, train_posterior=True)

        # ── Spatial → flat para regression predictions ────────────────
        pred = output["pred"]  # [B, C, *spatial] from the decoder

        # Verbose: imprime shapes una sola vez, independiente del task
        if self.verbose and not self._logged_shapes:
            print(
                f"[pl_ProbUnet] x.shape={tuple(x.shape)}  "
                f"pred.shape (pre-GAP)={tuple(pred.shape)}"
            )
            self._logged_shapes = True

        if self.task == "regression":
            # AdaptiveAvgPool: [B, C, *spatial] → [B, C]
            pred = self._global_average_pool(pred)

            # One-time shape sanity check
            if not self._shape_checked:
                C_pred, N_gt = pred.shape[1], y.shape[1]
                if C_pred != N_gt:
                    raise ValueError(
                        f"[pl_ProbUnet] Shape mismatch: decoder predicts {C_pred} "
                        f"channels but GT vector has {N_gt} elements. "
                        f"Set out_channels: {N_gt} in your YAML."
                    )
                if self.verbose:
                    print(
                        f"[pl_ProbUnet] Regression shapes OK  "
                        f"pred={tuple(pred.shape)}  y={tuple(y.shape)}"
                    )
                self._shape_checked = True

        # ── ELBO Loss ─────────────────────────────────────────────────
        current_ep = int(self.current_epoch)
        loss = self.criterion(
            logits=pred,
            y_true=y,
            prior_mu=output["prior_mu"],
            prior_logvar=output["prior_logvar"],
            post_mu=output["mu_post"],
            post_logvar=output["logvar_post"],
            epoch=current_ep,
        )
        return loss

    # ------------------------------------------------------------------
    def on_train_epoch_end(self):
        torch.cuda.synchronize()

    def on_validation_epoch_end(self):
        torch.cuda.synchronize()

    def training_step(self, batch, batch_idx):
        loss = self.run_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.run_step(batch)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        return loss
