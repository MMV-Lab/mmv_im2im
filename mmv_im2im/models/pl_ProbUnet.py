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

        self.model_info = model_info_xx
        self.verbose = verbose
        self.task = "segmentation"

        if "params" in model_info_xx.net and isinstance(
            model_info_xx.net["params"], dict
        ):
            self.task = model_info_xx.net["params"].get("task", "segmentation")

        if self.task not in ("regression", "segmentation"):
            raise ValueError(
                f"Task should be 'regression' or 'segmentation'; got '{self.task}'"
            )

        self.net = parse_config(model_info_xx.net)
        init_weights(self.net, init_type="kaiming")

        self._gap_2d = nn.AdaptiveAvgPool2d(1)
        self._gap_3d = nn.AdaptiveAvgPool3d(1)
        self._logged_shapes = False
        self._shape_checked = False

        if hasattr(model_info_xx, "criterion") and model_info_xx.criterion is not None:
            if (
                "params" not in model_info_xx.criterion
                or model_info_xx.criterion["params"] is None
            ):
                model_info_xx.criterion["params"] = {}
            if (
                "module_name" in model_info_xx.criterion
                and "utils.elbo_loss" in model_info_xx.criterion["module_name"]
            ):
                model_info_xx.criterion["params"]["task"] = self.task
            self.criterion = parse_config(self.model_info.criterion)
        else:
            self.criterion = None

        if hasattr(model_info_xx, "optimizer") and model_info_xx.optimizer is not None:
            self.optimizer_func = parse_config_func(model_info_xx.optimizer)
        else:
            self.optimizer_func = None

    def configure_optimizers(self):
        if self.optimizer_func is None:
            raise RuntimeError("Optimizer configuration is missing in the YAML.")

        optimizer = self.optimizer_func(self.parameters())
        if (
            not hasattr(self.model_info, "scheduler")
            or self.model_info.scheduler is None
        ):
            return optimizer

        scheduler_func = parse_config_func_without_params(self.model_info.scheduler)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler_func(
                    optimizer, **self.model_info.scheduler.get("params", {})
                ),
                "monitor": "val_loss",
            },
        }

    def forward(self, x, seg=None, train_posterior=False):
        return self.net(x, seg, train_posterior)

    def _global_average_pool(self, y_hat: torch.Tensor) -> torch.Tensor:
        ndim = y_hat.dim()
        if ndim == 4:
            return self._gap_2d(y_hat).squeeze(-1).squeeze(-1)
        elif ndim == 5:
            return self._gap_3d(y_hat).squeeze(-1).squeeze(-1).squeeze(-1)
        else:
            return y_hat.view(y_hat.size(0), y_hat.size(1), -1).mean(dim=-1)

    def run_step(self, batch):
        if self.criterion is None:
            raise RuntimeError(
                "Criterion is not initialized. Ensure 'criterion' is defined in your training YAML."
            )

        x, y = batch["IM"], batch["GT"]

        if x.ndim > 4 and x.shape[-1] == 1:
            x = x.squeeze(-1)

        if self.task == "regression":
            if y.ndim > 2 and y.shape[-1] == 1:
                y = y.squeeze(-1)
            y = y.view(y.size(0), -1).float()
        else:
            if y.ndim > 4 and y.shape[-1] == 1:
                y = y.squeeze(-1)
            if y.ndim == x.ndim - 1:
                y = y.unsqueeze(1)

        output = self(x, seg=y, train_posterior=True)
        pred = output["pred"]

        if self.verbose and not self._logged_shapes:
            print(
                f"[pl_ProbUnet] x.shape={tuple(x.shape)}  "
                f"pred.shape (pre-GAP)={tuple(pred.shape)}"
            )
            self._logged_shapes = True

        if self.task == "regression":
            pred = self._global_average_pool(pred)

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
