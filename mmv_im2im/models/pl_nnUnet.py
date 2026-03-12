from typing import Dict
import lightning as pl
import torch
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

        if self.task != "regression" and self.task != "segmentation":
            raise ValueError(
                f"Task should be regression/segmentation : {self.task} was given"
            )

        self.net = parse_config(model_info_xx.net)

        init_weights(self.net, init_type="kaiming")

        self.model_info = model_info_xx
        self.verbose = verbose
        self.weighted_loss = False
        self.seg_flag = False
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

    def configure_optimizers(self):
        optimizer = self.optimizer_func(self.parameters())
        if self.model_info.scheduler is None:
            return optimizer
        else:
            scheduler_func = parse_config_func_without_params(self.model_info.scheduler)
            lr_scheduler = scheduler_func(
                optimizer, **self.model_info.scheduler["params"]
            )
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

    def run_step(self, batch, validation_stage):
        x = batch["IM"]
        y = batch["GT"]

        if self.weighted_loss:
            assert (
                "CM" in batch.keys()
            ), "Costmap is detected, but no use_costmap param in criterion"
            cm = batch["CM"]

        if x.size()[-1] == 1 and x.ndim > (self.net.spatial_dims + 2):
            x = torch.squeeze(x, dim=-1)
            y = torch.squeeze(y, dim=-1)
            if cm is not None:
                cm = torch.squeeze(cm, dim=-1)

        y_hat = self(x)

        # Handle potential MONAI DynUNet deep supervision outputs safely
        if torch.is_tensor(y_hat) and y_hat.ndim == x.ndim + 1:
            # DynUNet interpolates and stacks intermediate predictions along dim=1.
            # We unbind and take the primary full-resolution output (index 0).
            y_hat = y_hat[:, 0, ...]
        elif isinstance(y_hat, (list, tuple)):
            y_hat = y_hat[0]

        if self.task == "regression":
            # Global Average Pooling  [B, C, H, W] -> [B, C]
            y_hat = y_hat.view(y_hat.size(0), y_hat.size(1), -1).mean(dim=-1)
            y = y.view(y.size(0), -1).float()
        else:
            if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
                y = torch.squeeze(y, dim=1)

        if isinstance(self.criterion, regularized):
            current_epoch = self.current_epoch
            loss = self.criterion(y_hat, y, epoch=current_epoch)
        else:
            if self.weighted_loss:
                loss = self.criterion(y_hat, y, cm)
            else:
                loss = self.criterion(y_hat, y)

        return loss

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
