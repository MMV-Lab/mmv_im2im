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
        if "CM" in batch.keys():
            assert (
                self.weighted_loss
            ), "Costmap is detected, but no use_costmap param in criterion"
            cm = batch["CM"]

        # only for badly formated data file
        if x.size()[-1] == 1:
            x = torch.squeeze(x, dim=-1)
            y = torch.squeeze(y, dim=-1)

        y_hat = self(x)

        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            # in case of CrossEntropy related error
            # see: https://discuss.pytorch.org/t/runtimeerror-expected-object-of-scalar-type-long-but-got-scalar-type-float-when-using-crossentropyloss/30542  # noqa E501
            y = torch.squeeze(y, dim=1)  # remove C dimension

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
