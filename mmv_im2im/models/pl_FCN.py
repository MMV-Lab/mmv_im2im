import os
import numpy as np
from typing import Dict
from pathlib import Path
import pytorch_lightning as pl
import torch
from monai.losses import MaskedLoss
from aicsimageio.writers import OmeTiffWriter

from mmv_im2im.utils.misc import (
    parse_config,
    parse_config_func,
    parse_config_func_without_params,
)


class Model(pl.LightningModule):
    def __init__(self, model_info_xx: Dict, train: bool = True, verbose: bool = False):
        super().__init__()
        self.net = parse_config(model_info_xx.net)
        # if "sliding_window_params" in model_info_xx:
        #    self.sliding_window = model_info_xx["sliding_window_params"]
        # else:
        #    self.sliding_window = None
        self.model_info = model_info_xx
        self.verbose = verbose
        if train:
            if "costmap" in model_info_xx.criterion and model_info_xx.criterion.pop(
                "costmap"
            ):
                self.criterion = MaskedLoss(parse_config(model_info_xx.criterion))
            else:
                self.criterion = parse_config(model_info_xx.criterion)
            self.optimizer_func = parse_config_func(model_info_xx.optimizer)

    def configure_optimizers(self):
        # https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.configure_optimizers  # noqa E501
        optimizer = self.optimizer_func(self.parameters())
        if self.model_info.scheduler is None:
            return optimizer
        else:
            scheduler_func = parse_config_func_without_params(self.model_info.scheduler)
            lr_scheduler = scheduler_func(
                optimizer, **self.model_info.scheduler["params"]
            )
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def prepare_batch(self, batch):
        return

    def forward(self, x):
        return self.net(x)

    def run_step(self, batch, validation_stage):
        # if "costmap" in batch:
        #    costmap = batch.pop("costmap")
        #    costmap = costmap[tio.DATA]
        # else:
        #    costmap = None

        # x = batch["source"][tio.DATA]
        # y = batch["target"][tio.DATA]
        x = batch["IM"]
        y = batch["GT"]

        if x.size()[-1] == 1:
            x = torch.squeeze(x, dim=-1)
            y = torch.squeeze(y, dim=-1)

        # if validation_stage and self.sliding_window is not None:
        #    y_hat = predict_piecewise(
        #        self,
        #        x[
        #            0,
        #        ],
        #        **self.sliding_window
        #    )
        # else:
        #    y_hat = self(x)

        y_hat = self(x)

        if isinstance(self.criterion, torch.nn.CrossEntropyLoss):
            # in case of CrossEntropy related error
            # see: https://discuss.pytorch.org/t/runtimeerror-expected-object-of-scalar-type-long-but-got-scalar-type-float-when-using-crossentropyloss/30542  # noqa E501
            y = torch.squeeze(y, dim=1)  # remove C dimension

        # if costmap is None:
        #    loss = self.criterion(y_hat, y)
        # else:
        #    loss = self.criterion(y_hat, y, costmap)

        loss = self.criterion(y_hat, y)

        return loss, y_hat

    def training_step(self, batch, batch_idx):
        loss, y_hat = self.run_step(batch, validation_stage=False)
        self.log("train_loss", loss, prog_bar=True)

        if self.verbose and batch_idx == 0:
            src = batch["IM"]  # batch["source"][tio.DATA]
            tar = batch["GT"]  # batch["target"][tio.DATA]

            # check if the log path exists, if not create one
            Path(self.trainer.log_dir).mkdir(parents=True, exist_ok=True)

            src_out = np.squeeze(src[0,].detach().cpu().numpy()).astype(np.float)
            tar_out = np.squeeze(tar[0,].detach().cpu().numpy()).astype(np.float)
            prd_out = np.squeeze(y_hat[0,].detach().cpu().numpy()).astype(np.float)

            if len(src_out.shape) == 2:
                src_order = "YX"
            elif len(src_out.shape) == 3:
                src_order = "ZYX"
            elif len(src_out.shape) == 4:
                src_order = "CZYX"
            else:
                raise ValueError("unexpected source dims")

            if len(tar_out.shape) == 2:
                tar_order = "YX"
            elif len(tar_out.shape) == 3:
                tar_order = "ZYX"
            elif len(tar_out.shape) == 4:
                tar_order = "CZYX"
            else:
                raise ValueError("unexpected target dims")

            if len(prd_out.shape) == 2:
                prd_order = "YX"
            elif len(prd_out.shape) == 3:
                prd_order = "ZYX"
            elif len(prd_out.shape) == 4:
                prd_order = "CZYX"
            else:
                raise ValueError(f"unexpected pred dims {prd_out.shape}")

            out_fn = (
                self.trainer.log_dir + os.sep + str(self.current_epoch) + "_src.tiff"
            )
            OmeTiffWriter.save(src_out, out_fn, dim_order=src_order)
            out_fn = (
                self.trainer.log_dir + os.sep + str(self.current_epoch) + "_tar.tiff"
            )
            OmeTiffWriter.save(tar_out, out_fn, dim_order=tar_order)
            out_fn = (
                self.trainer.log_dir + os.sep + str(self.current_epoch) + "_prd.tiff"
            )
            OmeTiffWriter.save(prd_out, out_fn, dim_order=prd_order)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat = self.run_step(batch, validation_stage=True)
        self.log("val_loss", loss)

        return loss
