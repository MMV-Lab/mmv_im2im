import numpy as np
from typing import Dict
from pathlib import Path
from random import randint
import lightning as pl
import torch
from aicsimageio.writers import OmeTiffWriter

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
        # if "sliding_window_params" in model_info_xx:
        #    self.sliding_window = model_info_xx["sliding_window_params"]
        # else:
        #    self.sliding_window = None

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
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

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

        if self.weighted_loss:
            loss = self.criterion(y_hat, y, cm)
        else:
            loss = self.criterion(y_hat, y)

        return loss, y_hat

    def training_step(self, batch, batch_idx):
        loss, y_hat = self.run_step(batch, validation_stage=False)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        if self.verbose and batch_idx == 0:
            src = batch["IM"]
            tar = batch["GT"]

            # check if the log path exists, if not create one
            save_path = Path(self.trainer.log_dir)
            save_path.mkdir(parents=True, exist_ok=True)

            # check if need to use softmax
            if self.seg_flag:
                act_layer = torch.nn.Softmax(dim=1)
                yhat_act = act_layer(y_hat)
            else:
                yhat_act = y_hat

            src_out = np.squeeze(src[0,].detach().cpu().numpy()).astype(float)
            tar_out = np.squeeze(tar[0,].detach().cpu().numpy()).astype(float)
            prd_out = np.squeeze(yhat_act[0,].detach().cpu().numpy()).astype(float)

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

            rand_tag = randint(1, 1000)
            out_fn = save_path / f"epoch_{self.current_epoch}_src_{rand_tag}.tiff"
            OmeTiffWriter.save(src_out, out_fn, dim_order=src_order)
            out_fn = save_path / f"epoch_{self.current_epoch}_tar_{rand_tag}.tiff"
            OmeTiffWriter.save(tar_out, out_fn, dim_order=tar_order)
            out_fn = save_path / f"epoch_{self.current_epoch}_prd_{rand_tag}_.tiff"
            OmeTiffWriter.save(prd_out, out_fn, dim_order=prd_order)

        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat = self.run_step(batch, validation_stage=True)
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
