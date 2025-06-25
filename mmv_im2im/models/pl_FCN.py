import numpy as np
from typing import Dict
from pathlib import Path
from random import randint
import lightning as pl
from bioio.writers import OmeTiffWriter

from mmv_im2im.utils.misc import (
    parse_config,
    parse_config_func,
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
                model_info_xx["lr_scheduler"] is not None
                and len(model_info_xx["lr_scheduler"]) > 0
            ):
                self.lr_scheduler_func = parse_config_func(model_info_xx.lr_scheduler)
            else:
                self.lr_scheduler_func = None

        # for segmentation, add flag to transform one hot encoding back to label image
        if "seg_output_to_label" in model_info_xx:
            self.seg_flag = model_info_xx["seg_output_to_label"]

        self.save_hyperparameters()

    def forward(self, x):
        return self.net(x)

    def configure_optimizers(self):
        optimizer = self.optimizer_func(
            self.parameters(), **self.hparams.model_info_xx.optimizer["params"]
        )
        if self.lr_scheduler_func is None:
            return optimizer
        else:
            scheduler = self.lr_scheduler_func(
                optimizer, **self.hparams.model_info_xx.lr_scheduler["params"]
            )
            return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        if self.weighted_loss:
            source, target, weight = batch
            weight_map = weight.float()
        else:
            source, target = batch
        source = source.float()
        target = target.float()

        pred = self.forward(source)
        if self.weighted_loss:
            loss = self.criterion(pred, target, weight_map)
        else:
            loss = self.criterion(pred, target)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # write to output
        if self.verbose and batch_idx == 0:
            # check if the log path exists, if not create one
            log_dir = Path(self.trainer.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            self.write_prediction_to_image(
                source, target, pred, log_dir, current_stage="train"
            )

        return loss

    def write_prediction_to_image(self, source, target, pred, save_path, current_stage):
        # assume batch size is 1
        src_out = source.detach().cpu().numpy().squeeze()
        tar_out = target.detach().cpu().numpy().squeeze()
        prd_out = pred.detach().cpu().numpy().squeeze()

        if self.seg_flag:
            # transform one hot encoding to label image
            tar_out = np.argmax(tar_out, axis=0)
            prd_out = np.argmax(prd_out, axis=0)

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

    def validation_step(self, batch, batch_idx):
        if self.weighted_loss:
            source, target, weight = batch
            weight_map = weight.float()
        else:
            source, target = batch
        source = source.float()
        target = target.float()

        pred = self.forward(source)
        if self.weighted_loss:
            loss = self.criterion(pred, target, weight_map)
        else:
            loss = self.criterion(pred, target)

        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # write to output
        if self.verbose and batch_idx == 0:
            # check if the log path exists, if not create one
            log_dir = Path(self.trainer.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            self.write_prediction_to_image(
                source, target, pred, log_dir, current_stage="val"
            )

        return loss

    def test_step(self, batch, batch_idx):
        if self.weighted_loss:
            source, target, weight = batch
            weight_map = weight.float()
        else:
            source, target = batch
        source = source.float()
        target = target.float()

        pred = self.forward(source)
        if self.weighted_loss:
            loss = self.criterion(pred, target, weight_map)
        else:
            loss = self.criterion(pred, target)
        self.log(
            "test_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        # write to output
        if self.verbose and batch_idx == 0:
            # check if the log path exists, if not create one
            log_dir = Path(self.trainer.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            self.write_prediction_to_image(
                source, target, pred, log_dir, current_stage="test"
            )

        return loss
