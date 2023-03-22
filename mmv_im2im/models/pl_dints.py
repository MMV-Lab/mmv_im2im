import os
import numpy as np
from typing import Dict
from pathlib import Path
import pytorch_lightning as pl
import torch
from aicsimageio.writers import OmeTiffWriter
import monai

from mmv_im2im.utils.misc import (
    parse_config,
    parse_config_func,
    parse_config_func_without_params,
)
from mmv_im2im.utils.model_utils import init_weights


class Model(pl.LightningModule):
    """
    Parameters:
    --------
    model_info_xx: Dict
        A dictionary of everything related to model
    train: bool
        a flag indicate if in training stage (True) of inference stage
    verbose: bool
        whether print out additional information for debug
    """
    def __init__(self, model_info_xx: Dict, train: bool = True, verbose: bool = False):
        super().__init__()

        # we need to do manual optimization
        self.automatic_optimization = False

        # determine the stage: train from scratch, finetune, or inference
        if not train:
            self.stage = "inference"
        elif model_info_xx.checkpoint is None:
            self.stage = "search"
        else:
            self.stage = "finetune"

        # TODO: placeholder
        # for inference, add support to plot the decoded NN
        if self.stage == "inference":
            # use this function: https://github.com/Project-MONAI/tutorials/blob/main/automl/DiNTS/decode_plot.py  # noqa E501
            pass
        
        ############################
        # set up according to stage
        ############################
        # define topology search space
        if self.stage == "search":
            self.dints_space = monai.networks.nets.TopologySearch(**model_info_xx.net.topo_param)
        else:
            ckpt = torch.load(model_info_xx.checkpoint)
            node_a = ckpt["node_a"]  # TODO: may need to update, as pytorch lightning may save addition params in ckpt
            arch_code_a = ckpt["arch_code_a"]
            arch_code_c = ckpt["arch_code_c"]

            self.dints_space = monai.networks.nets.TopologyInstance(
                arch_code=[arch_code_a, arch_code_c],
                **model_info_xx.net.topo_param
            )
        # define stem
        self.model = monai.networks.nets.DiNTS(
            dints_space=self.dints_space,
            **model_info_xx.net.stem_param
        )
        # TODO: need to check whether we still need SyncBatchNorm
        # ref: https://pytorch.org/docs/stable/generated/torch.nn.SyncBatchNorm.html
        # model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        if self.stage == "search" or self.stage == "finetune":
            self.criterion = parse_config(self.model_info.criterion)

        self.model_info = model_info_xx
        self.verbose = verbose

    def configure_optimizers(self):
        if self.stage == "search":
            base_optimizer_func = parse_config_func(self.model_info.optimizer["base"])
            a_optimizer_func = parse_config_func(self.model_info.optimizer["alpha_a"])
            c_optimizer_func = parse_config_func(self.model_info.optimizer["alpha_c"])
            optimizer_base = base_optimizer_func(self.parameters())
            optimizer_a = a_optimizer_func([self.dints_space.log_alpha_a])
            optimizer_c = c_optimizer_func([self.dints_space.log_alpha_c])        

            return optimizer_base, optimizer_a, optimizer_c

        elif self.stage == "finetune":
            optimizer_func = parse_config_func(self.model_info.optimizer)
            optimizer = optimizer_func(self.parameters())

            return optimizer

    def forward(self, x):
        return self.net(x)

    def run_finetune(self, batch, validation_stage):

    def run_search(self, batch, validation_stage):

        self.dints_space.log_alpha_a.requires_grad = False
        self.dints_space.log_alpha_c.requires_grad = False

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
        if self.stage == "search"
            loss, y_hat = self.run_search(batch, validation_stage=False)
        elif self.stage == "finetune":
            loss, y_hat = self.run_finetune(batch, validation_stage=False)
        else:
            raise ValueError("invalid stage value")
        self.log("train_loss_step", loss, prog_bar=True)

        if self.verbose and batch_idx == 0:
            src = batch["IM"]
            tar = batch["GT"]

            # check if the log path exists, if not create one
            Path(self.trainer.log_dir).mkdir(parents=True, exist_ok=True)

            # check if need to use softmax
            if self.seg_flag:
                act_layer = torch.nn.Softmax(dim=1)
                yhat_act = act_layer(y_hat)
            else:
                yhat_act = y_hat

            src_out = np.squeeze(src[0,].detach().cpu().numpy()).astype(np.float)
            tar_out = np.squeeze(tar[0,].detach().cpu().numpy()).astype(np.float)
            prd_out = np.squeeze(yhat_act[0,].detach().cpu().numpy()).astype(np.float)

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
        self.log("val_loss_step", loss)

        return loss

    def training_epoch_end(self, training_step_outputs):
        # be aware of future deprecation: https://github.com/Lightning-AI/lightning/issues/9968   # noqa E501
        training_step_outputs = [d["loss"] for d in training_step_outputs]
        loss_ave = torch.stack(training_step_outputs).mean().item()
        self.log("train_loss", loss_ave)

    def validation_epoch_end(self, validation_step_outputs):
        loss_ave = torch.stack(validation_step_outputs).mean().item()
        self.log("val_loss", loss_ave)
