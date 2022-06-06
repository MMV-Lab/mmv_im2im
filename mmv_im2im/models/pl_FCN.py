import os
import numpy as np
from typing import Dict
import pytorch_lightning as pl
import torch
from aicsimageio.writers import OmeTiffWriter

from mmv_im2im.utils.misc import (
    parse_config,
    parse_config_func,
    parse_config_func_without_params,
)


class Model(pl.LightningModule):
    def __init__(self, model_info_xx: Dict, train: bool = True, verbose: bool = False):
        super().__init__()
        self.net = parse_config(model_info_xx["net"])
        # if "sliding_window_params" in model_info_xx:
        #    self.sliding_window = model_info_xx["sliding_window_params"]
        # else:
        #    self.sliding_window = None
        self.model_info = model_info_xx
        self.verbose = verbose
        if train:
            self.criterion = parse_config(model_info_xx["criterion"])
            self.optimizer_func = parse_config_func(model_info_xx["optimizer"])

    def configure_optimizers(self):
        # https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.configure_optimizers  # noqa E501
        optimizer = self.optimizer_func(self.parameters())
        if "scheduler" in self.model_info:
            scheduler_func = parse_config_func_without_params(
                self.model_info["scheduler"]
            )
            lr_scheduler = scheduler_func(
                optimizer, **self.model_info["scheduler"]["params"]
            )
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}
        else:
            return optimizer

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

        ##########################################
        # check if the data is 2D or 3D
        ##########################################
        # torchio will add dummy dimension to 2D images to ensure 4D tensor
        # see: https://github.com/fepegar/torchio/blob/1c217d8716bf42051e91487ece82f0372b59d903/torchio/data/io.py#L402  # noqa E501
        # but in PyTorch, we usually follow the convention as C x D x H x W
        # or C x Z x Y x X, where the Z dimension of depth dimension is before
        # HW or YX. Padding the dummy dimmension at the end is okay and
        # actually makes data augmentation easier to implement. For example,
        # you have 2D image of Y x X, then becomes 1 x Y x X x 1. If you want
        # to apply a flip on the first dimension of your image, i.e. Y, you
        # can simply specify axes as [0], which is compatable
        # with the syntax for fliping along Y in 1 x Y x X x 1.
        # But, the FCN models do not like this. We just need to remove the
        # dummy dimension
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
            # remove C dimension
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
            if not os.path.exists(self.trainer.log_dir):
                os.mkdir(self.trainer.log_dir)

            src_out = np.squeeze(src.detach().cpu().numpy()).astype(np.float)
            tar_out = np.squeeze(tar.detach().cpu().numpy()).astype(np.float)
            prd_out = np.squeeze(y_hat.detach().cpu().numpy()).astype(np.float)

            if len(src_out.shape) == 2:
                src_order = "YX"
            elif len(src_out.shape) == 3:
                src_order = "ZYX"
            elif len(src_out.shape) == 4:
                src_order = "CZYX"

            if len(tar_out.shape) == 2:
                tar_order = "YX"
            elif len(tar_out.shape) == 3:
                tar_order = "ZYX"
            elif len(tar_out.shape) == 4:
                tar_order = "CZYX"

            if len(prd_out.shape) == 2:
                prd_order = "YX"
            elif len(prd_out.shape) == 3:
                prd_order = "ZYX"
            elif len(prd_out.shape) == 4:
                prd_order = "CZYX"

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
        """
        src = batch["source"][tio.DATA]
        tar = batch["target"][tio.DATA]
        from tifffile import imsave
        from random import randint
        fn_rand = randint(100,900)
        imsave("./val_src_"+str(fn_rand)+".tiff", src[0,0,].cpu().numpy())
        imsave("./val_tar_"+str(fn_rand)+".tiff", tar[0,0,].cpu().numpy())
        # tensorboard = self.logger.experiment
        # tensorboard.add_image("val_source", src)
        # tensorboard.add_image("val_target", tar)
        # tensorboard.add_image("val_predict", y_hat)
        """
        return loss
