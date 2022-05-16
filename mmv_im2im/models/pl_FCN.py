import os
from typing import Dict
import pytorch_lightning as pl
import torchio as tio
import torch
from tifffile import imsave

from mmv_im2im.utils.misc import (
    parse_config,
    parse_config_func,
    parse_config_func_without_params,
)
from mmv_im2im.utils.piecewise_inference import predict_piecewise


class Model(pl.LightningModule):
    def __init__(self, model_info_xx: Dict, train: bool = True, verbose: bool = False):
        super().__init__()
        self.net = parse_config(model_info_xx["net"])
        if "sliding_window_params" in model_info_xx:
            self.sliding_window = model_info_xx["sliding_window_params"]
        else:
            self.sliding_window = None
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
        if "costmap" in batch:
            costmap = batch.pop("costmap")
            costmap = costmap[tio.DATA]
        else:
            costmap = None

        x = batch["source"][tio.DATA]
        y = batch["target"][tio.DATA]

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

        if validation_stage and self.sliding_window is not None:
            y_hat = predict_piecewise(
                self,
                x[
                    0,
                ],
                **self.sliding_window
            )
        else:
            y_hat = self(x)

        if costmap is None:
            loss = self.criterion(y_hat, y)
        else:
            loss = self.criterion(y_hat, y, costmap)

        return loss, y_hat

    def training_step(self, batch, batch_idx):
        loss, y_hat = self.run_step(batch, validation_stage=False)
        self.log("train_loss", loss, prog_bar=True)
        """
        src = batch["source"][tio.DATA]
        tar = batch["target"][tio.DATA]
        from tifffile import imsave
        from random import randint
        fn_rand = randint(100,900)
        imsave("./train_src_"+str(fn_rand)+".tiff", src[0,0,].cpu().numpy())
        imsave("./train_tar_"+str(fn_rand)+".tiff", tar[0,0,].cpu().numpy())
        # tensorboard = self.logger.experiment
        # tensorboard.add_image("train_source", src)
        # tensorboard.add_image("train_target", tar)
        # tensorboard.add_image("train_predict", y_hat)
        """

        if self.verbose and batch_idx == 0:
            src = batch["source"][tio.DATA]
            tar = batch["target"][tio.DATA]
            if not os.path.exists(self.trainer.log_dir):
                os.mkdir(self.trainer.log_dir)
            out_fn = self.trainer.log_dir + os.sep + str(self.current_epoch) + "_src.tiff"
            imsave(out_fn, src.detach().cpu().numpy())
            out_fn = self.trainer.log_dir + os.sep + str(self.current_epoch) + "_tar.tiff"
            imsave(out_fn, tar.detach().cpu().numpy())
            out_fn = self.trainer.log_dir + os.sep + str(self.current_epoch) + "_prd.tiff"
            imsave(out_fn, y_hat.detach().cpu().numpy())

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
