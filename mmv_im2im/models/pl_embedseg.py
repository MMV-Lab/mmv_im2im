import os
from typing import Dict
from aicsimageio.writers import OmeTiffWriter
import random
import pytorch_lightning as pl
import torchio as tio
import torch
from mmv_im2im.postprocessing.embedseg_cluster import generate_instance_clusters

from mmv_im2im.utils.misc import (
    parse_config,
    parse_config_func,
    parse_config_func_without_params,
)


class Model(pl.LightningModule):
    def __init__(self, model_info_xx: Dict, train: bool = True, verbose: bool = False):
        super().__init__()
        self.verbose = verbose
        if verbose:
            self.clustering_params = model_info_xx["criterion"]["params"]
            self.clustering_params.pop("foreground_weight")
        self.net = parse_config(model_info_xx["net"])

        if "sliding_window_params" in model_info_xx:
            print("sliding window parameters are detected, but not needed for embedseg")
        self.model_info = model_info_xx
        if train:
            self.criterion = parse_config(model_info_xx["criterion"])
            self.optimizer_func = parse_config_func(model_info_xx["optimizer"])

    def configure_optimizers(self):
        # https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.configure_optimizers  # noqa E501
        optimizer = self.optimizer_func(self.parameters())
        print("optim done")
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

    def run_step(self, batch, validation_stage, save_path: str = None):

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

        im = batch["source"][tio.DATA]
        instances = batch["target"][tio.DATA]
        class_labels = batch["class_image"][tio.DATA]
        center_images = batch["center_image"][tio.DATA]

        if im.size()[-1] == 1:
            im = torch.squeeze(im, dim=-1).float()
            instances = torch.squeeze(instances, dim=-1)
            class_labels = torch.squeeze(class_labels, dim=-1)
            center_images = torch.squeeze(center_images, dim=-1)
        output = self(im)

        # TODO: need to handle args, try to receive the args in the definition step
        loss = self.criterion(output, instances, class_labels, center_images)
        loss = loss.mean()

        if validation_stage:
            # TODO: add validation step
            pass
        if save_path is not None:
            instances_map = generate_instance_clusters(output, **self.clustering_params)
            if len(im.size()) == 4:
                dim_order = "CYX"
            elif len(im.size()) == 5:
                dim_order = "CZYX"
            out_fn = save_path + "_raw.tiff"
            OmeTiffWriter.save(
                im.detach().cpu().numpy()[0,], out_fn, dim_order=dim_order
            )
            out_fn = save_path + "_gt.tiff"
            OmeTiffWriter.save(
                instances.detach().cpu().numpy()[0,], out_fn, dim_order=dim_order
            )
            out_fn = save_path + "_pred.tiff"
            OmeTiffWriter.save(
                instances_map, out_fn, dim_order=dim_order[1:]
            )

        return loss

    def training_step(self, batch, batch_idx):
        if self.verbose and batch_idx == 0:
            if not os.path.exists(self.trainer.log_dir):
                os.mkdir(self.trainer.log_dir)
            save_path_base = self.trainer.log_dir + os.sep + str(self.current_epoch)
            loss = self.run_step(
                batch, validation_stage=False, save_path=save_path_base
            )
        else:
            loss = self.run_step(batch, validation_stage=False)

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.run_step(batch, validation_stage=True)
        self.log("val_loss", loss)

        return loss
