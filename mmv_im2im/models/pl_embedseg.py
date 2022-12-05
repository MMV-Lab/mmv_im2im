import os
from typing import Dict
from aicsimageio.writers import OmeTiffWriter
import pytorch_lightning as pl
from mmv_im2im.postprocessing.embedseg_cluster import generate_instance_clusters
from mmv_im2im.utils.embedseg_utils import prepare_embedseg_tensor
from mmv_im2im.utils.model_utils import init_weights

from mmv_im2im.utils.misc import (
    parse_config,
    parse_config_func,
    parse_config_func_without_params,
)


class Model(pl.LightningModule):
    def __init__(self, model_info_xx: Dict, train: bool = True, verbose: bool = False):
        super().__init__()
        self.verbose = verbose

        # for training, we need the clustering parameters for the validation step,
        # the same set of parameters as in the criterion, except "foreground_weight", so
        # pop it out
        if train:
            self.clustering_params = model_info_xx.criterion["params"]
            self.clustering_params.pop("foreground_weight")
            if "use_costmap" in self.clustering_params:
                self.clustering_params.pop("use_costmap")

        self.net = parse_config(model_info_xx.net)

        if model_info_xx.net["func_name"].startswith("BranchedERFNet"):
            self.net.init_output(model_info_xx.criterion["params"]["n_sigma"])
        else:
            init_weights(self.net, init_type="kaiming")

        self.model_info = model_info_xx
        if train:
            self.criterion = parse_config(model_info_xx.criterion)
            self.optimizer_func = parse_config_func(model_info_xx.optimizer)

    def configure_optimizers(self):
        # https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.configure_optimizers  # noqa E501
        optimizer = self.optimizer_func(self.parameters())
        print("optim done")
        if self.model_info.scheduler is None:
            return optimizer
        else:
            scheduler_func = parse_config_func_without_params(self.model_info.scheduler)
            lr_scheduler = scheduler_func(
                optimizer, **self.model_info.scheduler["params"]
            )
            return {"optimizer": optimizer, "lr_scheduler": lr_scheduler}

    def forward(self, x):
        return self.net(x)

    def run_step(self, batch, validation_stage, save_path: str = None):
        im = batch["IM"]
        instances = batch["GT"]

        if len(im.size()) == 4:  # "BCYX"
            spatial_dim = 2
        elif len(im.size()) == 5:  # "BCZYX"
            spatial_dim = 3

        # decide if CL and CE are already generated
        if "CL" in batch and "CE" in batch:
            class_labels = batch["CL"]
            center_images = batch["CE"]
        else:
            class_labels, center_images = prepare_embedseg_tensor(
                instances,
                spatial_dim,
                self.model_info.model_extra["center_method"]
            )

        use_costmap = False
        if "CM" in batch:
            use_costmap = True
            costmap = batch["CM"]
        output = self(im)

        if use_costmap:
            loss = self.criterion(
                output, instances, class_labels, center_images, costmap
            )
        else:
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
                im.detach()
                .cpu()
                .numpy()[
                    0,
                ],
                out_fn,
                dim_order=dim_order,
            )
            out_fn = save_path + "_gt.tiff"
            OmeTiffWriter.save(
                instances.detach()
                .cpu()
                .numpy()[
                    0,
                ],
                out_fn,
                dim_order=dim_order,
            )
            out_fn = save_path + "_pred.tiff"
            OmeTiffWriter.save(instances_map, out_fn, dim_order=dim_order[1:])
            out_fn = save_path + "_out.tiff"
            OmeTiffWriter.save(
                output.detach()
                .cpu()
                .numpy()[
                    0,
                ]
                .astype(float),
                out_fn,
                dim_order=dim_order,
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
