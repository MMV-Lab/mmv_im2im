import numpy as np
from typing import Dict
from pathlib import Path
from bioio.writers import OmeTiffWriter
import lightning as pl
from mmv_im2im.postprocessing.embedseg_cluster import generate_instance_clusters
from mmv_im2im.utils.embedseg_utils import prepare_embedseg_tensor
from mmv_im2im.utils.model_utils import init_weights
import torch
from monai.inferers import sliding_window_inference

from mmv_im2im.utils.misc import (
    parse_config,
    parse_config_func,
    parse_config_func_without_params,
)
from mmv_im2im.utils.metrics import simplified_instance_IoU


class Model(pl.LightningModule):
    def __init__(self, model_info_xx: Dict, train: bool = True, verbose: bool = False):
        super().__init__()
        self.verbose = verbose

        # for training, we need the clustering parameters for the validation step,
        # the same set of parameters as in the criterion, except "foreground_weight", so
        # pop it out
        if train:
            self.clustering_params = model_info_xx.criterion["params"].copy()
            self.clustering_params.pop("foreground_weight")
            if "use_costmap" in self.clustering_params:
                self.clustering_params.pop("use_costmap")

        self.net = parse_config(model_info_xx.net)

        if model_info_xx.net["func_name"].startswith("BranchedERFNet"):
            if train:
                # if not train, not need to init
                self.net.init_output(model_info_xx.criterion["params"]["n_sigma"])
        else:
            init_weights(self.net, init_type="kaiming")

        self.model_info = model_info_xx
        if train:
            extra_clustering_param = [
                "min_mask_sum",
                "min_unclustered_sum",
                "min_object_size",
            ]
            for cp in extra_clustering_param:
                if cp in model_info_xx.criterion["params"]:
                    model_info_xx.criterion["params"].pop(cp)

            self.criterion = parse_config(model_info_xx.criterion)
            self.optimizer_func = parse_config_func(model_info_xx.optimizer)

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

    def forward(self, x):
        return self.net(x)

    def run_step(self, batch, validation_stage, save_path: str = None):
        im = batch["IM"]
        instances = batch["GT"]

        if len(im.size()) == 4:  # "BCYX"
            spatial_dim = 2
        elif len(im.size()) == 5:  # "BCZYX"
            spatial_dim = 3

        use_costmap = False
        if "CM" in batch:
            use_costmap = True
            costmap = batch["CM"]

        # forward
        if validation_stage:
            # generate predictions
            if (
                self.model_info.model_extra is not None
                and "validation_sliding_windows" in self.model_info.model_extra
            ):
                output = sliding_window_inference(
                    inputs=im,
                    predictor=self.net,
                    device=torch.device("cpu"),
                    **self.model_info.model_extra["validation_sliding_windows"],
                )
                # move back to CUDA
                output = output.cuda()
            else:
                output = self.net(im)

            # generate instance segmentation
            instances_map = generate_instance_clusters(output, **self.clustering_params)

            # calculate simplified IoU loss for validation
            instances = instances.detach().cpu().numpy()
            if len(instances.shape) > spatial_dim:
                instances = np.squeeze(instances)

            if use_costmap:
                sIoU = simplified_instance_IoU(
                    instances, instances_map, costmap.detach().cpu().numpy() == 0
                )
            else:
                sIoU = simplified_instance_IoU(instances, instances_map)

            loss = sIoU
        else:
            output = self(im)

            # decide if CL and CE are already generated
            if "CL" in batch and "CE" in batch:
                class_labels = batch["CL"]
                center_images = batch["CE"]
            else:
                class_labels, center_images = prepare_embedseg_tensor(
                    instances, spatial_dim, self.model_info.model_extra["center_method"]
                )

            if use_costmap:
                loss = self.criterion(
                    output, instances, class_labels, center_images, costmap
                )
            else:
                loss = self.criterion(output, instances, class_labels, center_images)
            loss = loss.mean()

        if save_path is not None:
            current_stage = "val"
            # get instance segmentation if not yet
            if not validation_stage:
                current_stage = "train"
                instances_map = generate_instance_clusters(
                    output, **self.clustering_params
                )
                # remove the batch dimension
                gt = instances.detach().cpu().numpy()[0,]
            else:
                # add back the C dimension
                gt = np.expand_dims(instances, axis=0)
            if len(im.size()) == 4:
                dim_order = "CYX"
            elif len(im.size()) == 5:
                dim_order = "CZYX"

            # save raw image
            out_fn = save_path / f"{self.current_epoch}_{current_stage}_raw.tiff"
            OmeTiffWriter.save(
                im.detach().cpu().numpy()[0,],
                out_fn,
                dim_order=dim_order,
            )

            # save ground truth
            out_fn = save_path / f"{self.current_epoch}_{current_stage}_gt.tiff"
            OmeTiffWriter.save(gt, out_fn, dim_order=dim_order)

            # # save instance segmentation
            out_fn = save_path / f"{self.current_epoch}_{current_stage}_seg.tiff"
            OmeTiffWriter.save(instances_map, out_fn, dim_order=dim_order[1:])
        return loss

    def training_step(self, batch, batch_idx):
        if self.verbose and batch_idx == 0:
            # check if the log path exists, if not create one
            log_dir = Path(self.trainer.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)

            loss = self.run_step(batch, validation_stage=False, save_path=log_dir)
        else:
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
        if self.verbose and batch_idx == 0:
            # check if the log path exists, if not create one
            log_dir = Path(self.trainer.log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            loss = self.run_step(batch, validation_stage=False, save_path=log_dir)
        else:
            loss = self.run_step(batch, validation_stage=False)
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
