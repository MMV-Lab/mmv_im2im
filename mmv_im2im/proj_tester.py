#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from importlib import import_module
from pathlib import Path

from aicsimageio import AICSImage
import pytorch_lightning as pl
import torch
from torchio.data.io import check_uint_to_int

from mmv_im2im.data_modules.dm_inference import Im2ImDataModule
from mmv_im2im.utils.misc import generate_test_dataset_dict
from mmv_im2im.utils.piecewise_inference import predict_piecewise

# https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction_guide.html#predicting
###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class ProjectTester(object):
    """
    entry for training models

    Parameters
    ----------
    cfg: configuration
    """

    def __init__(self, cfg):
        # extract the three major chuck of the config
        self.model_cfg = cfg.model
        self.data_cfg = cfg.data
        self.run_cfg = cfg.run

        # define variables
        self.model = None
        self.data = None

    def run_inference(self):
        # set up data
        # self.data = Im2ImDataModule(self.data_cfg)

        # set up model
        model_category = self.model_cfg.pop("category")
        model_module = import_module(f"mmv_im2im.models.{model_category}_basic")
        my_model_func = getattr(model_module, "Model")
        self.model = my_model_func.load_from_checkpoint(
            model_info_xx=self.model_cfg, train=False, **self.model_cfg["ckpt"]
        ).cuda()

        dataset_list = generate_test_dataset_dict(
            self.data_cfg["input"]["dir"], **self.data_cfg["input"]["params"]
        )

        for ds in dataset_list:
            #fn_core = Path(ds).stem
            #suffix = self.output["suffix"]
            #out_path = Path(self.output["path"]) / f"{fn_core}_{suffix}.tiff"
            img = AICSImage(ds).reader.get_image_dask_data("CZYX", C=[0],T=0)
            x = check_uint_to_int(img.compute())
            out = predict_piecewise(
                self.model,
                torch.from_numpy(x).float().cuda(),
                dims_max=[1, 32, 128, 128],
                overlaps=[0, 6, 12, 12]
            )
            print(out)

        #**self.sliding_window
        # set up trainer
        #trainer = pl.Trainer(**self.run_cfg)
        #trainer.predict(self.model, self.data)
