#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from importlib import import_module

import pytorch_lightning as pl

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
        from mmv_im2im.data_modules.dm_inference import Im2ImDataModule

        self.data = Im2ImDataModule(self.data_cfg)

        # set up model
        model_category = self.model_cfg.pop("category")
        model_module = import_module(f"mmv_im2im.models.{model_category}_basic")
        my_model_func = getattr(model_module, "Model")
        self.model = my_model_func.load_from_checkpoint(**self.model_cfg["ckpt"])

        # set up trainer
        trainer = pl.Trainer(**self.run_cfg)
        trainer.predict(self.model, self.data)
