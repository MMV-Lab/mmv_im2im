#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from importlib import import_module
from mmv_im2im.data_modules import get_data_module
from mmv_im2im.utils.misc import parse_ops_list
import pytorch_lightning as pl

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class ProjectTrainer(object):
    """
    entry for training models

    Parameters
    ----------
    cfg: configuration
    """

    def __init__(self, cfg):

        # seed everything before start
        pl.seed_everything(123, workers=True)

        # extract the three major chuck of the config
        self.model_cfg = cfg.model
        self.train_cfg = cfg.training
        self.data_cfg = cfg.data

        # define variables
        self.model = None
        self.data = None

    def run_training(self):
        # set up data
        self.data = get_data_module(self.data_cfg)

        # set up model
        model_category = self.model_cfg.pop("category")
        model_module = import_module(f"mmv_im2im.models.pl_{model_category}")
        my_model_func = getattr(model_module, "Model")
        self.model = my_model_func(self.model_cfg)

        # set up training
        if "callbacks" in self.train_cfg:
            callback_list = parse_ops_list(self.train_cfg["callbacks"])
        else:
            callback_list = []
        trainer = pl.Trainer(
            callbacks=callback_list, **self.train_cfg["params"]
        )  # noqa E501

        # start training
        trainer.fit(model=self.model, datamodule=self.data)
