#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from importlib import import_module
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
        # self.ckpt = cfg.checkpoint_path
        # self.ckpt_path = cfg.checkpoint_path
        # define variables
        self.model = None
        self.data = None

    def run_training(self):
        # set up data
        data_category = self.data_cfg.pop("category")
        data_cls_module = import_module(f"mmv_im2im.data_modules.dm_{data_category}")
        my_data_funcs = getattr(data_cls_module, "Im2ImDataModule")
        self.data = my_data_funcs(self.data_cfg)
        # self.callback = PrintTableMetricsCallback()
        # set up model
        model_category = self.model_cfg.pop("category")
        model_module = import_module(f"mmv_im2im.models.{model_category}_basic")
        my_model_func = getattr(model_module, "Model")
        self.model = my_model_func(self.model_cfg)
        # self.callbacks=[PrintTableMetricsCallback()]
        # print(self.train_cfg)
        # print("callbacks" in self.train_cfg)
        # # set up training
        # if "callbacks" in self.train_cfg:
        #     print("*********************")
        #     print(self.train_cfg)
        #     callback_list = parse_ops_list(self.train_cfg["callbacks"])
        # else:
        #     callback_list = []
        trainer = pl.Trainer(**self.train_cfg["params"])  # noqa E501
        
        # start training
        trainer.fit(model=self.model, datamodule=self.data)
