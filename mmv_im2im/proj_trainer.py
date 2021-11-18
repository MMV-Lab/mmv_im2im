#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from typing import Any, Tuple
from importlib import import_module

from mmv_im2im.utils.misc import parse_config

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

    # Static methods are available to the user regardless of if they have initialized
    # an instance of the class. They are useful when you have small portions of code
    # that while relevant to the class may not depend on entire class state.
    # In this case, this function isn't incredibly valuable outside of the usage of
    # this class and therefore we use the "Python" standard of prefixing the method
    # with an underscore.
    @staticmethod
    def _check_value(val: Any):
        """
        Check that the value is an integer. If not, raises a ValueError.
        """
        if not isinstance(val, int):
            raise ValueError(
                f"Provided value: {val} (type: {type(val)}, is not an integer.)"
            )

    def __init__(self, cfg):

        # extract the three major chuck of the config
        self.model_cfg = cfg.model
        self.train_cfg = cfg.training
        self.data_cfg = cfg.data
        
        # define variables
        self.model = None
        self.data = None

    def run_training(self):
        # set up data
        data_category = self.data_cfg.pop("category")
        data_cls_module = import_module(f"mmv_im2im.data_modules.dm_{data_category}")
        my_data_funcs = getattr(data_cls_module, "Im2ImDataModule")

        self.data = my_data_funcs(self.data_cfg)

        # set up model
        model_category = self.model_cfg.pop("category")
        model_module = import_module(f"mmv_im2im.models.{model_category}_basic")
        my_model_func = getattr(model_module, "Model")

        self.model = my_model_func(self.model_cfg)

        # set up training
        if "callbacks" in self.train_cfg:
            callback_list = parse_config(self.train_cfg["callback"])
        trainer = pl.Trainer(callbacks=callback_list, **self.train_cfg["params"])

        # start training
        trainer.fit(model=self.model, datamodule=self.data)

    # Representation's (reprs) are useful when using interactive Python sessions or
    # when you log an object. They are the shorthand of the object state. In this case,
    # our string method provides a good representation.
    def __repr__(self):
        return str(self)
