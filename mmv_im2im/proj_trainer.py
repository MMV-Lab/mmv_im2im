#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from pathlib import Path
from importlib import import_module
import lightning as pl
import torch
from mmv_im2im.data_modules import get_data_module
from mmv_im2im.utils.misc import parse_ops_list
import pyrallis

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class ProjectTrainer(object):
    """
    Entry point for training models.

    Parameters
    ----------
    cfg: configuration
    """

    def __init__(self, cfg):
        pl.seed_everything(123, workers=True)
        self.model_cfg = cfg.model
        self.train_cfg = cfg.trainer
        self.data_cfg = cfg.data
        self.model = None
        self.data = None

    def run_training(self):
        self.data = get_data_module(self.data_cfg)
        model_category = self.model_cfg.framework
        model_module = import_module(f"mmv_im2im.models.pl_{model_category}")
        my_model_func = getattr(model_module, "Model")
        self.model = my_model_func(self.model_cfg, verbose=self.train_cfg.verbose)

        if self.model_cfg.model_extra is not None:
            if "resume" in self.model_cfg.model_extra:
                self.model = self.model.load_from_checkpoint(
                    self.model_cfg.model_extra["resume"]
                )
            elif "pre-train" in self.model_cfg.model_extra:
                pre_train = torch.load(self.model_cfg.model_extra["pre-train"])

                if "extend" in self.model_cfg.model_extra:
                    if (
                        self.model_cfg.model_extra["extend"] is not None
                        and self.model_cfg.model_extra["extend"] is True
                    ):
                        pre_train["state_dict"].pop("criterion.xym", None)
                        model_state = self.model.state_dict()
                        pretrained_dict = pre_train["state_dict"]
                        filtered_dict = {}

                        for k, v in pretrained_dict.items():
                            if k in model_state and v.shape == model_state[k].shape:
                                filtered_dict[k] = v
                            else:
                                print(
                                    f"Skipped loading layer: {k} due to shape mismatch."
                                )

                        model_state.update(filtered_dict)
                        self.model.load_state_dict(model_state)
                else:
                    pre_train["state_dict"].pop("criterion.xym", None)
                    self.model.load_state_dict(pre_train["state_dict"])

        if self.train_cfg.callbacks is None:
            trainer = pl.Trainer(**self.train_cfg.params)
        else:
            callback_list = parse_ops_list(self.train_cfg.callbacks)
            trainer = pl.Trainer(callbacks=callback_list, **self.train_cfg.params)

        save_path = Path(trainer.log_dir)
        if trainer.local_rank == 0:
            save_path.mkdir(parents=True, exist_ok=True)
            pyrallis.dump(
                self.model_cfg, open(save_path / Path("model_config.yaml"), "w")
            )
            pyrallis.dump(
                self.train_cfg, open(save_path / Path("train_config.yaml"), "w")
            )
            pyrallis.dump(
                self.data_cfg, open(save_path / Path("data_config.yaml"), "w")
            )

        print("start training ... ")
        trainer.fit(model=self.model, datamodule=self.data)
        