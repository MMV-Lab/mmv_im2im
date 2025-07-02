#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This sample script will get deployed in the bin directory of the
users' virtualenv when the parent module is installed using pip.
"""
import logging
import sys
import traceback
import tempfile
from pathlib import Path

# import torch

from mmv_im2im import ProjectTester, ProjectTrainer
from mmv_im2im.configs.config_base import (
    ProgramConfig,
    parse_adaptor,
    configuration_validation,
)


###############################################################################

log = logging.getLogger()
logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)4s:%(lineno)4s %(asctime)s] %(message)s",  # noqa E501
)

###############################################################################
TRAIN_MODE = "train"
INFER_MODE = "inference"

###############################################################################


def main():
    cfg = parse_adaptor(config_class=ProgramConfig)

    # check the configurations to make sure no conflicting settings
    cfg = configuration_validation(cfg)

    try:
        # check gpu option
        # assert torch.cuda.is_available(), "GPU is not available."
        # torch.cuda.set_device(torch.device("cuda:0"))

        if cfg.mode.lower() == TRAIN_MODE:
            if (
                cfg.data.dataloader.train.dataloader_type["func_name"]
                == "PersistentDataset"
            ):
                # currently, train and val have to use persistent load together, or
                # both not use and their root cache dirs are the same. See the
                # validation part in config
                cache_root = Path(cfg.data.dataloader.train.dataset_params["cache_dir"])
                cache_root.mkdir(exist_ok=True)
                with tempfile.TemporaryDirectory(dir=cache_root) as tmp_exp:
                    train_cache = Path(tmp_exp) / "train"
                    val_cache = Path(tmp_exp) / "val"
                    cfg.data.dataloader.train.dataset_params["cache_dir"] = train_cache
                    cfg.data.dataloader.val.dataset_params["cache_dir"] = val_cache
                    exe = ProjectTrainer(cfg)
                    exe.run_training()
            else:
                exe = ProjectTrainer(cfg)
                exe.run_training()
        elif cfg.mode.lower() == INFER_MODE:
            exe = ProjectTester(cfg)
            exe.run_inference()
        else:
            log.error(f"Mode {cfg.mode} is not supported yet")
            sys.exit(1)

    except Exception as e:
        log.error("=============================================")
        log.error("\n\n" + traceback.format_exc())
        log.error("=============================================")
        log.error("\n\n" + str(e) + "\n")
        log.error("=============================================")
        sys.exit(1)


###############################################################################
# Allow caller to directly run this module (usually in development scenarios)

if __name__ == "__main__":
    main()
