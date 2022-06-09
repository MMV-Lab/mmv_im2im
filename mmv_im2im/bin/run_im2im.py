#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This sample script will get deployed in the bin directory of the
users' virtualenv when the parent module is installed using pip.
"""
import logging
import sys
import traceback

import torch

from mmv_im2im import ProjectTester, ProjectTrainer
from mmv_im2im.configs.config_base import ProgramConfig, parse_adaptor, configuration_validation


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
        assert torch.cuda.is_available(), "GPU is not available."
        # torch.cuda.set_device(torch.device("cuda:0"))

        if cfg.mode.lower() == TRAIN_MODE:
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
