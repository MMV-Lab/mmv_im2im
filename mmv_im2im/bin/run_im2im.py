#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This sample script will get deployed in the bin directory of the
users' virtualenv when the parent module is installed using pip.
"""

import argparse
import logging
import sys
import traceback

from mmv_im2im import get_module_version

###############################################################################

log = logging.getLogger()
logging.basicConfig(
    level=logging.INFO, format="[%(levelname)4s:%(lineno)4s %(asctime)s] %(message)s"
)

###############################################################################
TRAIN_MODE = "train"
INFER_MODE = "inference"

class Args(argparse.Namespace):

    def __init__(self):
        # Arguments that could be passed in through the command line
        self.debug = False
        self.__parse()

    def __parse(self):
        p = argparse.ArgumentParser(
            prog="run_im2im",
            description="running im2im",
        )

        p.add_argument(
            "-v",
            "--version",
            action="version",
            version="%(prog)s " + get_module_version(),
        )
        p.add_argument(
            "--config",
            dest="filename",
            required=True,
            help="path to configuration file",
        )
        p.add_argument(
            "--mode",
            required=True,
            help="the type of operation: train or inference",
        )
        p.add_argument(
            "--debug",
            action="store_true",
            dest="debug",
            help=argparse.SUPPRESS,
        )
        p.parse_args(namespace=self)


###############################################################################


def main():
    try:
        args = Args()
        dbg = args.debug

        # check gpu option
        assert torch.cuda.is_available(), "GPU is not available."
        torch.cuda.set_device(torch.device("cuda:0"))

        if args.mode == TRAIN_MODE or args.mode.lower() == TRAIN_MODE:
            opt = BaseOptions(args.filename, TRAIN_MODE).parse()
            exe = ProjectTrainer(opt)
            exe.run_training()
        elif args.mode == INFER_MODE or args.mode.lower() == INFER_MODE:
            opt = BaseOptions(args.filename, INFER_MODE).parse()
            exe = ProjectTester(opt)
            exe.run_inference()
        else:
            log.error(f"Mode {args.mode} is not supported yet")
            sys.exit(1)

    except Exception as e:
        log.error("=============================================")
        if dbg:
            log.error("\n\n" + traceback.format_exc())
            log.error("=============================================")
        log.error("\n\n" + str(e) + "\n")
        log.error("=============================================")
        sys.exit(1)


###############################################################################
# Allow caller to directly run this module (usually in development scenarios)

if __name__ == "__main__":
    main()
