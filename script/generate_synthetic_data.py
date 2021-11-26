#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
import sys
import traceback
from pathlib import Path
import numpy as np
from random import randint
from tqdm import tqdm

from skimage.morphology import dilation, ball
from skimage.util import random_noise
from scipy.ndimage import gaussian_filter
from tifffile import imsave

###############################################################################

log = logging.getLogger()
logging.basicConfig(
    level=logging.INFO, format="[%(levelname)4s:%(lineno)4s %(asctime)s] %(message)s"  # noqa E501
)


class Args(argparse.Namespace):

    def __init__(self):
        # Arguments that could be passed in through the command line
        self.debug = False
        self.__parse()

    def __parse(self):
        p = argparse.ArgumentParser(
            description="generate synthetic data",
        )
        p.add_argument(
            "--unpair",
            action="store_true",
            help="whether to generate unpair data",
        )
        p.add_argument(
            "--costmap",
            action="store_true",
            help="whether to generate costmap",
        )
        p.add_argument(
            "--type",
            default="mask",
            help="tyep of target: mask or im"
        )
        p.add_argument(
            "--large",
            action="store_true",
            help="whether to generate large image"
        )
        p.add_argument(
            "--num",
            default=10,
            type=int,
            help="number of synthetic data"
        )
        p.add_argument(
            "--dim",
            default=3,
            type=int,
            help="either 2 or 3 dimensional data"
        )
        p.add_argument(
            "--out",
            required=True,
            help="the path to save the data"
        )
        p.add_argument(
            "--debug",
            action="store_true",
            dest="debug",
            help=argparse.SUPPRESS,
        )
        p.parse_args(namespace=self)


###############################################################################
def generate_data(args):

    num_obj = 10

    out_path = Path(args.out).expanduser()
    for ii in tqdm(range(args.num)):
        out_raw_fn = out_path / f"img_{1000+ii}_IM.tiff"
        out_gt_fn = out_path / f"img_{1000+ii}_GT.tiff"
        if args.costmap:
            out_cm_fn = out_path / f"img_{1000+ii}_CM.tiff"

        if args.dim == 3:
            if args.large:
                im = np.zeros((64, 512, 512))
            else:
                im = np.zeros((64, 128, 128))
            for obj in range(num_obj):
                py = randint(15, im.shape[-2]-15)
                px = randint(15, im.shape[-1]-15)
                im[31, py, px] = 1
            im = dilation(im > 0, ball(2))

            if args.type == "mask" and not args.unpair:
                gt = im.astype(np.float32)
                gt[gt > 0] = 1
                imsave(out_gt_fn, gt)
            else:
                print("Not impletemented yet")
                exit(0)

            raw = gaussian_filter(im.astype(np.float32), 5)
            raw = random_noise(raw).astype(np.float32)
            imsave(out_raw_fn, raw)

            if args.costmap:
                costmap = np.ones_like(raw)
                imsave(out_cm_fn, costmap)
        else:
            print("Not impletemented yet")
            exit(0)


def main():
    try:
        args = Args()
        dbg = args.debug

        generate_data(args)

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
