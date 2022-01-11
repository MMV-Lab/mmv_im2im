#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import logging
import sys
import traceback
from pathlib import Path
import numpy as np
from tqdm import tqdm
# from skimage.morphology import dilation, ball
# from skimage.util import random_noise
# from scipy.ndimage import gaussian_filter
from PIL import Image, ImageDraw, ImageFilter
from tifffile import imsave
# from randimage import get_random_image

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

    # num_obj = 10

    out_path = Path(args.out).expanduser()
    for ii in tqdm(range(args.num)):
        out_raw_fn = out_path / f"img_{1000+ii}_IM.tiff"
        out_gt_fn = out_path / f"img_{1000+ii}_GT.tiff"
        # if args.costmap:
        #   # out_cm_fn = out_path / f"img_{1000+ii}_CM.tiff"

        if args.dim == 3:
            # if args.large:
            #     im = np.zeros((64, 512, 512))
            # else:
            #     im = np.zeros((1, 128, 128))

            # for obj in range(num_obj):
            #     py = randint(15, im.shape[-2]-15)
            #     px = randint(15, im.shape[-1]-15)
            #     im[31, py, px] = 1
            # im = dilation(im > 0, ball(10))

            if args.type == "im" and args.unpair:
                x, y = (128, 128)
                eX, eY = 60, 60
                im_a = Image.new("L", (128, 128), 0)
                draw = ImageDraw.Draw(im_a)
                draw.ellipse(
                        (x/2 - eX/2, y/2 - eY/2, x/2 + eX/2, y/2 + eY/2),
                        fill=255
                        )
                im_a_blur = im_a.filter(ImageFilter.GaussianBlur(15))
                im_GT = np.zeros((32, 128, 128))
                im_raw = np.zeros((32, 128, 128))
                c, _, _ = im_GT.shape
                for channel in range(c):
                    im_GT[channel, :, :] = im_a
                    im_raw[channel, :, :] = im_a_blur
                imsave(out_gt_fn, im_GT)
                imsave(out_raw_fn, im_raw)
            elif args.type == "mask" and (not args.unpair):
                w, h = (8, 8)
                img = Image.new("L", (w, h), 0)
                pixels = img.load()
                for i in range(w):
                    for j in range(h):
                        if (i+j) % 2 == 0:
                            pixels[i, j] = 255
                img = img.resize((128, 128), Image.NEAREST)
                im_GT = np.zeros((32, 128, 128))
                im_raw = np.zeros((32, 128, 128))
                c, w, h = im_GT.shape
                for channel in range(c):
                    im_GT[channel, :, :] = np.array(img)
                    im_raw[channel, :, :] = np.array(
                                            img.filter(ImageFilter.FIND_EDGES)
                                            )
                imsave(out_gt_fn, im_GT.astype(np.uint8))
                imsave(out_raw_fn, im_raw.astype(np.uint8))

            # raw = gaussian_filter(im.astype(np.float32), 3)
            # imsave(out_raw_fn, raw)

            # if args.costmap:
            #     costmap = np.ones_like(raw)
            #     imsave(out_cm_fn, costmap)
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
