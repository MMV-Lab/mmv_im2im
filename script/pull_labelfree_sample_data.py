#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pandas as pd
import quilt3
from pathlib import Path
import random
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
import os
import sys
import logging
import argparse
import traceback

log = logging.getLogger()
logging.basicConfig(
    level=logging.INFO, format="[%(levelname)4s:%(lineno)4s %(asctime)s] %(message)s"
)

###############################################################################

###############################################################################


class Args(object):
    """
    Use this to define command line arguments and use them later.
    For each argument do the following
    1. Create a member in __init__ before the self.__parse call.
    2. Provide a default value here.
    3. Then in p.add_argument, set the dest parameter to that variable name.
    See the debug parameter as an example.
    """

    def __init__(self, log_cmdline=True):
        self.debug = False
        self.__parse()

        if self.debug:
            log.setLevel(logging.DEBUG)
            log.debug("-" * 80)
            self.show_info()
            log.debug("-" * 80)

    @staticmethod
    def __no_args_print_help(parser):
        """
        This is used to print out the help if no arguments are provided.
        Note:
        - You need to remove it's usage if your script truly doesn't want arguments.
        - It exits with 1 because it's an error if this is used in a script with
          no args. That's a non-interactive use scenario - typically you don't want
          help there.
        """
        if len(sys.argv) == 1:
            parser.print_help()
            sys.exit(1)

    def __parse(self):
        p = argparse.ArgumentParser()
        # Add arguments
        p.add_argument(
            "-d",
            "--debug",
            action="store_true",
            dest="debug",
            help="If set debug log output is enabled",
        )
        p.add_argument(
            "--structure",
            default="LMNB1",
            help="which cell line to pull from the remote bucket",
        )
        p.add_argument(
            "--download_path",
            required=True,
            help="where to save the downloaded data",
        )
        p.add_argument(
            "--num",
            type=int,
            default=20,
            help="number of samples to download",
        )

        self.__no_args_print_help(p)
        p.parse_args(namespace=self)

    def show_info(self):
        log.debug("Working Dir:")
        log.debug("\t{}".format(os.getcwd()))
        log.debug("Command Line:")
        log.debug("\t{}".format(" ".join(sys.argv)))
        log.debug("Args:")
        for (k, v) in self.__dict__.items():
            log.debug("\t{}: {}".format(k, v))


class Executor(object):
    def __init__(self, args):
        pass

    def execute(self, args):

        cline = args.structure
        parent_path = Path(args.download_path)
        num_samples_per_cell_line = int(args.num)

        # prepare file path
        raw_path_base = parent_path / Path("raw_image")
        raw_path_base.mkdir(exist_ok=True)
        train_path_base = parent_path / Path("train")
        train_path_base.mkdir(exist_ok=True)
        holdout_path_base = parent_path / Path("holdout")
        holdout_path_base.mkdir(exist_ok=True)

        # connect to quilt and load meta table
        pkg = quilt3.Package.browse(
            "aics/hipsc_single_cell_image_dataset", registry="s3://allencell"
        )
        meta_df_obj = pkg["metadata.csv"]
        meta_df_obj.fetch(parent_path / "meta.csv")
        meta_df = pd.read_csv(parent_path / "meta.csv")

        # fetch the data of the specific cell line
        meta_df_line = meta_df.query("structure_name==@cline")

        # collapse the data table based on FOVId
        meta_df_line.drop_duplicates(subset="FOVId", inplace=True)

        # reset index
        meta_df_line.reset_index(drop=True, inplace=True)

        # prepare file paths
        raw_path = raw_path_base / Path(cline)
        raw_path.mkdir(exist_ok=True)
        train_path = train_path_base / Path(cline)
        train_path.mkdir(exist_ok=True)
        holdout_path = holdout_path_base / Path(cline)
        holdout_path.mkdir(exist_ok=True)

        # download all FOVs or a certain 
        if num_samples_per_cell_line > 0:
            num = num_samples_per_cell_line
        else:
            num = meta_df_line.shape[0]

        for row in meta_df_line.itertuples():

            if row.Index >= num:
                break

            # fetch the raw image
            subdir_name = row.fov_path.split("/")[0]
            file_name = row.fov_path.split("/")[1]

            local_fn = raw_path / f"{row.FOVId}_original.tiff"
            pkg[subdir_name][file_name].fetch(local_fn)

            # extract the bf and structures channel
            reader = AICSImage(local_fn)
            bf_img = reader.get_image_data(
                "ZYX", C=row.ChannelNumberBrightfield, S=0, T=0
            )
            str_img = reader.get_image_data(
                "ZYX", C=row.ChannelNumberStruct, S=0, T=0
            )

            if random.random() < 0.2:
                data_path = holdout_path
            else:
                data_path = train_path

            im_fn = data_path / f"{row.FOVId}_IM.tiff"
            gt_fn = data_path / f"{row.FOVId}_GT.tiff"
            OmeTiffWriter.save(bf_img, im_fn, dim_order="ZYX")
            OmeTiffWriter.save(str_img, gt_fn, dim_order="ZYX")


def main():
    dbg = False
    try:
        args = Args()
        dbg = args.debug

        exe = Executor(args)
        exe.execute(args)

    except Exception as e:
        log.error("=============================================")
        if dbg:
            log.error("\n\n" + traceback.format_exc())
            log.error("=============================================")
        log.error("\n\n" + str(e) + "\n")
        log.error("=============================================")
        sys.exit(1)


if __name__ == "__main__":
    main()
