#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from typing import Union
from dask.array.core import Array as DaskArray
from numpy import ndarray as NumpyArray
from importlib import import_module
from pathlib import Path
import tempfile
import shutil
import numpy as np
from bioio import BioImage
from bioio.writers import OmeTiffWriter
import torch
from mmv_im2im.utils.misc import generate_test_dataset_dict, parse_config
from mmv_im2im.utils.for_transform import parse_monai_ops_vanilla
from skimage.io import imsave as save_rgb

# from mmv_im2im.utils.piecewise_inference import predict_piecewise
from monai.inferers import sliding_window_inference

# https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction_guide.html#predicting
###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class ProjectTester(object):
    """
    entry for training models

    Parameters
    ----------
    cfg: configuration
    """

    def __init__(self, cfg):
        # extract the three major chuck of the config
        self.model_cfg = cfg.model
        self.data_cfg = cfg.data

        # define variables
        self.model = None
        self.data = None
        self.pre_process = None
        self.cpu = False
        self.spatial_dims = -1

    def setup_model(self):
        model_category = self.model_cfg.framework
        model_module = import_module(f"mmv_im2im.models.pl_{model_category}")
        my_model_func = getattr(model_module, "Model")
        self.model = my_model_func(self.model_cfg, train=False)

        if (
            self.model_cfg.model_extra is not None
            and "cpu_only" in self.model_cfg.model_extra
            and self.model_cfg.model_extra["cpu_only"]
        ):
            self.cpu = True
            pre_train = torch.load(
                self.model_cfg.checkpoint, map_location=torch.device("cpu")
            )
        else:
            pre_train = torch.load(self.model_cfg.checkpoint)

        # TODO: hacky solution to remove a wrongly registered key
        pre_train["state_dict"].pop("criterion.xym", None)
        pre_train["state_dict"].pop("criterion.xyzm", None)
        self.model.load_state_dict(pre_train["state_dict"])

        if not self.cpu:
            self.model.cuda()

        self.model.eval()

    def setup_data_processing(self):
        # determine spatial dimension from reader parameters
        if "Z" in self.data_cfg.inference_input.reader_params["dimension_order_out"]:
            self.spatial_dims = 3
        else:
            self.spatial_dims = 2

        # prepare data preprocessing if needed
        if self.data_cfg.preprocess is not None:
            # load preprocessing transformation
            self.pre_process = parse_monai_ops_vanilla(self.data_cfg.preprocess)

    def process_one_image(
        self, img: Union[DaskArray, NumpyArray], out_fn: Union[str, Path] = None
    ):

        if isinstance(img, DaskArray):
            # Perform the prediction
            x = img.compute()
        elif isinstance(img, NumpyArray):
            x = img

        else:
            raise ValueError("invalid image")

        # check if need to add channel dimension
        if len(x.shape) == self.spatial_dims:
            x = np.expand_dims(x, axis=0)

        # convert the numpy array to float tensor
        x = torch.tensor(x.astype(np.float32))

        # run pre-processing on tensor if needed

        if self.pre_process is not None:
            x = self.pre_process(x)

        # choose different inference function for different types of models
        # the input here is assumed to be a tensor
        with torch.no_grad():
            # add batch dimension and move to GPU

            if self.cpu:
                x = torch.unsqueeze(x, dim=0)
            else:
                x = torch.unsqueeze(x, dim=0).cuda()

            # TODO: add convert to tensor with proper type, similar to torchio check

            if (
                self.model_cfg.model_extra is not None
                and "sliding_window_params" in self.model_cfg.model_extra
            ):
                y_hat = sliding_window_inference(
                    inputs=x,
                    predictor=self.model,
                    device=torch.device("cpu"),
                    **self.model_cfg.model_extra["sliding_window_params"],
                )

                # currently, we keep sliding window stiching step on CPU, but assume
                # the output is on GPU (see note below). So, we manually move the data
                # back to GPU
                if not self.cpu:
                    y_hat = y_hat.cuda()
            else:
                y_hat = self.model(x)

        ###############################################################################
        #
        # Note: currently, we assume y_hat is still on gpu, because embedseg clustering
        # step is still only running on GPU (possible on CPU, need to some update on
        # grid loading). All the post-procesisng functions we tested so far can accept
        # tensor on GPU. If it is from mmv_im2im.post_processing, it will automatically
        # convert the tensor to a numpy array and return the result as numpy array; if
        # it is from monai.transforms, it is tensor in and tensor out. We have two items
        # as #TODO: (1) we will extend post-processing functions in mmv_im2im to work
        # similarly to monai transforms, ie. ndarray in ndarray out or tensor in tensor
        # out. (2) allow yaml config to control if we want to run post-processing on
        # GPU tensors or ndarrays
        #
        ##############################################################################

        # do post-processing on the prediction
        if self.data_cfg.postprocess is not None:
            pp_data = y_hat
            for pp_info in self.data_cfg.postprocess:
                pp = parse_config(pp_info)
                pp_data = pp(pp_data)
            if torch.is_tensor(pp_data):
                pred = pp_data.cpu().numpy()
            else:
                pred = pp_data
        else:
            pred = y_hat.cpu().numpy()

        if out_fn is None:
            return pred

        # determine output dimension orders
        if out_fn.suffix == ".npy":
            np.save(out_fn, pred)
        else:
            if len(pred.shape) == 2:
                OmeTiffWriter.save(pred, out_fn, dim_order="YX")
            elif len(pred.shape) == 3:
                # 3D output, for 2D data
                if self.spatial_dims == 2:
                    # save as RGB or multi-channel 2D
                    if pred.shape[0] == 3:
                        if out_fn.suffix != ".png":
                            out_fn = out_fn.with_suffix(".png")
                        save_rgb(out_fn, np.moveaxis(pred, 0, -1))
                    else:
                        OmeTiffWriter.save(pred, out_fn, dim_order="CYX")
                elif self.spatial_dims == 3:
                    OmeTiffWriter.save(pred, out_fn, dim_order="ZYX")
                else:
                    raise ValueError("Invalid spatial dimension of pred")
            elif len(pred.shape) == 4:
                if self.spatial_dims == 3:
                    OmeTiffWriter.save(pred, out_fn, dim_order="CZYX")
                elif self.spatial_dims == 2:
                    if pred.shape[0] == 1:
                        if pred.shape[1] == 1:
                            OmeTiffWriter.save(pred[0, 0], out_fn, dim_order="YX")
                        elif pred.shape[1] == 3:
                            if out_fn.suffix != ".png":
                                out_fn = out_fn.with_suffix(".png")
                            save_rgb(
                                out_fn,
                                np.moveaxis(
                                    pred[0,],
                                    0,
                                    -1,
                                ),
                            )
                        else:
                            OmeTiffWriter.save(
                                pred[0,],
                                out_fn,
                                dim_order="CYX",
                            )
                    else:
                        raise ValueError("invalid 4D output for 2d data")
            elif len(pred.shape) == 5:
                assert pred.shape[0] == 1, "error, found non-trivial batch dimension"
                OmeTiffWriter.save(
                    pred[0,],
                    out_fn,
                    dim_order="CZYX",
                )
            else:
                raise ValueError("error in prediction output shape")

    def run_inference(self):
        self.setup_model()
        self.setup_data_processing()
        # set up data filenames
        dataset_list = generate_test_dataset_dict(
            self.data_cfg.inference_input.dir, self.data_cfg.inference_input.data_type
        )
        dataset_length = len(dataset_list)

        # loop through all images and apply the model
        for i, ds in enumerate(dataset_list):

            # Read the image
            print(f"Reading the image {i}/{dataset_length}")

            # output file name info
            fn_core = Path(ds).stem
            suffix = self.data_cfg.inference_output.suffix

            timelapse_data = 0
            # if timelapse ...
            if (
                "T"
                in self.data_cfg.inference_input.reader_params["dimension_order_out"]
            ):
                if "T" in self.data_cfg.inference_input.reader_params:
                    raise NotImplementedError(
                        "processing a subset of all timepoint is not supported yet"
                    )
                tmppath = tempfile.mkdtemp()
                print(f"making a temp folder at {tmppath}")

                # get the number of time points
                reader = BioImage(ds)
                timelapse_data = reader.dims.T

                tmpfile_list = []
                for t_idx in range(timelapse_data):
                    img = BioImage(ds).get_image_data(
                        T=[t_idx], **self.data_cfg.inference_input.reader_params
                    )
                    print(f"Predicting the image timepoint {t_idx}")

                    # prepare output filename
                    out_fn = Path(tmppath) / f"{fn_core}_{t_idx}.npy"

                    self.process_one_image(img, out_fn)
                    tmpfile_list.append(out_fn)

                # gather all individual outputs and save as timelapse
                out_array = [np.load(tmp_one_file) for tmp_one_file in tmpfile_list]
                out_array = np.stack(out_array, axis=0)

                # prepare output filename
                if "." in suffix:
                    if ".tif" in suffix or ".tiff" in suffix or ".ome.tif" in suffix:
                        out_fn = (
                            Path(self.data_cfg.inference_output.path)
                            / f"{fn_core}{suffix}"
                        )
                    else:
                        raise ValueError(
                            "please check output suffix, either unexpected dot or unsupported fileformat"  # noqa E501
                        )
                else:
                    out_fn = (
                        Path(self.data_cfg.inference_output.path)
                        / f"{fn_core}{suffix}.tiff"
                    )

                if len(out_array.shape) == 3:
                    dim_order = "TYX"
                elif len(out_array.shape) == 4:
                    if self.spatial_dims == 3:
                        dim_order = "TZYX"
                    else:
                        dim_order = "TCYX"
                elif len(out_array.shape) == 5:
                    dim_order = "TCZYX"
                else:
                    raise ValueError(f"Unexpected pred of shape {out_array.shape}")

                # save the file output
                OmeTiffWriter.save(out_array, out_fn, dim_order=dim_order)

                # clean up temporary dir
                shutil.rmtree(tmppath)
            else:
                img = BioImage(ds).get_image_data(
                    **self.data_cfg.inference_input.reader_params
                )
                
                # prepare output filename
                if "." in suffix:
                    if (
                        ".png" in suffix
                        or ".tif" in suffix
                        or ".tiff" in suffix
                        or ".ome.tif" in suffix
                    ):
                        out_fn = (
                            Path(self.data_cfg.inference_output.path)
                            / f"{fn_core}{suffix}"
                        )
                    else:
                        raise ValueError(
                            "please check output suffix, either unexpected dot or unsupported fileformat"  # noqa E501
                        )
                else:
                    out_fn = (
                        Path(self.data_cfg.inference_output.path)
                        / f"{fn_core}{suffix}.tiff"
                    )

                print("Predicting the image")
                self.process_one_image(img, out_fn)
