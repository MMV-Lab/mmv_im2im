#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from typing import Union
from dask.array.core import Array as DaskArray
from numpy import ndarray as NumpyArray
from importlib import import_module
from pathlib import Path
import numpy as np
from bioio import BioImage
from bioio.writers import OmeTiffWriter
import torch
from mmv_im2im.utils.misc import parse_config
from mmv_im2im.utils.for_transform import parse_monai_ops_vanilla
from skimage.io import imsave as save_rgb
import bioio_tifffile
from mmv_im2im.utils.urcentainity_extractor import (
    Hole_Correction,
    Thickness_Corretion,
    Remove_objects,
    Extract_Uncertainty_Maps,
    perturb_image,
    Perycites_correction,
)
from monai.inferers import sliding_window_inference
import itertools
from mmv_im2im.utils.multi_pred import (
    variance_prediction,
    mean_prediction,
    max_prediction,
    add_prediction,
)
from bioio_base.types import PhysicalPixelSizes

# https://pytorch-lightning.readthedocs.io/en/latest/starter/introduction_guide.html#predicting
###############################################################################

log = logging.getLogger(__name__)

###############################################################################


class MapExtractor(object):
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
            checkpoint = torch.load(
                self.model_cfg.checkpoint,
                map_location=torch.device("cpu"),
                weights_only=False,
            )
        else:
            checkpoint = torch.load(self.model_cfg.checkpoint, weights_only=False)

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:

            pre_train = checkpoint
            pre_train["state_dict"].pop("criterion.xym", None)
            pre_train["state_dict"].pop("criterion.xyzm", None)
            self.model.load_state_dict(pre_train["state_dict"], strict=False)
        else:

            state_dict = checkpoint
            state_dict.pop("criterion.xym", None)
            state_dict.pop("criterion.xyzm", None)
            self.model.load_state_dict(state_dict, strict=False)

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
            x = x[0]

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

        if "pred_slice2vol" in self.model_cfg.net:

            if self.model_cfg.net["pred_slice2vol"] is not None:
                # handle multiple kind of elements tiff/tif
                if "," in self.data_cfg.inference_input.data_type:
                    types = self.data_cfg.inference_input.data_type.split(",")
                    extensions = [f"*{tipe}" for tipe in types]
                    filenames = sorted(
                        list(
                            itertools.chain.from_iterable(
                                self.data_cfg.inference_input.dir.glob(extension)
                                for extension in extensions
                            )
                        )
                    )
                else:
                    filenames = sorted(
                        self.data_cfg.inference_input.dir.glob(
                            "*" + self.data_cfg.inference_input.data_type
                        )
                    )

                vs_flag = PhysicalPixelSizes(1, 1, 1)
                if "pixel_dim" in self.model_cfg.net["pred_slice2vol"]:
                    if self.model_cfg.net["pred_slice2vol"]["pixel_dim"] is not None:
                        if isinstance(
                            self.model_cfg.net["pred_slice2vol"]["pixel_dim"], str
                        ):
                            vs_flag = "auto"
                        elif isinstance(
                            self.model_cfg.net["pred_slice2vol"]["pixel_dim"], tuple
                        ):
                            if (
                                len(self.model_cfg.net["pred_slice2vol"]["pixel_dim"])
                                == 3
                            ):
                                z, y, x = self.model_cfg.net["pred_slice2vol"][
                                    "pixel_dim"
                                ]
                                vs_flag = PhysicalPixelSizes(z, y, x)
                        elif isinstance(
                            self.model_cfg.net["pred_slice2vol"]["pixel_dim"], list
                        ):
                            if (
                                len(self.model_cfg.net["pred_slice2vol"]["pixel_dim"])
                                == 3
                            ):
                                z, y, x = self.model_cfg.net["pred_slice2vol"][
                                    "pixel_dim"
                                ]
                                vs_flag = PhysicalPixelSizes(z, y, x)

                max_proj = False
                if "max_proj" in self.model_cfg.net["pred_slice2vol"]:
                    if self.model_cfg.net["pred_slice2vol"]["max_proj"] is not None:
                        if isinstance(
                            self.model_cfg.net["pred_slice2vol"]["max_proj"], bool
                        ):
                            max_proj = self.model_cfg.net["pred_slice2vol"]["max_proj"]

                perycites_correction = False
                if "perycites_correction" in self.model_cfg.net["pred_slice2vol"]:
                    if (
                        self.model_cfg.net["pred_slice2vol"]["perycites_correction"]
                        is not None
                    ):
                        perycites_correction = self.model_cfg.net["pred_slice2vol"][
                            "perycites_correction"
                        ]

                print(
                    f"########################################### {len(filenames)} Volume(s) found for prediction ###########################################"
                )

                if (
                    self.model_cfg.net["pred_slice2vol"]["uncertainity_map"]
                    and "prob" not in self.model_cfg.net["func_name"].lower()
                ):
                    print(
                        "##################################################################### Warning #####################################################################"
                    )
                    print(f"Your selected Model is {self.model_cfg.net['func_name']}")
                    print(
                        "If the model is NOT probabilistic the uncertainity map and multiple prediction mode won't have sense"
                    )
                # save post process indicated by the user
                original_postprocess = self.data_cfg.postprocess
                # we set non postprocess we need logits for the model
                self.data_cfg.postprocess = None

                n_trunc = -1
                threshold_um = -1
                border_corr = False
                pert_opt = False
                # handle uncertainity maps generation
                if "uncertainity_map" in self.model_cfg.net["pred_slice2vol"]:
                    if (
                        self.model_cfg.net["pred_slice2vol"]["uncertainity_map"]
                        is not None
                    ):
                        if self.model_cfg.net["pred_slice2vol"]["uncertainity_map"]:

                            if "trunc" in self.model_cfg.net["pred_slice2vol"]:
                                if (
                                    self.model_cfg.net["pred_slice2vol"]["trunc"]
                                    is not None
                                ):
                                    if isinstance(
                                        self.model_cfg.net["pred_slice2vol"]["trunc"],
                                        bool,
                                    ):
                                        if not self.model_cfg.net["pred_slice2vol"][
                                            "trunc"
                                        ]:
                                            n_trunc = -1
                                        else:
                                            n_trunc = 4
                                    else:
                                        if (
                                            type(
                                                self.model_cfg.net["pred_slice2vol"][
                                                    "trunc"
                                                ]
                                            )
                                            is int
                                            and self.model_cfg.net["pred_slice2vol"][
                                                "trunc"
                                            ]
                                            >= 0
                                        ):
                                            n_trunc = self.model_cfg.net[
                                                "pred_slice2vol"
                                            ]["trunc"]
                                        else:
                                            raise ValueError(
                                                f"Unexpected Value for trunc: {self.model_cfg.net['pred_slice2vol']['trunc']}. It should be a positive integer"
                                            )
                                else:
                                    n_trunc = 4
                            else:
                                n_trunc = 4

                            if "threshold" in self.model_cfg.net["pred_slice2vol"]:
                                if (
                                    self.model_cfg.net["pred_slice2vol"]["threshold"]
                                    is not None
                                ):
                                    if isinstance(
                                        self.model_cfg.net["pred_slice2vol"][
                                            "threshold"
                                        ],
                                        bool,
                                    ):
                                        if not self.model_cfg.net["pred_slice2vol"][
                                            "threshold"
                                        ]:
                                            threshold_um = -1
                                        else:
                                            raise ValueError(
                                                f"Unexpected Value for threshold: {self.model_cfg.net['pred_slice2vol']['threshold']}. It should be a positive float"
                                            )
                                    else:
                                        if (
                                            type(
                                                self.model_cfg.net["pred_slice2vol"][
                                                    "threshold"
                                                ]
                                            )
                                            is float
                                            and self.model_cfg.net["pred_slice2vol"][
                                                "threshold"
                                            ]
                                            > 0
                                        ):
                                            threshold_um = self.model_cfg.net[
                                                "pred_slice2vol"
                                            ]["threshold"]
                                        else:
                                            raise ValueError(
                                                f"Unexpected Value for threshold: {self.model_cfg.net['pred_slice2vol']['threshold']}. It should be a positive float"
                                            )
                                else:
                                    threshold_um = -1
                            else:
                                threshold_um = -1

                            if (
                                "border_correction"
                                in self.model_cfg.net["pred_slice2vol"]
                            ):
                                if (
                                    self.model_cfg.net["pred_slice2vol"][
                                        "border_correction"
                                    ]
                                    is not None
                                ):
                                    if isinstance(
                                        self.model_cfg.net["pred_slice2vol"][
                                            "border_correction"
                                        ],
                                        bool,
                                    ):
                                        if not self.model_cfg.net["pred_slice2vol"][
                                            "border_correction"
                                        ]:
                                            border_corr = False
                                        else:
                                            raise ValueError(
                                                f"Unexpected Value for border_correction: {self.model_cfg.net['pred_slice2vol']['border_correction']}."
                                            )
                                    else:
                                        if (
                                            type(
                                                self.model_cfg.net["pred_slice2vol"][
                                                    "border_correction"
                                                ]
                                            )
                                            is int
                                            and self.model_cfg.net["pred_slice2vol"][
                                                "border_correction"
                                            ]
                                            >= 0
                                        ):
                                            if (
                                                self.model_cfg.net["pred_slice2vol"][
                                                    "border_correction"
                                                ]
                                                == 0
                                            ):
                                                border_corr = False
                                            else:
                                                border_corr = [
                                                    self.model_cfg.net[
                                                        "pred_slice2vol"
                                                    ]["border_correction"]
                                                ] * 2
                                        elif (
                                            type(
                                                self.model_cfg.net["pred_slice2vol"][
                                                    "border_correction"
                                                ]
                                            )
                                            is list
                                            and len(
                                                self.model_cfg.net["pred_slice2vol"][
                                                    "border_correction"
                                                ]
                                            )
                                            <= 2
                                        ):
                                            if self.model_cfg.net["pred_slice2vol"][
                                                "border_correction"
                                            ] != [0, 0] and self.model_cfg.net[
                                                "pred_slice2vol"
                                            ][
                                                "border_correction"
                                            ] != [
                                                0
                                            ]:
                                                if (
                                                    len(
                                                        self.model_cfg.net[
                                                            "pred_slice2vol"
                                                        ]["border_correction"]
                                                    )
                                                    == 1
                                                ):
                                                    border_corr = (
                                                        self.model_cfg.net[
                                                            "pred_slice2vol"
                                                        ]["border_correction"]
                                                        * 2
                                                    )
                                                else:
                                                    border_corr = self.model_cfg.net[
                                                        "pred_slice2vol"
                                                    ]["border_correction"]
                                            else:
                                                border_corr = False
                                        else:
                                            raise ValueError(
                                                f"Unexpected Value for border_correction: {self.model_cfg.net['pred_slice2vol']['border_correction']}."
                                            )
                                else:
                                    border_corr = False
                            else:
                                border_corr = False

                            if self.model_cfg.net["pred_slice2vol"]["n_samples"] <= 1:
                                print(
                                    "Number of samples are less or equal to 1 more are required for uncertainty generation"
                                )
                                print(
                                    "Automatically 5 samples will use to uncetainity calculation"
                                )
                                self.model_cfg.net["pred_slice2vol"]["n_samples"] = 5

                            if "var_reductor" in self.model_cfg.net["pred_slice2vol"]:
                                if (
                                    self.model_cfg.net["pred_slice2vol"]["var_reductor"]
                                    is None
                                ):
                                    self.model_cfg.net["pred_slice2vol"][
                                        "var_reductor"
                                    ] = True
                            else:
                                self.model_cfg.net["pred_slice2vol"][
                                    "var_reductor"
                                ] = True

                            if "relative_MI" in self.model_cfg.net["pred_slice2vol"]:
                                if (
                                    self.model_cfg.net["pred_slice2vol"]["relative_MI"]
                                    is None
                                ):
                                    self.model_cfg.net["pred_slice2vol"][
                                        "relative_MI"
                                    ] = True
                            else:
                                self.model_cfg.net["pred_slice2vol"][
                                    "relative_MI"
                                ] = True

                            if "compute_mode" in self.model_cfg.net["pred_slice2vol"]:
                                if (
                                    self.model_cfg.net["pred_slice2vol"]["compute_mode"]
                                    is None
                                ):
                                    self.model_cfg.net["pred_slice2vol"][
                                        "compute_mode"
                                    ] = "mutual_inf"
                            else:
                                self.model_cfg.net["pred_slice2vol"][
                                    "compute_mode"
                                ] = "mutual_inf"

                            if "estabilizer" in self.model_cfg.net["pred_slice2vol"]:
                                if (
                                    self.model_cfg.net["pred_slice2vol"]["estabilizer"]
                                    is None
                                ):
                                    self.model_cfg.net["pred_slice2vol"][
                                        "estabilizer"
                                    ] = False
                            else:
                                self.model_cfg.net["pred_slice2vol"][
                                    "estabilizer"
                                ] = False

                            if "pertubations" in self.model_cfg.net["pred_slice2vol"]:
                                if (
                                    self.model_cfg.net["pred_slice2vol"]["pertubations"]
                                    is not None
                                ):
                                    if isinstance(
                                        self.model_cfg.net["pred_slice2vol"][
                                            "pertubations"
                                        ],
                                        str,
                                    ):
                                        pert_opt = True
                                    elif isinstance(
                                        self.model_cfg.net["pred_slice2vol"][
                                            "pertubations"
                                        ],
                                        bool,
                                    ):
                                        pert_opt = True
                                    elif isinstance(
                                        self.model_cfg.net["pred_slice2vol"][
                                            "pertubations"
                                        ],
                                        list,
                                    ):
                                        if (
                                            len(
                                                self.model_cfg.net["pred_slice2vol"][
                                                    "pertubations"
                                                ]
                                            )
                                            != 0
                                        ):
                                            pert_opt = True
                    else:
                        self.model_cfg.net["pred_slice2vol"]["uncertainity_map"] = False
                else:
                    self.model_cfg.net["pred_slice2vol"]["uncertainity_map"] = False

                # handle multi pred generation
                if "multi_pred_mode" in self.model_cfg.net["pred_slice2vol"]:
                    if (
                        self.model_cfg.net["pred_slice2vol"]["multi_pred_mode"]
                        is not None
                    ):
                        if (
                            self.model_cfg.net["pred_slice2vol"][
                                "multi_pred_mode"
                            ].lower()
                            != "single"
                        ):
                            if self.model_cfg.net["pred_slice2vol"]["n_samples"] <= 1:
                                print(
                                    "Number of samples are less or equal to 1 more are required for multi prediction usage"
                                )
                                print("Automatically 5 samples will use to prediction")
                                self.model_cfg.net["pred_slice2vol"]["n_samples"] = 5
                    else:
                        self.model_cfg.net["pred_slice2vol"][
                            "multi_pred_mode"
                        ] = "single"

                else:
                    self.model_cfg.net["pred_slice2vol"]["multi_pred_mode"] = "single"

                if "jupyter" in self.model_cfg.net["pred_slice2vol"]:
                    from tqdm.notebook import tqdm
                else:
                    from tqdm import tqdm

                for fn in tqdm(filenames, desc="Prediction for the Volume"):
                    try:
                        img = BioImage(fn, reader=bioio_tifffile.Reader).get_image_data(
                            self.data_cfg.inference_input.reader_params[
                                "dimension_order_out"
                            ],
                            T=self.data_cfg.inference_input.reader_params["T"],
                        )
                    except Exception:
                        try:
                            img = BioImage(fn).get_image_data(
                                self.data_cfg.inference_input.reader_params[
                                    "dimension_order_out"
                                ],
                                T=self.data_cfg.inference_input.reader_params["T"],
                            )
                        except Exception as e:
                            print(f"Error: {e}")
                            print(
                                f"Image {fn} failed at read process check the format."
                            )

                    voxel_sizes = PhysicalPixelSizes(1, 1, 1)
                    if vs_flag == "auto":
                        pps = getattr(BioImage(fn), "physical_pixel_sizes", None)
                        if pps is None:
                            voxel_sizes = PhysicalPixelSizes(None, None, None)
                        elif isinstance(pps, tuple):
                            voxel_sizes = pps  # tuple like (Z, Y, X)
                        else:
                            voxel_sizes = (
                                getattr(pps, "Z", None),
                                getattr(pps, "Y", None),
                                getattr(pps, "X", None),
                            )
                            voxel_sizes = PhysicalPixelSizes(
                                voxel_sizes[0], voxel_sizes[1], voxel_sizes[2]
                            )

                        voxel_sizes = [
                            1.0 if v is None else float(v) for v in voxel_sizes
                        ]
                        voxel_sizes = PhysicalPixelSizes(
                            voxel_sizes[0], voxel_sizes[1], voxel_sizes[2]
                        )
                    else:
                        voxel_sizes = vs_flag

                    if max_proj:
                        img = np.max(img, axis=1)
                        img = np.expand_dims(img, axis=1)

                    out_list = []
                    uncertainity_map = []
                    n = fn.name
                    if len(img.shape) == 3:
                        # chanel dummy
                        img = img[None, ...]

                    for zz in tqdm(
                        range(img.shape[1]), desc="infering slice", leave=False
                    ):
                        samplesz = []
                        im_input = img[:, zz, :, :]
                        for i in range(
                            self.model_cfg.net["pred_slice2vol"]["n_samples"]
                        ):

                            if pert_opt and i != 0:
                                inp = perturb_image(
                                    im_input,
                                    self.model_cfg.net["pred_slice2vol"][
                                        "pertubations"
                                    ],
                                )
                            else:
                                inp = im_input

                            logits = self.process_one_image(inp)
                            samplesz.append(np.squeeze(logits))

                        if (
                            self.model_cfg.net["pred_slice2vol"][
                                "multi_pred_mode"
                            ].lower()
                            == "single"
                        ):
                            seg = samplesz[0]
                            seg = seg[None, ...]
                        elif (
                            self.model_cfg.net["pred_slice2vol"][
                                "multi_pred_mode"
                            ].lower()
                            == "max"
                        ):
                            seg = max_prediction(samplesz)
                            seg = seg[None, ...]
                        elif (
                            self.model_cfg.net["pred_slice2vol"][
                                "multi_pred_mode"
                            ].lower()
                            == "mean"
                        ):
                            seg = mean_prediction(samplesz)
                            seg = seg[None, ...]
                        elif (
                            self.model_cfg.net["pred_slice2vol"][
                                "multi_pred_mode"
                            ].lower()
                            == "variance"
                        ):
                            seg = variance_prediction(samplesz)
                            seg = seg[None, ...]
                        elif (
                            self.model_cfg.net["pred_slice2vol"][
                                "multi_pred_mode"
                            ].lower()
                            == "sum"
                        ):
                            seg = add_prediction(samplesz)
                            seg = seg[None, ...]
                        else:
                            raise ValueError(
                                f"{self.model_cfg.net['pred_slice2vol']['multi_pred_mode']} is not a valid method for multi prediction use."
                            )

                        if original_postprocess is not None:
                            pp_data = seg
                            for pp_info in original_postprocess:
                                pp = parse_config(pp_info)
                                pp_data = pp(pp_data)
                            if torch.is_tensor(pp_data):
                                seg = pp_data.cpu().numpy()
                            else:
                                seg = pp_data
                        out_list.append(np.squeeze(seg))
                        if self.model_cfg.net["pred_slice2vol"]["uncertainity_map"]:
                            Uc_zmap = Extract_Uncertainty_Maps(
                                logits_samples=samplesz,
                                compute_mode=self.model_cfg.net["pred_slice2vol"][
                                    "compute_mode"
                                ],
                                relative_MI=self.model_cfg.net["pred_slice2vol"][
                                    "relative_MI"
                                ],
                                var_reductor=self.model_cfg.net["pred_slice2vol"][
                                    "var_reductor"
                                ],
                                estabilizer=self.model_cfg.net["pred_slice2vol"][
                                    "estabilizer"
                                ],
                            )
                            uncertainity_map.append(Uc_zmap)

                            # elimina _IM del name para vessqc
                            if "_IM" in n:
                                n = n.replace("_IM", "")

                    seg_full = np.stack(out_list, axis=0)
                    if self.model_cfg.net["pred_slice2vol"]["uncertainity_map"]:
                        UM_full = np.stack(uncertainity_map, axis=0)

                    if "remove_object_size" in self.model_cfg.net["pred_slice2vol"]:
                        if (
                            self.model_cfg.net["pred_slice2vol"]["remove_object_size"]
                            is not None
                        ):
                            seg_full = Remove_objects(
                                seg_full=seg_full,
                                n_classes=self.model_cfg.net["pred_slice2vol"][
                                    "n_class_correction"
                                ],
                                remove_object_size=self.model_cfg.net["pred_slice2vol"][
                                    "remove_object_size"
                                ],
                                voxel_sizes=tuple(voxel_sizes),
                            )

                    if "hole_size_threshold" in self.model_cfg.net["pred_slice2vol"]:
                        if (
                            self.model_cfg.net["pred_slice2vol"]["hole_size_threshold"]
                            is not None
                        ):
                            seg_full = Hole_Correction(
                                seg_full=seg_full,
                                n_classes=self.model_cfg.net["pred_slice2vol"][
                                    "n_class_correction"
                                ],
                                hole_size_threshold=self.model_cfg.net[
                                    "pred_slice2vol"
                                ]["hole_size_threshold"],
                                voxel_sizes=tuple(voxel_sizes),
                            )

                    if "min_thickness_list" in self.model_cfg.net["pred_slice2vol"]:
                        if (
                            self.model_cfg.net["pred_slice2vol"]["min_thickness_list"]
                            is not None
                        ):
                            seg_full = Thickness_Corretion(
                                seg_full=seg_full,
                                n_classes=self.model_cfg.net["pred_slice2vol"][
                                    "n_class_correction"
                                ],
                                min_thickness_physical=self.model_cfg.net[
                                    "pred_slice2vol"
                                ]["min_thickness_list"],
                                voxel_sizes=tuple(voxel_sizes),
                            )

                    if perycites_correction:
                        seg_full = Perycites_correction(seg_full=seg_full)

                    if ".tiff" in n:

                        out_fn = self.data_cfg.inference_output.path / n.replace(
                            ".tiff", "_segPred.tiff"
                        )
                        UM_out_fn = self.data_cfg.inference_output.path / n.replace(
                            ".tiff", "_uncertainty.tiff"
                        )
                    elif ".tif" in n:
                        out_fn = self.data_cfg.inference_output.path / n.replace(
                            ".tif", "_segPred.tif"
                        )
                        UM_out_fn = self.data_cfg.inference_output.path / n.replace(
                            ".tif", "_uncertainty.tif"
                        )

                    OmeTiffWriter.save(
                        data=seg_full,
                        uri=out_fn,
                        dim_order="ZYX",
                        physical_pixel_sizes=voxel_sizes,
                        physical_pixel_units="micron",
                    )

                    if self.model_cfg.net["pred_slice2vol"]["uncertainity_map"]:

                        if n_trunc >= 0:
                            UM_full = np.trunc(UM_full * (10**n_trunc)) / (10**n_trunc)

                        if threshold_um >= 0:
                            UM_full[UM_full < threshold_um] = 0

                        if border_corr:
                            nx, ny = border_corr
                            nx = abs(nx)
                            ny = abs(ny)
                            Z, X, Y = UM_full.shape
                            UM_full[:, : nx + 1, :] = 0
                            UM_full[:, X - (nx + 1) :, :] = 0
                            UM_full[:, :, : (ny + 1)] = 0
                            UM_full[:, :, Y - (ny + 1) :] = 0

                        if self.model_cfg.net["pred_slice2vol"]["var_reductor"]:
                            OmeTiffWriter.save(UM_full, UM_out_fn, dim_order="ZYX")

                        else:
                            UM_full_CZYX = np.moveaxis(UM_full, 1, 0)
                            OmeTiffWriter.save(
                                UM_full_CZYX, UM_out_fn, dim_order="CZYX"
                            )

            else:
                raise ValueError("Please provide params for the volumetric prediction.")
