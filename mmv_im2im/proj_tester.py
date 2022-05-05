#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from importlib import import_module
from pathlib import Path
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
import torch
from torchio.data.io import check_uint_to_int
from mmv_im2im.utils.misc import generate_test_dataset_dict, parse_config_func
from mmv_im2im.utils.for_transform import parse_tio_ops
from mmv_im2im.utils.piecewise_inference import predict_piecewise

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
        self.run_cfg = cfg.run

        # define variables
        self.model = None
        self.data = None

    def run_inference(self):

        # set up model
        model_category = self.model_cfg.pop("category")
        model_module = import_module(f"mmv_im2im.models.basic_{model_category}")
        my_model_func = getattr(model_module, "Model")
        self.model = my_model_func.load_from_checkpoint(
            model_info_xx=self.model_cfg, train=False, **self.model_cfg["ckpt"]
        ).cuda()
        self.model.eval()

        # set up data
        dataset_list = generate_test_dataset_dict(
            self.data_cfg["input"]["dir"], **self.data_cfg["input"]["params"]
        )

        dataset_length = len(dataset_list)
        if "Z" in self.data_cfg["input"]["reader_params"]["dimension_order_out"]:
            self.spatial_dims = 3
        else:
            self.spatial_dims = 2

        # load preprocessing transformation
        pre_process = parse_tio_ops(self.data_cfg["preprocess"])

        # loop through all images and apply the model
        for i, ds in enumerate(dataset_list):
            # Read the image
            print(f"Reading the image {i}/{dataset_length}")
            img = AICSImage(ds).reader.get_image_dask_data(
                **self.data_cfg["input"]["reader_params"]
            )
            x = torch.tensor(check_uint_to_int(img.compute()))
            # Perform the prediction
            print("Predicting the image")
            if self.spatial_dims == 2:
                # if data is 2D
                # Initial shape (1, W, H)
                x = torch.unsqueeze(x, dim=-1)
                # Adding dimension for torchio to handle 2D data
                # to become (1, W, H, 1)
                x = pre_process(x)  # (1, resized_W, resized_H, 1)

                # TODO: some model requires (1, W, H), some requires (1, 1, W, H)
                # solution: update all 2D models to accept (1, W, H)
                # x = torch.unsqueeze(x, dim=0)
                x = torch.squeeze(x, dim=-1)

            # choose different inference function for different types of models
            with torch.no_grad():
                if "sliding_window_params" in self.model_cfg:
                    y_hat = predict_piecewise(
                        self.model,
                        x.float().cuda(),
                        **self.model_cfg["sliding_window_params"],
                    )
                else:
                    y_hat = self.model(x.float().cuda())

            # do post-processing on the prediction
            if "post_processing" in self.model_cfg:
                pp_data = y_hat
                for pp_info in self.model_cfg["post_processing"]:
                    pp = parse_config_func(pp_info)
                    pp_data = pp(pp_data)
                if torch.is_tensor(pp_data):
                    pred = pp_data.cpu().numpy()
                else:
                    pred = pp_data
            else:
                pred = y_hat.cpu().numpy()

            # prepare output filename
            fn_core = Path(ds).stem
            suffix = self.data_cfg["output"]["suffix"]
            out_fn = Path(self.data_cfg["output"]["path"]) / f"{fn_core}_{suffix}.tiff"

            # determine output dimension orders
            if len(pred.shape) == 2:
                OmeTiffWriter.save(pred, out_fn, dim_order="YX")
            elif len(pred.shape) == 3:
                if self.spatial_dims == 2:
                    OmeTiffWriter.save(pred, out_fn, dim_order="CYX")
                else:
                    OmeTiffWriter.save(pred, out_fn, dim_order="ZYX")
            elif len(pred.shape) == 4:
                if self.spatial_dims == 3:
                    OmeTiffWriter.save(pred, out_fn, dim_order="CZYX")
                else:
                    raise ValueError("4D output detected for 2d problem")
            elif len(pred.shape) == 5:
                assert pred.shape[0] == 1, "find non-trivial batch dimension"
                OmeTiffWriter.save(
                    pred[
                        0,
                    ],
                    out_fn,
                    dim_order="CZYX",
                )
            else:
                raise ValueError("error in prediction output shape")
