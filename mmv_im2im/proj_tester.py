#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from importlib import import_module
from pathlib import Path
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
import torch
from torchio.data.io import check_uint_to_int
from mmv_im2im.utils.misc import generate_test_dataset_dict
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
        model_module = import_module(f"mmv_im2im.models.{model_category}_basic")
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
        self.spatial_dims = str(self.data_cfg["spatial_dims"])

        # load preprocessing transformation
        pre_process = parse_tio_ops(self.data_cfg["preprocess"])

        # loop through all images and apply the model
        for i, ds in enumerate(dataset_list):
            # Read the image
            print(f"Reading the image {i}/{dataset_length}")
            img = AICSImage(ds).reader.get_image_dask_data(
                **self.data_cfg["input"]["reader_params"]
            )
            x = check_uint_to_int(img.compute())
            # Perform the prediction
            print("Predicting the image")
            if self.spatial_dims == "2":
                # if data is 2D
                # Initial shape (1, W, H)
                x = torch.tensor(x)
                x = torch.unsqueeze(x, dim=-1)
                # Adding dimension for torchio to handle 2D data
                # to become (1, W, H, 1)
                x = pre_process(x)
                # preprocessed shape (1, resized_W, resized_H, 1)
                # The below operations are needed to reshape the data
                # for the model
                x = torch.unsqueeze(x, dim=0)
                x = torch.squeeze(x, dim=-1)
                with torch.no_grad():
                    y_hat = self.model(x.float().cuda())
                    pred = y_hat.cpu().detach().numpy()
            elif self.spatial_dims == "3":
                # if data is 3D
                x = pre_process(x)
                with torch.no_grad():
                    y_hat = predict_piecewise(
                        self.model,
                        torch.from_numpy(x).float().cuda(),
                        **self.model_cfg["sliding_window_params"],
                    )
                    pred = y_hat.cpu().detach().numpy()
            else:
                print("Data must be either 2D or 3D")
                # TO DO

            # prepare output
            fn_core = Path(ds).stem
            suffix = self.data_cfg["output"]["suffix"]
            out_fn = (
                Path(self.data_cfg["output"]["path"]) / f"{fn_core}_{suffix}.tiff"
            )  # noqa E501

            if len(pred.shape) == 4:
                OmeTiffWriter.save(pred, out_fn, dim_order="CZYX")
            elif len(pred.shape) == 3:
                OmeTiffWriter.save(pred, out_fn, dim_order="CYX")
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
                print("error in prediction")
