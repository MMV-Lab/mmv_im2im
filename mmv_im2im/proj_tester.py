#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from importlib import import_module
from pathlib import Path
import torchvision.transforms as transforms
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
import torch
from torchio.data.io import check_uint_to_int
import torchio as tio
from mmv_im2im.utils.for_transform import parse_tio_ops
from mmv_im2im.utils.misc import generate_test_dataset_dict
from mmv_im2im.utils.piecewise_inference import predict_piecewise
from skimage.morphology import remove_small_objects
from scipy.ndimage import median_filter



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
        pre_process = parse_tio_ops(self.data_cfg["preprocess"])

        # loop through all images and apply the model
        for ds in dataset_list:
            img = AICSImage(ds).reader.get_image_dask_data(
                **self.data_cfg["input"]["reader_params"]
            )
            x = check_uint_to_int(img.compute())
            print("Hello")
            print(x.shape)
            x = pre_process(x)
            # from tifffile import imsave
            # imsave("test_input_"+str(i)+".tiff", x)
            # import pdb
            # pdb.set_trace()
            y_hat = predict_piecewise(
                self.model,
                torch.from_numpy(x).float().cuda(),
                **self.model_cfg["sliding_window_params"],
            )
            print(y_hat.shape)

            # if '3d' in data_category:
            #     x_transformed = self.transform(x)
            #     y_hat = predict_piecewise(
            #         self.model,
            #         torch.from_numpy(x_transformed).float().cuda(),
            #         **self.model_cfg["sliding_window_params"],
            #     )
            #     print(y_hat.shape)
            #     log_softmax_layer = torch.nn.LogSoftmax(dim=1)
            #     y_hat = log_softmax_layer(y_hat)
            #     y_hat[y_hat >= -0.1] = 1.0
            #     y_hat[y_hat < -0.1] = 0.0
            #     y_hat = y_hat[:, 1, :, :, :]
            #     y_hat = remove_small_objects(y_hat.detach().cpu().numpy().astype(bool), min_size=5000)
            #     y_hat = median_filter(y_hat, size=3)
            #     y_hat = torch.from_numpy(y_hat).float().cuda()

            # elif '2d' in data_category:
            #     x = torch.tensor(x, dtype=float)
            #     x = torch.unsqueeze(x, dim=0)
            #     x_transformed = self.transform(x)
            #     y_hat = self.model(x_transformed.float().cuda())
            fn_core = Path(ds).stem
            suffix = self.data_cfg["output"]["suffix"]
            out_fn = (
                Path(self.data_cfg["output"]["path"]) / f"{fn_core}_{suffix}.tiff"
            )
            pred = y_hat.cpu().detach().numpy()
            # from tifffile import imsave
            # imsave("test_output_"+str(i)+".tiff", pred)
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
