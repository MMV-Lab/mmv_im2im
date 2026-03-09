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
logging.getLogger("bioio").setLevel(logging.ERROR)

###############################################################################


class MapExtractor(object):
    """
    Entry for training/inference models.

    Parameters
    ----------
    cfg: configuration
    """

    def __init__(self, cfg):
        # extract the three major chuck of the config
        self.model_cfg = cfg.model
        self.data_cfg = cfg.data

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
        self,
        img: Union[DaskArray, NumpyArray],
        dim: int = 2,
        out_fn: Union[str, Path] = None,
    ):

        if isinstance(img, DaskArray):
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
            if dim == 2:
                x = x[0]

        # choose different inference function for different types of models
        with torch.no_grad():
            if self.cpu:
                x = torch.unsqueeze(x, dim=0)
            else:
                x = torch.unsqueeze(x, dim=0).cuda()

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

                if not self.cpu:
                    y_hat = y_hat.cuda()
            else:
                y_hat = self.model(x)

            if isinstance(y_hat, dict):
                try:
                    y_hat = y_hat["pred"]
                except Exception:
                    raise ValueError(
                        f"y_hat is a dictionary but the key 'pred' it's not found the y_hat output is:  {y_hat}"
                    )

        # do post-processing on the prediction (if not handled by caller)
        # Note: In the refactored run_inference, post-processing is largely handled externally
        # for aggregation, but we keep this for direct calls or simple single-inference
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

        # Save logic (kept from original for compatibility with direct calls)
        self._save_to_disk(pred, out_fn)

    def _save_to_disk(self, pred, out_fn):
        """Helper to save prediction based on shape."""
        if out_fn.suffix == ".npy":
            np.save(out_fn, pred)
        else:
            if len(pred.shape) == 2:
                OmeTiffWriter.save(pred, out_fn, dim_order="YX")
            elif len(pred.shape) == 3:
                if self.spatial_dims == 2:
                    if pred.shape[0] == 3:
                        if out_fn.suffix != ".png":
                            out_fn = out_fn.with_suffix(".png")
                        save_rgb(out_fn, np.moveaxis(pred, 0, -1))
                    else:
                        OmeTiffWriter.save(pred, out_fn, dim_order="CYX")
                elif self.spatial_dims == 3:
                    OmeTiffWriter.save(pred, out_fn, dim_order="ZYX")
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
                            save_rgb(out_fn, np.moveaxis(pred[0,], 0, -1))
                        else:
                            OmeTiffWriter.save(pred[0,], out_fn, dim_order="CYX")
            elif len(pred.shape) == 5:
                assert pred.shape[0] == 1
                OmeTiffWriter.save(pred[0,], out_fn, dim_order="CZYX")

    def _get_filenames(self):
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
        return filenames

    def _determine_voxel_size(self, fn, pred_cfg):
        vs_flag = PhysicalPixelSizes(1, 1, 1)
        if "pixel_dim" in pred_cfg and pred_cfg["pixel_dim"] is not None:
            if isinstance(pred_cfg["pixel_dim"], str):
                vs_flag = "auto"
            elif (
                isinstance(pred_cfg["pixel_dim"], (tuple, list))
                and len(pred_cfg["pixel_dim"]) == 3
            ):
                z, y, x = pred_cfg["pixel_dim"]
                vs_flag = PhysicalPixelSizes(z, y, x)

        voxel_sizes = PhysicalPixelSizes(1, 1, 1)
        if vs_flag == "auto":
            try:
                pps = getattr(BioImage(fn), "physical_pixel_sizes", None)
                if pps is None:
                    voxel_sizes = PhysicalPixelSizes(None, None, None)
                elif isinstance(pps, tuple):
                    voxel_sizes = pps
                else:
                    voxel_sizes = PhysicalPixelSizes(
                        getattr(pps, "Z", None),
                        getattr(pps, "Y", None),
                        getattr(pps, "X", None),
                    )

                # Cleanup Nones
                v_list = [
                    1.0 if v is None else float(v)
                    for v in (voxel_sizes.Z, voxel_sizes.Y, voxel_sizes.X)
                ]
                voxel_sizes = PhysicalPixelSizes(*v_list)
            except Exception:
                pass
        else:
            voxel_sizes = vs_flag
        return voxel_sizes

    def _post_process_volume(self, seg_full, UM_full, voxel_sizes, pred_cfg):
        """Common post-processing logic for the 3D volume (after assembly or direct prediction)."""

        # Remove object size
        if (
            "remove_object_size" in pred_cfg
            and pred_cfg["remove_object_size"] is not None
        ):
            seg_full = Remove_objects(
                seg_full=seg_full,
                n_classes=pred_cfg["n_class_correction"],
                remove_object_size=pred_cfg["remove_object_size"],
                voxel_sizes=tuple((voxel_sizes.Z, voxel_sizes.Y, voxel_sizes.X)),
            )

        # Hole correction
        if (
            "hole_size_threshold" in pred_cfg
            and pred_cfg["hole_size_threshold"] is not None
        ):
            seg_full = Hole_Correction(
                seg_full=seg_full,
                n_classes=pred_cfg["n_class_correction"],
                hole_size_threshold=pred_cfg["hole_size_threshold"],
                voxel_sizes=tuple((voxel_sizes.Z, voxel_sizes.Y, voxel_sizes.X)),
            )

        # Thickness correction
        if (
            "min_thickness_list" in pred_cfg
            and pred_cfg["min_thickness_list"] is not None
        ):
            seg_full = Thickness_Corretion(
                seg_full=seg_full,
                n_classes=pred_cfg["n_class_correction"],
                min_thickness_physical=pred_cfg["min_thickness_list"],
                voxel_sizes=tuple((voxel_sizes.Z, voxel_sizes.Y, voxel_sizes.X)),
            )

        # Pericytes correction
        perycites_correction = pred_cfg.get("perycites_correction", False)
        if perycites_correction:
            seg_full = Perycites_correction(seg_full=seg_full)

        # Uncertainty Map Post-processing
        if pred_cfg.get("uncertainity_map", False) and UM_full is not None:
            n_trunc = pred_cfg.get("trunc", 4)
            if not isinstance(n_trunc, int):
                n_trunc = 4  # Default fallback

            threshold_um = pred_cfg.get("threshold", -1)
            if not isinstance(threshold_um, (int, float)):
                threshold_um = -1

            border_corr = pred_cfg.get("border_correction", False)

            if n_trunc >= 0:
                UM_full = np.trunc(UM_full * (10**n_trunc)) / (10**n_trunc)

            if threshold_um >= 0:
                UM_full[UM_full < threshold_um] = 0

            if border_corr:
                # Parse border correction params
                nx, ny = 0, 0
                if isinstance(border_corr, int) and border_corr > 0:
                    nx = ny = border_corr
                elif isinstance(border_corr, list) and len(border_corr) > 0:
                    if len(border_corr) == 1:
                        nx = ny = border_corr[0]
                    else:
                        nx, ny = border_corr[0], border_corr[1]

                if nx > 0 or ny > 0:
                    Z, X, Y = UM_full.shape
                    UM_full[:, : nx + 1, :] = 0
                    UM_full[:, X - (nx + 1) :, :] = 0
                    UM_full[:, :, : (ny + 1)] = 0
                    UM_full[:, :, Y - (ny + 1) :] = 0

        return seg_full, UM_full

    def _save_results(self, seg_full, UM_full, fn, voxel_sizes, pred_cfg, output_dir):
        n = fn.name
        if ".tiff" in n:
            out_fn = output_dir / n.replace(".tiff", "_segPred.tiff")
            UM_out_fn = output_dir / n.replace(".tiff", "_uncertainty.tiff")
        elif ".tif" in n:
            out_fn = output_dir / n.replace(".tif", "_segPred.tif")
            UM_out_fn = output_dir / n.replace(".tif", "_uncertainty.tif")
        else:
            # Fallback
            out_fn = output_dir / (n + "_segPred.tif")
            UM_out_fn = output_dir / (n + "_uncertainty.tif")

        OmeTiffWriter.save(
            data=seg_full,
            uri=out_fn,
            dim_order="ZYX",
            physical_pixel_sizes=voxel_sizes,
            physical_pixel_units="micron",
        )

        if pred_cfg.get("uncertainity_map", False) and UM_full is not None:
            if pred_cfg.get("var_reductor", True):
                OmeTiffWriter.save(UM_full, UM_out_fn, dim_order="ZYX")
            else:
                UM_full_CZYX = np.moveaxis(UM_full, 1, 0)
                OmeTiffWriter.save(UM_full_CZYX, UM_out_fn, dim_order="CZYX")

    def _process_vol2slice(self, img, pred_cfg, original_postprocess, pert_opt):
        """Original slice-by-slice processing logic."""
        out_list = []
        uncertainity_map = []

        # Handle dummy channel if missing
        if len(img.shape) == 3:
            img = img[None, ...]  # (C, Z, Y, X)

        if "jupyter" in self.model_cfg.net["pred_slice2vol"]:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm

        for zz in tqdm(range(img.shape[1]), desc="infering slice", leave=False):
            samplesz = []
            im_input = img[:, zz, :, :]  # (C, Y, X)

            for i in range(pred_cfg["n_samples"]):
                if pert_opt and i != 0:
                    inp = perturb_image(im_input, pred_cfg["pertubations"])
                else:
                    inp = im_input

                logits = self.process_one_image(inp, dim=2)
                samplesz.append(np.squeeze(logits))

            # Multi-prediction aggregation
            mode = pred_cfg.get("multi_pred_mode", "single").lower()
            if mode == "single":
                seg = samplesz[0][None, ...]
            elif mode == "max":
                seg = max_prediction(samplesz)[None, ...]
            elif mode == "mean":
                seg = mean_prediction(samplesz)[None, ...]
            elif mode == "variance":
                seg = variance_prediction(samplesz)[None, ...]
            elif mode == "sum":
                seg = add_prediction(samplesz)[None, ...]
            else:
                raise ValueError(f"{mode} is not valid.")

            # Per-slice Post-process (e.g. from config)
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

            # Uncertainty map
            if pred_cfg.get("uncertainity_map", False):
                Uc_zmap = Extract_Uncertainty_Maps(
                    logits_samples=samplesz,
                    compute_mode=pred_cfg.get("compute_mode", "mutual_inf"),
                    relative_MI=pred_cfg.get("relative_MI", True),
                    var_reductor=pred_cfg.get("var_reductor", True),
                    estabilizer=pred_cfg.get("estabilizer", False),
                )
                uncertainity_map.append(Uc_zmap)

        seg_full = np.stack(out_list, axis=0)
        UM_full = None
        if pred_cfg.get("uncertainity_map", False):
            UM_full = np.stack(uncertainity_map, axis=0)

        return seg_full, UM_full

    def _process_vol2vol(self, img, pred_cfg, original_postprocess, pert_opt):
        """New direct 3D volume processing logic."""
        # Input img is (C, Z, Y, X) or (Z, Y, X)
        # Handle dummy channel if missing
        if len(img.shape) == 3:
            img = img[None, ...]  # (C, Z, Y, X)

        samples_vol = []

        # In vol2vol, we perturb the entire 3D volume
        for i in range(pred_cfg["n_samples"]):
            if pert_opt and i != 0:
                inp = perturb_image(img, pred_cfg["pertubations"])
            else:
                inp = img

            # Process one image will handle 4D input by adding batch dim -> (1, C, Z, Y, X)
            # Ensure spatial_dims is set to 3 in setup
            logits = self.process_one_image(inp, dim=3)
            samples_vol.append(np.squeeze(logits))

        # Multi-prediction aggregation (3D)
        mode = pred_cfg.get("multi_pred_mode", "single").lower()
        if mode == "single":
            seg = samples_vol[0]
        elif mode == "max":
            seg = max_prediction(samples_vol)
        elif mode == "mean":
            seg = mean_prediction(samples_vol)
        elif mode == "variance":
            seg = variance_prediction(samples_vol)
        elif mode == "sum":
            seg = add_prediction(samples_vol)
        else:
            raise ValueError(f"{mode} is not valid.")

        # Post-process (Volume level)
        if original_postprocess is not None:
            pp_data = seg
            for pp_info in original_postprocess:
                pp = parse_config(pp_info)
                pp_data = pp(pp_data)
            if torch.is_tensor(pp_data):
                seg = pp_data.cpu().numpy()
            else:
                seg = pp_data

        seg_full = seg  # Shape (Z, Y, X) or (C, Z, Y, X)
        # Ensure seg_full is (Z, Y, X) if single class, usually squeezed.
        if seg_full.ndim == 4 and seg_full.shape[0] == 1:
            seg_full = seg_full[0]

        UM_full = None
        if pred_cfg.get("uncertainity_map", False):
            UM_full = Extract_Uncertainty_Maps(
                logits_samples=samples_vol,
                compute_mode=pred_cfg.get("compute_mode", "mutual_inf"),
                relative_MI=pred_cfg.get("relative_MI", True),
                var_reductor=pred_cfg.get("var_reductor", True),
                estabilizer=pred_cfg.get("estabilizer", False),
            )

        return seg_full, UM_full

    def run_inference(self):
        self.setup_model()
        self.setup_data_processing()

        if (
            "pred_slice2vol" not in self.model_cfg.net
            or self.model_cfg.net["pred_slice2vol"] is None
        ):
            raise ValueError(
                "Please provide params for the volumetric prediction in 'pred_slice2vol'."
            )

        pred_cfg = self.model_cfg.net["pred_slice2vol"]
        filenames = self._get_filenames()

        print(
            f"########################################### {len(filenames)} Volume(s) found for prediction ###########################################"
        )

        if (
            pred_cfg.get("uncertainity_map", False)
            and "prob" not in self.model_cfg.net["func_name"].lower()
        ):
            print(
                "Warning: Model might not be probabilistic. Uncertainty maps may be invalid."
            )

        # Save post process indicated by the user
        original_postprocess = self.data_cfg.postprocess
        # We set non postprocess because we need logits for the uncertainty calculation
        self.data_cfg.postprocess = None

        # Defaults for configuration if missing
        if pred_cfg.get("n_samples", 0) <= 1:
            if (
                pred_cfg.get("uncertainity_map", False)
                or pred_cfg.get("multi_pred_mode", "single") != "single"
            ):
                print("Setting n_samples to 5 for uncertainty/multi-pred.")
                pred_cfg["n_samples"] = 5

        # Check perturbation options
        pert_opt = False
        p = pred_cfg.get("pertubations", None)
        if isinstance(p, (str, bool)) and p:
            pert_opt = True
        elif isinstance(p, list) and len(p) > 0:
            pert_opt = True

        # Determine Inference Mode
        # Default: "vol2slice"
        inference_mode = pred_cfg.get("inference_mode", "vol2slice").lower()

        if "jupyter" in self.model_cfg.net["pred_slice2vol"]:
            from tqdm.notebook import tqdm
        else:
            from tqdm import tqdm

        for fn in tqdm(filenames, desc="Prediction for the Volume"):
            # Load Image
            try:
                reader_params = self.data_cfg.inference_input.reader_params
                img = BioImage(fn, reader=bioio_tifffile.Reader).get_image_data(
                    reader_params["dimension_order_out"], T=reader_params.get("T", None)
                )
            except Exception:
                try:
                    img = BioImage(fn).get_image_data(
                        reader_params["dimension_order_out"],
                        T=reader_params.get("T", None),
                    )
                except Exception as e:
                    print(f"Error: {e}. Image {fn} failed at read.")
                    continue

            voxel_sizes = self._determine_voxel_size(fn, pred_cfg)

            # Pre-calc Max Projection if requested (Usually specific to 2D logic, but check)
            if pred_cfg.get("max_proj", False):
                # (Z, Y, X) -> max over Z (axis 0) or axis 1?
                # Original code used axis=1 on shape (C, Z, Y, X) -> Z dim.
                # Assuming standard C, Z, Y, X input after get_image_data if dims match
                img = np.max(img, axis=1)
                img = np.expand_dims(img, axis=1)

            # DISPATCHER
            if inference_mode == "vol2vol":
                seg_full, UM_full = self._process_vol2vol(
                    img, pred_cfg, original_postprocess, pert_opt
                )
            else:
                seg_full, UM_full = self._process_vol2slice(
                    img, pred_cfg, original_postprocess, pert_opt
                )

            # Post-Processing (Morphology, Hole Filling, etc.)
            seg_full, UM_full = self._post_process_volume(
                seg_full, UM_full, voxel_sizes, pred_cfg
            )

            # Save
            self._save_results(
                seg_full,
                UM_full,
                fn,
                voxel_sizes,
                pred_cfg,
                self.data_cfg.inference_output.path,
            )
