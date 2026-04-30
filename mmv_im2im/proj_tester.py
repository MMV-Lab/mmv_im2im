#!/usr/bin/env python
# -*- coding: utf-8 -*-
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
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
import bioio_tifffile
from tqdm.auto import tqdm
from monai.inferers import sliding_window_inference
from monai.transforms import Compose

###############################################################################

log = logging.getLogger(__name__)
logging.getLogger("bioio").setLevel(logging.ERROR)

###############################################################################

# Module/function name pairs that activate the shared-state padding mechanism.
# Both the short name and the full mmv_im2im package path are accepted so that
# the YAML can use either style.
_PAD_PREPROCESS_TRIGGERS = {
    ("inverse_transforms", "RecordShapeAndPad"),
    ("custom_transforms", "RecordShapeAndPad"),
    ("mmv_im2im.utils.custom_transforms", "RecordShapeAndPad"),
    ("custom_transforms", "DivisiblePadWithGTAdjustd"),
    ("mmv_im2im.utils.custom_transforms", "DivisiblePadWithGTAdjustd"),
}
_PAD_POSTPROCESS_TRIGGERS = {
    ("inverse_transforms", "RemovePadFromPrediction"),
    ("custom_transforms", "RemovePadFromPrediction"),
    ("mmv_im2im.utils.custom_transforms", "RemovePadFromPrediction"),
}


def _get_trigger_key(cfg_entry: dict) -> tuple:
    """Devuelve (module_name, func_name) de una entrada de config."""
    return (
        cfg_entry.get("module_name", ""),
        cfg_entry.get("func_name", ""),
    )


###############################################################################


class ProjectTester(object):
    """
    Entry point for model inference.

    Parameters
    ----------
    cfg: configuration
    """

    def __init__(self, cfg):
        self.model_cfg = cfg.model
        self.data_cfg = cfg.data

        self.model = None
        self.data = None
        self.pre_process = None
        self.post_process_ops = None  # pre-built postprocess pipeline
        self.cpu = False
        self.spatial_dims = -1
        self.pad_state = None  # shared state for variable-size padding

        # Read task BEFORE setup_model() is called.
        # pl_FCN / pl_nnUnet / pl_ProbUnet all do net["params"].pop("task")
        # in __init__, so by the time setup_data_processing() runs the key
        # is gone from the dict.  Reading it here from the raw cfg is safe.
        net_params = cfg.model.net.get("params", {}) if cfg.model.net else {}
        self.is_regression = net_params.get("task", "segmentation") == "regression"

    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    def _detect_and_wire_pad_state(self):
        """
        Escanea los configs de preprocess y postprocess buscando nuestras
        transforms personalizadas de padding variable.

        Si se detecta alguna:
          - Crea un PadStateBuffer en self.pad_state.
          - Construye self.pre_process con el estado inyectado donde corresponde.
          - Construye self.post_process_ops (lista de instancias) con el estado
            inyectado en RemovePadFromPrediction.

        Si NO se detecta nada:
          - No modifica nada; setup_data_processing() usará el camino estándar.
        """
        pre_cfgs = self.data_cfg.preprocess or []
        post_cfgs = self.data_cfg.postprocess or []

        # ── 1. Detectar si hay algún trigger en preprocess ─────────────
        pad_pre_idx = None
        for i, cfg in enumerate(pre_cfgs):
            if _get_trigger_key(cfg) in _PAD_PREPROCESS_TRIGGERS:
                pad_pre_idx = i
                break

        if pad_pre_idx is None:
            return  # nada que hacer; comportamiento original

        # ── 2. Crear estado compartido ────────────────────────────────
        try:
            from mmv_im2im.utils.custom_transforms import (
                PadStateBuffer,
                RecordShapeAndPad,
            )
        except ImportError as e:
            raise ImportError(
                "Could not import PadStateBuffer/RecordShapeAndPad from "
                "mmv_im2im.utils.custom_transforms."
            ) from e

        self.pad_state = PadStateBuffer()
        print(
            f"[ProjectTester] PadStateBuffer created — "
            f"detected '{pre_cfgs[pad_pre_idx]['func_name']}' "
            f"in preprocess[{pad_pre_idx}]."
        )

        # ── 3. Construir pipeline de preprocess con estado inyectado ──
        pre_ops = []
        for i, cfg in enumerate(pre_cfgs):
            key = _get_trigger_key(cfg)

            if key in _PAD_PREPROCESS_TRIGGERS:
                # Extraer k, mode y constant_value del params original si existen
                params = cfg.get("params", {})
                k = params.get("k", 16)
                mode = params.get("mode", "constant")
                constant_value = params.get("constant_value", 0.0)

                op = RecordShapeAndPad(
                    state=self.pad_state,
                    k=k,
                    mode=mode,
                    constant_value=constant_value,
                )
                print(
                    f"[ProjectTester] preprocess[{i}] '{cfg['func_name']}' → "
                    f"RecordShapeAndPad(k={k}, mode='{mode}') with shared state."
                )
            else:
                # Transform estándar MONAI u otro: instanciar normalmente
                op = parse_config(cfg)

            pre_ops.append(op)

        self.pre_process = Compose(pre_ops)

        # ── 4. Construir pipeline de postprocess con estado inyectado ──
        if not post_cfgs:
            self.post_process_ops = []
            return

        try:
            from mmv_im2im.utils.custom_transforms import RemovePadFromPrediction
        except ImportError as e:
            raise ImportError(
                "Could not import RemovePadFromPrediction from "
                "mmv_im2im.utils.custom_transforms."
            ) from e

        post_ops = []
        for i, cfg in enumerate(post_cfgs):
            key = _get_trigger_key(cfg)

            if key in _PAD_POSTPROCESS_TRIGGERS:
                params = cfg.get("params", {})
                k = params.get("k", 16)
                n_coord_dims = params.get("n_coord_dims", 3)

                op = RemovePadFromPrediction(
                    state=self.pad_state,
                    k=k,
                    n_coord_dims=n_coord_dims,
                )
                print(
                    f"[ProjectTester] postprocess[{i}] 'RemovePadFromPrediction' "
                    f"with shared state (k={k}, n_coord_dims={n_coord_dims})."
                )
            else:
                op = parse_config(cfg)

            post_ops.append(op)

        self.post_process_ops = post_ops

    # ------------------------------------------------------------------
    def setup_data_processing(self):
        # Determinar dimensión espacial
        if "Z" in self.data_cfg.inference_input.reader_params["dimension_order_out"]:
            self.spatial_dims = 3
        else:
            self.spatial_dims = 2

        # Wire the shared PadStateBuffer if variable-size padding transforms are
        # present in the YAML. If no trigger is detected this call does nothing.
        self._detect_and_wire_pad_state()

        # Standard preprocess path — only if _detect_and_wire_pad_state did
        # not already build self.pre_process.
        if self.pre_process is None and self.data_cfg.preprocess is not None:
            self.pre_process = parse_monai_ops_vanilla(self.data_cfg.preprocess)

        # Standard postprocess path.
        # post_process_ops is None  → no trigger detected; transforms are
        #                             instantiated per image (legacy behaviour).
        # post_process_ops is []    → trigger detected but postprocess is empty.

    # ------------------------------------------------------------------
    def process_one_image(
        self, img: Union[DaskArray, NumpyArray], out_fn: Union[str, Path] = None
    ):
        if isinstance(img, DaskArray):
            x = img.compute()
        elif isinstance(img, NumpyArray):
            x = img
        else:
            raise ValueError("invalid image")

        if len(x.shape) == self.spatial_dims:
            x = np.expand_dims(x, axis=0)

        x = torch.tensor(x.astype(np.float32))

        # ── Preprocess ────────────────────────────────────────────────
        # Si pre_process es un Compose con RecordShapeAndPad, el estado
        # se actualiza aquí para esta imagen concreta.
        if self.pre_process is not None:
            x = self.pre_process(x)

        # ── Inferencia ────────────────────────────────────────────────
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
                        f"y_hat is a dict but key 'pred' was not found. "
                        f"y_hat keys: {list(y_hat.keys())}"
                    )

            # Global Average Pool — regression only, model-agnostic
            # -----------------------------------------------------------------
            # During training, GAP is applied inside each pl_*.run_step(),
            # collapsing [B, C, *spatial] → [B, C] before the loss.
            # During inference those training steps are bypassed: the raw
            # network output is [B, C, *spatial] regardless of which
            # architecture is used (AttentionUnet, DynUNet, ProbUnet all
            # share this contract).  GAP is therefore applied here once,
            # before any postprocess transform runs.
            if self.is_regression and y_hat.dim() > 2:
                import torch.nn.functional as F

                spatial_dims = y_hat.dim() - 2
                if spatial_dims == 2:
                    y_hat = F.adaptive_avg_pool2d(y_hat, 1).squeeze(-1).squeeze(-1)
                elif spatial_dims == 3:
                    y_hat = (
                        F.adaptive_avg_pool3d(y_hat, 1)
                        .squeeze(-1)
                        .squeeze(-1)
                        .squeeze(-1)
                    )
                # Remove the batch dimension → 1-D vector (C,)
                if y_hat.shape[0] == 1:
                    y_hat = y_hat.squeeze(0)

        # Postprocess
        if self.post_process_ops is not None:
            # Stateful path: pre-built pipeline with shared PadStateBuffer.
            # RemovePadFromPrediction already holds a reference to pad_state,
            # which was updated by RecordShapeAndPad during preprocess of this
            # image — the correct pad_before offsets are applied automatically.
            pp_data = y_hat
            for pp in self.post_process_ops:
                pp_data = pp(pp_data)
            pred = pp_data.cpu().numpy() if torch.is_tensor(pp_data) else pp_data

        elif self.data_cfg.postprocess is not None:
            # Standard path: instantiate each postprocess transform per image.
            pp_data = y_hat
            for pp_info in self.data_cfg.postprocess:
                pp = parse_config(pp_info)
                pp_data = pp(pp_data)
            pred = pp_data.cpu().numpy() if torch.is_tensor(pp_data) else pp_data

        else:
            pred = y_hat.cpu().numpy()

        if out_fn is None:
            return pred

        # Save result.
        # Regression predictions are always saved as .npy vectors regardless
        # of the suffix specified in the output config.
        if self.is_regression or out_fn.suffix == ".npy":
            out_fn = out_fn.with_suffix(".npy")
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
                            save_rgb(out_fn, np.moveaxis(pred[0,], 0, -1))
                        else:
                            OmeTiffWriter.save(pred[0,], out_fn, dim_order="CYX")
                    else:
                        raise ValueError("invalid 4D output for 2d data")
            elif len(pred.shape) == 5:
                assert pred.shape[0] == 1, "error, found non-trivial batch dimension"
                OmeTiffWriter.save(pred[0,], out_fn, dim_order="CZYX")
            else:
                raise ValueError("error in prediction output shape")

    # ------------------------------------------------------------------
    def run_inference(self):
        self.setup_model()
        self.setup_data_processing()

        dataset_list = generate_test_dataset_dict(
            self.data_cfg.inference_input.dir, self.data_cfg.inference_input.data_type
        )

        for ds in tqdm(dataset_list, desc="Predicting images"):
            fn_core = Path(ds).stem
            suffix = self.data_cfg.inference_output.suffix
            timelapse_data = 0

            if (
                "T"
                in self.data_cfg.inference_input.reader_params["dimension_order_out"]
            ):
                if "T" in self.data_cfg.inference_input.reader_params:
                    raise NotImplementedError(
                        "processing a subset of all timepoints is not supported yet"
                    )
                tmppath = tempfile.mkdtemp()
                print(f"making a temp folder at {tmppath}")

                try:
                    reader = BioImage(ds, reader=bioio_tifffile.Reader)
                except Exception:
                    try:
                        reader = BioImage(ds)
                    except Exception as e:
                        print(f"Error: {e}")
                        print(f"Image {ds} failed at read process check the format.")

                timelapse_data = reader.dims.T
                tmpfile_list = []

                for t_idx in tqdm(
                    range(timelapse_data), desc="Predicting image timepoint"
                ):
                    try:
                        img = BioImage(ds, reader=bioio_tifffile.Reader).get_image_data(
                            T=[t_idx], **self.data_cfg.inference_input.reader_params
                        )
                    except Exception:
                        try:
                            img = BioImage(ds).get_image_data(
                                T=[t_idx], **self.data_cfg.inference_input.reader_params
                            )
                        except Exception as e:
                            print(f"Error: {e}")
                            print(
                                f"Image {ds} failed at read process check the format."
                            )

                    out_fn = Path(tmppath) / f"{fn_core}_{t_idx}.npy"
                    self.process_one_image(img, out_fn)
                    tmpfile_list.append(out_fn)

                out_array = [np.load(f) for f in tmpfile_list]
                out_array = np.stack(out_array, axis=0)

                if "." in suffix:
                    if ".tif" in suffix or ".tiff" in suffix or ".ome.tif" in suffix:
                        out_fn = (
                            Path(self.data_cfg.inference_output.path)
                            / f"{fn_core}{suffix}"
                        )
                    else:
                        raise ValueError(
                            "please check output suffix, either unexpected dot or "
                            "unsupported fileformat"
                        )
                else:
                    out_fn = (
                        Path(self.data_cfg.inference_output.path)
                        / f"{fn_core}{suffix}.tiff"
                    )

                if len(out_array.shape) == 3:
                    dim_order = "TYX"
                elif len(out_array.shape) == 4:
                    dim_order = "TZYX" if self.spatial_dims == 3 else "TCYX"
                elif len(out_array.shape) == 5:
                    dim_order = "TCZYX"
                else:
                    raise ValueError(f"Unexpected pred of shape {out_array.shape}")

                OmeTiffWriter.save(out_array, out_fn, dim_order=dim_order)
                shutil.rmtree(tmppath)

            else:
                try:
                    img = BioImage(ds, reader=bioio_tifffile.Reader).get_image_data(
                        **self.data_cfg.inference_input.reader_params
                    )
                except Exception:
                    try:
                        img = BioImage(ds).get_image_data(
                            **self.data_cfg.inference_input.reader_params
                        )
                    except Exception as e:
                        print(f"Error: {e}")
                        print(f"Image {ds} failed at read process check the format.")

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
                            "please check output suffix, either unexpected dot or "
                            "unsupported fileformat"
                        )
                else:
                    out_fn = (
                        Path(self.data_cfg.inference_output.path)
                        / f"{fn_core}{suffix}.tiff"
                    )

                self.process_one_image(img, out_fn)
