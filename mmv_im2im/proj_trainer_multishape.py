#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
---------------
Entry point for training the spherical-harmonic regression model with
VARIABLE-SIZE inputs.

Key changes vs. the original proj_trainer.py
---------------------------------------------
1.  The data module is wrapped in VariableSizeDataModule so that every
    batch is dynamically padded to a common size that is divisible by 16,
    and the x0 coordinates in the GT vector are corrected for the padding
    offset.

2.  The per-sample transform DivisiblePadWithGTAdjustd is added to the
    preprocess pipeline in the YAML (see Attention_Unet_Reg_variable.yaml),
    so that SINGLE-SAMPLE operations (batch_size=1, augmentation caching,
    etc.) also work correctly.

3.  When batch_size=1 is used (the safest choice) the custom collate is
    technically not needed, but VariableSizeDataModule is still applied
    for consistency and forward compatibility.

"""

import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)
import logging
from importlib import import_module
from pathlib import Path

import lightning as pl
import pyrallis
import torch

from mmv_im2im.data_modules import get_data_module
from mmv_im2im.utils.misc import parse_ops_list
from mmv_im2im.utils.nnHeuristic import get_nnunet_plans

# Variable-size additions
from mmv_im2im.utils.variable_datamodule import VariableSizeDataModule

torch.set_float32_matmul_precision("medium")

log = logging.getLogger(__name__)
logging.getLogger("bioio").setLevel(logging.ERROR)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class VariableSizeProjectTrainer:
    """
    Trainer with support for variable spatial-size inputs.

    The only behavioural difference from the original ProjectTrainer is
    that the DataModule is wrapped in VariableSizeDataModule which injects
    a custom collate function that:
      - Pads all images in each batch to the same spatial size (max dims
        in the batch, rounded up to the nearest multiple of 16).
      - Adjusts the first 3 elements of the GT vector (x0: z, y, x centre
        coordinates) to reflect the padding offset.
      - Leaves all spherical-harmonic coefficients untouched.

    Parameters
    ----------
    cfg : ProgramConfig
    """

    def __init__(self, cfg):
        pl.seed_everything(123, workers=True)
        self.model_cfg = cfg.model
        self.train_cfg = cfg.trainer
        self.data_cfg = cfg.data
        self.model = None
        self.data = None

    # ------------------------------------------------------------------
    @staticmethod
    def _k_from_strides(strides) -> int:
        """
        Computes the minimum spatial divisibility k required by a UNet
        whose encoder strides are ``strides`` (list of per-layer stride
        lists, e.g. [[1,1],[2,2],[2,2],[2,2]]).

        k = product of all strides along each spatial axis, then take
        the maximum across axes (handles anisotropic downsampling).

        Examples
        --------
        AttentionUnet strides=[1,2,2,2,2]         -> k = 2^4 = 16
        DynUNet 6 DS  strides=[[1,1],[2,2]x6]     -> k = 2^6 = 64
        """
        # Normalise: accept both flat [1,2,2,2] and nested [[1,1],[2,2],...]
        if not isinstance(strides[0], (list, tuple)):
            strides = [[s] for s in strides]

        n_dims = len(strides[0])
        k_per_dim = [1] * n_dims
        for stride_layer in strides:
            for i, s in enumerate(stride_layer):
                k_per_dim[i] *= s
        return int(max(k_per_dim))

    # ------------------------------------------------------------------
    def run_training(self):

        # ── 1. Build the base data module ─────────────────────────────
        base_data = get_data_module(self.data_cfg)

        dynunet_info = None
        collate_k = 16
        if self.model_cfg.net["func_name"] == "DynUNet":
            extra_params = (
                self.data_cfg.extra if self.data_cfg.extra is not None else {}
            )
            patch_size = extra_params.get("patch_size", [256, 256])
            spacing = extra_params.get("spacing", [1.0, 1.0])
            modality = extra_params.get("modality", "non-CT")
            min_size = extra_params.get("min_size", 8)
            plans = get_nnunet_plans(patch_size, spacing, modality, min_size=min_size)
            self.model_cfg.net["params"].update(
                {
                    "kernel_size": plans["kernel_size"],
                    "strides": plans["strides"],
                    "filters": plans["filters"],
                    "upsample_kernel_size": plans["upsample_kernel_size"],
                }
            )
            collate_k = self._k_from_strides(plans["strides"])

            dynunet_info = (
                f"nnU-Net configured for {len(patch_size)}D.\n"
                f"[VariableSizeProjectTrainer] DynUNet: {len(plans['strides'])-1} downsampling stages -> collate k={collate_k}\n"
                f"Filters: {plans['filters']}\n"
                f"Strides: {plans['strides']}\n"
                f"Kernel size: {plans['kernel_size']}\n"
                f"Upsample Kernel size: {plans['upsample_kernel_size']}\n"
            )
            print(f"✅ nnU-Net configured for {len(patch_size)}D.")
            print(
                f"[VariableSizeProjectTrainer] DynUNet: "
                f"{len(plans['strides'])-1} downsampling stages "
                f"-> collate k={collate_k}"
            )
            print(f"Filters: {plans['filters']}")
            print(f"Strides: {plans['strides']}")
            print(f"Kernel size: {plans['kernel_size']}")
            print(f"Upsample Kernel size: {plans['upsample_kernel_size']}")

        # ── 2. Wrap data module with the correct k ─────────────────────
        # DivisiblePadWithGTAdjustd (per-sample, in YAML) may use a smaller
        # k. The collate pads further to collate_k and re-adjusts GT coords.
        self.data = VariableSizeDataModule(
            base_data,
            k=collate_k,
            mode="constant",
            constant_value=0.0,
            n_coord_dims=3,
        )

        model_category = self.model_cfg.framework
        model_module = import_module(f"mmv_im2im.models.pl_{model_category}")
        my_model_func = getattr(model_module, "Model")
        self.model = my_model_func(self.model_cfg, verbose=self.train_cfg.verbose)

        # ── 4. Optional weight loading (unchanged) ─────────────────────
        if self.model_cfg.model_extra is not None:
            if "resume" in self.model_cfg.model_extra:
                self.model = self.model.load_from_checkpoint(
                    self.model_cfg.model_extra["resume"]
                )
            elif "pre-train" in self.model_cfg.model_extra:
                pre_train = torch.load(
                    self.model_cfg.model_extra["pre-train"], weights_only=False
                )
                if "extend" in self.model_cfg.model_extra:
                    if (
                        self.model_cfg.model_extra["extend"] is not None
                        and self.model_cfg.model_extra["extend"] is True
                    ):
                        pre_train["state_dict"].pop("criterion.xym", None)
                        model_state = self.model.state_dict()
                        pretrained_dict = pre_train["state_dict"]
                        filtered_dict = {
                            k: v
                            for k, v in pretrained_dict.items()
                            if k in model_state and v.shape == model_state[k].shape
                        }
                        model_state.update(filtered_dict)
                        self.model.load_state_dict(model_state, strict=False)
                else:
                    pre_train["state_dict"].pop("criterion.xym", None)
                    self.model.load_state_dict(pre_train["state_dict"], strict=False)

        # ── 5. Build the Lightning Trainer (unchanged) ─────────────────
        if self.train_cfg.callbacks is None:
            trainer = pl.Trainer(**self.train_cfg.params)
        else:
            callback_list = parse_ops_list(self.train_cfg.callbacks)
            trainer = pl.Trainer(callbacks=callback_list, **self.train_cfg.params)

        # ── 6. Save configs ────────────────────────────────────────────
        save_path = Path(trainer.log_dir)
        if trainer.local_rank == 0:
            save_path.mkdir(parents=True, exist_ok=True)
            pyrallis.dump(
                self.model_cfg,
                open(save_path / "model_config.yaml", "w"),
            )
            pyrallis.dump(
                self.train_cfg,
                open(save_path / "train_config.yaml", "w"),
            )
            pyrallis.dump(
                self.data_cfg,
                open(save_path / "data_config.yaml", "w"),
            )
            if dynunet_info is not None:
                nnunet_cfg_path = save_path / "nnUnet_parameter_generation.txt"
                with open(nnunet_cfg_path, "w") as f:
                    f.write(dynunet_info)
                print(
                    f"Inferred model configuration saved in -> {nnunet_cfg_path} for inference configuration"
                )

        print("Starting training with variable-size input support...")
        trainer.fit(model=self.model, datamodule=self.data)
