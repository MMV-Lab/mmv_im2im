import numpy as np
from typing import Dict
from pathlib import Path
from random import randint
import lightning as pl
import torch
from bioio.writers import OmeTiffWriter

from mmv_im2im.utils.misc import (
    parse_config,
    parse_config_func,
    parse_config_func_without_params,
)
from mmv_im2im.utils.model_utils import init_weights


class Model(pl.LightningModule):
    def __init__(self, model_info_xx: Dict, train: bool = True, verbose: bool = False):
        super().__init__()
        self.net = parse_config(model_info_xx.net)
        init_weights(self.net, init_type="kaiming")

        self.model_info = model_info_xx
        self.verbose = verbose
        self.weighted_loss = False
        if train:
            self.criterion = parse_config(model_info_xx.criterion)
            self.optimizer_func = parse_config_func(model_info_xx.optimizer)

        # Store these as attributes for access in run_step/training_step/validation_step
        self.last_prior_mu = None
        self.last_prior_logvar = None
        self.last_post_mu = None
        self.last_post_logvar = None

    def forward(self, x, y=None):
        # The underlying ProbabilisticUNet returns multiple values.
        # Capture them here and store them as instance attributes.
        logits, prior_mu, prior_logvar, post_mu, post_logvar = self.net(x, y)

        # Store for use in run_step (which calculates loss)
        self.last_prior_mu = prior_mu
        self.last_prior_logvar = prior_logvar
        self.last_post_mu = post_mu
        self.last_post_logvar = post_logvar

        # For the 'Model' (LightningModule) forward, only return the logits
        # This makes the API consistent with other models in your framework.
        return logits

    def configure_optimizers(self):
        optimizer = self.optimizer_func(self.parameters())
        if self.model_info.scheduler is None:
            return optimizer
        else:
            scheduler_func = parse_config_func_without_params(self.model_info.scheduler)
            lr_scheduler = scheduler_func(
                optimizer, **self.model_info.scheduler["params"]
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": lr_scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                    "strict": True,
                },
            }

    def run_step(self, batch, validation_stage):
        x = batch["IM"]
        y = batch["GT"]

        if x.size(-1) == 1:
            x = torch.squeeze(x, dim=-1)
            y = torch.squeeze(y, dim=-1)

        # Call forward pass of the LightningModule.
        # This will internally call self.net(x,y) and store the extra outputs.
        logits = self(x, y)  # This is now just 'logits'

        # Calculate loss using the stored attributes
        # Ensure post_mu and post_logvar are not None if y was provided
        # The ELBOLoss expects these to be tensors, not None.
        if self.last_post_mu is None or self.last_post_logvar is None:
            raise ValueError(
                "Posterior distributions (mu, logvar) were not computed. Ensure 'y' is provided during training."
            )

        loss = self.criterion(
            logits,
            y,
            self.last_prior_mu,
            self.last_prior_logvar,
            self.last_post_mu,
            self.last_post_logvar,
        )

        return loss, logits

    def training_step(self, batch, batch_idx):
        loss, y_hat = self.run_step(batch, validation_stage=False)
        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        if self.verbose and batch_idx == 0:
            self.log_images(batch, y_hat, "train")

        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat = self.run_step(batch, validation_stage=True)
        self.log(
            "val_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        if self.verbose and batch_idx == 0:
            self.log_images(batch, y_hat, "val")

        return loss

    def log_images(self, batch, y_hat, stage):
        src = batch["IM"]
        tar = batch["GT"]

        save_path = Path(self.trainer.log_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        act = torch.nn.Softmax(dim=1)
        yhat_act = act(y_hat)

        src_out = np.squeeze(src[0].detach().cpu().numpy()).astype(float)
        tar_out = np.squeeze(tar[0].detach().cpu().numpy()).astype(float)
        prd_out = np.squeeze(yhat_act[0].detach().cpu().numpy()).astype(float)

        def get_dim_order(arr):
            dims = len(arr.shape)
            return {2: "YX", 3: "ZYX", 4: "CZYX"}.get(dims, "YX")

        rand_tag = randint(1, 1000)

        out_fn = save_path / f"epoch_{self.current_epoch}_{stage}_src_{rand_tag}.tiff"
        OmeTiffWriter.save(src_out, out_fn, dim_order=get_dim_order(src_out))

        out_fn = save_path / f"epoch_{self.current_epoch}_{stage}_tar_{rand_tag}.tiff"
        OmeTiffWriter.save(tar_out, out_fn, dim_order=get_dim_order(tar_out))

        out_fn = save_path / f"epoch_{self.current_epoch}_{stage}_prd_{rand_tag}.tiff"
        OmeTiffWriter.save(prd_out, out_fn, dim_order=get_dim_order(prd_out))
