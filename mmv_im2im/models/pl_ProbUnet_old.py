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
        if train:
            self.criterion = parse_config(model_info_xx.criterion)
            self.optimizer_func = parse_config_func(model_info_xx.optimizer)

    def forward(self, x, y=None):
        outputs = self.net(x, y)
        return outputs[0]

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

        logits, prior_mu, prior_logvar, post_mu, post_logvar = self.net(x, y)

        if not validation_stage and (post_mu is None or post_logvar is None):
            raise ValueError(
                "Posterior distributions (mu, logvar) are None during training. "
                "Ensure 'y' is passed correctly to the network."
            )

        loss = self.criterion(
            logits,
            y,
            prior_mu,
            prior_logvar,
            post_mu,
            post_logvar,
            self.current_epoch,
        )
        return loss, logits.detach()

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
        with torch.no_grad():
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
        task = self.model_info.net["params"].get("task", "segment")

        save_path = Path(self.trainer.log_dir)
        save_path.mkdir(parents=True, exist_ok=True)

        if task == "segment":
            act = torch.nn.Softmax(dim=1)
            yhat_act = act(y_hat)

            if yhat_act.ndim > 1:
                prd_out = np.squeeze(yhat_act[0].cpu().numpy().argmax(axis=0)).astype(
                    float
                )
            else:
                prd_out = np.squeeze(yhat_act.cpu().numpy().argmax(axis=0)).astype(
                    float
                )

            tar_out = np.squeeze(tar[0].cpu().numpy()).astype(float)

        elif task == "regression":
            prd_out = np.squeeze(y_hat[0].cpu().numpy()).astype(float)
            tar_out = np.squeeze(tar[0].cpu().numpy()).astype(float)
        else:
            raise ValueError(f"Unknown task type for logging: {task}")

        def get_dim_order(arr):
            dims = len(arr.shape)
            return {2: "YX", 3: "CYX"}.get(dims, "YX")

        rand_tag = randint(1, 1000)

        src_out = np.squeeze(src[0].cpu().numpy()).astype(float)

        out_fn = save_path / f"epoch_{self.current_epoch}_{stage}_src_{rand_tag}.tiff"
        OmeTiffWriter.save(src_out, out_fn, dim_order=get_dim_order(src_out))

        out_fn = save_path / f"epoch_{self.current_epoch}_{stage}_tar_{rand_tag}.tiff"
        OmeTiffWriter.save(tar_out, out_fn, dim_order=get_dim_order(tar_out))

        out_fn = save_path / f"epoch_{self.current_epoch}_{stage}_prd_{rand_tag}.tiff"
        OmeTiffWriter.save(prd_out, out_fn, dim_order=get_dim_order(prd_out))
