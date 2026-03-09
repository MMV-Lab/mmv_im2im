import lightning as pl
import torch
from mmv_im2im.utils.misc import (
    parse_config,
    parse_config_func,
    parse_config_func_without_params,
)
from mmv_im2im.utils.model_utils import init_weights


class Model(pl.LightningModule):
    def __init__(self, model_info_xx, train=True, verbose=False):
        super().__init__()
        self.net = parse_config(model_info_xx.net)
        init_weights(self.net, init_type="kaiming")
        self.model_info = model_info_xx
        self.verbose = verbose
        if train:
            self.criterion = parse_config(model_info_xx.criterion)
            self.optimizer_func = parse_config_func(model_info_xx.optimizer)

    def configure_optimizers(self):
        optimizer = self.optimizer_func(self.parameters())
        if self.model_info.scheduler is None:
            return optimizer
        scheduler_func = parse_config_func_without_params(self.model_info.scheduler)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler_func(
                    optimizer, **self.model_info.scheduler["params"]
                ),
                "monitor": "val_loss",
            },
        }

    def forward(self, x, seg=None, train_posterior=False):
        return self.net(x, seg, train_posterior)

    def run_step(self, batch):
        x, y = batch["IM"], batch["GT"]

        if x.ndim > 4 and x.shape[-1] == 1:
            x = x.squeeze(-1)

        if y.ndim > 4 and y.shape[-1] == 1:
            y = y.squeeze(-1)

        if y.ndim == x.ndim - 1:
            y = y.unsqueeze(1)

        # Forward pass (Train Posterior)
        output = self(x, seg=y, train_posterior=True)

        current_ep = int(self.current_epoch)

        loss = self.criterion(
            logits=output["pred"],
            y_true=y,  # (B, 1, H, W) Integer labels usually
            prior_mu=output["prior_mu"],
            prior_logvar=output["prior_logvar"],
            post_mu=output["mu_post"],
            post_logvar=output["logvar_post"],
            epoch=current_ep,
        )
        return loss

    def on_train_epoch_end(self):
        torch.cuda.synchronize()

    def on_validation_epoch_end(self):
        torch.cuda.synchronize()

    def training_step(self, batch, batch_idx):
        loss = self.run_step(batch)
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, logger=True)

        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.run_step(batch)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True, logger=True)

        return loss
