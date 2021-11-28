from typing import Dict
import pytorch_lightning as pl
import torchio as tio
from mmv_im2im.utils.misc import parse_config, parse_config_func
from mmv_im2im.utils.piecewise_inference import predict_piecewise


class Model(pl.LightningModule):
    def __init__(self, model_info_xx: Dict):
        super().__init__()
        self.net = parse_config(model_info_xx["net"])
        self.criterion = parse_config(model_info_xx["criterion"])
        self.optimizer_func = parse_config_func(model_info_xx["optimizer"])

    def configure_optimizers(self):
        optimizer = self.optimizer_func(self.parameters())
        return optimizer

    def prepare_batch(self, batch):
        return

    def forward(self, x):
        return self.net(x)

    def run_step(self, batch, validation_stage):
        if "costmap" in batch:
            costmap = batch.pop("costmap")
            costmap = costmap[tio.DATA]
        else:
            costmap = None

        x = batch["source"][tio.DATA]
        y = batch["target"][tio.DATA]

        if validation_stage:
            y_hat = predict_piecewise(
                self,
                x[0,],
                dims_max=[1, 64, 128, 128],
                overlaps=[0, 6, 12, 12]
            )
        else:
            y_hat = self(x)

        if costmap is None:
            loss = self.criterion(y_hat, y)
        else:
            loss = self.criterion(y_hat, y, costmap)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.run_step(batch, validation_stage=False)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.run_step(batch, validation_stage=True)
        self.log("val_loss", loss)
        return loss
