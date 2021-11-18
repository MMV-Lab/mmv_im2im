from typing import Dict
import pytorch_lightning as pl
import torchio as tio
from mmv_im2im.utils.misc import parse_config, parse_config_func


class Model(pl.LightningModule):
    def __init__(self, model_info_xx: Dict):
        super().__init__()
        self.net = parse_config(model_info_xx["net"])
        self.criterion = parse_config(model_info_xx["criterion"])
        self.optimizer_func = parse_config_func(model_info_xx["optimizer"])

    def configure_optimizers(self):
        optimizer = self.optimizer_func(self.net.parameters())
        return optimizer

    def prepare_batch(self, batch):
        return 

    def forward(self, x):
        return self.net(x)

    def run_step(self, batch):
        if "costmap" in batch:
            costmap = batch.pop("costmap")
            costmap = costmap[tio.DATA]
        else:
            costmap = None

        x = batch['source'][tio.DATA]
        y = batch['target'][tio.DATA]

        y_hat = self(x)

        if costmap is None:
            loss = self.criterion(y_hat, y)
        else:
            loss = self.criterion(y_hat, y, costmap)

        return loss

    def training_step(self, batch, batch_idx):
        loss = self.run_step(batch)
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.run_step(batch)
        self.log('val_loss', loss)
        return loss
