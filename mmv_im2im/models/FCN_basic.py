from typing import Dict
import pytorch_lightning as pl
import torchio as tio

from mmv_im2im.utils.misc import parse_config, parse_config_func
from mmv_im2im.utils.piecewise_inference import predict_piecewise


class Model(pl.LightningModule):
    def __init__(self, model_info_xx: Dict, train: bool = True):
        super().__init__()
        self.net = parse_config(model_info_xx["net"])
        self.sliding_window = model_info_xx["sliding_window_params"]
        if train:
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
                x[0,:,:,:,0],
                **self.sliding_window
            ) # TODO HACK for now
        else:
            y_hat = self(x)

        y = y[:, :, :, :, 0]

        if costmap is None:
            loss = self.criterion(y_hat, y)
        else:
            loss = self.criterion(y_hat, y, costmap)

        return loss, y_hat

    def training_step(self, batch, batch_idx):
        loss, y_hat = self.run_step(batch, validation_stage=False)
        self.log("train_loss", loss, prog_bar=True)
        """
        src = batch["source"][tio.DATA]
        tar = batch["target"][tio.DATA]
        from tifffile import imsave
        from random import randint
        fn_rand = randint(100,900)
        imsave("./train_src_"+str(fn_rand)+".tiff", src[0,0,].cpu().numpy())
        imsave("./train_tar_"+str(fn_rand)+".tiff", tar[0,0,].cpu().numpy())
        # tensorboard = self.logger.experiment
        # tensorboard.add_image("train_source", src)
        # tensorboard.add_image("train_target", tar)
        # tensorboard.add_image("train_predict", y_hat)
        """
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat = self.run_step(batch, validation_stage=True)
        self.log("val_loss", loss)
        """
        src = batch["source"][tio.DATA]
        tar = batch["target"][tio.DATA]
        from tifffile import imsave
        from random import randint
        fn_rand = randint(100,900)
        imsave("./val_src_"+str(fn_rand)+".tiff", src[0,0,].cpu().numpy())
        imsave("./val_tar_"+str(fn_rand)+".tiff", tar[0,0,].cpu().numpy())
        # tensorboard = self.logger.experiment
        # tensorboard.add_image("val_source", src)
        # tensorboard.add_image("val_target", tar)
        # tensorboard.add_image("val_predict", y_hat)
        """
        return loss
