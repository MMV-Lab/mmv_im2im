import pytorch_lightning as pl
import torchio as tio
from mmv_im2im.utils.misc import parse_config, parse_config_func


class Model(pl.LightningModule):
    def __init__(self, model_cfg):
        super().__init__()
        self.net = parse_config[model_cfg["net"]]
        self.criterion = parse_config[model_cfg["criterion"]]
        self.optimizer_func = parse_config_func[model_cfg["optimizer"]]

    def configure_optimizers(self):
        optimizer = self.optimizer_func(self.net.parameters())
        return optimizer

    def prepare_batch(self, batch):
        return batch['source'][tio.DATA], batch['target'][tio.DATA]

    def infer_batch(self, batch):
        x, y = self.prepare_batch(batch)
        y_hat = self.net(x)
        return y_hat, y

    def run_step(self, batch):
        if "costmap" in batch:
            costmap = batch.pop("costmap")
            costmap = costmap[tio.DATA]
        else:
            costmap = None

        y_hat, y = self.infer_batch(batch)

        if costmap is None:
            loss = self.criterion(y_hat, y)
        else:
            loss = self.criterion(y_hat, y, costmap)

        return loss

    def training_step(self, batch, batch_idx):
        self.optimizer.zero_grad()
        loss = self.run_step(batch)
        self.log('train_loss', loss, prog_bar=True)
        loss.backward()
        self.optimizer.step()
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.run_step(batch)
        self.log('val_loss', loss)
        return loss
