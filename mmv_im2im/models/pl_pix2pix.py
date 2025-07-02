from pathlib import Path
from typing import Dict
from importlib import import_module
import numpy as np
from tifffile import imsave
from skimage.io import imsave as save_rgb
import lightning as pl
import torch
from mmv_im2im.utils.misc import parse_config_func
from mmv_im2im.utils.model_utils import init_weights, state_dict_simplification
from mmv_im2im.models.nets.gans import define_generator, define_discriminator


class Model(pl.LightningModule):
    def __init__(self, model_info_xx: Dict, train: bool = True, verbose: bool = False):
        super(Model, self).__init__()

        # Important: This property activates manual optimization.
        self.automatic_optimization = False

        self.verbose = verbose
        gen_init = None
        dis_init = None
        if "init_weight" in model_info_xx.net["generator"]:
            gen_init = model_info_xx.net["generator"].pop("init_weight")
        if "init_weight" in model_info_xx.net["discriminator"]:
            dis_init = model_info_xx.net["discriminator"].pop("init_weight")
        self.generator = define_generator(model_info_xx.net["generator"])
        self.discriminator = define_discriminator(model_info_xx.net["discriminator"])
        # self.sliding_window = model_info_xx["sliding_window_params"]
        if train:
            # initialize model weights
            if gen_init is not None:
                if Path(gen_init).is_file:
                    print(f"loading pre-train from {gen_init}")
                    try:
                        pre_train = torch.load(Path(gen_init))
                    except RuntimeError:
                        pre_train = torch.load(
                            Path(gen_init), map_location=torch.device("cpu")
                        )
                    try:
                        self.generator.load_state_dict(pre_train["state_dict"])
                    except Exception:
                        cleaned_state_dict = state_dict_simplification(
                            pre_train["state_dict"], cut="generator."
                        )
                        self.generator.load_state_dict(cleaned_state_dict, strict=False)
                else:
                    init_weights(self.generator, init_type=gen_init)
            else:
                init_weights(self.generator, init_type="normal")

            if dis_init is not None:
                if Path(dis_init).is_file:
                    try:
                        pre_train = torch.load(Path(dis_init))
                    except RuntimeError:
                        pre_train = torch.load(
                            Path(dis_init), map_location=torch.device("cpu")
                        )
                    self.discriminator.load_state_dict(pre_train["state_dict"])
                else:
                    init_weights(self.discriminator, init_type=dis_init)
            else:
                init_weights(self.discriminator, init_type="normal")

            # get loss functions
            loss_category = model_info_xx.criterion.pop("loss_type")
            loss_module = import_module("mmv_im2im.utils.gan_losses")
            loss_func = getattr(loss_module, loss_category)
            self.loss = loss_func(**model_info_xx.criterion)

            # get info of optimizer and scheduler
            self.optimizer_info = model_info_xx.optimizer
            self.scheduler_info = model_info_xx.scheduler

    def forward(self, x):
        return self.generator(x)

    def configure_optimizers(self):
        discriminator_optimizer_func = parse_config_func(
            self.optimizer_info["discriminator"]
        )
        discriminator_scheduler_func = parse_config_func(
            self.scheduler_info["discriminator"]
        )
        discriminator_optimizer = discriminator_optimizer_func(
            self.discriminator.parameters()
        )
        discriminator_scheduler = discriminator_scheduler_func(discriminator_optimizer)

        generator_optimizer_func = parse_config_func(self.optimizer_info["generator"])
        generator_scheduler_func = parse_config_func(self.scheduler_info["generator"])
        generator_optimizer = generator_optimizer_func(
            self.generator.parameters(),
        )
        generator_scheduler = generator_scheduler_func(generator_optimizer)

        return (
            [discriminator_optimizer, generator_optimizer],
            [
                discriminator_scheduler,
                generator_scheduler,
            ],
        )

    def save_pix2pix_output(self, image_A, image_B, fake_image, current_stage):
        # check if the log path exists, if not create one
        log_dir = Path(self.trainer.log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        current_epoch = self.current_epoch

        if len(fake_image.shape) == 3 and fake_image.shape[0] == 3:
            out_fn = log_dir / f"{current_stage}_epoch_{current_epoch}_fake_B.png"
            save_rgb(out_fn, np.moveaxis(fake_image, 0, -1))

            out_fn = log_dir / f"{current_stage}_epoch_{current_epoch}_real_B.png"
            save_rgb(out_fn, np.moveaxis(image_B, 0, -1))

            out_fn = log_dir / f"{current_stage}_epoch_{current_epoch}_real_A.png"
            save_rgb(out_fn, np.moveaxis(image_A, 0, -1))
        else:
            out_fn = log_dir / f"{current_stage}_epoch_{current_epoch}_fake_B.tiff"
            imsave(out_fn, fake_image)

            out_fn = log_dir / f"{current_stage}_epoch_{current_epoch}_real_B.tiff"
            imsave(out_fn, image_B)

            out_fn = log_dir / f"{current_stage}_epoch_{current_epoch}_real_A.tiff"
            imsave(out_fn, image_A)

    def training_step(self, batch):
        # imageB : real image
        # imageA : condition image
        # conditional GAN refers generating a realistic image (image B as ground truth),
        # conditioned on image A
        image_A = batch["IM"]
        image_B = batch["GT"]

        # get optimizer:
        d_opt, g_opt = self.optimizers()

        #######################
        # calculate loss
        #######################
        # for generator
        self.toggle_optimizer(g_opt)
        fake_B = self.generator(image_A)
        g_loss = self.loss.generator_step(self.discriminator, image_A, image_B, fake_B)

        self.log(
            "train_g_loss",
            g_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        g_opt.zero_grad()
        self.manual_backward(g_loss)
        g_opt.step()
        self.untoggle_optimizer(g_opt)

        # for discriminator
        self.toggle_optimizer(d_opt)
        fake_B = self.generator(image_A).detach()
        d_loss = self.loss.discriminator_step(
            self.discriminator, image_A, image_B, fake_B
        )

        self.log(
            "train_d_loss",
            d_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        d_opt.zero_grad()
        self.manual_backward(d_loss)
        d_opt.step()
        self.untoggle_optimizer(d_opt)

        sch_d, sch_g = self.lr_schedulers()
        if self.trainer.is_last_batch:
            sch_g.step()
            sch_d.step()

    def validation_step(self, batch, batch_idx):
        image_A = batch["IM"]
        image_B = batch["GT"]

        fake_B = self.generator(image_A).detach()
        d_loss = self.loss.discriminator_step(
            self.discriminator, image_A, image_B, fake_B
        )
        self.log(
            "val_d_loss",
            d_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        g_loss = self.loss.generator_step(self.discriminator, image_A, image_B, fake_B)
        self.log(
            "val_g_loss",
            g_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )

        if self.verbose and batch_idx == 0:
            fake_images = fake_B.detach()
            fake_image = fake_images[0].cpu().numpy()

            image_A0 = image_A[0].detach().cpu().numpy()
            image_B0 = image_B[0].detach().cpu().numpy()

            self.save_pix2pix_output(image_A0, image_B0, fake_image, "val")
