########################################################
# ####       general data module for          ########
# ####   paired and unpaired 3D/2D images         #######
# ###  (mostly for FCN or CGAN-like models)     ######
#
#
# About data in a batch:
# We woudl expect 3 parts, image_source, image_target,
# and image_cmap (cmap: cost map), where image_cmap
# can be optional. Note that image_target could be masks
# (e.g. for segmentation) or images (e.g. for labelfree)
########################################################
from typing import Union
from pathlib import Path
from torch.utils.data import random_split, DataLoader
import pytorch_lightning as pl
from mmv_im2im.utils.for_transform import parse_monai_ops  # , custom_preproc_to_tio
from mmv_im2im.utils.misc import (
    generate_dataset_dict_monai,
    parse_config_func_without_params,
)
import monai
from monai.data import list_data_collate


class Im2ImDataModule(pl.LightningDataModule):
    def __init__(self, data_cfg, cache_path: Union[str, Path] = None):
        super().__init__()

        if cache_path is None:
            self.data_path = data_cfg.data_path
        else:
            # use the cache path as the directory to load data
            self.data_path = cache_path

        # train/val split
        self.train_val_ratio = data_cfg.dataloader.train_val_ratio or 0.2
        self.train_set = None
        self.val_set = None

        if data_cfg.preprocess is None:
            self.preproc = None
        else:
            self.preproc = parse_monai_ops(data_cfg.preprocess)

        if data_cfg.augmentation is None:
            self.augment = None
        else:
            self.augment = parse_monai_ops(data_cfg.augmentation)

        if self.preproc is None and self.augment is not None:
            self.transform = self.augment
        elif self.preproc is not None and self.augment is None:
            self.transform = self.preproc
        elif self.preproc is None and self.augment is None:
            self.transform = None
        else:
            self.transform = monai.transforms.Compose([self.preproc, self.augment])

        # parameters for dataloader
        self.dataloader_info = data_cfg.dataloader

    def prepare_data(self):
        dataset_list = generate_dataset_dict_monai(self.data_path)

        """
        if self.category == "unpair":
            shuffled_dataset_list = dataset_list.copy()
            random.shuffle(shuffled_dataset_list)
            for i, shuffled_ds in enumerate(shuffled_dataset_list):
                dataset_list[i]["target_fn"] = shuffled_ds["target_fn"]
        """
        self.data = dataset_list

    def setup(self, stage=None):
        num_subjects = len(self.data)
        num_val_subjects = int(round(num_subjects * self.train_val_ratio))
        num_train_subjects = num_subjects - num_val_subjects
        splits = num_train_subjects, num_val_subjects
        train_subjects, val_subjects = random_split(self.data, splits)
        self.val_data = val_subjects
        self.train_data = train_subjects

    def train_dataloader(self):
        train_loader_info = self.dataloader_info.train
        train_dataset_func = parse_config_func_without_params(
            train_loader_info.dataloader_type
        )
        # gather the data filepath into dataset
        train_data = self.train_data
        if train_loader_info.partial_loader < 1.0:
            num_load = int(train_loader_info.partial_loader * len(self.train_data))
            from sklearn.utils import shuffle

            train_data = shuffle(train_data)
            train_data = train_data[:num_load]
        train_dataset = train_dataset_func(
            data=train_data,
            transform=self.transform,
            **train_loader_info.dataset_params
        )
        # wrap the dataset into dataloader
        train_dataloader = DataLoader(
            train_dataset,
            shuffle=True,
            collate_fn=list_data_collate,
            **train_loader_info.dataloader_params
        )
        return train_dataloader

    def val_dataloader(self):
        val_loader_info = self.dataloader_info.val
        val_dataset_func = parse_config_func_without_params(
            val_loader_info.dataloader_type
        )
        val_dataset = val_dataset_func(
            data=self.val_data,
            transform=self.preproc,
            **val_loader_info.dataset_params
        )
        val_dataloader = DataLoader(
            val_dataset,
            shuffle=True,
            collate_fn=list_data_collate,
            **val_loader_info.dataloader_params
        )
        return val_dataloader
