########################################################
# ########     data module for embedseg         ########
#
# About transformation:
# We use TorchIO, which can handle 3D/2D data in a more
# efficient way than torchvision
#
# About data in a batch:
# We woudl expect 3 parts, image_source, instance, center
########################################################
from functools import partial
import numpy as np
from torch.utils.data import random_split
import torchio as tio
import pytorch_lightning as pl
from mmv_im2im.utils.for_transform import parse_tio_ops  # , custom_preproc_to_tio
from mmv_im2im.utils.misc import generate_dataset_dict, aicsimageio_reader
from mmv_im2im.utils.embedseg_utils import generate_center_image
import logging


class Im2ImDataModule(pl.LightningDataModule):
    def __init__(self, data_cfg):
        super().__init__()

        self.target_type = data_cfg["target_type"]
        self.source_type = data_cfg["source_type"]
        self.target_reader_param = data_cfg["target_reader_params"]
        self.source_reader_param = data_cfg["source_reader_params"]
        self.data_path = data_cfg["data_path"]

        # check category
        category = data_cfg["category"]
        if not category == "embedseg":
            raise NotImplementedError("only catergory=embedsegXD is supported")

        # all subjects
        self.subjects = None

        # train/val split
        self.train_val_ratio = data_cfg["train_val_ratio"] or 0.2
        self.train_set = None
        self.val_set = None

        if "preprocess" in data_cfg:
            self.preproc = parse_tio_ops(data_cfg["preprocess"])
        else:
            self.preproc = None

        if "augmentation" in data_cfg:
            self.augment = parse_tio_ops(data_cfg["augmentation"])
        else:
            self.augment = None

        if self.preproc is None and self.augment is not None:
            self.transform = self.augment
        elif self.preproc is not None and self.augment is None:
            self.transform = self.preproc
        elif self.preproc is None and self.augment is None:
            self.transform is None
        else:
            self.transform = tio.Compose([self.preproc, self.augment])

        if "Z" in data_cfg["source_reader_params"]["dimension_order_out"]:
            self.spatial_dim = 3
        else:
            self.spatial_dim = 2

        # reserved for test data
        self.test_subjects = None
        self.test_set = None

    def prepare_data(self):
        dataset_list = generate_dataset_dict(self.data_path)

        target_reader = partial(aicsimageio_reader, **self.target_reader_param)
        source_reader = partial(aicsimageio_reader, **self.source_reader_param)

        self.subjects = []
        for ds in dataset_list:
            subject = tio.Subject(
                source=tio.ScalarImage(ds["source_fn"], reader=source_reader),
                target=tio.LabelMap(ds["target_fn"], reader=target_reader),
            )
            # TODO: add costmap support?

            # generate center image and label image
            instance = subject["target"][tio.DATA]
            instance_np = np.array(instance, copy=False)  #  TODO: why resave?
            object_mask = instance_np > 0

            ids = np.unique(instance_np[object_mask])
            ids = ids[ids != 0]
            center_image = generate_center_image(
                instance,
                "Centroid",
                ids,
                anisotropy_factor=1,
                speed_up=1,
            )
            subject.add_image(tio.LabelMap(center_image), "center_image")
            subject.add_image(tio.LabelMap(object_mask), "class_image")

            self.subjects.append(subject)

    def setup(self, stage=None):
        num_subjects = len(self.subjects)
        num_val_subjects = int(round(num_subjects * self.train_val_ratio))
        num_train_subjects = num_subjects - num_val_subjects
        splits = num_train_subjects, num_val_subjects
        train_subjects, val_subjects = random_split(self.subjects, splits)
        self.val_set = tio.SubjectsDataset(val_subjects, transform=self.preproc)
        self.train_set = tio.SubjectsDataset(train_subjects, transform=self.transform)
        if self.patch_loader:
            raise NotImplementedError(
                "patch sampler is not implemented for embedseg yet"
            )
