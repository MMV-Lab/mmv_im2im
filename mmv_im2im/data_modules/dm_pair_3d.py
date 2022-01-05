########################################################
# #### general data module for paired 3D images ########
# ####   (mostly for FCN or CGAN-like models)     ######
#
# About transformation:
# We use TorchIO, which can handle 3D data in a more
# efficient way than torchvision
#
# About data in a batch:
# We woudl expect 3 parts, image_source, image_target,
# and image_cmap (cmap: cost map), where image_cmap
# can be optional. Note that image_target could be masks
# (e.g. for segmentation) or images (e.g. for labelfree)
########################################################
from importlib import import_module
from functools import partial
from torch.utils.data import random_split, DataLoader
import torchio as tio
import pytorch_lightning as pl

from mmv_im2im.utils.for_transform import parse_tio_ops
from mmv_im2im.utils.misc import generate_dataset_dict, aicsimageio_reader


class Im2ImDataModule(pl.LightningDataModule):
    def __init__(self, data_cfg):
        super().__init__()

        self.target_type = data_cfg["target_type"]
        self.source_type = data_cfg["source_type"]
        self.target_reader_param = data_cfg["target_reader_params"]
        self.source_reader_param = data_cfg["source_reader_params"]
        self.data_path = data_cfg["data_path"]

        # all subjects
        self.subjects = None

        # train/val split
        self.train_val_ratio = data_cfg["train_val_ratio"] or 0.2
        self.train_set = None
        self.val_set = None

        # transformation
        self.preproc = parse_tio_ops(data_cfg["preprocess"])
        self.augment = parse_tio_ops(data_cfg["augmentation"])
        self.transform = tio.Compose([self.preproc, self.augment])

        # parameters for dataloader
        self.loader_params = data_cfg["dataloader_params"]
        if "dataloader_patch_queue" in data_cfg:
            self.patch_loader = True
            self.patch_loader_params = data_cfg["dataloader_patch_queue"][
                "params"
            ]  # noqa E501
            self.patch_loader_sampler = data_cfg["dataloader_patch_queue"][
                "sampler"
            ]  # noqa E501
        else:
            self.patch_loader = False

        # reserved for test data
        self.test_subjects = None
        self.test_set = None

    def prepare_data(self):
        dataset_list = generate_dataset_dict(self.data_path)

        # parse source and target type
        tio_image_module = import_module("torchio")
        if self.target_type.lower() == "label":
            target_image_class = getattr(tio_image_module, "LabelMap")
        elif self.target_type.lower() == "image":
            target_image_class = getattr(tio_image_module, "ScalarImage")
        else:
            print("unsupported target type")
        target_reader = partial(aicsimageio_reader, **self.target_reader_param)

        if self.source_type.lower() == "label":
            source_image_class = getattr(tio_image_module, "LabelMap")
        elif self.source_type.lower() == "image":
            source_image_class = getattr(tio_image_module, "ScalarImage")
        else:
            print("unsupported source type")
        source_reader = partial(aicsimageio_reader, **self.source_reader_param)

        self.subjects = []
        for ds in dataset_list:
            if "costmap_fn" in ds:
                subject = tio.Subject(
                    source=source_image_class(ds["source_fn"], reader=source_reader),
                    target=target_image_class(ds["target_fn"], reader=target_reader),
                    costmap=tio.ScalarImage(ds["costmap_fn"]),
                )
            else:
                subject = tio.Subject(
                    source=source_image_class(ds["source_fn"], reader=source_reader),
                    target=target_image_class(ds["target_fn"], reader=target_reader),
                )
            self.subjects.append(subject)

    def setup(self, stage=None):
        num_subjects = len(self.subjects)
        num_val_subjects = int(round(num_subjects * self.train_val_ratio))
        num_train_subjects = num_subjects - num_val_subjects
        splits = num_train_subjects, num_val_subjects
        train_subjects, val_subjects = random_split(self.subjects, splits)

        self.val_set = tio.SubjectsDataset(val_subjects, transform=self.preproc)

        train_set = tio.SubjectsDataset(train_subjects, transform=self.transform)
        if self.patch_loader:
            # define sampler
            sampler_module = import_module("torchio.data")
            sampler_func = getattr(
                sampler_module, self.patch_loader_sampler["name"]
            )  # noqa E501
            train_sampler = sampler_func(**self.patch_loader_sampler["params"])
            self.train_set = tio.Queue(
                train_set, sampler=train_sampler, **self.patch_loader_params
            )
        else:
            self.train_set = train_set

    def train_dataloader(self):
        return DataLoader(self.train_set, shuffle=False, **self.loader_params["train"])

    def val_dataloader(self):
        return DataLoader(self.val_set, shuffle=False, **self.loader_params["val"])

    def test_dataloader(self):
        # need to be overwritten in a test script for specific test case
        pass
