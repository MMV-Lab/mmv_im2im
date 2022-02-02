########################################################
# ####       general data module for          ########
# ####   paired and unpaired 3D images         #######
# ###  (mostly for FCN or CGAN-like models)     ######
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
import sys
import pytorch_lightning as pl
from mmv_im2im.utils.for_transform import parse_tio_ops
from mmv_im2im.utils.misc import generate_dataset_dict, aicsimageio_reader
import random


class Im2ImDataModule(pl.LightningDataModule):
    def __init__(self, data_cfg):
        super().__init__()

        self.target_type = data_cfg["target_type"]
        self.source_type = data_cfg["source_type"]
        self.target_reader_param = data_cfg["target_reader_params"]
        self.source_reader_param = data_cfg["source_reader_params"]
        self.data_path = data_cfg["data_path"]
        self.shuffle_data = data_cfg["shuffle_data"]

        # all subjects
        self.subjects = None

        # train/val split
        self.train_val_ratio = data_cfg["train_val_ratio"] or 0.2
        self.train_set = None
        self.val_set = None
        # transformation
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

        self.spatial_dims = str(data_cfg["spatial_dims"])

        # parameters for dataloader
        self.loader_params = data_cfg["dataloader_params"]
        if ("dataloader_patch_queue" in data_cfg) \
                and (self.spatial_dims == "3"):
            print("The dimensions of the data is 3D")
            self.patch_loader = True
            self.patch_loader_params = data_cfg["dataloader_patch_queue"][
                "params"
            ]  # noqa E501
            self.patch_loader_sampler = data_cfg["dataloader_patch_queue"][
                "sampler"
            ]  # noqa E501
        elif ("dataloader_patch_queue" not in data_cfg) \
                and (self.spatial_dims == "2"):
            print("The dimensions of the data is 2D")
            self.patch_loader = False
        else:
            print("The dimensions of the data are invalid")

        # reserved for test data
        self.test_subjects = None
        self.test_set = None

    def prepare_data(self):
        dataset_list = generate_dataset_dict(self.data_path)
        dataset_list = generate_dataset_dict(self.data_path)
        shuffled_dataset_list = dataset_list.copy()
        if self.shuffle_data:
            print("Printing shuffled datalist")
            random.shuffle(shuffled_dataset_list)
            for i, shuffled_ds in enumerate(shuffled_dataset_list):
                dataset_list[i]["target_fn"] = shuffled_ds["target_fn"]
        # parse source and target type
        tio_image_module = import_module("torchio")
        if self.target_type.lower() == "label":
            target_image_class = getattr(tio_image_module, "LabelMap")
        elif self.target_type.lower() == "image":
            target_image_class = getattr(tio_image_module, "ScalarImage")
        else:
            print("unsupported target type")
            sys.exit(0)
        target_reader = partial(aicsimageio_reader, **self.target_reader_param)

        if self.source_type.lower() == "label":
            source_image_class = getattr(tio_image_module, "LabelMap")
        elif self.source_type.lower() == "image":
            source_image_class = getattr(tio_image_module, "ScalarImage")
        else:
            print("unsupported source type")
            sys.exit(0)
        source_reader = partial(aicsimageio_reader, **self.source_reader_param)

        self.subjects = []
        for ds in dataset_list:
            if "costmap_fn" in ds:
                subject = tio.Subject(
                    source=source_image_class(ds["source_fn"], reader=source_reader),  # noqa: E501
                    target=target_image_class(ds["target_fn"], reader=target_reader),  # noqa: E501
                    costmap=tio.ScalarImage(ds["costmap_fn"]),
                )
            else:
                subject = tio.Subject(
                    source=source_image_class(ds["source_fn"], reader=source_reader),  # noqa: E501
                    target=target_image_class(ds["target_fn"], reader=target_reader),  # noqa: E501
                )
            self.subjects.append(subject)

    def setup(self, stage=None):
        num_subjects = len(self.subjects)
        num_val_subjects = int(round(num_subjects * self.train_val_ratio))
        num_train_subjects = num_subjects - num_val_subjects
        splits = num_train_subjects, num_val_subjects
        train_subjects, val_subjects = random_split(self.subjects, splits)
        self.val_set = tio.SubjectsDataset(val_subjects, transform=self.preproc)  # noqa: E501
        train_set = tio.SubjectsDataset(train_subjects, transform=self.transform)  # noqa: E501
        if self.patch_loader:
            # define sampler
            sampler_module = import_module("torchio.data")
            sampler_func = getattr(
                sampler_module, self.patch_loader_sampler["name"]
            )   # noqa: E501
            train_sampler = sampler_func(**self.patch_loader_sampler["params"])
            self.train_set = tio.Queue(
                train_set, sampler=train_sampler, **self.patch_loader_params
            )   # noqa: E501
        else:
            self.train_set = train_set

    def train_dataloader(self):
        return DataLoader(self.train_set, shuffle=True, **self.loader_params["train"])  # noqa: E501

    def val_dataloader(self):
        return DataLoader(self.val_set, shuffle=False, **self.loader_params["val"])   # noqa: E501

    def test_dataloader(self):
        # need to be overwritten in a test script for specific test case
        pass
