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
from torch.utils.data import random_split, DataLoader
import torchio as tio
import pytorch_lightning as pl
from mmv_im2im.utils.for_transform import parse_tio_ops
from mmv_im2im.utils.misc import generate_dataset_dict, aicsimageio_reader
from mmv_im2im.utils.embedseg_utils import generate_center_image


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
            self.transform = None
        else:
            self.transform = tio.Compose([self.preproc, self.augment])

        if "Z" in data_cfg["source_reader_params"]["dimension_order_out"]:
            self.spatial_dim = 3
        else:
            self.spatial_dim = 2

        # parameters for dataloader
        self.loader_params = data_cfg["dataloader_params"]
        # check patch load
        if "patch_loader" in data_cfg:
            raise NotImplementedError(
                "patch sampler is not implemented for embedseg yet"
            )

        # reserved for test data
        self.test_subjects = None
        self.test_set = None

    def prepare_data(self):
        dataset_list = generate_dataset_dict(self.data_path)

        target_reader = partial(aicsimageio_reader, **self.target_reader_param)
        source_reader = partial(aicsimageio_reader, **self.source_reader_param)

        self.subjects = []
        for ds in dataset_list:
            # the data need to be in format of CXYZ, so that torchio transformation can be applied
            subject = tio.Subject(
                source=tio.ScalarImage(ds["source_fn"], reader=source_reader),
                target=tio.LabelMap(ds["target_fn"], reader=target_reader),
                #center_image=tio.LabelMap(ds["target_fn"], reader=target_reader),
                #class_image=tio.LabelMap(ds["target_fn"], reader=target_reader),
            )
            # TODO: add costmap support?

            # try a simple version to test torchio subjects
            instance = subject["target"][tio.DATA]
            center_image = instance
            object_mask = instance
            subject.add_image(tio.LabelMap(tensor=center_image), "center_image")
            subject.add_image(tio.LabelMap(tensor=object_mask), "class_image")
            """
            # generate center image and label image
            instance = np.squeeze(subject["target"][tio.DATA])
            instance_np = np.array(instance, copy=False)  # TODO: why resave?
            object_mask = instance_np > 0

            ids = np.unique(instance_np[object_mask])
            ids = ids[ids != 0]
            center_image = generate_center_image(
                instance,
                "centroid",
                ids,
                anisotropy_factor=1,
                speed_up=1,
            )
            if len(center_image.shape) == 2:
                # pad Z and C dimensions
                # this is just a temporary solution, we may want to switch to MONAI
                center_image = center_image[np.newaxis, :, :, np.newaxis]
                object_mask = object_mask[np.newaxis, :, :, np.newaxis]
            subject.add_image(tio.LabelMap(tensor=center_image), "center_image")
            subject.add_image(tio.LabelMap(tensor=object_mask), "class_image")
            """

            self.subjects.append(subject)

    def setup(self, stage=None):
        num_subjects = len(self.subjects)
        print("Total exmaples: ....")
        print(num_subjects)
        num_val_subjects = int(round(num_subjects * self.train_val_ratio))
        num_train_subjects = num_subjects - num_val_subjects
        splits = num_train_subjects, num_val_subjects
        train_subjects, val_subjects = random_split(self.subjects, splits)
        self.val_set = tio.SubjectsDataset(val_subjects, transform=self.preproc)
        self.train_set = tio.SubjectsDataset(train_subjects, transform=self.transform)

    def train_dataloader(self):
        print(" ... dm train loader ....")
        dl = DataLoader(self.train_set, shuffle=True, **self.loader_params["train"])
        print("train loader done")
        return dl

    def val_dataloader(self):
        print(" ... dm val loader ....")
        dl= DataLoader(self.val_set, shuffle=False, **self.loader_params["val"])
        print("val loader done ..")
        return dl

    def test_dataloader(self):
        print(" ... dm test loader ....")
        # need to be overwritten in a test script for specific test case
        pass
