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
from glob import glob
import os
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
        self.target_reader = partial(aicsimageio_reader, **self.target_reader_param)
        self.source_reader = partial(aicsimageio_reader, **self.source_reader_param)
        self.data_path = data_cfg["data_path"]
        if "cache_path" in data_cfg:
            self.cache_path = data_cfg["cache_path"]
        else:
            self.cache_path = None

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
        """Note from Pytorch Lightning documentation:
        prepare_data is called from the main process. It is not recommended
        to assign state here (e.g. self.x = y).
        """

        # check if cropped images have been cached.
        if self.cache_path:
            raw_count = len(glob(self.cache_path + os.sep + "*_IM.tiff"))
            center_count = len(glob(self.cache_path + os.sep + "*_CE.tiff"))
            class_count = len(glob(self.cache_path + os.sep + "*_CL.tiff"))
            if raw_count == center_count == class_count > 0:
                print("cache is found. Start running ...")
                return

        # If cache is not found, generate crop images
        dataset_list = generate_dataset_dict(self.data_path)

        # determine crop size
        # (TODO: currently only support single channel input 2D/3D)
        from aicsimageio import AICSImage
        from aicsimageio.writers import OmeTiffWriter
        from tqdm import tqdm

        min_xy = 65535
        min_z = 65535
        for ds in dataset_list:
            reader = AICSImage(ds["source_fn"])
            min_xy = min((reader.dims.X, reader.dims.Y, min_xy))
            min_z = min((reader.dims.Z, min_z))
        assert min_xy >= 128, "some images have dimension smaller than 128"
        crop_size = 128 * (min_xy // 128)
        if self.spatial_dim == 3:
            assert min_z >= 16, "some 3D data has less than 16 Z slices"
            crop_size_z = min((32, min_z))

        print("cache intermediate results ...")
        for ds in tqdm(dataset_list):
            instance, _ = self.target_reader(ds["target_fn"])
            image, _ = self.source_reader(ds["source_fn"])
            fn_base = str(os.path.basename(ds["target_fn"]))
            if fn_base.endswith("GT.tiff"):
                fn_base = fn_base[:-7]
            else:
                fn_base = os.path.splitext(fn_base)[0]

            instance_np = np.array(instance, copy=False)
            object_mask = instance_np > 0

            ids = np.unique(instance_np[object_mask])
            ids = ids[ids != 0]

            # loop over instances
            for j, id in enumerate(ids):
                if self.spatial_dim == 2:
                    h, w = image.shape
                    y, x = np.where(instance_np == id)
                    ym, xm = np.mean(y), np.mean(x)

                    jj = int(np.clip(ym - crop_size / 2, 0, h - crop_size))
                    ii = int(np.clip(xm - crop_size / 2, 0, w - crop_size))

                    if image[jj : jj + crop_size, ii : ii + crop_size].shape == (
                        crop_size,
                        crop_size,
                    ):
                        im_crop = image[jj : jj + crop_size, ii : ii + crop_size]
                        instance_crop = instance[
                            jj : jj + crop_size, ii : ii + crop_size
                        ]
                        center_image_crop = generate_center_image(
                            instance_crop, "centroid", ids
                        )
                        class_image_crop = object_mask[
                            jj : jj + crop_size, ii : ii + crop_size
                        ]
                        dim_order = "YX"

                elif self.spatial_dim == 3:
                    d, h, w = image.shape
                    z, y, x = np.where(instance_np == id)
                    zm, ym, xm = np.mean(z), np.mean(y), np.mean(x)
                    kk = int(np.clip(zm - crop_size_z / 2, 0, d - crop_size_z))
                    jj = int(np.clip(ym - crop_size / 2, 0, h - crop_size))
                    ii = int(np.clip(xm - crop_size / 2, 0, w - crop_size))

                    if image[
                        kk : kk + crop_size_z, jj : jj + crop_size, ii : ii + crop_size
                    ].shape == (crop_size_z, crop_size, crop_size):
                        im_crop = image[
                            kk : kk + crop_size_z,
                            jj : jj + crop_size,
                            ii : ii + crop_size,
                        ]
                        instance_crop = instance[
                            kk : kk + crop_size_z,
                            jj : jj + crop_size,
                            ii : ii + crop_size,
                        ]
                        center_image_crop = generate_center_image(
                            instance_crop,
                            "centroid",
                            ids,
                            anisotropy_factor=1,
                            speed_up=1,
                        )
                        class_image_crop = object_mask[
                            kk : kk + crop_size_z,
                            jj : jj + crop_size,
                            ii : ii + crop_size,
                        ]
                        dim_order = "ZYX"

                OmeTiffWriter.save(
                    im_crop,
                    self.cache_path + os.sep + fn_base + f"_{j:04d}_IM.tiff",
                    dim_order=dim_order,
                )
                OmeTiffWriter.save(
                    instance_crop.astype(np.uint16),
                    self.cache_path + os.sep + fn_base + f"_{j:04d}_GT.tiff",
                    dim_order=dim_order,
                )
                OmeTiffWriter.save(
                    center_image_crop.astype(np.uint8),
                    self.cache_path + os.sep + fn_base + f"_{j:04d}_CE.tiff",
                    dim_order=dim_order,
                )
                OmeTiffWriter.save(
                    class_image_crop.astype(np.uint8),
                    self.cache_path + os.sep + fn_base + f"_{j:04d}_CL.tiff",
                    dim_order=dim_order,
                )

    def setup(self, stage=None):
        dataset_list = generate_dataset_dict(self.cache_path)
        subjects = []
        for ds in dataset_list:
            # the data need to be in format of CXYZ, so that torchio
            # transformation can be applied
            subject = tio.Subject(
                source=tio.ScalarImage(ds["source_fn"], reader=self.source_reader),
                target=tio.LabelMap(ds["target_fn"], reader=self.target_reader),
                center_image=tio.LabelMap(
                    str(ds["target_fn"])[:-7] + "CE.tiff", reader=self.target_reader
                ),
                class_image=tio.LabelMap(
                    str(ds["target_fn"])[:-7] + "CL.tiff", reader=self.target_reader
                ),
            )
            # TODO: add costmap support?

            """
            # generate center image and label image
            instance, _ = target_reader(ds["target_fn"])
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

            subjects.append(subject)

        num_subjects = len(subjects)
        num_val_subjects = int(round(num_subjects * self.train_val_ratio))
        num_train_subjects = num_subjects - num_val_subjects
        splits = num_train_subjects, num_val_subjects
        train_subjects, val_subjects = random_split(subjects, splits)
        self.val_set = tio.SubjectsDataset(val_subjects, transform=self.preproc)
        self.train_set = tio.SubjectsDataset(train_subjects, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_set, shuffle=True, **self.loader_params["train"])

    def val_dataloader(self):
        return DataLoader(self.val_set, shuffle=False, **self.loader_params["val"])

    def test_dataloader(self):
        pass
