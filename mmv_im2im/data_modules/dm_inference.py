from pathlib import Path
from torch.utils.data import DataLoader
import torchio as tio
import pytorch_lightning as pl

from mmv_im2im.utils.for_transform import parse_tio_ops
from mmv_im2im.utils.misc import generate_test_dataset_dict


class Im2ImDataModule(pl.LightningDataModule):
    def __init__(self, data_cfg):
        super().__init__()

        self.data_path = data_cfg["input"]
        self.output = data_cfg["output"]

        # all subjects
        self.subjects = None
        self.test_set = None

        # transformation
        self.preproc = parse_tio_ops(data_cfg["preprocess"])

        # parameters for dataloader
        self.loader_params = data_cfg["dataloader_params"]

    def prepare_data(self):
        dataset_list = generate_test_dataset_dict(
            self.data_path["dir"], **self.data_path["params"]
        )
        self.subjects = []
        for ds in dataset_list:
            fn_core = Path(ds).stem
            suffix = self.output["suffix"]
            out_path = self.output["path"] / f"{fn_core}_{suffix}.tiff"
            subject = tio.Subject(source=tio.ScalarImage(ds), save_path=out_path)
            self.subjects.append(subject)

    def setup(self, stage=None):
        self.test_set = tio.SubjectsDataset(self.subjects, transform=self.preproc)

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        return DataLoader(self.test_set, shuffle=False, **self.loader_params)
