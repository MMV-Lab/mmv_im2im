from dataclasses import dataclass
from pathlib import Path
from pyrallis import field
import tempfile

import argparse
import dataclasses
import sys
import warnings
from argparse import HelpFormatter, Namespace
from collections import defaultdict
from logging import getLogger
from typing import Dict, List, Sequence, Text, Type, Union, TypeVar, Generic, Optional

from pyrallis import utils, cfgparsing
from pyrallis.help_formatter import SimpleHelpFormatter
from pyrallis.parsers import decoding
from pyrallis.utils import Dataclass, PyrallisException
from pyrallis.wrappers import DataclassWrapper

logger = getLogger(__name__)

T = TypeVar("T")


class ArgumentParser(Generic[T], argparse.ArgumentParser):
    def __init__(
        self,
        config_class: Type[T],
        config: Optional[str] = None,
        formatter_class: Type[HelpFormatter] = SimpleHelpFormatter,
        *args,
        **kwargs,
    ):
        """Creates an ArgumentParser instance."""
        kwargs["formatter_class"] = formatter_class
        super().__init__(*args, **kwargs)

        # constructor arguments for the dataclass instances.
        # (a Dict[dest, [attribute, value]])
        self.constructor_arguments: Dict[str, Dict] = defaultdict(dict)

        self._wrappers: List[DataclassWrapper] = []

        self.config = config
        self.config_class = config_class

        self._assert_no_conflicts()
        self.add_argument(
            f"--{utils.CONFIG_ARG}",
            type=str,
            help="Path for a config file to parse with pyrallis",
        )
        self.set_dataclass(config_class)

    def set_dataclass(
        self,
        dataclass: Union[Type[Dataclass], Dataclass],
        prefix: str = "",
        default: Union[Dataclass, Dict] = None,
        dataclass_wrapper_class: Type[DataclassWrapper] = DataclassWrapper,
    ):
        """Adds command-line arguments for the fields of `dataclass`."""
        if not isinstance(dataclass, type):
            default = dataclass if default is None else default
            dataclass = type(dataclass)

        new_wrapper = dataclass_wrapper_class(dataclass, prefix=prefix, default=default)
        self._wrappers.append(new_wrapper)
        self._wrappers += new_wrapper.descendants

        for wrapper in self._wrappers:
            logger.debug(
                f"Adding arguments for dataclass: {wrapper.dataclass} "
                f"at destination {wrapper.dest}"
            )
            wrapper.add_arguments(parser=self)

    def _assert_no_conflicts(self):
        """Checks for a field name that conflicts with utils.CONFIG_ARG"""
        if utils.CONFIG_ARG in [
            field.name for field in dataclasses.fields(self.config_class)
        ]:
            raise PyrallisException(
                f"{utils.CONFIG_ARG} is a reserved word for pyrallis"
            )

    def parse_args(self, args=None, namespace=None) -> T:
        return super().parse_args(args, namespace)

    def parse_known_args(
        self,
        args: Sequence[Text] = None,
        namespace: Namespace = None,
        attempt_to_reorder: bool = False,
    ):
        # NOTE: since the usual ArgumentParser.parse_args() calls
        # parse_known_args, we therefore just need to overload the
        # parse_known_args method to support both.
        if args is None:
            # args default to the system args
            args = sys.argv[1:]
        else:
            # make sure that args are mutable
            args = list(args)

        if "--help" not in args:
            for action in self._actions:
                # TODO: Find a better way to do that?
                action.default = (
                    argparse.SUPPRESS
                )  # To avoid setting of defaults in actual run
                action.type = (
                    str  # In practice, we want all processing to happen with yaml
                )
        parsed_args, unparsed_args = super().parse_known_args(args, namespace)

        parsed_args = self._postprocessing(parsed_args)
        return parsed_args, unparsed_args

    def print_help(self, file=None):
        return super().print_help(file)

    def _postprocessing(self, parsed_args: Namespace) -> T:
        logger.debug("\nPOST PROCESSING\n")
        logger.debug(f"(raw) parsed args: {parsed_args}")

        parsed_arg_values = vars(parsed_args)

        for key in parsed_arg_values:
            parsed_arg_values[key] = cfgparsing.parse_string(parsed_arg_values[key])

        config = self.config  # Could be NONE

        if utils.CONFIG_ARG in parsed_arg_values:
            new_config = parsed_arg_values[utils.CONFIG_ARG]
            if config is not None:
                warnings.warn(
                    UserWarning(f"Overriding default {config} with {new_config}")
                )
            ######################################################################
            # adapted from original implementation in pyrallis
            ######################################################################
            if Path(new_config).is_file():
                # pass in a absolute path
                config = new_config
            else:
                new_config = str(new_config)
                print(f"trying to locate preset config for {new_config} ...")

                config = Path(__file__).parent / f"preset_{new_config}.yaml"
            del parsed_arg_values[utils.CONFIG_ARG]

        if config is not None:
            print(f"loading configuration from {config} ...")
            file_args = cfgparsing.load_config(open(config, "r"))
            file_args = utils.flatten(file_args, sep=".")
            file_args.update(parsed_arg_values)
            parsed_arg_values = file_args
            print("configuration loading is completed")

        deflat_d = utils.deflatten(parsed_arg_values, sep=".")
        cfg = decoding.decode(self.config_class, deflat_d)

        return cfg


def parse_adaptor(
    config_class: Type[T],
    config: Optional[Union[Path, str]] = None,
    args: Optional[Sequence[str]] = None,
) -> T:

    parser = ArgumentParser(config_class=config_class, config=config)
    return parser.parse_args(args)


@dataclass
class DataloaderModuleConfig:
    """Config for the dataloader module"""

    # the module information
    dataloader_type: Dict = field(
        default={"module_name": "monai.data", "func_name": "PersistentDataset"},
        is_mutable=True,
    )

    # the parameters to be passed in when making the dataset (e.g. PersistentDataset)
    # find full API in the corresponding Dataset function
    # e.g. https://docs.monai.io/en/stable/data.html#persistentdataset
    dataset_params: Dict = field(default={"cache_dir": "./tmp"}, is_mutable=True)

    # the parameters to be passed in when making the DataLoader
    # find full API here:
    # https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    dataloader_params: Dict = field(
        default={
            "batch_size": 1,
            "pin_memory": True,
            "num_workers": 2,
        },
        is_mutable=True,
    )

    # whether to load part of the dataset, and periodically reload
    partial_loader: float = field(default=1.0)


@dataclass
class DataloaderConfig:
    """Config for training resources"""

    # The ratio of train/validation split
    train_val_ratio: float = field(default=0.1)

    # config for the training dataloader
    train: DataloaderModuleConfig = field(default_factory=DataloaderModuleConfig)

    # config for the validation dataloader
    val: DataloaderModuleConfig = field(default_factory=DataloaderModuleConfig)


@dataclass
class DataConfig:
    """Config for data and data loaders"""

    # The type of data: "pair" | "unpair" | "embedseg"
    category: str = field(default=None)

    # The data path
    data_path: Union[Path, str] = field(default=None)

    # save pre-processed data into a cache folder (currently, only for embedseg)
    cache_path: Union[Path, str] = field(default=None)

    # config for dataloader
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)

    # what to do in pre-processing
    preprocess: List[Dict] = field(default=None)

    # what to do in data augmentation
    augmentation: List[Dict] = field(default=None)

    # global variable that can be used to verify or overwrite other related settings
    patch_size: List = field(default=None, is_mutable=True)

    # extra parameters for specific methods:
    extra: Dict = field(default=None, is_mutable=True)

    # @property
    # def exp_dir(self) -> Path:
    #    # Properties are great for arguments that can be derived from existing ones
    #    return self.exp_root / self.exp_name


@dataclass
class ModelConfig:
    """Config for model"""

    # the type of model
    framework: str = field(default=None)

    # the exact network configuration
    net: Dict = field(default=None, is_mutable=True)

    # the config for criterion
    criterion: Dict = field(default=None, is_mutable=True)

    # the config for optimizer
    optimizer: Dict = field(default=None, is_mutable=True)

    # the config for learning scheduler
    scheduler: Dict = field(default=None, is_mutable=True)

    # extra for special parameters of specific method
    model_extra: Dict = field(default=None, is_mutable=True)


@dataclass
class TrainingConfig:
    """Config for how to run the training"""

    # whether to save sample outputs at the beginning of each epoch
    verbose: bool = field(default=False)

    # the parameters to be passed into pytorch-lightning trainer
    params: Dict = field(default=None)

    # the config for callbacks
    callbacks: List[Dict] = field(
        default=[
            {
                "module_name": "pytorch_lightning.callbacks",
                "func_name": "ModelCheckpoint",
                "params": {"save_last": True},
            }
        ],
        is_mutable=True,
    )

    # global variable that can be used to overwrite gpus in trainer
    gpus: Union[int, List[int]] = field(default=None, is_mutable=True)


@dataclass
class ProgramConfig:
    """the main configuration for the whole program"""

    # mode: "train" | "infernece" (#TODO future version will add "evaluation")
    mode: str = field(default=None)

    # the configuration for data (path, dataloader, transform, etc.)
    data: DataConfig = field(default_factory=DataConfig)

    # the configuration for model (e.g., network, loss, etc.)
    model: ModelConfig = field(default_factory=ModelConfig)

    # the configuration for trainer
    training: TrainingConfig = field(default_factory=TrainingConfig)


def configuration_validation(cfg):
    # check 1: partial_loader should be a value between 0 (excluded) and 1 (included)
    assert (
        cfg.data.dataloader.train.partial_loader <= 1.0
        and cfg.data.dataloader.train.partial_loader > 0
    ), "partial loading percentage for training loader is not valid"

    # check 2: if partial loading is used, make sure reloading is enabled in trainer
    if cfg.data.dataloader.train.partial_loader < 1.0:
        if "reload_dataloaders_every_n_epochs" not in cfg.training.params:
            cfg.training.params["reload_dataloaders_every_n_epochs"] = 5

    # check 3: partial_loader for validation dataloader should be 1.0
    if cfg.data.dataloader.val.partial_loader != 1.0:
        cfg.data.dataloader.val.partial_loader = 1.0
        warnings.warn(
            UserWarning("partial loading for validation step is not allowed.")
        )

    # check 4: for embedseg, patch_size needs to be consistent with grid in criterion
    if cfg.data.patch_size is not None:
        grid = cfg.model.criterion["params"]["grid_x"]
        assert (
            grid == cfg.data.patch_size[-1]
        ), f"grid_x is set as {grid}, needs to be the same as the dimX of patch_size {cfg.data.patch_size}"  # noqa E501

        grid = cfg.model.criterion["params"]["grid_y"]
        assert (
            grid == cfg.data.patch_size[-2]
        ), f"grid_y is set as {grid}, needs to be the same as the dimY of patch_size {cfg.data.patch_size}"  # noqa E501

        if len(cfg.data.patch_size) == 3:
            grid = cfg.model.criterion["params"]["grid_z"]
            assert (
                grid == cfg.data.patch_size[0]
            ), f"grid_z is set as {grid}, needs to be the same as the dimZ of patch_size {cfg.data.patch_size}"  # noqa E501

    # check 5, if a global GPU number is set, update the value in trainer
    if cfg.training.gpus is not None:
        cfg.training.params["gpus"] = cfg.training.gpu

    # check 5, if PersistentDataset is used, make sure add a tmpdir in subdirectory (otherwise may cause hash errors)
    if cfg.data.dataloader.train.dataloader_type["func_name"] == "PersistentDataset":
        if "cache_dir" in cfg.data.dataloader.train.dataset_params:
            cache_dir = cfg.data.dataloader.train.dataset_params["cache_dir"]
            cache_dir_tmp = tempfile.mkdtemp(dir=cache_dir)
            cfg.data.dataloader.train.dataset_params["cache_dir"] = cache_dir_tmp
        else:
            warnings.warn(
                UserWarning(
                    "The cache dir of PersistentDataset for training was overwritten to None. No caching ..."
                )
            )
    if cfg.data.dataloader.val.dataloader_type["func_name"] == "PersistentDataset":
        if "cache_dir" in cfg.data.dataloader.val.dataset_params:
            cache_dir = cfg.data.dataloader.val.dataset_params["cache_dir"]
            cache_dir_tmp = tempfile.mkdtemp(dir=cache_dir)
            cfg.data.dataloader.val.dataset_params["cache_dir"] = cache_dir_tmp
        else:
            warnings.warn(
                UserWarning(
                    "The cache dir of PersistentDataset for val was overwritten to None. No caching ..."
                )
            )

    return cfg
