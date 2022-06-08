from dataclasses import dataclass
from pathlib import Path
from pyrallis import field

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
        config_path: Optional[str] = None,
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

        self.config_path = config_path
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

        config_path = self.config_path  # Could be NONE

        if utils.CONFIG_ARG in parsed_arg_values:
            new_config_path = parsed_arg_values[utils.CONFIG_ARG]
            if config_path is not None:
                warnings.warn(
                    UserWarning(
                        f"Overriding default {config_path} with {new_config_path}"
                    )
                )
            ######################################################################
            # adapted from original implementation in pyrallis
            ######################################################################
            if Path(new_config_path).is_file():
                # pass in a absolute path
                config_path = new_config_path
            else:
                new_config_path = str(new_config_path)
                print(f"trying to locate preset config for {new_config_path} ...")

                config_path = Path(__file__).parent / f"preset_{new_config_path}.yaml"
            del parsed_arg_values[utils.CONFIG_ARG]

        if config_path is not None:
            file_args = cfgparsing.load_config(open(config_path, "r"))
            file_args = utils.flatten(file_args, sep=".")
            file_args.update(parsed_arg_values)
            parsed_arg_values = file_args

        deflat_d = utils.deflatten(parsed_arg_values, sep=".")
        cfg = decoding.decode(self.config_class, deflat_d)

        return cfg


def parse_adaptor(
    config_class: Type[T],
    config_path: Optional[Union[Path, str]] = None,
    args: Optional[Sequence[str]] = None,
) -> T:

    parser = ArgumentParser(config_class=config_class, config_path=config_path)
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
    # find full API here: https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
    dataloader_params: Dict = field(
        default={
            "batch_size": 1,
            "pin_memory": True,
            "num_workers": 4,
        },
        is_mutable=True,
    )

    # whether to load part of the dataset, and periodically reload
    partial_loader: Dict = field(default={"load_percentage": 1.0}, is_mutable=True)


@dataclass
class DataloaderConfig:
    """Config for training resources"""

    # The ratio of train/validation split
    train_val_ratio: float = field(default=0.1)

    # config for the training dataloader
    train: DataloaderModuleConfig = field(default_factory=DataloaderModuleConfig)

    # config for the validation dataloader
    val: DataloaderModuleConfig = field(default=DataloaderModuleConfig)


@dataclass
class DataConfig:
    """Config for data and data loaders"""

    # The type of data: "pair" | "unpair"
    category: str = field(default=None)

    # The data path
    data_path: Union[Path, str] = field(default=None)

    # config for dataloader
    dataloader: DataloaderConfig = field(default_factory=DataloaderConfig)

    # what to do in pre-processing
    preprocess: List[Dict] = field(default=None)

    # what to do in data augmentation
    augmentation: List[Dict] = field(default=None)

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
    net: Dict = field(default=None)

    # the config for criterion
    criterion: Dict = field(default=None)

    # the config for optimizer
    optimizer: Dict = field(default=None)

    # the config for learning scheduler
    scheduler: Dict = field(default=None)


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
                "params": {"save_last": True, "save_top_k": 5},
            }
        ],
        is_mutable=True,
    )


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
