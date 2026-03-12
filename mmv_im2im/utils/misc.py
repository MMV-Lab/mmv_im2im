from typing import Union, Dict, List, Sequence, Tuple
from pathlib import Path
from functools import partial
import importlib
import numpy as np
import inspect
from bioio import BioImage
from monai.data import ImageReader
from monai.utils import ensure_tuple, require_pkg
from monai.config import PathLike
from monai.data.image_reader import _stack_images
import bioio_tifffile


@require_pkg(pkg_name="bioio")
class monai_bio_reader(ImageReader):
    def __init__(self, **kwargs):
        super().__init__()
        self.kwargs = kwargs

    def verify_suffix(self, filename: Union[Sequence[PathLike], PathLike]) -> bool:
        return True

    def read(self, data: Union[Sequence[PathLike], PathLike]):
        filenames: Sequence[PathLike] = ensure_tuple(data)
        img_ = []
        for name in filenames:
            if str(name).endswith(".npy"):
                img_.append(np.load(str(name)))
            else:
                try:
                    img_.append(BioImage(f"{name}", reader=bioio_tifffile.Reader))
                except Exception:
                    try:
                        img_.append(BioImage(f"{name}"))
                    except Exception as e:
                        print(f"Error: {e}")
                        print(f"Image {name} failed at read process check the format.")

        return img_ if len(filenames) > 1 else img_[0]

    def get_data(self, img) -> Tuple[np.ndarray, Dict]:
        if isinstance(img, np.ndarray):
            return img, {}

        img_array: List[np.ndarray] = []

        for img_obj in ensure_tuple(img):
            if isinstance(img_obj, np.ndarray):
                img_array.append(img_obj)
            else:
                data = img_obj.get_image_data(**self.kwargs)
                img_array.append(data)

        return _stack_images(img_array, {}), {}


"""
def aicsimageio_reader(fn, **kwargs):
    img = AICSImage(fn).reader.get_image_dask_data(**kwargs)
    img_data = tio.data.io.check_uint_to_int(img.compute())
    return img_data, np.eye(4)


def load_yaml_cfg(yaml_path):
    with open(yaml_path, "r") as stream:
        opt_dict = yaml.safe_load(stream)

    # convert dictionary to attribute-like object
    opt = Munch(opt_dict)

    return opt


def get_max_shape(subjects):
    dataset = tio.SubjectsDataset(subjects)
    shapes = np.array([s.spatial_shape for s in dataset])
    return shapes.max(axis=0)
"""


def parse_config_func_without_params(info):
    # TODO add docstring
    my_module = importlib.import_module(info["module_name"])
    my_func = getattr(my_module, info["func_name"])
    return my_func


def parse_config(info):
    # universal configuration parser
    my_func = parse_config_func_without_params(info)
    if "params" in info:
        if inspect.isclass(my_func):
            return my_func(**info["params"])
        else:
            return partial(my_func, **info["params"])
    else:
        if inspect.isclass(my_func):
            return my_func()
        else:
            return my_func


def parse_config_func(info):
    # TODO: remove this function, use the universal parse_config above
    my_func = parse_config_func_without_params(info)
    if "params" in info:
        return partial(my_func, **info["params"])
    else:
        return my_func


def parse_ops_list(trans_func: List[Dict]):
    op_list = []
    # {"module_name":"numpy.random",
    #  "func_name":"randint",
    #  "params":{"low":0,"high":1}}
    # load functions according to config
    for trans_dict in trans_func:
        op_list.append(parse_config(trans_dict))
    return op_list


def generate_test_dataset_dict(data: Union[str, Path], data_type: str = None) -> List:
    """
    different options for "data":
    - one CSV
    - one folder
    Return
        a list of filename
    """
    dataset_list = []
    data = Path(data).expanduser()
    if data.is_file():
        # should be a csv of dataframe
        import pandas as pd

        df = pd.read_csv(data)
        for row in df.iterrows():
            dataset_list.append(row[data_type])

    elif data.is_dir():
        if "*" in data_type:
            all_filename = sorted(data.glob(data_type))
        else:
            all_filename = sorted(data.glob(f"*{data_type}"))
        assert len(all_filename) > 0, f"no file found in {data}"
        print(f"{len(all_filename)} files are found at {data}")
        all_filename.sort()
        for fn in all_filename:
            dataset_list.append(fn)
    else:
        print(f"{data} is not a valid file or directory")

    return dataset_list


def generate_dataset_dict(data: Union[str, Path, Dict]) -> List[Dict]:
    """
    different options for "data":
    - one CSV (columns: source, target, cmap), then split
    - one folder (_IM.tiff, _GT.tiff, _CM.tiff), then split
    - a dictionary of two or three folders (Im, GT, CM), then split

    Return
        a list of dict, each dict contains 2 or 3 keys
        "source_fn", "target_fn", "costmap_fn" (optional)
    """
    dataset_list = []
    if isinstance(data, str) or isinstance(data, Path):
        data = Path(data).expanduser()
        if data.is_file():
            # should be a csv of dataframe
            import pandas as pd

            df = pd.read_csv(data)
            assert "source_path" in df.columns, "column source_path not found"
            assert "target_path" in df.columns, "column target_path not found"
            cm_flag = "costmap_path" in df.columns

            for row in df.itertuples():
                item = {
                    "source_fn": row.source_path,
                    "target_fn": row.target_path,
                }
                if cm_flag:
                    item["costmap_fn"] = row.costmap_path
                dataset_list.append(item)

        elif data.is_dir():
            all_filename = sorted(data.glob("*_IM.*"))
            assert len(all_filename) > 0, f"no file found in {data}"
            all_filename.sort()

            for fn in all_filename:
                # Extension agnostic matching
                basename = fn.name[: fn.name.rfind("_IM.")]
                target_fn_list = list(data.glob(f"{basename}_GT.*"))
                costmap_fn_list = list(data.glob(f"{basename}_CM.*"))

                item = {"source_fn": fn}
                if target_fn_list:
                    item["target_fn"] = target_fn_list[0]
                else:
                    # Fallback to original behavior
                    item["target_fn"] = data / fn.name.replace("_IM.", "_GT.")

                if costmap_fn_list:
                    item["costmap_fn"] = costmap_fn_list[0]
                else:
                    # Fallback to original behavior, only add if exists
                    costmap_fn_fallback = data / fn.name.replace("_IM.", "_CM.")
                    if costmap_fn_fallback.is_file():
                        item["costmap_fn"] = costmap_fn_fallback

                dataset_list.append(item)
        else:
            print(f"{data} is not a valid file or directory")

    elif isinstance(data, Dict):
        # assume 3~4 keys: "source_dir", "target_dir", and
        # "image_type", "costmap_dir" (optional)
        if "costmap_dir" in data:
            cm_path = Path(data["costmap_dir"]).expanduser()
        else:
            cm_path = None

        source_path = Path(data["source_dir"]).expanduser()
        target_path = Path(data["target_dir"]).expanduser()
        data_type = data["image_type"]

        all_filename = sorted(source_path.glob(f"*.{data_type}"))
        assert len(all_filename) > 0, f"no file found in {source_path}"
        all_filename.sort()

        for fn in all_filename:
            target_fn = target_path / fn.name

            # Allow fallback if extensions differ (e.g. source is .tiff, target is .npy)
            if not target_fn.is_file():
                matched_target = list(target_path.glob(f"{fn.stem}.*"))
                if matched_target:
                    target_fn = matched_target[0]

            item = {"source_fn": fn, "target_fn": target_fn}

            if cm_path is not None:
                costmap_fn = cm_path / fn.name
                if not costmap_fn.is_file():
                    matched_cm = list(cm_path.glob(f"{fn.stem}.*"))
                    if matched_cm:
                        costmap_fn = matched_cm[0]
                item["costmap_fn"] = costmap_fn

            dataset_list.append(item)
    else:
        print("unsupported data type")

    assert len(dataset_list) > 0, f"empty dataset in {data}"
    return dataset_list


def load_one_folder(data) -> List:
    dataset_list = []
    data = Path(data)
    all_filename = sorted(data.glob("*_IM.*"))
    assert len(all_filename) > 0, f"no file found in {data}"

    # parse how many images associated with one subject
    first_fn = all_filename[0]
    basename_template = first_fn.name[: first_fn.name.rfind("_IM")]
    subject_files = sorted(data.glob(f"{basename_template}_*.*"))

    all_tags = list(
        set([sfile.stem[sfile.stem.rfind("_") + 1 :] for sfile in subject_files])
    )
    all_tags.sort()

    all_filename.sort()
    for fn in all_filename:
        path_list = {}
        current_basename = fn.name[: fn.name.rfind("_IM")]

        for tag_name in all_tags:
            tag_files = list(data.glob(f"{current_basename}_{tag_name}.*"))
            if len(tag_files) > 0:
                path_list[tag_name] = tag_files[0]
            else:
                # Original behavior fallback
                path_list[tag_name] = data / fn.name.replace("_IM.", f"_{tag_name}.")

        if len(path_list) > 0:
            dataset_list.append(path_list)

    return dataset_list


def load_subfolders(data) -> List:
    dataset_list = []

    # parse how many images associated with one subject
    all_tags = [
        d.name
        for d in sorted(data.iterdir())
        if d.is_dir() and not d.name.startswith(".")
    ]

    data_p1 = data / Path(all_tags[0])
    all_filename = sorted(data_p1.glob("*.*"))
    assert len(all_filename) > 0, f"no file found in {data}"

    all_filename.sort()
    for fn in all_filename:
        path_list = {}
        for tag_name in all_tags:
            target_dir = data / Path(tag_name)
            fn_exact = target_dir / fn.name

            # If extensions are identical, standard match works
            if fn_exact.is_file():
                path_list[tag_name] = fn_exact
            else:
                # If they differ (e.g. .npy vs .tiff), search by stem
                matched = list(target_dir.glob(f"{fn.stem}.*"))
                if matched:
                    path_list[tag_name] = matched[0]
                else:
                    path_list[tag_name] = fn_exact
        dataset_list.append(path_list)

    return dataset_list


def generate_dataset_dict_monai(data: Union[str, Path, Dict]) -> List[Dict]:
    """
    different options for "data":
    - one CSV (columns: source, target, cmap), then split
    - one folder (_IM.tiff, _GT.tiff, _CM.tiff), then split
    - one folder with two or three subfolders (Im, GT, CM), then split
    - a dictionary of train/val

    Return
        a list of dict, each dict contains 2 or more keys, such as "IM",
        "GT", and more (optional, e.g. "LM", "CM", etc.)
    """
    if isinstance(data, str):
        try:
            data = eval(data)
        except Exception as e:
            print(f"data path is recognized as a string ... due to {e}")

    dataset_list = []
    if isinstance(data, str) or isinstance(data, Path):
        data = Path(data).expanduser()
        if data.is_file():
            # Support for CSV loading
            import pandas as pd

            df = pd.read_csv(data)
            for row in df.itertuples(index=False):
                row_dict = row._asdict()
                item = {}
                if "IM" in row_dict:
                    item["IM"] = Path(row_dict["IM"])
                if "GT" in row_dict:
                    item["GT"] = Path(row_dict["GT"])
                dataset_list.append(item)

        elif data.is_dir():
            if len(sorted(data.glob("*_IM.*"))) > 0:
                dataset_list = load_one_folder(data)
            else:
                # one folder with multiple subfolders (Im, GT, CM)
                dataset_list = load_subfolders(data)
        else:
            print(f"{data} is not a valid file or directory")

    elif isinstance(data, Dict):
        # a dictionary of train/val
        if "train" in data and "val" in data:
            train_path = Path(data["train"])
            train_list = load_one_folder(train_path)
            val_path = Path(data["val"])
            val_list = load_one_folder(val_path)
            dataset_list = {"train": train_list, "val": val_list}
        else:
            raise NotImplementedError(
                "dictionary based loading only takes keywords: train and val"
            )
    else:
        print("unsupported data type")

    assert len(dataset_list) > 0, "empty dataset"
    return dataset_list
