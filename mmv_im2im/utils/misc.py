from typing import Union, Dict, List
from pathlib import Path
from functools import partial
import importlib
import numpy as np
import inspect
from bioio import BioImage
from typing import Sequence, Tuple
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

    def verify_suffix(self, name: PathLike) -> bool:
        if str(name).endswith(".npy"):
            return True
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


def parse_config_func_without_params(info):
    my_module = importlib.import_module(info["module_name"])
    my_func = getattr(my_module, info["func_name"])
    return my_func


def parse_config(info):
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
    my_func = parse_config_func_without_params(info)
    if "params" in info:
        return partial(my_func, **info["params"])
    else:
        return my_func


def parse_ops_list(trans_func: List[Dict]):
    op_list = []
    for trans_dict in trans_func:
        op_list.append(parse_config(trans_dict))
    return op_list


def generate_test_dataset_dict(data: Union[str, Path], data_type: str = None) -> List:
    dataset_list = []
    data = Path(data).expanduser()
    if data.is_file():
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
    dataset_list = []
    if isinstance(data, str) or isinstance(data, Path):
        data = Path(data).expanduser()
        if data.is_file():
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
                target_fn = list(data.glob(f"{basename}_GT.*"))
                costmap_fn = list(data.glob(f"{basename}_CM.*"))

                item = {"source_fn": fn}
                if target_fn:
                    item["target_fn"] = target_fn[0]
                else:
                    item["target_fn"] = data / fn.name.replace("_IM.", "_GT.")

                if costmap_fn:
                    item["costmap_fn"] = costmap_fn[0]

                dataset_list.append(item)
        else:
            print(f"{data} is not a valid file or directory")

    elif isinstance(data, Dict):
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
            item = {"source_fn": fn, "target_fn": target_fn}
            if cm_path is not None:
                item["costmap_fn"] = cm_path / fn.name
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

    first_fn = all_filename[0]
    basename_template = first_fn.name[: first_fn.name.rfind("_IM")]
    subject_files = sorted(data.glob(f"{basename_template}_*.*"))

    all_tags = list(
        set([sfile.stem[sfile.stem.rfind("_") + 1 :] for sfile in subject_files])
    )

    for fn in all_filename:
        path_list = {}
        current_basename = fn.name[: fn.name.rfind("_IM")]

        for tag_name in all_tags:
            tag_files = list(data.glob(f"{current_basename}_{tag_name}.*"))
            if len(tag_files) > 0:
                path_list[tag_name] = tag_files[0]

        if len(path_list) > 0:
            dataset_list.append(path_list)

    return dataset_list


def load_subfolders(data) -> List:
    dataset_list = []
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
            fn_full = data / Path(tag_name) / fn.name
            path_list[tag_name] = fn_full
        dataset_list.append(path_list)

    return dataset_list


def generate_dataset_dict_monai(data: Union[str, Path, Dict]) -> List[Dict]:
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
                dataset_list = load_subfolders(data)
        else:
            print(f"{data} is not a valid file or directory")

    elif isinstance(data, Dict):
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
