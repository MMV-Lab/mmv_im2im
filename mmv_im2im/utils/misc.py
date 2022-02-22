from typing import Union, Dict, List
from pathlib import Path
from functools import partial
import importlib
import yaml
import numpy as np
from munch import Munch
from aicsimageio import AICSImage
import torchio as tio


def aicsimageio_reader(fn, **kwargs):
    img = AICSImage(fn).reader.get_image_dask_data(**kwargs)
    img_data = tio.data.io.check_uint_to_int(img.compute())
    return img_data, np.eye(4)
    """
    if len(img_data.shape) == 3:
        return img_data, np.eye(4)
    elif len(img_data.shape) == 2:
        return img_data, np.eye(3)
    else:
        print("error in aicsimage loader")
        sys.exit(0)
    """


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


def parse_config_func_without_params(info):
    my_module = importlib.import_module(info["module_name"])
    my_func = getattr(my_module, info["func_name"])
    return my_func


def parse_config(info):
    my_func = parse_config_func_without_params(info)
    if "params" in info:
        return my_func(**info["params"])
    else:
        return my_func()


def parse_config_func(info):
    my_module = importlib.import_module(info["module_name"])
    my_func = getattr(my_module, info["func_name"])
    child_func = partial(my_func, **info["params"])
    return child_func


def parse_ops_list(trans_func: List[Dict]):
    op_list = []
    # {"module_name":"numpy.random",
    #  "func_name":"randint",
    #  "params":{"low":0,"high":1}}
    # load functions according to config
    for trans_dict in trans_func:
        op_list.append(parse_config(trans_dict))
        # trans_module = importlib.import_module(trans_dict["module_name"])
        # trans_func = getattr(trans_module, trans_dict["func_name"])
        # op_list.append(trans_func(**trans_dict["params"]))
    return op_list


def generate_test_dataset_dict(
    data: Union[str, Path], data_column: str = None, data_type: str = None
) -> List:
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
        data_column

        for row in df.iterrows():
            dataset_list.append(row[data_column])

    elif data.is_dir():
        all_filename = sorted(data.glob(f"*{data_type}"))
        assert len(all_filename) > 0, f"no file found in {data}"
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
    if isinstance(data, str):
        data = Path(data).expanduser()
        if data.is_file():
            # should be a csv of dataframe
            import pandas as pd

            df = pd.read_csv(data)
            assert "source_path" in df.columns, "column source_path not found"
            assert "target_path" in df.columns, "column target_path not found"

            # check if costmap is needed
            if "costmap_path" in df.columns:
                cm_flag = True
            else:
                cm_flag = False

            for row in df.itertuples():
                if cm_flag:
                    dataset_list.append(
                        {
                            "source_fn": row.source_path,
                            "target_fn": row.target_path,
                            "costmap_fn": row.costmap_path,
                        }
                    )
                else:
                    dataset_list.append(
                        {
                            "source_fn": row.source_path,
                            "target_fn": row.target_path,
                        }  # noqa E501
                    )
        elif data.is_dir():
            all_filename = sorted(data.glob("*_IM.*"))
            assert len(all_filename) > 0, f"no file found in {data}"

            all_filename.sort()
            for fn in all_filename:
                target_fn = data / fn.name.replace("_IM.", "_GT.")
                costmap_fn = data / fn.name.replace("_IM.", "_CM.")
                if costmap_fn.is_file():
                    dataset_list.append(
                        {
                            "source_fn": fn,
                            "target_fn": target_fn,
                            "costmap_fn": costmap_fn,
                        }
                    )
                else:
                    dataset_list.append(
                        {
                            "source_fn": fn,
                            "target_fn": target_fn,
                        }
                    )
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
            if cm_path is not None:
                costmap_fn = cm_path / fn.name
                dataset_list.append(
                    {
                        "source_fn": fn,
                        "target_fn": target_fn,
                        "costmap_fn": costmap_fn,
                    }  # noqa E501
                )
            else:
                dataset_list.append({"source_fn": fn, "target_fn": target_fn})

    else:
        print("unsupported data type")

    assert len(dataset_list) > 0, "empty dataset"

    return dataset_list
