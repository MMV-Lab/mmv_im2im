from typing import List, Dict
from functools import partial
from mmv_im2im.utils.misc import parse_config, parse_config_func_without_params
from monai.transforms import Compose, Lambdad, Lambda, MapTransform
import inspect


def center_crop(img, target_shape):
    # target_shape is smaller than img.shape
    target_x = target_shape[-1]
    img_x = img.shape[-1]
    diff_x = img_x - target_x
    half_x = diff_x // 2

    target_y = target_shape[-2]
    img_y = img.shape[-2]
    diff_y = img_y - target_y
    half_y = diff_y // 2

    if len(target_shape) == 3:
        target_z = target_shape[-3]
        img_z = img.shape[-3]
        diff_z = img_z - target_z
        half_z = diff_z // 2

        return img[
            half_z : -(diff_z - half_z),
            half_y : -(diff_y - half_y),
            half_x : -(diff_x - half_x),
        ]
    else:
        return img[half_y : -(diff_y - half_y), half_x : -(diff_x - half_x)]


def parse_monai_ops(trans_func: List[Dict]):

    trans_list = []

    # loop through the config
    for func_info in trans_func:
        if func_info["module_name"] == "monai.transforms":
            if func_info["func_name"] == "LoadImaged":
                # Here, we handle the LoadImaged separately to allow bio-reader
                from mmv_im2im.utils.misc import monai_bio_reader
                from monai.transforms import LoadImaged

                trans_list.append(
                    LoadImaged(reader=monai_bio_reader, **func_info["params"])
                )
            else:
                trans_list.append(parse_config(func_info))
        else:
            my_func = parse_config_func_without_params(func_info)

            # MapTransform subclasses operate on the full data dict and manage
            # their own keys internally — wrap them like native MONAI transforms.
            if inspect.isclass(my_func) and issubclass(my_func, MapTransform):
                trans_list.append(parse_config(func_info))
            else:
                # Single-key function: use Lambdad wrapper (original behaviour)
                func_params = func_info["params"]
                apply_keys = func_params.pop("keys")

                # check if any other params
                if len(func_params) > 0:
                    if inspect.isclass(my_func):
                        callable_func = my_func(**func_params)
                    else:
                        callable_func = partial(my_func, **func_params)
                else:
                    callable_func = my_func

                trans_list.append(Lambdad(keys=apply_keys, func=callable_func))

    return Compose(trans_list)


def parse_monai_ops_vanilla(trans_func: List[Dict]):
    trans_list = []
    # loop through the config
    for func_info in trans_func:
        if func_info["module_name"] == "monai.transforms":
            trans_list.append(parse_config(func_info))
        else:
            my_func = parse_config_func_without_params(func_info)
            if "params" in func_info:
                func_params = func_info["params"]
                callable_func = partial(my_func, **func_params)
            else:
                callable_func = my_func
            trans_list.append(Lambda(func=callable_func))

    return Compose(trans_list)
