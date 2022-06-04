from typing import List, Dict
from functools import partial
from mmv_im2im.utils.misc import parse_config, parse_config_func_without_params
from monai.transforms import Compose, Lambdad


def parse_tio_ops(trans_func: List[Dict]):
    import torchio as tio

    # Here, we will use the Compose function in torchio to merge
    # all transformations. If any trnasformation not from torchio,
    # a torchio Lambda function will be used to wrap around it.
    trans_list = []
    for func_info in trans_func:
        if func_info["module_name"] == "torchio":
            trans_list.append(parse_config(func_info))
        else:
            my_func = parse_config_func_without_params(func_info)
            if "params" in func_info:
                callable_func = partial(my_func, **func_info["params"])
            else:
                callable_func = my_func

            if "extra_kwargs" in func_info:
                trans_list.append(
                    tio.Lambda(callable_func, **func_info["extra_kwargs"])
                )
            else:
                trans_list.append(tio.Lambda(callable_func))

    return tio.Compose(trans_list)


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

    # Here, we will use the Compose function in MONAI to merge
    # all transformations. If any trnasformation not from MONAI,
    # a torchio Lambda function will be used to wrap around it.
    trans_list = []
    reader = trans_func.pop(0)
    if reader["module_name"] == "monai.transforms" and reader["func_name"] == "LoadImaged":
        from mmv_im2im.utils.misc import monai_bio_reader
        from monai.transforms import LoadImaged
        trans_list.append(LoadImaged(reader=monai_bio_reader, **reader["params"]))
    for func_info in trans_func:
        if func_info["module_name"] == "monai.transforms":
            trans_list.append(parse_config(func_info))
        else:
            my_func = parse_config_func_without_params(func_info)
            if "params" in func_info:
                callable_func = partial(my_func, **func_info["params"])
            else:
                callable_func = my_func

            if "extra_kwargs" in func_info:
                trans_list.append(
                    Lambdad(callable_func, **func_info["extra_kwargs"])
                )
            else:
                trans_list.append(Lambdad(callable_func))

    return Compose(trans_list)
