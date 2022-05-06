from typing import List, Dict
from functools import partial
from mmv_im2im.utils.misc import parse_config, parse_config_func_without_params
import torchio


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
                    torchio.Lambda(callable_func, **func_info["extra_kwargs"])
                )
            else:
                trans_list.append(torchio.Lambda(my_func))

    return tio.Compose(trans_list)
