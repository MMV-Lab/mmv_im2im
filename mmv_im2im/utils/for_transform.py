from typing import List, Dict
from mmv_im2im.utils.misc import parse_ops_list


def parse_tio_ops(trans_func: List[Dict]):
    import torchio as tio

    return tio.Compose(parse_ops_list(trans_func))
