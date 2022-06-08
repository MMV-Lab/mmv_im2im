from mmv_im2im.data_modules.data_loader_embedseg import Im2ImDataModule as dm_embedseg
from mmv_im2im.data_modules.data_loader import Im2ImDataModule as dm_basic


def get_data_module(cfg):
    if cfg["category"] == "embedseg":
        return dm_embedseg(cfg)
    else:
        return dm_basic(cfg)
