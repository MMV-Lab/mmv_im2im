from pathlib import Path
from mmv_im2im.data_modules.data_loader_basic import Im2ImDataModule as dm_basic
from mmv_im2im.utils.embedseg_utils import prepare_embedseg_cache


def get_data_module(cfg):
    if cfg.category == "embedseg":
        # check cache path
        cache_path = Path(cfg.cache_path)
        if not any(cache_path.iterdir()):
            print("cache for embedseg is empty, start generating ...")
            prepare_embedseg_cache(cfg.data_path, cache_path)
        return dm_basic(cfg, cache_path=cache_path)
    else:
        return dm_basic(cfg)
