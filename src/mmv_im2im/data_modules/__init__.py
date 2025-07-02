from mmv_im2im.data_modules.data_loader_basic import Im2ImDataModule as dm_basic
from mmv_im2im.utils.embedseg_utils import prepare_embedseg_cache


def get_data_module(cfg):
    if cfg.category == "embedseg":
        # if no cache_path, use random patch generation on-the-fly
        if cfg.cache_path is None:
            return dm_basic(cfg)

        # if cache_path is set, but empty, then generate pre-cropped patches
        if not any(cfg.cache_path.iterdir()):
            print("cache for embedseg is empty, start generating ...")
            prepare_embedseg_cache(cfg.data_path, cfg.cache_path, cfg)
        cfg.data_path = cfg.cache_path

    return dm_basic(cfg)
