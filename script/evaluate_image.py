# import numpy as np
# import os
# from aicsimageio import AICSImage
# from aicsimageio.writers import OmeTiffWriter
# import numpy as np

# # load_datadir = "/mnt/data/ISAS.DE/sai.sata/ambiomgroupdrive/
# Sai/data/paired_dataset/T_CELLS_training_data"
# image_files = sorted(os.listdir(load_datadir))

# img_reader = AICSImage(os.path.join(load_datadir, image_files[0]))

# im = img_reader.get_image_data("ZYX", C=0, T=0)
# im = im.astype(np.uint8)
# im[im > 0] = 1
# OmeTiffWriter.save(im, os.path.join(load_datadir+"img001_GT.tiff"),
#                    dim_order="ZYX")
