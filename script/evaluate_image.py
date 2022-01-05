



import numpy as np
import os
from aicsimageio import AICSImage

load_datadir = "/mnt/data/ISAS.DE/sai.sata/ambiomgroupdrive/Sai/data/unpaired_dataset/training_data"
image_files=sorted(os.listdir(load_datadir))

img = AICSImage(os.path.join(load_datadir,image_files[0]))
print(img.shape)