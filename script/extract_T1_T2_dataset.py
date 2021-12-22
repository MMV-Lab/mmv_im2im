


import tarfile
import os
datadir ="/mnt/data/ISAS.DE/sai.sata/ambiomgroupdrive/Sai/data/unpaired_dataset/training_data_T1_T2"

tf = tarfile.open(os.path.join(datadir,'IXI-T2.tar'))
tf.extractall(datadir)
