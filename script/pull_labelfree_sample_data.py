import pandas as pd
import quilt3
from pathlib import Path
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter

parent_path = Path("~/ambiomgroupdrive/Jianxu/data/labelfree_lamin/").expanduser()  # noqa E501

# connect to quilt and load meta table
pkg = quilt3.Package.browse(
    "aics/hipsc_single_cell_image_dataset", registry="s3://allencell"
)
meta_df_obj = pkg["metadata.csv"]
meta_df_obj.fetch(parent_path / "meta.csv")
meta_df = pd.read_csv(parent_path / "meta.csv")

# we use lamin B1 cell line for example (structure_name=='LMNB1')
meta_df_lamin = meta_df.query("structure_name=='LMNB1'")

# collapse the data table based on FOVId
meta_df_lamin.drop_duplicates(subset="FOVId", inplace=True)

# reset index
meta_df_lamin.reset_index(drop=True, inplace=True)

# prepare file paths
raw_path = parent_path / Path("raw_image")
raw_path.mkdir(exist_ok=True)
data_path = parent_path / Path("training_data")
data_path.mkdir(exist_ok=True)

# download all FOVs or a certain number
num = 100  # meta_df_lamin.shape[0]
out_list = []
for row in meta_df_lamin.itertuples():

    if row.Index >= num:
        break

    # fetch the raw image
    subdir_name = row.fov_path.split("/")[0]
    file_name = row.fov_path.split("/")[1]

    local_fn = raw_path / f"{row.FOVId}_original.tiff"
    pkg[subdir_name][file_name].fetch(local_fn)

    # extract the bf and lamin channel
    reader = AICSImage(local_fn)
    import pdb
    pdb.set_trace()
    bf_img = reader.get_image_data("ZYX", C=row.ChannelBrightfield, S=0, T=0)
    str_img = reader.get_image_data("ZYX", C=row.ChannelStructure, S=0, T=0)

    im_fn = data_path / f"{row.FOVId}_IM.tiff"
    gt_fn = data_path / f"{row.FOVId}_GT.tiff"
    OmeTiffWriter.save(bf_img, im_fn, dim_order="ZYX")
    OmeTiffWriter.save(str_img, gt_fn, dim_order="ZYX")
