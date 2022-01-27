import pandas as pd
import quilt3
from pathlib import Path
import random
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter

# parameters
cell_lines = [
    "LMNB1", "AAVS1", "FBL", "NPM1", "TOMM20", "SEC61B", "NUP153", "TUBA1B"
]
num_samples_per_cell_line = 100

# prepare file path
# parent_path = Path("~/ambiomgroupdrive/Jianxu/data/labelfree_lamin/").expanduser()  # noqa E501
parent_path = Path("/mnt/eternus/project_data/labelfree/")
raw_path_base = parent_path / Path("raw_image")
raw_path_base.mkdir(exist_ok=True)
train_path_base = parent_path / Path("train")
train_path_base.mkdir(exist_ok=True)
holdout_path_base = parent_path / Path("holdout")
holdout_path_base.mkdir(exist_ok=True)


# connect to quilt and load meta table
pkg = quilt3.Package.browse(
    "aics/hipsc_single_cell_image_dataset", registry="s3://allencell"
)
meta_df_obj = pkg["metadata.csv"]
meta_df_obj.fetch(parent_path / "meta.csv")
meta_df = pd.read_csv(parent_path / "meta.csv")

# loop through each cell line and fetch the data
for cline in cell_lines:
    meta_df_line = meta_df.query("structure_name==@cline")

    # collapse the data table based on FOVId
    meta_df_line.drop_duplicates(subset="FOVId", inplace=True)

    # reset index
    meta_df_line.reset_index(drop=True, inplace=True)

    # prepare file paths
    raw_path = raw_path_base / Path(cline)
    raw_path.mkdir(exist_ok=True)
    train_path = train_path_base / Path(cline)
    train_path.mkdir(exist_ok=True)
    holdout_path = holdout_path_base / Path(cline)
    holdout_path.mkdir(exist_ok=True)

    # download all FOVs or a certain number
    if num_samples_per_cell_line > 0:
        num = num_samples_per_cell_line
    else:
        num = meta_df_line.shape[0]
    out_list = []
    for row in meta_df_line.itertuples():

        if row.Index >= num:
            break

        # fetch the raw image
        subdir_name = row.fov_path.split("/")[0]
        file_name = row.fov_path.split("/")[1]

        local_fn = raw_path / f"{row.FOVId}_original.tiff"
        pkg[subdir_name][file_name].fetch(local_fn)

        # extract the bf and structures channel
        reader = AICSImage(local_fn)
        bf_img = reader.get_image_data(
            "ZYX", C=row.ChannelNumberBrightfield, S=0, T=0
        )
        str_img = reader.get_image_data(
            "ZYX", C=row.ChannelNumberStruct, S=0, T=0
        )

        if random.random() < 0.2:
            data_path = holdout_path
        else:
            data_path = train_path

        im_fn = data_path / f"{row.FOVId}_IM.tiff"
        gt_fn = data_path / f"{row.FOVId}_GT.tiff"
        OmeTiffWriter.save(bf_img, im_fn, dim_order="ZYX")
        OmeTiffWriter.save(str_img, gt_fn, dim_order="ZYX")
