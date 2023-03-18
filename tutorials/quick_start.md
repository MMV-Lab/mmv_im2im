# Quick Start

We use a [3D labelfree prediction](https://www.allencell.org/label-free-determination.html#:~:text=The%20Label-Free%20Determination%20model%20can%20leverage%20the%20specificity,structures.%20How%20does%20the%20label-free%20determination%20model%20work%3F) as an example. (If you followed [basic installation](#basic-installation), you will need to install the `quilt3` package by `pip install quilt3`, in order to download the data programmatically.)

**step 1 (data preparation):** Running the following command will pull 50 examples of 3D lamin B1 images and the corresponding brightfield images from the [hiPSC single cell image dataset](https://open.quiltdata.com/b/allencell/packages/aics/hipsc_single_cell_image_dataset) released by the Allen Institute for Cell Science. 

```bash
# Suppose the current working directory is the root of mmv_im2im
python  scripts/pull_labelfree_sample_data.py --download_path /path/to/save/the/downloaded/images/ 
```

**step 2 (train):** Now, we can train a labelfree model like this:
```bash
# Suppose the current working directory is the root of mmv_im2im
run_im2im --config train_labelfree_3d  --data.data_path /path/to/save/the/downloaded/train
```

This will apply all the default settings to train the 3D labelfree model. The training will stop after 1000 epochs or meeting the early stopping criteria (by default, when validation loss does not decrease for 50 epochs). 

**step 3 (test):** 

Suppose you run the training under the root of mmv_im2im. Then you will find a folder `mmv_im2im/lightning_logs/checkpoints`, where you can find several trained model. Here, we test with the model with the checkpoint of lowest validation loss `best.ckpt` (other models are intermediate results for debugging purpose) by running:

```bash
# Suppose the current working directory is the root of mmv_im2im
run_im2im --config inference_labelfree_3d --data.inference_input.dir /path/to/save/the/downloaded/holdout --data.inference_output.path /path/to/save/predictions/ --model.checkpoint ./lightning_logs/checkpoints/best.ckpt
```

Now, you should be able to see predicted 