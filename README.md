# MMV Im2Im Transformation

[![Build Status](https://github.com/MMV-Lab/mmv_im2im/workflows/Build%20Main/badge.svg)](https://github.com/MMV-Lab/mmv_im2im/actions)
[![Documentation](https://github.com/MMV-Lab/mmv_im2im/workflows/Documentation/badge.svg)](https://MMV-Lab.github.io/mmv_im2im/)

A generic python package for deep learing based image-to-image transformation in biomedical applications

---

## Overview

MMV_Im2Im an open source python package for image-to-image transformation in bioimaging applications. The overall package is designed with a generic image-to-image transformation framework, which could be directly used for semantic segmentation, instance segmentation, image restoration, image generation, labelfree prediction, etc.. The implementation takes advantage of the state-of-the-art machine learning engineering techniques for users to focus on the research without worrying about the engineering details. In our pre-print [arxiv link](https://arxiv.org/abs/2209.02498), we demonstrated the effectiveness of MMV_Im2Im in more than ten different biomedical problems. 

* For biomedical machine learning researchers, we hope this new package could serve as the starting point for their specific problems to stimulate new biomedical image analysis or machine learning methods. 
* For experimental biomedical researchers, we hope this work can provide a holistic view of the image-to-image transformation concept with diverse examples, so that deep learning based image-to-image transformation could be further integrated into the assay development process and permit new biomedical studies that can hardly be done only with traditional experimental methods


## Installation

We recommend to [create a new conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) or [a virtual environment](https://docs.python.org/3/library/venv.html) with Python 3.9 or higher.

### basic installation

(for users only using this package, not planning to change any code or make any extension):

**Stable Release:** `pip install mmv_im2im`<br>
**Development Head:** `pip install git+https://github.com/MMV-Lab/mmv_im2im.git`

### build from source

(for users planning to extend the methods or improve the code):

```
git clone https://github.com/MMV-Lab/mmv_im2im.git
cd mmv_im2im
pip install -e .[all]
```

Note: The `-e` option is the so-called "editable" mode. This will allow code changes taking effect immediately.


## Quick Start / Try it out

Here, we use a [3D labelfree prediction](https://www.allencell.org/label-free-determination.html#:~:text=The%20Label-Free%20Determination%20model%20can%20leverage%20the%20specificity,structures.%20How%20does%20the%20label-free%20determination%20model%20work%3F) as an example. (If you followed [basic installation](#basic-installation), you will need to install the `quilt3` package by `pip install quilt3`, in order to download the data programmatically.)

**step 1 (data preparation):**, we pull 100 examples of lamin B1 images and the corresponding brightfield images from the AllenCell quilt bucket by running the following in the command line. 

```bash
# Suppose the current working directory is the root of mmv_im2im
python  scripts/pull_labelfree_sample_data.py --download_path /path/to/save/the/downloaded/images/ --structure LMNB1 --num 100 
```

**step 2 (train):** Now, we can train a labelfree model like this:
```bash
# Suppose the current working directory is the root of mmv_im2im
run_im2im --config train_labelfree_3d  --data.data_path /path/to/save/the/downloaded/train
```

This will apply all the default settings to train the 3D labelfree model. The training will stop after 100 epochs, 

**step 3 (test):** 

Suppose you run the training under the root of mmv_im2im. Then you will find a folder `mmv_im2im/lightning_logs/checkpoints`, where you can find several trained model. Here, we test with the model after the full training `last.ckpt` (other models are intermediate results for debugging purpose) by running:

```bash
# Suppose the current working directory is the root of mmv_im2im
run_im2im --config inference_labelfree_3d --data.inference_input.dir /path/to/save/the/downloaded/holdout --data.inference_output.path /path/to/save/predictions/ --model.checkpoint lightning_logs/checkpoints/last.ckpt
```


## Documentation

For full package API (i.e., the technical details of each function), please visit [MMV-Lab.github.io/mmv_im2im](https://MMV-Lab.github.io/mmv_im2im).

## Notes on the package design


1. The four main packages we build upon: [pytorch-lightning](https://www.pytorchlightning.ai/), [MONAI](https://monai.io/), [pyrallis](https://eladrich.github.io/pyrallis/), and [aicsimageio](https://github.com/AllenCellModeling/aicsimageio).

The whole package uses [pytorch-lightning](https://www.pytorchlightning.ai/) as the core of its backend, in the sense that the package is implemented following the biolerplate components in pytorch-lightning, such as `LightningModule`, `DataModule` and `Trainer`. All small building blocks, like network architecture, optimizer, etc., can be swapped easily without changing the boilerplate. 

We adopt the [PersistentDataset](https://docs.monai.io/en/stable/data.html#persistentdataset) in [MONAI](https://monai.io) as the default dataloader, which combines the efficiency and flexibility in data handling for biomedical applications. E.g., able to efficiently handle: when any single file in training data is large, sampling mutiple patches from each of the large image in training data, and when there are a huge number of files and have to load a small portion to memory in each epoch and periodically refresh the data in memory, etc.

[Pyrallis](https://eladrich.github.io/pyrallis/) provides a handy configuration system. Combining pyrallis and the boilerplate concepts in pytorch-lightning, it is very easy to configurate your method at any level of details (as high level as only providing the path to the training data, all the way to as low level as changing which type of normalization to use in the model). 

Finally, [aicsimageio](https://github.com/AllenCellModeling/aicsimageio) is adopted for efficient data I/O, which not only supports all major bio-formats and OME-TIFF, but also makes it painless to handle hugh data by delayed loading.


2. There are three levels of abstraction: 
- `mmv_im2im/proj_trainer.py` (main entry point to define data module, pytorch-lightning module, and trainer)
    -  `mmv_im2im/data_modules/data_loaders.py` (Currently, all 2D/3D paired/unpaired data loader can share the same universal dataloader. We may add other generic dataloaders when really needed, otherwise, the current one is general enough to cover current applications)
    -  `mmv_im2im/models/pl_XYZ.py` (the middle level wrapper for different categories of models: FCN, pix2pix (conditional GAN), CycleGAN, embedseg. We can add more when needed, e.g., for denoising or cellpose. This is the pytorch-lightning module specific for this category of models. The specific model backbone, loss function, etc. can be easily specified via parameters in yaml config)
        -  Other scripts under `mmv_im2im/models/nets/` and `mmv_im2im/preprocessing`, as well as `mmv_im2im/utils`, are low level functions to instantiate the pytorch-lightining module defined in `pl_XYZ.py` or `data_loader.py`


3. Understand the configuration system and make customizations

`run_im2im --help` can print out all the configuration details, including their default values. We pre-compiled a collection of configuration for various tasks, see `/mmv_im2im/configs/preset_XXYYZZ.yaml`. To use a preset configuration, one can simply do `run_im2im --config XXYYZZ` Or, you can run on your customized configuration file `run_im2im --config /full/path/to/your/config.yaml`. You can specify everything in the yaml file, or you can also pass in additional args via the command line. For example, we have a general configuration file at `/home/example/my_config.yaml`, which specific the model details and training details for one specific project. But, when you train different models for this same project with different data, you don't have to change the yaml file. Instead, you can simply pass in the extra data path in command line, like `run_im2im --config /home/example/my_config.yaml --data.data_path /path/to/my/data`. So, it is a good practice to make different generic configurations for different projects, and leave the runtime specific args, like data_path, to the command line. 

A special note on dictionary parameters. After printing out all the details from `run_im2im --help`, if you see one parameter is a Dict, for example `training.params`, you need to be careful if you want to overwrite the values specified in yaml by values passed in command line. Specifically, you have to overwrite the whole dictionary, and cannot overwrite only a specific key. In this situation, I would recommed to change the config file directly, instead of overwriting with command line args.  


4. About the training data directory

Currently, the package supports three types of training data organizations. First, one single folder with "X1_IM.tiff", "X1_GT.tiff", "X1_CM.tiff", "X2_IM.tiff", etc.. Second, one single folder with multiple sub-folders "IM", "GT", "CM" (optional), where the name of each file is exactly the same accross different subfolders. The first two scenario will incur a train/validatio split on all files. If you already have train/validation split done, then you can pass in the something like `--data.data_path "{'train': '/path/to/train', 'val': '/path/to/val'}"`, where `/path/to/train` and `/path/to/val` are similar to the first scenario above.

**preset keys**: "IM" (raw images), "GT" (ground truth or training target), "CM" (costmap, e.g. to exclude certain pixels by setting corresponding value in costmap as 0)

**Additional note for EmbedSeg**: When doing `run_im2im --config train_embedseg_3d --data.data_path /path/to/train`, we expect "IM" and "GT" in the folder `/path/to/train` ("CM" is optional). However, the program will automatically convert this dataset into a special format needed for EmbedSeg and save them at the folder specified by `data.cache_path` (default is `./tmp`). Two new types of files wil be created in the cache_path, including "CL" (CLass image) and "CE" (CEnter image). Since preparing this compatible dataset takes time, one can directly specify the cache_path to load without re-generating from original training data. Namely `run_im2im --config train_embegseg_3d --data.cache_path ./tmp/` even without passing the `data.data_path`.  

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.



**MIT license**

