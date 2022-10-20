# MMV Im2Im Transformation

[![Build Status](https://github.com/MMV-Lab/mmv_im2im/workflows/Build%20Main/badge.svg)](https://github.com/MMV-Lab/mmv_im2im/actions)
[![Documentation](https://github.com/MMV-Lab/mmv_im2im/workflows/Documentation/badge.svg)](https://MMV-Lab.github.io/mmv_im2im/)

A generic python package for deep learning based image-to-image transformation in biomedical applications

(We are actively working on the documentation and tutorials. Submit a feature request if there is anything you need.)

---

## Overview

The overall package is designed with a generic image-to-image transformation framework, which could be directly used for semantic segmentation, instance segmentation, image restoration, image generation, labelfree prediction, staining transformation, etc.. The implementation takes advantage of the state-of-the-art ML engineering techniques for users to focus on researches without worrying about the engineering details. In our pre-print [arxiv link](https://arxiv.org/abs/2209.02498), we demonstrated the effectiveness of *MMV_Im2Im* in more than ten different biomedical problems/datasets. 

* For biomedical machine learning researchers, we hope this new package could serve as the starting point for their specific problems to stimulate new biomedical image analysis or machine learning methods. 
* For experimental biomedical researchers, we hope this work can provide a holistic view of the image-to-image transformation concept with diverse examples, so that deep learning based image-to-image transformation could be further integrated into the assay development process and permit new biomedical studies that can hardly be done only with traditional experimental methods


## Installation

We recommend to [create a new conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) or [a virtual environment](https://docs.python.org/3/library/venv.html) with Python 3.9.

### Install PyTorch before installing our package

Follow the instruction from the official website: https://pytorch.org/get-started/locally/. E.g..

To install the stable version (accessed on Oct 20, 2022) with conda for CUDA 11.3 (also, fine with 11.4), use the following command. Make sure check the website to find the command suitable for your system.

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

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


## Quick start / Try out on a simple example

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

## Walk-through Guide

We provide [a tutorial](tutorials/README.md) for users to understand how to enjoy both the simplicity and the full flexibility of the package. 

The scripts and configurations for reproducing our experiments in the [pre-print](https://arxiv.org/abs/2209.02498) can also be used as examples for walking through different applications.

| Application | data preparation | training config | inference config |
| :---: |  :---: |  :---: |  :---: |
| 3D labelfree prediction  | [script](script/pull_labelfree_sample_data.py) | [supervisedFCN](./paper_configs/labelfree_3d_FCN_train.yaml), [supervisedPix2pix-TBA](), [unsupervised-TBA]() |  [supervisedFCN](./paper_configs/labelfree_3d_FCN_inference.yaml), [supervisedPix2pix-TBA](), [unsupervised-TBA]() |
| 2D labelfree prediction  | [script-TBA]() | [supervisedFCN-TBA]() |  [supervisedFCN-TBA]() |
| 2D semantic segmentation | [script-TBA]() | [supervisedFCN-TBA]() |  [supervisedFCN-TBA]() |
| 3D semantic segmentation | [script-TBA]() | [supervisedFCN-TBA]() |  [supervisedFCN-TBA]() |
| 2D instance segmentation | [script-TBA]() | [supervisedEmbedSeg-TBA]() |  [supervisedEmbedseg-TBA]() |
| 3D instance segmentation | [script-TBA]() | [supervisedEmbedSeg_fluo](./paper_configs/train_embedseg_3d.yaml), [supervisedEmbedSeg_bf](./paper_configs/train_embedseg_3d_BF.yaml) |  [supervisedEmbedseg_fluo](./paper_configs/inference_embedseg_3d.yaml), [supervisedEmbedSeg_bf](./paper_configs/inference_embedseg_3d_bf.yaml) | 
| 2D unsupervised segmentation | [script-TBA]() | [unsupervised-TBA]() |  [unsupervised-TBA]() | 
| 3D unsupervised segmentation | [script-TBA]() | [unsupervised-TBA]() |  [unsupervised-TBA]() | 
| 2D synthetic image generation | [script-TBA]() | [supervisedPix2pix-TBA](), [unsupervised-TBA]() | [supervisedPix2pix-TBA](), [unsupervised-TBA]() |
| 3D synthetic image generation | [script-TBA]() | [supervisedPix2pix-TBA]() | [supervisedPix2pix-TBA]() |
| image restoration / denoising  | [script-TBA]() | [supervisedFCN-TBA]() | [supervisedFCN-TBA]()  |
| image modality transfer | [script-TBA]() | [supervisedFCN-TBA]() | [supervisedFCN-TBA]()  |
| staining transformation | [script-TBA]() | [supervisedPix2pix-TBA]() | [supervisedPix2pix-TBA]() |


## API Documentation

For full package API (i.e., the technical details of each function), please visit [MMV-Lab.github.io/mmv_im2im](https://MMV-Lab.github.io/mmv_im2im).


## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.


**MIT license**

