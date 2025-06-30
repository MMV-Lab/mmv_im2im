# MMV Im2Im Transformation

[![Build Status](https://github.com/MMV-Lab/mmv_im2im/workflows/Build%20Main/badge.svg)](https://github.com/MMV-Lab/mmv_im2im/actions)

A generic python package for deep learning based image-to-image transformation in biomedical applications

The main branch will be further developed in order to be able to use the latest state of the art techniques and methods in the future. To reproduce the results of our manuscript, we refer to the branch [paper_version](https://github.com/MMV-Lab/mmv_im2im/tree/paper_version).
(We are actively working on the documentation and tutorials. Submit a feature request if there is anything you need.)

---

## Overview

The overall package is designed with a generic image-to-image transformation framework, which could be directly used for semantic segmentation, instance segmentation, image restoration, image generation, labelfree prediction, staining transformation, etc.. The implementation takes advantage of the state-of-the-art ML engineering techniques for users to focus on researches without worrying about the engineering details. In our pre-print [arxiv link](https://arxiv.org/abs/2209.02498), we demonstrated the effectiveness of *MMV_Im2Im* in more than ten different biomedical problems/datasets. 

* For computational biomedical researchers (e.g., AI algorithm development or bioimage analysis workflow development), we hope this package could serve as the starting point for their specific problems, since the image-to-image "boilerplates" can be easily extended further development or adapted for users' specific problems.
* For experimental biomedical researchers, we hope this work provides a comprehensive view of the image-to-image transformation concept through diversified examples and use cases, so that deep learning based image-to-image transformation could be integrated into the assay development process and permit new biomedical studies that can hardly be done only with traditional experimental methods


## Installation

Before starting, we recommend to [create a new conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) or [a virtual environment](https://docs.python.org/3/library/venv.html) with Python 3.10+.

```bash
conda create -y -n im2im -c conda-forge python=3.11
conda activate im2im
```

Please note that the proper setup of hardware is beyond the scope of this pacakge. This package was tested with GPU/CPU on Linux/Windows and CPU on MacOS. [Special note for MacOS users: Directly pip install in MacOS may need [additional setup of xcode](https://developer.apple.com/forums/thread/673827).]

### Install MONAI

To reproduce our results, we need to install MONAI's code version of a specific commit. To do this:
```bash
git clone https://github.com/Project-MONAI/MONAI.git
cd ./MONAI
git checkout 37b58fcec48f3ec1f84d7cabe9c7ad08a93882c0
pip install .
```

We will remove this step for the main branch in the future to ensure a simplified installation of our tool.

### Install MMV_Im2Im for basic usage:

(For users only using this package, not planning to change any code or make any extension):

**Option 1: core functionality only** `pip install mmv_im2im`<br>
**Option 2: advanced functionality (core + logger)** `pip install mmv_im2im[advance]`<br>
**Option 3: to reproduce paper:** `pip install mmv_im2im[paper]`<br>
**Option 4: install everything:** `pip install mmv_im2im[all]`<br>

For MacOS users, additional ' ' marks are need when using installation tags in zsh. For example, `pip install mmv_im2im[paper]` should be `pip install mmv_im2im'[paper]'` in MacOS.

### Install MMV_Im2Im for customization or extension:


```bash
git clone https://github.com/MMV-Lab/mmv_im2im.git
cd mmv_im2im
pip install -e .[all]
```

Note: The `-e` option is the so-called "editable" mode. This will allow code changes taking effect immediately. The installation tags, `advance`, `paper`, `all`, are be selected based on your needs.

### (Optional) Install using Docker

It is also possible to use our package through [docker](https://www.docker.com/). The installation tutorial is [here](docker/tutorial.md). Specifically, for MacOS users, please refer to [this tutorial](tutorials/docker/mmv_im2im_docker_tutorial.md).

### (Optional) Use MMV_Im2Im with Google Colab

We provide a web-based demo, if cloud computing is preferred. you can [![Open a 2D labelfree DEMO in Google Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MMV-Lab/mmv_im2im/blob/main/tutorials/colab/labelfree_2d.ipynb). The same demo can de adapted for different applications.

## Quick start

You can try out on a simple example following [the quick start guide](tutorials/quick_start.md)

Basically, you can specify your training configuration in a yaml file and run training with `run_im2im --config /path/to/train_config.yaml`. Then, you can specify the inference configuration in another yaml file and run inference with `run_im2im --config /path/to/inference_config.yaml`. You can also run the inference as a function with the provided API. This will be useful if you want to run the inference within another python script or workflow.  Here is an example:

```python
from pathlib import Path
from bioio import BioImage
from bioio.writers import OmeTiffWriter
from mmv_im2im.configs.config_base import ProgramConfig, parse_adaptor, configuration_validation
from mmv_im2im import ProjectTester

# load the inference configuration
cfg = parse_adaptor(config_class=ProgramConfig, config="./paper_configs/semantic_seg_2d_inference.yaml")
cfg = configuration_validation(cfg)

# define the executor for inference
executor = ProjectTester(cfg)
executor.setup_model()
executor.setup_data_processing()

# get the data, run inference, and save the result
fn = Path("./data/img_00_IM.tiff")
img = BioImage(fn).get_image_data("YX", Z=0, C=0, T=0)
# or using delayed loading if the data is large
# img = BioImage(fn).get_image_dask_data("YX", Z=0, C=0, T=0)
seg = executor.process_one_image(img)
OmeTiffWriter.save(seg, "output.tiff", dim_orders="YX")
```


## Tutorials, examples, demonstrations and documentations

The overall package aims to achieve both simplicty and flexibilty with the modularized image-to-image boilerplates. To help different users to best use this package, we provide documentations from four different aspects:

* [Examples (i.e., scripts and config files)](tutorials/example_by_use_case.md) for reproducing all the experiments in our [pre-print](https://arxiv.org/abs/2209.02498)
* A bottom-up tutorials on [how to understand the modularized image-to-image boilerplates](tutorials/how_to_understand_boilerplates.md) (for extending or adapting the package) and [how to understand the configuration system in details](tutorials/how_to_understand_config.md) (for advance usage to make specific customization).
* A top-down tutorials as [FAQ](tutorials/FAQ.md), which will continuously grow as we receive more questions.
* All the models used in the manuscript and sample data can be found here: [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10034416.svg)](https://doi.org/10.5281/zenodo.10034416)


### Contribute models to [BioImage Model Zoo](https://bioimage.io/#/)

We highly appreciate the BioImage Model Zoo's initiative to provide a comprehensive collection of pre-trained models for a wide range of applications. To make MMV_Im2Im trained models available as well, the first step involves extracting the state_dict from the PyTorch Lightning checkpoint.
This can be done via:

```python
import torch

ckpt_path = "./lightning_logs/version_0/checkpoints/last.ckpt"
checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
state_dict = checkpoint['state_dict']
torch.save(state_dict, "./state_dict.pt")
```

All further steps to provide models can be found in the [official documentation](https://bioimage.io/docs/#/contribute_models/README).

## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.


**MIT license**
