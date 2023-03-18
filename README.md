# MMV Im2Im Transformation

[![Build Status](https://github.com/MMV-Lab/mmv_im2im/workflows/Build%20Main/badge.svg)](https://github.com/MMV-Lab/mmv_im2im/actions)
[![Documentation](https://github.com/MMV-Lab/mmv_im2im/workflows/Documentation/badge.svg)](https://MMV-Lab.github.io/mmv_im2im/)

A generic python package for deep learning based image-to-image transformation in biomedical applications

(We are actively working on the documentation and tutorials. Submit a feature request if there is anything you need.)

---

## Overview

The overall package is designed with a generic image-to-image transformation framework, which could be directly used for semantic segmentation, instance segmentation, image restoration, image generation, labelfree prediction, staining transformation, etc.. The implementation takes advantage of the state-of-the-art ML engineering techniques for users to focus on researches without worrying about the engineering details. In our pre-print [arxiv link](https://arxiv.org/abs/2209.02498), we demonstrated the effectiveness of *MMV_Im2Im* in more than ten different biomedical problems/datasets. 

* For computational biomedical researchers (e.g., AI algorithm development or bioimage analysis workflow development), we hope this package could serve as the starting point for their specific problems, since the image-to-image "boilerplates" can be easily extended further development or adapted for users' specific problems.
* For experimental biomedical researchers, we hope this work provides a comprehensive view of the image-to-image transformation concept through diversified examples and use cases, so that deep learning based image-to-image transformation could be integrated into the assay development process and permit new biomedical studies that can hardly be done only with traditional experimental methods


## Installation

Before starting, we recommend to [create a new conda environment](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#creating-an-environment-with-commands) or [a virtual environment](https://docs.python.org/3/library/venv.html) with Python 3.9+.

### Install PyTorch before installing our package

Follow the instruction from the official website: https://pytorch.org/get-started/locally/. 

To install the stable version (accessed on Oct 20, 2022) with conda for CUDA 11.3 (also, fine with 11.4), use the following command. Make sure check the website to find the command suitable for your system.

(note: we haven't tested the latest PyTorch 2.0, will update accordingly if necessary.)

```bash
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

### Install MMV_Im2Im (basic):

(For users only using this package, not planning to change any code or make any extension):

**Stable Release:** `pip install mmv_im2im`<br>
**Development Head:** `pip install git+https://github.com/MMV-Lab/mmv_im2im.git`

### Install MMV_Im2Im (build from source):

(For users planning to extend the methods or improve the code):

```
git clone https://github.com/MMV-Lab/mmv_im2im.git
cd mmv_im2im
pip install -e .[all]
```

Note: The `-e` option is the so-called "editable" mode. This will allow code changes taking effect immediately.


## Quick start

You can try out on a simple example following [the quick start guide](tutorials/quick_start.md)

Basically, you can specify your training configuration in a yaml file and run training with `run_im2im --config /path/to/train_config.yaml`. Then, you can specify the inference configuration in another yaml file and run inference with `run_im2im --config /path/to/inference_config.yaml`. You can also run the inference as a function with the provided API. This will be useful if you want to run the inference within another python script or workflow.  Here is an example:

```
from pathlib import Path
from aicsimageio import AICSImage
from aicsimageio.writers import OmeTiffWriter
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
img = AICSImage(fn).get_image_data("YX", Z=0, C=0, T=0)
# or using delayed loading if the data is large
# img = AICSImage(fn).get_image_dask_data("YX", Z=0, C=0, T=0)
seg = executor.process_one_image(img)
OmeTiffWriter.save(seg, "output.tiff", dim_orders="YX")
```


## Tutorials, examples, demonstrations and documentations

The overall package aims to achieve both simplicty and flexibilty with the modularized image-to-image boilerplates. To help different users to best use this package, we provide documentations from four different aspects:

* [Examples (i.e., scripts and config files)](tutorials/example_by_use_case.md) for reproducing all the experiments in our [pre-print](https://arxiv.org/abs/2209.02498)
* A bottom-up tutorials on [how to understand the modularized image-to-image boilerplates](tutorials/how_to_understand_boilerplates.md) (for extending or adapting the package) and [how to understand the configuration system in details](tutorials/how_to_understand_config.md) (for advance usage to make specific customization).
* A top-down tutorials as [FAQ](tutorials/FAQ.md), which will continuously grow as we receive more questions.
* Full package API (i.e., the technical details of each function) [MMV-Lab.github.io/mmv_im2im](https://MMV-Lab.github.io/mmv_im2im).



## Development

See [CONTRIBUTING.md](CONTRIBUTING.md) for information related to developing the code.


**MIT license**

