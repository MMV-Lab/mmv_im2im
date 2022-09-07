# MMV_Im2Im package tutorial

(This is a long tutorial. If you are reading on Github, you can use [the table-of-content button](https://github.blog/changelog/2021-04-13-table-of-contents-support-in-markdown-files/) to navigate the tutorial.)

## How to use the configuration system to run experiments?

In general, the package adopts a "predefine + YAML + command line" method to achieve a balance between simplicity and flexibility, customized from [pyrallis](https://eladrich.github.io/pyrallis/). In order to maintain high flexibility, the package is highly modularized and deeply configurable, there are a lot of "knobs" (i.e., hyper-parameters) you can tweak by configuration without touching any line of code. But, we cannot manually set every single hyper-parameter in each experiment. That is just too much work. For this reason, a lot of the hyper-parameters have been predefined with a default value, but can still be changed if necessary. For example, data.dataloader.train_val_ratio controls how to split the full dataset into train and validation in each experiment. The default value is 0.1 (i.e., 10% of the full dataset will be used for validation).

**To update any of the hyper-parameters**, the program will first look for if there is anything set in a YAML file, if so, the default values will be overwritten. Then, the program will check if there is any more args in the command line, if so, the specific values (even if having been overwritten once by YAML) will be overwritten. In short, the order of "authority" is predefine < YAML < command line. 

Here is an example. Suppose you have a YAML file, called `example.yaml` like this

```yaml
data:
    dataloader:
        train_val_ratio: 0.3
```

When you run `run_im2im --config example.yaml --data.dataloader.train_val_ratio 0.5`, the program will first load all default hyperparameters, so `train_val_ratio` is 0.1. Then the program receives a YAML configuration and finds that `train_val_ratio` is set as 0.3 now. Finally, the program checks all command line inputs and finds that `train_val_ratio` is set as 0.5. At the end, the program takes `train_val_ratio` as 0.5. This is just an example to explain how the configuration system works to avoid accidental hyper-parameter misuse. 

**YAML files can be used in two different ways.** First, all YAML file under `mmv_im2im/configs/` whose filename contains prefix `preset_` can be used in a simplified way (note: have to build from source). For example, if you create a YAML file `mmv_im2im/configs/preset_train_my_problem.yaml`, then you can simply do `run_im2im --config train_my_problem --data.data_path /path/to/data`. This is useful when you create a few generic YAML files for different problems and in each run, you simply pass in the training data path in the command line. Second, you can also put every customized hyperparameter of each run to keep the full record of your experiment, maybe saved in a central location with version control. For example, you have a YAML file at `/home/Luci/exp_log/proj_1_set_2.yaml`. Then, you can run with only `run_im2im --config /home/Luci/exp_log/proj_1_set_2.yaml`, no more command line args.


**Help function**
`run_im2im --help` can print out all the configuration details, including their default values. Note that if you see one parameter is a Dict, for example `training.params`, you need to be careful if you want to overwrite the values specified in yaml by values passed in command line. Specifically, you have to overwrite the whole dictionary, and cannot overwrite only a specific key. In this situation, I would recommend to change the config file directly, instead of overwriting with command line args.  


## Walk-through the configuration systems step by step

After knowing how to use this configuration system, now we can look at what exactly can be configured. You can find a lot of example configurations [HERE](https://github.com/MMV-Lab/mmv_im2im/tree/main/paper_configs), which can be used to reproduce all experiments in our [pre-print](https://arxiv.org/abs/2209.02498). You can always use one of these as a start and do your changes. 


There are four major parts in the configuration system: mode, data, model, trainer. We will go through these parts one by one. 

### part 1: mode

Currently, only supporting two values "train" or "inference". In our future version, we plan to add "evaluation", which can automatically generate evaluation reports for you. This will be useful for benchmarking different methods.

### part 2: data

All fields under `data` can be found [HERE](https://github.com/MMV-Lab/mmv_im2im/blob/main/mmv_im2im/configs/config_base.py#L250). The parts you may need to adjust for your problems are as follows.


#### only for training

* data.category: You can specify your data are "paired" or "unpaired" (e.g. for CycleGAN), or "embedseg" (for instance segmentation).

* data.data_path: the path to your training data. Currently, the package supports three types of training data organizations. First, one single folder with "X1_IM.tiff", "X1_GT.tiff", "X1_CM.tiff", "X2_IM.tiff", etc.. Second, one single folder with multiple sub-folders "IM", "GT", "CM" (optional), where the name of each file is exactly the same across different sub-folders. The first two scenario will incur a train/validation split on all files. If you already have train/validation split done, then you can pass in the something like `--data.data_path "{'train': '/path/to/train', 'val': '/path/to/val'}"`, where `/path/to/train` and `/path/to/val` are similar to the first scenario above. It is important to use the **preset keys** as in the examples above. "IM" (raw images), "GT" (ground truth or training target), "CM" (costmap, e.g. to exclude certain pixels by setting corresponding value in costmap as 0). The program relies on these keys to load the correct data.

* data.cache_path: This is only used for instance segmentation. When doing `run_im2im --config train_embedseg_3d --data.data_path /path/to/train`, we expect "IM" and "GT" in the folder `/path/to/train` ("CM" is optional). However, the program will automatically convert this dataset into a special format needed for EmbedSeg and save them at the folder specified by `data.cache_path` (default is `./tmp`). Two new types of files wil be created in the cache_path, including "CL" (CLass image) and "CE" (CEnter image). Since preparing this compatible dataset takes time, one can directly specify the cache_path to load without re-generating from original training data. Namely `run_im2im --config train_embegseg_3d --data.cache_path ./tmp/` even without passing the `data.data_path`.  


#### only for inference

* data.inference_output: where to save the output
* data.inference_input:
    * data.inference_input.dir: where to load the test data
    * data.inference_input.reader_params: how to read the data with aicsimageio, see [example configs](https://github.com/MMV-Lab/mmv_im2im/tree/main/paper_configs)


#### Special note about preprocess, augmentation, postprocess

TBA


### part 3: model

All fields under `model` can be found [HERE](https://github.com/MMV-Lab/mmv_im2im/blob/main/mmv_im2im/configs/config_base.py#L288)


### part 4: trainer

We adopted the trainer from pytorch-lightning. You can find detailed API from [the official documentation](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-class-api). Then, you can find examples of how to pass in these parameters in our example configuration files.