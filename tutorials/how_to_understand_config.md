# Everything about the configuration system

(This is a long tutorial. If you are reading on Github, you can use [the table-of-content button](https://github.blog/changelog/2021-04-13-table-of-contents-support-in-markdown-files/) to navigate the tutorial.)

## How to use the configuration system to run experiments?

In general, the package adopts a "pre-define + YAML + command line" method to achieve a balance between simplicity and flexibility, customized from [pyrallis](https://eladrich.github.io/pyrallis/). In order to maintain high flexibility, the package is highly modularized and deeply configurable, there are a lot of "knobs" (i.e., hyper-parameters) you can tweak by configuration without touching any line of code. But, we cannot manually set every single hyper-parameter in each experiment. That is just too much work. For this reason, a lot of the hyper-parameters have been predefined with a default value, but can still be changed when necessary. For example, data.dataloader.train_val_ratio controls how to split the full dataset into train and validation in each experiment. The default value is 0.1 (i.e., 10% of the full dataset will be used for validation), but you can easily override this value when you need to use a different ratio.

**To update any of the hyper-parameters**, the program will first look for if there is anything set in a provided YAML file, if so, the default values will be overwritten. Then, the program will check if there is any more args in the command line, if so, the specific values (even if having been overwritten once by YAML) will be overwritten. In short, the order of "authority" is pre-define < YAML < command line. 

Here is an example. Suppose you have a YAML file, called `example.yaml` like this

```yaml
data:
    dataloader:
        train_val_ratio: 0.3
```

When you run `run_im2im --config example.yaml --data.dataloader.train_val_ratio 0.5`, the program will first load all default hyperparameters, so `train_val_ratio` is 0.1. Then the program receives a YAML configuration and finds that `train_val_ratio` is set as 0.3 now. Finally, the program checks all command line inputs and finds that `train_val_ratio` is set as 0.5. At the end, the program takes `train_val_ratio` as 0.5. This is just an example to explain how the configuration system works to avoid accidental hyper-parameter mis-use. 

**YAML files can be used in two different ways.** First, all YAML file under `mmv_im2im/configs/` whose filename contains prefix `preset_` can be used in a simplified way. You check all existing preset configs [HERE](https://github.com/MMV-Lab/mmv_im2im/tree/main/mmv_im2im/configs). If you built the package from source, you can also add your own preset for easy re-use. For example, if you create a YAML file `mmv_im2im/configs/preset_train_my_problem.yaml`, then you can simply do `run_im2im --config train_my_problem --data.data_path /path/to/data`. This is useful when you create a few generic YAML files for different problems and in each run, you simply pass in the training data path in the command line. Second, you can also put every customized hyperparameter of each run to keep the full record of your experiment, maybe saved in a central location with version control. For example, you have a YAML file at `/home/Luci/exp_log/proj_1_set_2.yaml`. Then, you can run with only `run_im2im --config /home/Luci/exp_log/proj_1_set_2.yaml`, no more command line args.


**Help function**
`run_im2im --help` can print out all the configuration details, including their default values. Note that if you see one parameter is a Dict, for example `training.params`, you need to be careful if you want to overwrite the values specified in yaml by values passed in command line. Specifically, you have to overwrite the whole dictionary, and cannot overwrite only a specific key. In this situation, I would recommend to change the config file directly, instead of overwriting with command line args.  


## Walk-through the configuration systems step by step

After knowing how to use this configuration system, now we can look at what exactly can be configured. You can find a lot of example configurations [HERE](https://github.com/MMV-Lab/mmv_im2im/tree/main/paper_configs), which can be used to reproduce all experiments in our [pre-print](https://arxiv.org/abs/2209.02498). You can always use one of these as a start and do your changes. 

There are four major parts in the configuration system: mode, data, model, trainer. We will go through these parts one by one. 

### part 1: mode

Currently, only two values "train" or "inference" are supported. In our future version, we plan to add an "evaluation" mode, which can automatically generate evaluation reports for you. This will be useful for benchmarking different methods.

### part 2: data

All fields under `data` can be found [HERE](https://github.com/MMV-Lab/mmv_im2im/blob/main/mmv_im2im/configs/config_base.py#L250). There are mainly four types of hyper-parameters: directory, transformation, dataloader, and extras. 

#### About directory

##### training

* data.category: You can specify your data are "paired" or "unpaired" (e.g. for CycleGAN), or "embedseg" (for instance segmentation).

* `data.data_path`: the path to your training data. Currently, the package supports three types of training data organizations. First, one single folder with "X1_IM.tiff", "X1_GT.tiff", "X1_CM.tiff", "X2_IM.tiff", etc.. Second, one single folder with multiple sub-folders "IM", "GT", "CM" (optional), where the name of each file is exactly the same across different sub-folders. The first two scenario will incur a train/validation split on all files. If you already have train/validation split done, then you can pass in the something like `--data.data_path "{'train': '/path/to/train', 'val': '/path/to/val'}"`, where `/path/to/train` and `/path/to/val` are similar to the first scenario above. It is important to use the **preset keywords** as in the examples above. "IM" (raw images), "GT" (ground truth or training target), "CM" (costmap, e.g. to exclude certain pixels by setting corresponding value in costmap as 0). The program relies on these keywords to load the correct data.

* `data.cache_path`: This is only used for instance segmentation with Embedseg in a pre-train mode (see more details about [Embedseg Training](note_embedseg.md)). When doing `run_im2im --config train_embedseg_3d --data.data_path /path/to/train`, we expect "IM" and "GT" in the folder `/path/to/train` ("CM" is optional). When `cach_path` is set, the program will automatically convert this dataset into a special format needed for EmbedSeg and save them at the folder specified by `data.cache_path` (default is `./tmp`). Two new types of files wil be created in the cache_path, including "CL" (CLass image) and "CE" (CEnter image). Since preparing this compatible dataset takes time, one can directly specify the cache_path (if have been generated in a previous run) to load without re-generating. Namely `run_im2im --config train_embegseg_3d --data.cache_path ./tmp/` even without passing the `data.data_path`. When `cach_path` is not set, we refer this as fine-tuning mode, only the `data_path` is enough, and the patches will be randomly generated on-the-fly.

##### inference

* `data.inference_input`:
    * data.inference_input.dir: where to load the test data
    * data.inference_input.data_type: default is "tiff", where all tiff files in the test data folder will be used. This is useful to only run inference on a specific set of images by regular expression, like "CD64_2023_*.tif" or "*_DAPI.czi". 
    * data.inference_input.reader_params: how to read the data with aicsimageio, see [example configs](https://github.com/MMV-Lab/mmv_im2im/tree/main/paper_configs)

* `data.inference_output`:
    * data.inference_output.path: where to save the prediction output
    * data.inference_output.suffix: default is "pred". For example, when applying the model on a file named "sample1.tiff", the output will be saved with a name "sample1_pred.tiff". When a different suffix is preferred, it can be changed here.


#### transformation

##### trans for training (`preprocess` and `augmentation`)

For all transformations during training, we follow dictionary transform syntax in [MONAI transforms](https://docs.monai.io/en/stable/transforms.html), but are not restricted to use the transforms in MONAI. (Recall the keywords, like "IM" and "GT", explained in `data.data_path`. They correspond to the dictionary keys here.) 

You can list as many operations as you want under `preprocess` and `augmentation` following the same syntax. The only difference is that all operations under `preprocess` will be applied to both training and validation data, but the operations under `validation` will only be used on training data.  Here is a typical example:

```yaml
preprocess:
  - module_name: monai.transforms
    func_name: LoadImaged
    params:
      keys: ["IM", "GT"]
      dimension_order_out: "ZYX"
      T: 0
      C: 0
    # Note: we wrote a special bioformat reader with aicsimageio and wrapped it to be compatible with MONAI data reading
    # The reading function above follows the parameter here: https://allencellmodeling.github.io/aicsimageio/aicsimageio.html?highlight=get_image_dask_data#aicsimageio.aics_image.AICSImage.get_image_dask_data
  - module_name: monai.transforms
    func_name: AddChanneld
    params:
      keys: ["IM", "GT"]
  # we can also add transforms not in monai
  - module_name: mmv_im2im.preprocessing.transforms
    func_name: norm_around_center
    params:
      keys: ["IM"]
      min_z: 32
  # Note that by dictionary transforms, you can easily apply different operations on image and ground truth
  - module_name: monai.transforms
      func_name: ScaleIntensityRangePercentilesd
      params:
        keys: ["GT"]
        lower: 0.1
        upper: 99.9
        b_min: 0
        b_max: 1
  # apply the same random sampling on both "IM" and "GT" so that their correspondence can be maintained. 
  # When different cropping from "IM" and "CT" are needed, e.g. for CycleGAN, you can use two different samplers, each with a different key.
  # You can also choose the put sampler in preprocess or in augmentation, depending on your use case. 
  - module_name: monai.transforms
    func_name: RandSpatialCropSamplesd
    params:
      keys: ["IM", "GT"]
      random_size: False
      num_samples: 4
      roi_size: [32, 128, 128]
  - module_name: monai.transforms
    func_name: EnsureTyped
    params:
      keys: ["IM", "GT"]
augmentation:
  - module_name: monai.transforms
    func_name: RandFlipd
    params:
      prob: 0.5
      keys: ["IM", "GT"]
  - module_name: monai.transforms
    func_name: RandStdShiftIntensityd
    params:
      prob: 0.25
      factors: 1.0
      keys: ["IM", "GT"]
```

##### trans for inference (`preprocess` and `postprocess`)

For inference, we only need the input data (no ground truth), so we don't need to use the dictionary transforms. Usually, it is important to use the same intensity normalization methods in inference as in training, unless in special use cases where you are sure about specific processing works better. For example, when `ScaleIntensityRangePercentilesd` from MONAI is used to preprocess your data in training, then you usually need to do `ScaleIntensityRangePercentiles` in inference (Note: be careful with the letter "d" at the end of the function name, if from MONAI, referring the "dictionary" version of a specific transform).

For `postprocess`, one can use chain up a list of different operations as for `preprocess`, e.g., some [post-processing transforms from MONAI](https://docs.monai.io/en/stable/transforms.html#post-processing) or a few [more customized functions provided by **MMV_Im2IM**](https://mmv-lab.github.io/mmv_im2im/mmv_im2im.postprocessing.html).

```yaml
preprocess:      
  - module_name: monai.transforms
    func_name: NormalizeIntensity
postprocess:
  - module_name: mmv_im2im.postprocessing.embedseg_cluster
    func_name: generate_instance_clusters
    params:
      grid_x: 1024
      grid_y: 1024
      grid_z: 128
      pixel_x: 1
      pixel_y: 1
      pixel_z: 1
      n_sigma: 3 
      seed_thresh: 0.5
      min_mask_sum: 2
      min_unclustered_sum: 2
      min_object_size: 2
```

#### data loaders

The `dataloader` contains a hierarchy of multiple levels information. The top level contains three parameters: `dataloader.train`, `dataloader.val` and `dataloader.train_val_ratio`, which defines the training dataloader, validation dataloader, and how to split the data into train set and validation set (default 0.1). Both `dataloader.train` and `dataloader.val` have the following default setting:

```yaml
  dataloader_type:  # could be any dataload in [MONAI data](https://docs.monai.io/en/stable/data.html#generic-interfaces)
    module_name: monai.data
    func_name: PersistentDataset
  dataloader_params:
    batch_size: 1
    pin_memory: True
    num_workers: 2
  dataset_params: # parameter for PersistentDataset (https://docs.monai.io/en/stable/data.html#persistentdataset) or any dataload you choose to use
    cache_dir: "./tmp"
    pickle_protocol: 5
  partial_loader: 1.0  # only load certain percentage of the dataset, value between 0 and 1.0.
```

You can overwrite some of the default values like this:

```yaml
dataloader:
  train:
    dataloader_params:
      batch_size: 2
      pin_memory: True
      num_workers: 4
  train_val_ratio: 0.2
```


#### extras

This is reserved for additional parameters for future extension, specially when needing to add special operations to customized the data loading process (e.g., extending [the current universal data module](https://github.com/MMV-Lab/mmv_im2im/blob/main/mmv_im2im/data_modules/data_loader_basic.py) with specific needs). 

### part 3: model

All fields under `model` can be found [HERE](https://github.com/MMV-Lab/mmv_im2im/blob/main/mmv_im2im/configs/config_base.py#L288)

* model.framework: currenly only four options, FCN, pix2pix, cyclegan, embedseg.
* model.net: a dictionary defining your backbone network
* model.criterion: a dictionary defining which loss function(s) to use
* model.optimizer: a dictionary defining which optimizor(s) and optimization parameters (e.g., learning rate or weight decay) to use
* model.scheduler: a dictionary defining which scheduler(s) to use
* model.checkpoint: **this is only for inference**, defining with model to use. 
* model.model_extra: This is reserved for special parameters or for future customizations. Two very common use cases are loading a pre-train model and defining how to apply sliding window inference during the validation step. See example in how to [finetune an embedseg model for 3D instance segmentation](https://github.com/MMV-Lab/mmv_im2im/blob/main/paper_configs/instance_seg_3d_train_bf_finetune.yaml#L71).

Please refer to [the collection of sample configs](example_by_use_case.md) for examples how to set `model` in different applications.


### part 4: trainer (only for training)

We adopted the trainer from pytorch-lightning. You can find detailed API from [the official documentation](https://pytorch-lightning.readthedocs.io/en/stable/common/trainer.html#trainer-class-api) and [the callbacks it supports](https://pytorch-lightning.readthedocs.io/en/latest/extensions/callbacks.html#built-in-callbacks). You can find examples of how to pass in these parameters in our [example configuration files](example_by_use_case.md). A `verbose` option is available. When setting to `True`, one example data (e.g., input + output + ground truth) will be saved out in the lightning log folder at the beginning of each epoch. A typical setting for `trainer` is as follows:

```yaml
trainer:
  verbose: True # or False
  params:   # anything in pytorch-lightning trainer API
    gpus: 1
    precision: 16
    max_epochs: 1000
    detect_anomaly: True
  callbacks:  # any callback for pytorch-lightning trainer
    - module_name: pytorch_lightning.callbacks.early_stopping
      func_name: EarlyStopping
      params:
        monitor: 'val_loss'
        patience: 50 
        verbose: True
    - module_name: pytorch_lightning.callbacks.model_checkpoint
      func_name: ModelCheckpoint
      params:
        monitor: 'val_loss'
        filename: 'best'
```