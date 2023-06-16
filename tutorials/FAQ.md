# FAQ

**We will keep adding more sections on this page, as we receive more questions from users**

## 1. how to load a pre-train model for transfer learning?

Usually, you can do this by setting `model.model_extra.pre-train` with the checkpoint path. A special case is for loading a pre-trained FCN as the generator of a GAN model, while the discriminator will still be initialized regularly with Kaiming method. In this case, you need to set `model.net.generator.init_weight` with the path to the pretrained FCN model. See an example [HERE](../paper_configs/labelfree_3d_pix2pix_finetune.yaml).

## 2. how to use multi-GPU training or half precision training to better utilize GPU(s)?

This is very simple, just setting the parameters in trainer.

```
trainer:
  verbose: True
  params:
    gpus: 5
    precision: 16
    max_epochs: 10000
    detect_anomaly: True
```

## 3. How to monitor training progress?

Tensorboard is used for tracking the training progress by default. Everything is saved under `lightning_logs/version_X`. Then after installing tensorboard, you can simply run `tensorboard --logdir=./lightning_logs/version_X`. For advacned users, if you wish to change what is being logged with tensorboard, you can just modify the corresponding lightning module file, like [here](https://github.com/MMV-Lab/mmv_im2im/blob/main/mmv_im2im/models/pl_pix2pix.py#L160).

# How to select different GPUs?

By default, when you run `run_im2im --config myconfig.yaml`, all GPUs available on your machine are usable by the program. Then, if you select to use 1 GPU, then the first GPU will be used. If you want to run on a specific GPU(s), you can do `CUDA_VISIBLE_DEVICES=3 run_im2im --config myconfig.yaml` or `CUDA_VISIBLE_DEVICES=1,3,5 run_im2im --config myconfig.yaml` 


# How to automatically select which GPU to use?

set `device = "auto"` in `trainer`. For example:
```
trainer:
  verbose: True
  params:
    accelerator: "gpu"
    devices: "auto"
    precision: 16
    max_epochs: 2000
    detect_anomaly: True
```