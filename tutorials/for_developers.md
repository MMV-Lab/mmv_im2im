## Notes on the package design


1. The four main packages we build upon: [pytorch-lightning](https://www.pytorchlightning.ai/), [MONAI](https://monai.io/), [pyrallis](https://eladrich.github.io/pyrallis/), and [aicsimageio](https://github.com/AllenCellModeling/aicsimageio).

The whole package uses [pytorch-lightning](https://www.pytorchlightning.ai/) as the core of its backend, in the sense that the package is implemented following the boilerplate components in pytorch-lightning, such as `LightningModule`, `DataModule` and `Trainer`. All small building blocks, like network architecture, optimizer, etc., can be swapped easily without changing the boilerplate. 

We adopt the [PersistentDataset](https://docs.monai.io/en/stable/data.html#persistentdataset) in [MONAI](https://monai.io) as the default dataloader, which combines the efficiency and flexibility in data handling for biomedical applications. E.g., able to efficiently handle: when any single file in training data is large, sampling multiple patches from each of the large image in training data, and when there are a huge number of files and have to load a small portion to memory in each epoch and periodically refresh the data in memory, etc.

[Pyrallis](https://eladrich.github.io/pyrallis/) provides a handy configuration system. Combining pyrallis and the boilerplate concepts in pytorch-lightning, it is very easy to configure your method at any level of details (as high level as only providing the path to the training data, all the way to as low level as changing which type of normalization to use in the model). 

Finally, [aicsimageio](https://github.com/AllenCellModeling/aicsimageio) is adopted for efficient data I/O, which not only supports all major bio-formats and OME-TIFF, but also makes it painless to handle hugh data by delayed loading.


2. There are three levels of abstraction: 
- `mmv_im2im/proj_trainer.py` (main entry point to define data module, pytorch-lightning module, and trainer)
    -  `mmv_im2im/data_modules/data_loaders.py` (Currently, all 2D/3D paired/unpaired data loader can share the same universal dataloader. We may add other generic dataloaders when really needed, otherwise, the current one is general enough to cover current applications)
    -  `mmv_im2im/models/pl_XYZ.py` (the middle level wrapper for different categories of models: FCN, pix2pix (conditional GAN), CycleGAN, embedseg. We can add more when needed, e.g., for denoising or cellpose. This is the pytorch-lightning module specific for this category of models. The specific model backbone, loss function, etc. can be easily specified via parameters in yaml config)
        -  Other scripts under `mmv_im2im/models/nets/` and `mmv_im2im/preprocessing`, as well as `mmv_im2im/utils`, are low level functions to instantiate the pytorch-lightning module defined in `pl_XYZ.py` or `data_loader.py`
