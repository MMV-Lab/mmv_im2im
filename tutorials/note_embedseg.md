# Special note on Embedseg training

To train an EmbedSeg model for 2D/3D instance segmentation, we implemented two modes: pretrain and finetune.

### How the "pretrain" and "finetune" modes work?

* pretrain mode: Before training starts, for each image, one patch will be cropped around the center of each instance and a class image (semantic classes of the segmentation) and center image (center of each intance) corresponding to the patch will be generated and saved to at `cache_path`. Then, during each epoch, the dataloader will deal with the patch directly without referring to the full images. This is similar to the implementation in the original EmbedSeg paper.

* finetune mode: No pre-cropping is done. During each epoch, the full images are loaded, while the patches, center images, class images are generated on-the-fly. The sampling is not necessarily around the instance centers as in pre-train, and can be fully random or weighted (checkout different [sampling crops in MONAI](https://docs.monai.io/en/stable/transforms.html#randweightedcropd)). 

### When to use what?

In geneneral, the "pretrain" model is faster (no on-the-fly cropping and centroid calculation in each iteration), while the "finetune" model can usually make more effective training. This is because the "finetune" model will do training on more diverse samples (not only around instance centers) and can do more data augmentations (e.g., spatial deformation) that pre-cropping cannot. So, when training images are not very large and the number of training images is small, a direct "finetune" mode is enough; when the training set is large, it is better to run the "pretrain" model first, and then followed by a "finetune" mode.

### What is the difference in configurations?

One key difference is when `data.cache_path` is set, the "pretrain" model will be triggered; otherwise, the training goes to "finetune" mode directly. 

When using "pretrain" mode, you can choose how to calculate the instance centers by setting `data.extra.center_method` as `centroid` or `approximate-medoid` or `medoid`. See [original Embedseg code](https://github.com/juglab/EmbedSeg/blob/main/EmbedSeg/utils/generate_crops.py#L145) for details. Also, in "pretrain" mode, all transformation should be done with dictionary keys: "IM", "GT", "CL", "CE", which are generated in the pre-cropping step. "CL" referrs class images, while "CE" represents center images. When costmap is used, you can also include "CM" in the key, while making sure to set `model.criterion.params.use_costmap` as `True` (default is `False`).

When using "finetune" mode, you need to make sure all transformation is done with dictionary keys, only "IM" and "GT". When costmap is used, you can also include "CM" in the key, while making sure to set `model.criterion.params.use_costmap` as `True` (default is `False`). In order to define how to calculate instance centers on-the-fly in each iteration, you can set `model.model_extra` as `data.extra.center_method` as `centroid` or `approximate-medoid` or `medoid`. If you want to load a pre-trained model from earlier "pretrain" model, just pass in the checkpoint directory at `model.model_extra.pre-train`.

See examples: [pretrain mode](../paper_configs/instance_seg_3d_train_bf_pretrain.yaml) and [finetune mode](../paper_configs/instance_seg_3d_train_bf_finetune.yaml).
