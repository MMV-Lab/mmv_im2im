########################################################
# #### general data module for paired 3D images ########
# ####   (mostly for FCN or CGAN-like models)     ######
#
# About transformation:
# We use TorchIO, which can handle 3D data in a more
# efficient way than torchvision
#
# About data in a batch:
# We woudl expect 3 parts, image_source, image_target,
# and image_cmap (cmap: cost map), where image_cmap
# can be optional. Note that image_target could be masks
# (e.g. for segmentation) or images (e.g. for labelfree)
########################################################
