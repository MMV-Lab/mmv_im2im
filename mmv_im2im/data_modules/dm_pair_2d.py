########################################################
# #### general data module for paired 2D images ########
# ####   (mostly for FCN or CGAN-like models)     ######
#
# About transformation:
# For 2D images, the combination of PIL, torchvision and
# albumentation would be sufficient for pre-processing
# and data augmentation
#
# About data in a batch:
# We woudl expect 3 parts, image_source, image_target,
# and image_cmap (cmap: cost map), where image_cmap
# can be optional. Note that image_target could be masks
# (e.g. for segmentation) or images (e.g. for labelfree)
########################################################
