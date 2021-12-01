########################################################
# #### general data module for unpaired 2D images ######
# ####     (mostly for Cycle-GAN-like models)     ######
#
# About transformation:
# We use TorchIO, which can handle 2D data in a more
# efficient way than torchvision
#
# About data in a batch:
# We woudl expect 2 parts, image_source and image_target
# Note that image_target could be masks (e.g., for
# segmentation) or images (e.g., for labelfree)
########################################################
