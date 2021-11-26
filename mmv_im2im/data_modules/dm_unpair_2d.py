########################################################
# #### general data module for unpaired 2D images ######
# ####     (mostly for Cycle-GAN-like models)     ######
#
# About transformation:
# For 2D images, the combination of PIL, torchvision and
# albumentation would be sufficient for pre-processing
# and data augmentation
#
# About data in a batch:
# We woudl expect 2 parts, image_source and image_target
# Note that image_target could be masks (e.g., for
# segmentation) or images (e.g., for labelfree)
########################################################
