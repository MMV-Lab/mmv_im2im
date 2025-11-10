import numpy as np
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.measure import label, regionprops
from mmv_im2im.utils.utils import topology_preserving_thinning
import torch
from scipy.ndimage import shift, rotate
import random


def perturb_image(
    im_input,
    opts,
    gaussian_std=0.01,
    sp_prob=0.01,
    speckle_std=0.1,
    color_jitter_factor=0.1,
    max_shift=2,
    max_angle=2,
    scale_range=(0.98, 1.02),
    dropout_rate=0.02,
):
    """
    Applies a random combination of small perturbations to an image (C, X, Y) in NumPy.

    This function is intended for Data Augmentation *before* the model inference
    to generate slightly varied inputs for Monte Carlo Dropout or similar methods.

    Args:
        im_input (np.ndarray): Input image with shape (channels, height, width).
                               Assumed to be a float array (e.g., normalized 0.0 to 1.0).
        opts: List with the option tranformations or string indicatting 'all' ranodm text-> all
        gaussian_std (float): Standard deviation for Gaussian Noise.
        sp_prob (float): Probability for Salt and Pepper Noise (controls the density).
        speckle_std (float): Standard deviation for Speckle Noise (multiplicative).
        color_jitter_factor (float): Max factor for color perturbation (brightness/contrast).
        max_shift (int): Maximum displacement in pixels (for both X and Y axes).
        max_angle (float): Maximum rotation angle in degrees (e.g., ±2 degrees).
        scale_range (tuple): Range (min, max) of the scaling factor (e.g., 0.98 to 1.02).
        dropout_rate (float): Probability that an individual pixel will be dropped (set to 0).

    Returns:
        np.ndarray: The perturbed image.
    """

    # Create a copy to avoid modifying the original array
    im_out = im_input.copy()
    C, X, Y = im_out.shape

    if isinstance(opts, str):
        opts = [
            "gauss_noise",
            "impulse_noise",
            "speckle_noise",
            "color_jitter",
            "shift",
            "rotation",
            "pixel_dropout",
        ]

    # Gaussian Noise (Additive)
    def add_gaussian_noise(img):
        # Generate Gaussian noise array
        noise = np.random.normal(0, gaussian_std, img.shape)
        return img + noise

    # Salt and Pepper Noise (Impulse Noise)
    def add_salt_and_pepper_noise(img):
        out = img.copy()
        # Calculate number of salt (max value) and pepper (min value) points
        num_sp_points = int(sp_prob * X * Y * C / 2)  # Divide by 2 for Salt and Pepper

        # Salt Noise: max values (assuming normalized data 0.0 to 1.0)
        coords_salt = [np.random.randint(0, s, num_sp_points) for s in img.shape]
        out[tuple(coords_salt)] = 1.0

        # Pepper Noise: min values
        coords_pepper = [np.random.randint(0, s, num_sp_points) for s in img.shape]
        out[tuple(coords_pepper)] = 0.0

        return out

    # Speckle Noise (Multiplicative Noise)
    def add_speckle_noise(img):
        # Generate multiplicative noise component
        noise = np.random.normal(0, speckle_std, img.shape)
        return img * (1 + noise)

    # Color Jitter (Small random change in brightness/contrast)
    def apply_color_jitter(img):
        # Choose a random small scale factor (contrast)
        scale = 1.0 + np.random.uniform(-color_jitter_factor, color_jitter_factor)
        # Choose a random small offset (brightness)
        offset = np.random.uniform(-color_jitter_factor / 5, color_jitter_factor / 5)

        # Apply transformation: img * scale + offset
        return (img * scale) + offset

    # Shift (Translation)
    def apply_shift(img):
        # Random shift by 1 or 2 pixels in X and Y (axes 1 and 2)
        shift_x = np.random.randint(-max_shift, max_shift + 1)
        shift_y = np.random.randint(-max_shift, max_shift + 1)

        # Shift is applied to axes X and Y (1 and 2). Channel (axis 0) shift is 0.
        return shift(img, (0, shift_x, shift_y), mode="nearest")

    # Rotation
    def apply_rotation(img):
        # Random very small angle (e.g., ±1° or ±2°)
        angle = np.random.uniform(-max_angle, max_angle)

        # Rotate in the X-Y plane (axes 1 and 2).
        # reshape=False ensures the output shape is the same as input.
        return rotate(img, angle, axes=(1, 2), reshape=False, mode="nearest")

    # Pixel Dropout (Setting random pixels to 0)
    def apply_pixel_dropout(img):
        # Create a binary mask where True (1) pixels are kept and False (0) are dropped
        mask = np.random.binomial(1, 1 - dropout_rate, size=img.shape)
        return img * mask

    # List of all transformation functions
    # Each function takes im_out and returns the transformed array
    transformations = []
    if "gauss_noise" in opts:
        transformations.append(add_gaussian_noise)
    if "impulse_noise" in opts:
        transformations.append(add_salt_and_pepper_noise)
    if "speckle_noise" in opts:
        transformations.append(add_speckle_noise)
    if "color_jitter" in opts:
        transformations.append(apply_color_jitter)
    if "shift" in opts:
        transformations.append(apply_shift)
    if "rotation" in opts:
        transformations.append(apply_rotation)
    if "pixel_dropout" in opts:
        transformations.append(apply_pixel_dropout)

    if len(transformations) == 0:
        raise ValueError("Invalid transformations")

    # --- Random Application of Transformations ---

    # Choose how many transformations to apply (1 to N, where N is the total number of defined transforms)
    num_transforms_to_apply = random.randint(1, len(transformations))

    # Randomly select the transformations (without replacement)
    selected_transforms = random.sample(transformations, num_transforms_to_apply)

    # Apply the selected transformations in random order
    for transform_func in selected_transforms:
        im_out = transform_func(im_out)

    # Crucial: Ensure data remains within a valid range (e.g., 0.0 to 1.0)
    # The clip operation prevents extreme values generated by noise/jitter from breaking the model.
    # Note: If your input images are not normalized to [0, 1], adjust this clipping range accordingly.
    im_out = np.clip(im_out, 0.0, 1.0)

    return im_out


def Perycites_correction(seg_full):
    seg_2 = remove_small_objects(seg_full == 2, min_size=30)
    seg_2_mid = np.logical_xor(seg_2, remove_small_objects(seg_2, min_size=300))

    for zz in range(seg_2_mid.shape[0]):
        seg_label, num_obj = label(seg_2_mid[zz, :, :], return_num=True)
        if num_obj > 0:
            stats = regionprops(seg_label)
            for ii in range(num_obj):
                if (
                    stats[ii].eccentricity < 0.88
                    and stats[ii].solidity > 0.85
                    and stats[ii].area < 150
                ):
                    seg_z = seg_2[zz, :, :]
                    seg_z[seg_label == (ii + 1)] = 0
                    seg_2[zz, :, :] = seg_z

    seg_full[seg_full == 2] = 1
    seg_full[seg_2 > 0] = 2
    return seg_full


def Remove_objects(seg_full, n_classes, remove_object_size, voxel_sizes=(1, 1, 1)):
    """
    Applies removal of small objects to all object classes (1 to n_classes-1)
    in a 3D segmentation volume, allowing for class-specific size thresholds.

    Args:
        seg_full (np.ndarray): The 3D segmentation volume with integer class values.
        n_classes (int): The total number of classes in the segmentation (including background).
        remove_object_size (list or int): A single minimum size (int) or a list of
                                          minimum sizes (list) for objects to be kept.
                                          If a list, its length must be 1 or equal
                                          to the number of object classes (n_classes - 1).

    Returns:
        np.ndarray: The segmentation volume with small objects removed for each class.

    Raises:
        ValueError: If the length of remove_object_size list is invalid.
    """
    pz, py, px = voxel_sizes
    voxel_volume = pz * py * px

    classes_to_process = range(1, n_classes)
    num_target_classes = len(classes_to_process)

    thresholds = []

    if not isinstance(remove_object_size, list):
        physical_thresholds = [remove_object_size] * num_target_classes
    else:
        list_len = len(remove_object_size)

        if list_len == 1:
            physical_thresholds = [remove_object_size[0]] * num_target_classes

        elif list_len == num_target_classes:
            physical_thresholds = remove_object_size

        else:
            raise ValueError(
                f"The list 'remove_object_size' has {list_len} elements, "
                f"but {num_target_classes} (or 1) were expected for the {num_target_classes} classes to process "
                f"(Class 1 to {n_classes - 1}). The background (Class 0) is ignored."
            )

    for physical_size in physical_thresholds:
        min_voxel_count = int(np.ceil(physical_size / voxel_volume))
        thresholds.append(min_voxel_count)

    seg_cleaned = np.zeros_like(seg_full)

    for i, class_id in enumerate(classes_to_process):
        min_size_threshold = thresholds[i]

        seg_class_mask = seg_full == class_id

        if seg_class_mask.any():
            seg_class_clean = remove_small_objects(
                seg_class_mask, min_size=min_size_threshold
            )
            seg_cleaned[seg_class_clean] = class_id

    return seg_cleaned


def Hole_Correction(seg_full, n_classes, hole_size_threshold, voxel_sizes=(1, 1, 1)):
    """
    Applies hole correction to multiple classes in a segmentation volume.

    The correction is applied to object classes (typically 1 up to n_classes-1).
    Each class can have a different hole size threshold. It also includes
    an initial removal of small objects for all classes.

    Args:
        seg_full (np.ndarray): The 3D segmentation volume with integer class values.
        n_classes (int): The total number of classes in the segmentation (including background).
        hole_size_threshold (list or int): A single threshold (int) or a list of
                                           thresholds (list) for hole correction.
                                           If a list, its length must be 1 or
                                           equal to the number of object classes (n_classes - 1).

    Returns:
        np.ndarray: The corrected segmentation volume.

    Raises:
        ValueError: If the length of hole_size_threshold is not 1 and is less than n_classes - 1.
    """
    pz, py, px = voxel_sizes
    pixel_area = py * px

    classes_to_correct = range(1, n_classes)
    num_target_classes = len(classes_to_correct)

    thresholds = []

    if not isinstance(hole_size_threshold, list):
        physical_thresholds = [hole_size_threshold] * num_target_classes
    else:
        list_len = len(hole_size_threshold)

        if list_len == 1:
            physical_thresholds = [hole_size_threshold[0]] * num_target_classes

        elif list_len == num_target_classes:
            physical_thresholds = hole_size_threshold

        else:
            raise ValueError(
                f"The list 'hole_size_threshold' has {list_len} elements, "
                f"but {num_target_classes} (or 1) were expected for the {num_target_classes} classes to correct "
                f"(Class 1 to {n_classes - 1}). The background (Class 0) does not need a threshold."
            )

    for physical_area in physical_thresholds:
        area_threshold = int(np.ceil(physical_area / pixel_area))
        thresholds.append(area_threshold)

    seg_corrected = seg_full.copy()

    for i, class_id in enumerate(classes_to_correct):
        threshold = thresholds[i]
        seg_obj_mask = seg_corrected == class_id

        if seg_obj_mask.any():
            seg_obj_slice_corrected = seg_obj_mask.copy()

            for zz in range(seg_full.shape[0]):
                s_v = remove_small_holes(
                    seg_obj_slice_corrected[zz, :, :], area_threshold=threshold
                )
                seg_obj_slice_corrected[zz, :, :] = s_v[:, :]
            seg_corrected[seg_corrected == class_id] = 0
            seg_corrected[seg_obj_slice_corrected] = class_id

    return seg_corrected


def Thickness_Corretion(
    seg_full, n_classes, min_thickness_physical, voxel_sizes=(1, 1, 1)
):
    """
    Applies topology-preserving thinning (thickness correction) to all object
    classes (1 to n_classes-1) in a 3D segmentation volume, using a specific
    minimum thickness for each class. Class 0 (background) is automatically ignored.

    Args:
        seg_full (np.ndarray): The 3D segmentation volume with integer class values.
        n_classes (int): The total number of classes in the segmentation (including Class 0).
        min_thickness_physical (list or np.ndarray): A list or array of minimum
                                                thickness values. The index 'i'
                                                corresponds to the minimum thickness for Class i+1.
                                                (e.g., index 0 is for Class 1).

    Returns:
        np.ndarray: The segmentation volume where each object class has been thinned,
                    preserving its original class label.
    """
    pz, py, px = voxel_sizes
    distance_unit = (py + px) / 2

    classes_to_process = range(1, n_classes)
    num_object_classes = len(classes_to_process)

    # 1. Validate the length of the minimum thickness list
    if len(min_thickness_physical) != num_object_classes:
        raise ValueError(
            f"The length of 'min_thickness_list' ({len(min_thickness_physical)}) does not match "
            f"the number of object classes to process ({num_object_classes}). "
            "Class 0 (background) is ignored, so {num_object_classes} values are expected "
            " (one for each class from 1 to {n_classes-1})."
        )

    min_thickness_list = []
    for physical_distance in min_thickness_physical:
        min_thickness_pixel = int(np.ceil(physical_distance / distance_unit))
        min_thickness_list.append(min_thickness_pixel)

    seg_corrected = np.zeros_like(seg_full)

    for i, class_id in enumerate(classes_to_process):

        current_min_thickness = min_thickness_list[i]
        seg_class_mask = seg_full == class_id

        seg_thinned = topology_preserving_thinning(
            seg_class_mask, min_thickness=current_min_thickness, thin=1
        )

        seg_corrected[seg_thinned > 0] = class_id

    return seg_corrected


def adjust_volume(volume: np.ndarray) -> np.ndarray:
    """
    Applies the square root (sqrt) to a NumPy volume (2D or 3D) based on
    the percentage of values within specific orders of magnitude.

    Args:
        volume (np.ndarray): The input volume with shape (c, y, x) or (y, x).

    Returns:
        np.ndarray: The transformed volume.
    """

    # 1. Condition: 80% or more of the values are of the order e^-6 or smaller
    # Range: [0, 1e-6]

    threshold_e_neg_6 = 1e-6
    # Create a mask for values less than or equal to 1e-6 (including 0)
    mask_e_neg_6_or_less = np.less_equal(volume, threshold_e_neg_6)

    # Count how many elements meet the condition
    count_e_neg_6_or_less = np.count_nonzero(mask_e_neg_6_or_less)

    # Calculate the percentage
    total_elements = volume.size
    percentage_e_neg_6_or_less = (count_e_neg_6_or_less / total_elements) * 100

    # 2. Condition: 80% or more of the values are of the order 0.000n or 0.00n
    # Range: (1e-6, 0.01] (Strictly greater than 1e-6 and less than or equal to 0.01)

    lower_threshold_00n = 1e-6
    upper_threshold_00n = 1e-2  # 0.01

    mask_00n = np.logical_and(
        np.greater(volume, lower_threshold_00n),
        np.less_equal(volume, upper_threshold_00n),
    )

    # Count how many elements meet the condition
    count_00n = np.count_nonzero(mask_00n)

    # Calculate the percentage
    percentage_00n = (count_00n / total_elements) * 100

    # Apply double sqrt (e^{-6} or less)
    if percentage_e_neg_6_or_less >= 80:
        volume = np.sqrt(np.sqrt(volume))

    # Apply single sqrt (0.000n or 0.00n)
    elif percentage_00n >= 80:
        # sqrt(x)
        volume = np.sqrt(volume)

    else:
        return volume

    return volume


def Extract_Uncertainty_Maps(
    logits_samples, compute_mode, relative_MI=True, var_reductor=True, estabilizer=False
):
    """
    Generates an uncertainty map based on the compute_mode.

    Args:
        logits_samples (list[np.ndarray]): List of N logit samples, where each
                                           sample has shape (C, Y, X).
        compute_mode (str): Uncertainty calculation mode:
                            'variance' (Returns variance) or
                            'mutual_inf' (Returns Mutual Information).
                            'entropy' (Returns Total uncertainity)
                            'prob_in' (Returns 1-Prob)
        relative_MI (bool): If True and compute_mode='mutual_inf', MI is normalized
                            by ln(C) to the range [0, 1]. Ignored for 'variance'.
        var_reductor (bool): ONLY APPLIES TO 'variance' mode.
                             If True, returns the minimum variance across classes
                             (shape: (Y, X)).
                             If False, returns the variance for all classes
                             (shape: (C, Y, X)).

    Returns:
        np.ndarray: The resulting uncertainty map. Shape is (Y, X) for 'mutual_inf'
                    and reduced 'variance', or (C, Y, X) for unreduced 'variance'.

    Raises:
        ValueError: If the list of logit samples is empty or the computation mode is invalid.
    """

    if not logits_samples:
        raise ValueError("The list of logit samples cannot be empty.")

    # Convert Logits to Probabilities ---

    # Stack samples along a new axis (axis 0). Shape: (N_samples, C, Y, X)
    stacked_logits = np.stack(logits_samples, axis=0)
    N, C, Y, X = stacked_logits.shape

    # Apply Softmax along the class axis (axis 1) to get probabilities P.
    logits_tensor = torch.from_numpy(stacked_logits).float()

    # stacked_probs.shape: (N_samples, C, Y, X)
    stacked_probs = torch.nn.functional.softmax(logits_tensor, dim=1).numpy()

    # CALCULATE AND MERGE UNCERTAINTY ---

    if compute_mode == "variance":
        # Calculate Variance of probabilities (class-wise uncertainty)
        # uncertainty_map_split shape: (C, Y, X)
        uncertainty_map_split = np.var(stacked_probs, axis=0)

        if var_reductor:
            # Merge: Take the MINIMUM uncertainty across classes (axis 0)
            # merged_uncertainty shape: (Y, X)
            merged_uncertainty = np.min(uncertainty_map_split, axis=0)

            if estabilizer:
                merged_uncertainty = adjust_volume(merged_uncertainty)
            return merged_uncertainty
        else:
            # Return per-class uncertainty map
            # shape: (C, Y, X)
            if estabilizer:
                uncertainty_map_split = adjust_volume(uncertainty_map_split)
            return uncertainty_map_split
    elif compute_mode == "prob_inv":
        # Probability map
        # stacked_max_probs.shape: (N_samples, Y, X)
        prob_comp = 1 - np.max(stacked_probs, axis=1)
        # averaged_max_probs.shape: (Y, X)
        prob_comp = np.mean(prob_comp, axis=0)

        if estabilizer:
            prob_comp = adjust_volume(prob_comp)

        return prob_comp
    elif compute_mode == "mutual_inf" or compute_mode == "entropy":
        # Calculate Mutual Information (MI)

        # helps to avoid log(0)
        epsilon = 1e-12

        # a) Average Predictive Probability (E[P(y|x)])
        avg_probs = np.mean(stacked_probs, axis=0)
        # Cliping para la entropía total (H[E[P(y|x)]])
        avg_probs_clipped = np.clip(avg_probs, a_min=epsilon, a_max=None)

        # b) Total Predictive Entropy (H[E[P(y|x)]])
        entropy_total = -np.sum(avg_probs_clipped * np.log(avg_probs_clipped), axis=0)

        # c) Average Conditional Entropy (E[H[P(y|x, w)]])

        # Cliping para la entropía condicional (P(y|x,w) * log(P(y|x,w)))
        stacked_probs_clipped = np.clip(stacked_probs, a_min=epsilon, a_max=None)

        # Cálculo de la entropía por muestra: -sum(P log P)
        per_sample_entropy = -np.sum(
            stacked_probs_clipped * np.log(stacked_probs_clipped), axis=1
        )

        # Promedio de la entropía por muestra (E[H[P(y|x, w)]])
        entropy_avg_conditional = np.mean(per_sample_entropy, axis=0)

        if compute_mode == "entropy":
            # Entropy
            mutual_information_map = entropy_total
        else:
            # d) Mutual Information (MI)
            mutual_information_map = entropy_total - entropy_avg_conditional

        #  Apply Relative Normalization if requested
        if relative_MI:
            # Normalization factor: ln(C), the max possible entropy (MI max theoretical bound)
            max_mi = np.log(C)

            if max_mi == 0:  # Handle C=1 case (though unlikely for segmentation)
                return np.zeros_like(mutual_information_map)

            # Clip result to ensure strict [0, 1] range due to floating point arithmetic
            normalized_mi = np.clip(mutual_information_map / max_mi, 0.0, 1.0)

            if estabilizer:
                normalized_mi = adjust_volume(normalized_mi)

            return normalized_mi
        else:
            # Return original MI (in nats), range [0, ln(C)]
            if estabilizer:
                mutual_information_map = adjust_volume(mutual_information_map)
            return mutual_information_map

    else:
        raise ValueError("Invalid computation mode.")
