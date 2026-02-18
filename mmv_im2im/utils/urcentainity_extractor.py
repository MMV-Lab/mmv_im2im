import numpy as np
from skimage.morphology import remove_small_objects, remove_small_holes
from skimage.measure import label, regionprops
from mmv_im2im.utils.utils import topology_preserving_thinning
import torch
from scipy.ndimage import shift, rotate
import random
from typing import List, Union, Tuple


def perturb_image(
    im_input: np.ndarray,
    opts: Union[List[str], str],
    gaussian_std: float = 0.01,
    sp_prob: float = 0.01,
    speckle_std: float = 0.1,
    color_jitter_factor: float = 0.1,
    max_shift: int = 2,
    max_angle: float = 2,
    scale_range: Tuple[float, float] = (0.98, 1.02),
    dropout_rate: float = 0.02,
) -> np.ndarray:
    """
    Applies a random combination of small perturbations to an image.
    Supports both 2D inputs (C, Y, X) and 3D inputs (C, Z, Y, X).
    """

    # Create a copy to avoid modifying the original array
    im_out = im_input.copy()
    ndim = im_out.ndim

    # Determine axes based on dimensions
    # If 3D (C, Y, X): Shift/Rotate axes (1, 2)
    # If 4D (C, Z, Y, X): Shift/Rotate axes (2, 3) (The spatial plane)
    if ndim == 3:
        C, H, W = im_out.shape
        rot_axes = (1, 2)
    elif ndim == 4:
        C, D, H, W = im_out.shape
        rot_axes = (2, 3)
    else:
        raise ValueError(
            f"Unsupported input shape dimension: {ndim}. Expected 3 (C,Y,X) or 4 (C,Z,Y,X)."
        )

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
        noise = np.random.normal(0, gaussian_std, img.shape)
        return img + noise

    # Salt and Pepper Noise (Impulse Noise)
    def add_salt_and_pepper_noise(img):
        out = img.copy()
        # Total pixels * channels
        total_pixels = img.size
        num_sp_points = int(sp_prob * total_pixels / 2)

        # Salt
        coords_salt = [np.random.randint(0, s, num_sp_points) for s in img.shape]
        out[tuple(coords_salt)] = 1.0

        # Pepper
        coords_pepper = [np.random.randint(0, s, num_sp_points) for s in img.shape]
        out[tuple(coords_pepper)] = 0.0
        return out

    # Speckle Noise (Multiplicative)
    def add_speckle_noise(img):
        noise = np.random.normal(0, speckle_std, img.shape)
        return img * (1 + noise)

    # Color Jitter
    def apply_color_jitter(img):
        scale = 1.0 + np.random.uniform(-color_jitter_factor, color_jitter_factor)
        offset = np.random.uniform(-color_jitter_factor / 5, color_jitter_factor / 5)
        return (img * scale) + offset

    # Shift (Translation)
    def apply_shift(img):
        shift_x = np.random.randint(-max_shift, max_shift + 1)
        shift_y = np.random.randint(-max_shift, max_shift + 1)

        if ndim == 3:
            # Shift Y and X
            shift_tuple = (0, shift_x, shift_y)
        elif ndim == 4:
            # Shift Y and X, leave Z and C alone
            shift_tuple = (0, 0, shift_x, shift_y)

        return shift(img, shift_tuple, mode="nearest")

    # Rotation
    def apply_rotation(img):
        angle = np.random.uniform(-max_angle, max_angle)
        # Rotate in the spatial plane defined by rot_axes
        return rotate(img, angle, axes=rot_axes, reshape=False, mode="nearest")

    # Pixel Dropout
    def apply_pixel_dropout(img):
        mask = np.random.binomial(1, 1 - dropout_rate, size=img.shape)
        return img * mask

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

    num_transforms_to_apply = random.randint(1, len(transformations))
    selected_transforms = random.sample(transformations, num_transforms_to_apply)

    for transform_func in selected_transforms:
        im_out = transform_func(im_out)

    im_out = np.clip(im_out, 0.0, 1.0)
    return im_out


def Perycites_correction(seg_full):
    seg_2 = remove_small_objects(seg_full == 2, max_size=30)
    seg_2_mid = np.logical_xor(seg_2, remove_small_objects(seg_2, max_size=300))

    # Iterate over Z dimension (slice-by-slice correction logic)
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
            raise ValueError("Invalid remove_object_size list length.")

    for physical_size in physical_thresholds:
        min_voxel_count = int(np.ceil(physical_size / voxel_volume))
        thresholds.append(min_voxel_count)

    seg_cleaned = np.zeros_like(seg_full)
    for i, class_id in enumerate(classes_to_process):
        min_size_threshold = thresholds[i]
        seg_class_mask = seg_full == class_id
        if seg_class_mask.any():
            seg_class_clean = remove_small_objects(
                seg_class_mask, max_size=min_size_threshold
            )
            seg_cleaned[seg_class_clean] = class_id

    return seg_cleaned


def Hole_Correction(seg_full, n_classes, hole_size_threshold, voxel_sizes=(1, 1, 1)):
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
            raise ValueError("Invalid hole_size_threshold list length.")

    for physical_area in physical_thresholds:
        area_threshold = int(np.ceil(physical_area / pixel_area))
        thresholds.append(area_threshold)

    seg_corrected = seg_full.copy()
    for i, class_id in enumerate(classes_to_correct):
        threshold = thresholds[i]
        seg_obj_mask = seg_corrected == class_id

        if seg_obj_mask.any():
            seg_obj_slice_corrected = seg_obj_mask.copy()
            # Apply per-slice
            for zz in range(seg_full.shape[0]):
                s_v = remove_small_holes(
                    seg_obj_slice_corrected[zz, :, :], max_size=threshold
                )
                seg_obj_slice_corrected[zz, :, :] = s_v[:, :]
            seg_corrected[seg_corrected == class_id] = 0
            seg_corrected[seg_obj_slice_corrected] = class_id

    return seg_corrected


def Thickness_Corretion(
    seg_full, n_classes, min_thickness_physical, voxel_sizes=(1, 1, 1)
):
    pz, py, px = voxel_sizes
    distance_unit = (py + px) / 2
    classes_to_process = range(1, n_classes)
    num_object_classes = len(classes_to_process)

    if len(min_thickness_physical) != num_object_classes:
        raise ValueError("min_thickness_list length mismatch.")

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
    threshold_e_neg_6 = 1e-6
    mask_e_neg_6_or_less = np.less_equal(volume, threshold_e_neg_6)
    count_e_neg_6_or_less = np.count_nonzero(mask_e_neg_6_or_less)
    percentage_e_neg_6_or_less = (count_e_neg_6_or_less / volume.size) * 100

    lower_threshold_00n = 1e-6
    upper_threshold_00n = 1e-2
    mask_00n = np.logical_and(
        np.greater(volume, lower_threshold_00n),
        np.less_equal(volume, upper_threshold_00n),
    )
    count_00n = np.count_nonzero(mask_00n)
    percentage_00n = (count_00n / volume.size) * 100

    if percentage_e_neg_6_or_less >= 80:
        volume = np.sqrt(np.sqrt(volume))
    elif percentage_00n >= 80:
        volume = np.sqrt(volume)

    return volume


def Extract_Uncertainty_Maps(
    logits_samples, compute_mode, relative_MI=True, var_reductor=True, estabilizer=False
):
    if not logits_samples:
        raise ValueError("The list of logit samples cannot be empty.")

    # Stack samples: (N, C, Z, Y, X) for 3D or (N, C, Y, X) for 2D
    stacked_logits = np.stack(logits_samples, axis=0)

    # Softmax on Channel axis (dim 1)
    logits_tensor = torch.from_numpy(stacked_logits).float()
    stacked_probs = torch.nn.functional.softmax(logits_tensor, dim=1).numpy()

    # For normalization later
    C = stacked_logits.shape[1]

    if compute_mode == "variance":
        # Variance across samples (axis 0)
        uncertainty_map_split = np.var(stacked_probs, axis=0)

        if var_reductor:
            # Min uncertainty across classes (axis 0 of the result)
            merged_uncertainty = np.min(uncertainty_map_split, axis=0)
            if estabilizer:
                merged_uncertainty = adjust_volume(merged_uncertainty)
            return merged_uncertainty
        else:
            if estabilizer:
                uncertainty_map_split = adjust_volume(uncertainty_map_split)
            return uncertainty_map_split

    elif compute_mode == "prob_inv":
        prob_comp = 1 - np.max(stacked_probs, axis=1)
        prob_comp = np.mean(prob_comp, axis=0)
        if estabilizer:
            prob_comp = adjust_volume(prob_comp)
        return prob_comp

    elif compute_mode == "mutual_inf" or compute_mode == "entropy":
        epsilon = 1e-12
        avg_probs = np.mean(stacked_probs, axis=0)
        avg_probs_clipped = np.clip(avg_probs, a_min=epsilon, a_max=None)
        entropy_total = -np.sum(avg_probs_clipped * np.log(avg_probs_clipped), axis=0)

        stacked_probs_clipped = np.clip(stacked_probs, a_min=epsilon, a_max=None)
        # Sum over classes (axis 1)
        per_sample_entropy = -np.sum(
            stacked_probs_clipped * np.log(stacked_probs_clipped), axis=1
        )
        entropy_avg_conditional = np.mean(per_sample_entropy, axis=0)

        if compute_mode == "entropy":
            mutual_information_map = entropy_total
        else:
            mutual_information_map = entropy_total - entropy_avg_conditional

        if relative_MI:
            max_mi = np.log(C)
            if max_mi == 0:
                return np.zeros_like(mutual_information_map)
            normalized_mi = np.clip(mutual_information_map / max_mi, 0.0, 1.0)
            if estabilizer:
                normalized_mi = adjust_volume(normalized_mi)
            return normalized_mi
        else:
            if estabilizer:
                mutual_information_map = adjust_volume(mutual_information_map)
            return mutual_information_map

    else:
        raise ValueError("Invalid computation mode.")
