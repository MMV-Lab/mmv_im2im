from typing import Optional
import numpy as np


def simplified_instance_IoU(
    mask: np.ndarray, pred: np.ndarray, exclusion_mask: Optional[np.ndarray] = None
):
    if exclusion_mask is not None:
        exclusion_mask = np.squeeze(exclusion_mask)
        assert (
            exclusion_mask.shape == mask.shape
        ), "exclustion mask and gt have different sizes"
        mask[exclusion_mask > 0] = 0
        pred[exclusion_mask > 0] = 0

    # clean up ground truth by removing all objects touching boundary
    # also, boudary is defined by 5 pixels within the actual border
    boundary_template = np.zeros_like(mask)
    if len(mask.shape) == 3:
        boundary_template[:, :5, :] = 1
        boundary_template[:, -5:, :] = 1
        boundary_template[:, :, :5] = 1
        boundary_template[:, :, -5:] = 1
    elif len(mask.shape) == 2:
        boundary_template[:5, :] = 1
        boundary_template[-5:, :] = 1
        boundary_template[:, :5] = 1
        boundary_template[:, -5:] = 1
    else:
        raise ValueError("bad image size in IoU")

    bd_idx = list(np.unique(mask[boundary_template > 0]))
    adjusted_mask = mask.copy()
    for idx in bd_idx:
        adjusted_mask[mask == idx] = 0

    # loop through all instances in GT and compare with pred
    valid_idx = list(np.unique(adjusted_mask[adjusted_mask > 0]))
    iou_list = []
    for gt_idx in valid_idx:
        gt_obj = adjusted_mask == gt_idx

        # find corresponding prediction mask
        eval_overlap = pred.copy()
        eval_overlap[gt_obj == 0] = 0
        pred_candidates = list(np.unique(eval_overlap[eval_overlap > 0]))
        if len(pred_candidates) == 0:
            continue
        pred_match = []
        for pred_idx in pred_candidates:
            overlap_size = np.count_nonzero(eval_overlap == pred_idx)
            predict_size = np.count_nonzero(pred == pred_idx) + 0.0000001
            if overlap_size / predict_size > 0.25:
                pred_match.append(pred_idx)
        if len(pred_match) == 0:
            continue
        pred_mask = np.zeros_like(mask)
        for pred_idx in pred_match:
            pred_mask[pred == pred_idx] = 1

        # calculate IOU
        i_size = np.count_nonzero(np.logical_and(pred_mask > 0, gt_obj > 0))
        u_size = np.count_nonzero(np.logical_or(pred_mask > 0, gt_obj > 0))
        score = i_size / (u_size + 0.000001)
        iou_list.append(score)

    if len(iou_list) == 0:
        return 0
    else:
        return sum(iou_list) / len(iou_list)
