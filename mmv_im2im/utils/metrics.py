from typing import Optional
import numpy as np


def simplified_instance_IoU_3D(
    mask: np.ndarray, pred: np.ndarray, costmap: Optional[np.ndarray]
):
    # clean up ground truth by removing all objects touching boundary
    # also, boudary is defined by 5 pixels within the actual border
    boundary_template = np.zeros_like(mask)
    boundary_template[:, :5, :] = 1
    boundary_template[:, -5:, :] = 1
    boundary_template[:, :, :5] = 1
    boundary_template[:, :, -5:] = 1

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
