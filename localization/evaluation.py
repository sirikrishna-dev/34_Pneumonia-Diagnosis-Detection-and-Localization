"""
Evaluate predicted pneumonia localization boxes using IoU.
"""

import numpy as np
import pandas as pd


def calculate_iou(box_a, box_b):
    """
    Calculate Intersection over Union.

    Box format:
        (x_min, y_min, x_max, y_max)
    """

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    intersection = (
        max(0, ix2 - ix1) *
        max(0, iy2 - iy1)
    )

    area_a = (
        max(0, ax2 - ax1) *
        max(0, ay2 - ay1)
    )

    area_b = (
        max(0, bx2 - bx1) *
        max(0, by2 - by1)
    )

    union = area_a + area_b - intersection

    if union <= 0:
        return 0.0

    return float(intersection / union)


def load_ground_truth(csv_path, patient_id, image_size=128):
    """
    Load and scale ground-truth boxes from the RSNA CSV.

    Original coordinates refer to 1024x1024 images.
    """

    df = pd.read_csv(csv_path)

    patient_rows = df[
        df["patientId"].astype(str) == str(patient_id)
    ]

    if patient_rows.empty:
        raise ValueError(
            f"Patient ID not found: {patient_id}"
        )

    boxes = []

    scale = image_size / 1024.0

    for _, row in patient_rows.iterrows():

        # Normal cases and rows without annotations have no box.
        if (
            pd.isna(row["x"])
            or pd.isna(row["y"])
            or pd.isna(row["width"])
            or pd.isna(row["height"])
        ):
            continue

        x1 = float(row["x"]) * scale
        y1 = float(row["y"]) * scale

        x2 = (
            float(row["x"]) + float(row["width"])
        ) * scale

        y2 = (
            float(row["y"]) + float(row["height"])
        ) * scale

        boxes.append((x1, y1, x2, y2))

    return boxes


def best_iou(predicted_boxes, ground_truth_boxes):
    """
    Return the best IoU for each predicted box.

    Note:
        This metric alone does not penalize missed ground-truth boxes.
    """

    if not predicted_boxes or not ground_truth_boxes:
        return []

    return [
        max(
            calculate_iou(predicted, ground_truth)
            for ground_truth in ground_truth_boxes
        )
        for predicted in predicted_boxes
    ]


def evaluate_localization(
    predicted_boxes,
    ground_truth_boxes,
    iou_threshold=0.5,
):
    """
    Evaluate box localization using one-to-one matching.

    A predicted box can match at most one ground-truth box,
    and each ground-truth box can match at most one prediction.

    Returns:
        TP: matched boxes
        FP: unmatched predictions
        FN: missed ground-truth boxes
        precision, recall, F1
        mean_matched_iou
    """

    if not 0 <= iou_threshold <= 1:
        raise ValueError(
            "IoU threshold must be between 0 and 1."
        )

    # Calculate all predicted-to-ground-truth IoUs.
    pairs = []

    for pred_index, predicted in enumerate(predicted_boxes):
        for gt_index, ground_truth in enumerate(ground_truth_boxes):
            score = calculate_iou(predicted, ground_truth)

            if score >= iou_threshold:
                pairs.append(
                    (score, pred_index, gt_index)
                )

    # Match highest-IoU pairs first without reusing boxes.
    pairs.sort(reverse=True)

    matched_predictions = set()
    matched_ground_truth = set()
    matched_ious = []

    for score, pred_index, gt_index in pairs:
        if (
            pred_index in matched_predictions
            or gt_index in matched_ground_truth
        ):
            continue

        matched_predictions.add(pred_index)
        matched_ground_truth.add(gt_index)
        matched_ious.append(score)

    tp = len(matched_predictions)
    fp = len(predicted_boxes) - tp
    fn = len(ground_truth_boxes) - tp

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )

    mean_matched_iou = (
        float(np.mean(matched_ious))
        if matched_ious
        else 0.0
    )

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "mean_matched_iou": mean_matched_iou,
    }