"""
Experimental peak-based pneumonia localization.

Uses the existing classification model and CAM.
Detects prominent heatmap peaks and creates candidate
bounding boxes around their activation regions.

This is an experiment, not a trained object detector.
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from scipy.ndimage import maximum_filter, label


# --------------------------------------------------
# Project imports
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCALIZATION_DIR = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(LOCALIZATION_DIR) not in sys.path:
    sys.path.insert(0, str(LOCALIZATION_DIR))

from cam import load_model, preprocess_image, generate_cam
from evaluation import load_ground_truth, calculate_iou


# --------------------------------------------------
# Peak detection and region generation
# --------------------------------------------------

def peak_based_boxes(
    cam,
    fraction=0.70,
    peak_fraction=0.80,
    min_distance=10,
    min_pixels=10,
):
    """
    Detect prominent CAM peaks and generate candidate boxes.

    Each peak is associated with its connected component in
    a lower-threshold mask. If multiple peaks fall in the
    same component, that component is split by nearest peak.

    Returns:
        boxes: List of (x1, y1, x2, y2) tuples.
        peaks: List of (row, column) peak coordinates.
    """

    cam = np.asarray(cam, dtype=np.float32)

    if cam.ndim != 2:
        raise ValueError("CAM must be a 2D array.")

    if not np.isfinite(cam).all():
        raise ValueError("CAM contains invalid values.")

    if not 0 <= fraction <= 1:
        raise ValueError("fraction must be between 0 and 1.")

    if not 0 <= peak_fraction <= 1:
        raise ValueError("peak_fraction must be between 0 and 1.")

    if min_distance < 1 or min_pixels < 1:
        raise ValueError("min_distance and min_pixels must be positive.")

    maximum = float(cam.max())

    if maximum <= 0:
        return [], []

    # Detect local maxima in the heatmap.
    neighborhood = 2 * min_distance + 1

    local_max = cam == maximum_filter(
        cam,
        size=neighborhood,
        mode="nearest",
    )

    peak_mask = (
        local_max
        & (cam >= peak_fraction * maximum)
        & (cam > 0)
    )

    peak_rows, peak_cols = np.where(peak_mask)

    if len(peak_rows) == 0:
        return [], []

    # Merge nearby peak pixels into peak groups.
    peak_groups, group_count = label(
        peak_mask,
        structure=np.ones((3, 3), dtype=np.uint8),
    )

    peaks = []

    for group_id in range(1, group_count + 1):
        rows, cols = np.where(peak_groups == group_id)

        if len(rows) == 0:
            continue

        # Choose the strongest activation in this group.
        values = cam[rows, cols]
        index = int(np.argmax(values))

        peaks.append((
            int(rows[index]),
            int(cols[index]),
        ))

    # Use a lower threshold to capture surrounding activations.
    region_mask = cam >= fraction * maximum

    if not region_mask.any():
        return [], peaks

    component_labels, component_count = label(
        region_mask,
        structure=np.ones((3, 3), dtype=np.uint8),
    )

    boxes = []

    for component_id in range(1, component_count + 1):

        rows, cols = np.where(
            component_labels == component_id
        )

        if len(rows) < min_pixels:
            continue

        component_peaks = [
            peak for peak in peaks
            if component_labels[peak] == component_id
        ]

        if not component_peaks:
            continue

        # Assign each component pixel to its nearest peak.
        pixel_coords = np.column_stack((rows, cols))
        peak_coords = np.asarray(component_peaks)

        distances = (
            (pixel_coords[:, None, :] - peak_coords[None, :, :])
            ** 2
        ).sum(axis=2)

        assignments = np.argmin(distances, axis=1)

        for peak_index in range(len(component_peaks)):

            selected = assignments == peak_index

            selected_rows = rows[selected]
            selected_cols = cols[selected]

            if len(selected_rows) < min_pixels:
                continue

            x1 = int(selected_cols.min())
            y1 = int(selected_rows.min())
            x2 = int(selected_cols.max()) + 1
            y2 = int(selected_rows.max()) + 1

            boxes.append((x1, y1, x2, y2))

    return boxes, peaks


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Experimental peak-based localization"
    )

    parser.add_argument(
        "--image",
        default=str(
            PROJECT_ROOT
            / "data"
            / "00436515-870c-4b36-a041-de91049b9ab4.dcm"
        ),
    )

    parser.add_argument(
        "--fraction",
        type=float,
        default=0.70,
    )

    parser.add_argument(
        "--peak-fraction",
        type=float,
        default=0.80,
    )

    parser.add_argument(
        "--min-distance",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--min-pixels",
        type=int,
        default=10,
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    checkpoint = PROJECT_ROOT / "classification" / "cnn_best.pt"

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    model, mean, std, device = load_model(checkpoint)

    image_tensor = preprocess_image(
        image_path,
        mean,
        std,
        device,
    )

    cam, predicted_class, probabilities = generate_cam(
        model,
        image_tensor,
    )

    boxes, peaks = peak_based_boxes(
        cam,
        fraction=args.fraction,
        peak_fraction=args.peak_fraction,
        min_distance=args.min_distance,
        min_pixels=args.min_pixels,
    )

    print("\nPEAK-BASED LOCALIZATION EXPERIMENT")
    print("=" * 60)

    print("Image:", image_path.name)
    print("CAM shape:", cam.shape)
    print(
        "Prediction:",
        "Pneumonia" if int(predicted_class) == 1 else "Normal",
    )
    print("Detected peaks:", len(peaks))
    print("Peaks (row, column):", peaks)
    print("Predicted boxes:", len(boxes))

    for index, box in enumerate(boxes, start=1):
        print(f"Box {index}: {box}")

    labels_path = (
        PROJECT_ROOT / "data" / "stage_2_train_labels_actual.csv"
    )

    ground_truth = load_ground_truth(
        labels_path,
        image_path.stem,
        image_size=128,
    )

    print("\nGround-truth boxes:", ground_truth)

    if ground_truth:
        best_ious = []

        for index, box in enumerate(boxes, start=1):
            overlaps = [
                calculate_iou(box, gt)
                for gt in ground_truth
            ]

            best_iou = max(overlaps, default=0.0)
            best_ious.append(best_iou)

            print(
                f"Box {index} IoUs:",
                [round(value, 4) for value in overlaps],
            )

            print(f"Box {index} best IoU: {best_iou:.4f}")

        if best_ious:
            print(
                "Mean best IoU per predicted box:",
                f"{np.mean(best_ious):.4f}",
            )

    # Visualize the heatmap, peaks and boxes.
    fig, axes = plt.subplots(1, 2, figsize=(13, 6))

    axes[0].imshow(cam, cmap="jet", vmin=0, vmax=1)
    axes[0].set_title("CAM Heatmap")

    axes[1].imshow(cam, cmap="jet", vmin=0, vmax=1)

    for row, col in peaks:
        axes[1].plot(col, row, "w+", markersize=10, markeredgewidth=2)

    for box in boxes:
        x1, y1, x2, y2 = box

        axes[1].add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor="red",
                linewidth=2,
            )
        )

    for box in ground_truth:
        x1, y1, x2, y2 = box

        axes[1].add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor="lime",
                linestyle="--",
                linewidth=2,
            )
        )

    axes[0].axis("off")
    axes[1].axis("off")

    axes[1].set_title(
        "Peak-based Boxes\n"
        "White: Peaks | Red: Predicted | Green dashed: Ground truth"
    )

    fig.suptitle(
        f"Prediction: {'Pneumonia' if int(predicted_class) == 1 else 'Normal'}"
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()