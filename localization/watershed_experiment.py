"""
Experimental Pneumonia Localization using Watershed Segmentation.

Pipeline:
1. Load the trained classification model.
2. Generate a CAM heatmap.
3. Threshold the heatmap.
4. Apply watershed segmentation.
5. Generate bounding boxes.
6. Compare predicted boxes with ground-truth annotations.
7. Visualize the results.
"""

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle


# --------------------------------------------------
# Configure project imports
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
# CAM thresholding
# --------------------------------------------------

def cam_to_mask(cam, fraction=0.70):
    """
    Convert a normalized CAM into a binary mask.

    Pixels are activated when their CAM value is at least
    fraction * maximum CAM activation.
    """

    cam = np.asarray(cam, dtype=np.float32)

    if cam.ndim != 2:
        raise ValueError("CAM must be a 2D array.")

    if not np.isfinite(cam).all():
        raise ValueError("CAM contains invalid values.")

    if not 0 <= fraction <= 1:
        raise ValueError("Fraction must be between 0 and 1.")

    maximum = float(cam.max())

    if maximum <= 0:
        return np.zeros(cam.shape, dtype=np.uint8)

    return (
        cam >= fraction * maximum
    ).astype(np.uint8)


# --------------------------------------------------
# Watershed localization
# --------------------------------------------------

def watershed_boxes(cam, fraction=0.70, min_area=20):
    """
    Generate experimental bounding boxes using OpenCV watershed.

    Returns:
        List of boxes in (x1, y1, x2, y2) format.
    """

    if min_area < 1:
        raise ValueError("min_area must be at least 1.")

    binary_mask = cam_to_mask(cam, fraction)

    if not binary_mask.any():
        return []

    binary = binary_mask * 255

    # Close tiny gaps in the activation mask.
    kernel = np.ones((3, 3), dtype=np.uint8)

    closed = cv2.morphologyEx(
        binary,
        cv2.MORPH_CLOSE,
        kernel,
    )

    # Find connected components.
    component_count, component_labels = cv2.connectedComponents(
        (closed > 0).astype(np.uint8)
    )

    if component_count <= 1:
        return []

    # Watershed needs a three-channel image.
    height, width = binary.shape

    watershed_image = np.zeros(
        (height, width, 3),
        dtype=np.uint8,
    )

    watershed_image[:] = 255

    # Background is marker 1; foreground components get
    # unique marker IDs starting at 2.
    markers = np.zeros(
        (height, width),
        dtype=np.int32,
    )

    markers[closed == 0] = 1

    for component_id in range(1, component_count):
        markers[component_labels == component_id] = (
            component_id + 1
        )

    # Watershed modifies the markers in place.
    cv2.watershed(
        watershed_image,
        markers,
    )

    boxes = []

    # Ignore background (1) and watershed boundaries (-1).
    for region_id in np.unique(markers):

        if region_id <= 1:
            continue

        rows, cols = np.where(markers == region_id)

        if len(rows) < min_area:
            continue

        x1 = int(cols.min())
        y1 = int(rows.min())
        x2 = int(cols.max()) + 1
        y2 = int(rows.max()) + 1

        boxes.append((x1, y1, x2, y2))

    return boxes


# --------------------------------------------------
# Main experiment
# --------------------------------------------------

def main():

    parser = argparse.ArgumentParser(
        description="Experimental watershed localization"
    )

    parser.add_argument(
        "--image",
        type=str,
        default=str(
            PROJECT_ROOT
            / "data"
            / "00436515-870c-4b36-a041-de91049b9ab4.dcm"
        ),
        help="Path to the X-ray image",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(
            PROJECT_ROOT
            / "classification"
            / "cnn_best.pt"
        ),
        help="Path to the trained model checkpoint",
    )

    parser.add_argument(
        "--fraction",
        type=float,
        default=0.70,
        help="Relative CAM threshold between 0 and 1",
    )

    parser.add_argument(
        "--min-area",
        type=int,
        default=20,
        help="Minimum number of pixels in a watershed region",
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    checkpoint_path = Path(args.checkpoint)

    if not image_path.exists():
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    # --------------------------------------------------
    # 1. Load trained model
    # --------------------------------------------------

    print("Loading trained model...")

    model, mean, std, device = load_model(
        checkpoint_path
    )

    print("Model loaded successfully!")

    # --------------------------------------------------
    # 2. Generate CAM
    # --------------------------------------------------

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

    predicted_class = int(predicted_class)

    prediction = (
        "Pneumonia"
        if predicted_class == 1
        else "Normal"
    )

    print("\n" + "=" * 65)
    print("EXPERIMENTAL WATERSHED LOCALIZATION")
    print("=" * 65)

    print("Device:", device)
    print("Image:", image_path)
    print("Input tensor shape:", tuple(image_tensor.shape))
    print("CAM shape:", cam.shape)

    print("\nCLASSIFICATION")
    print("-" * 65)
    print("Prediction:", prediction)
    print(
        f"P(Normal): {float(probabilities[0]):.4f}"
    )
    print(
        f"P(Pneumonia): {float(probabilities[1]):.4f}"
    )

    # --------------------------------------------------
    # 3. Generate watershed boxes
    # --------------------------------------------------

    boxes = watershed_boxes(
        cam,
        fraction=args.fraction,
        min_area=args.min_area,
    )

    print("\nWATERSHED SETTINGS")
    print("-" * 65)
    print("Relative threshold:", args.fraction)
    print("Minimum region area:", args.min_area)
    print("Predicted boxes:", len(boxes))

    for index, box in enumerate(boxes, start=1):
        print(f"Box {index}: {box}")

    # --------------------------------------------------
    # 4. Load ground-truth boxes
    # --------------------------------------------------

    labels_path = (
        PROJECT_ROOT
        / "data"
        / "stage_2_train_labels_actual.csv"
    )

    if not labels_path.exists():
        raise FileNotFoundError(
            f"Ground-truth CSV not found: {labels_path}"
        )

    ground_truth_boxes = load_ground_truth(
        labels_path,
        image_path.stem,
        image_size=128,
    )

    print("\nGROUND-TRUTH BOXES")
    print("-" * 65)

    if ground_truth_boxes:
        for index, box in enumerate(
            ground_truth_boxes,
            start=1,
        ):
            print(f"Ground truth {index}: {box}")
    else:
        print("No ground-truth bounding boxes found.")

    # --------------------------------------------------
    # 5. Evaluate IoU
    # --------------------------------------------------

    print("\nLOCALIZATION EVALUATION")
    print("-" * 65)

    if ground_truth_boxes and boxes:

        best_ious = []

        for index, predicted_box in enumerate(
            boxes,
            start=1,
        ):

            overlaps = [
                calculate_iou(
                    predicted_box,
                    ground_truth_box,
                )
                for ground_truth_box in ground_truth_boxes
            ]

            best_iou = max(overlaps, default=0.0)

            best_ious.append(best_iou)

            print(f"Predicted box {index}: {predicted_box}")
            print(
                "IoUs with ground truth:",
                [round(value, 4) for value in overlaps],
            )
            print(f"Best IoU: {best_iou:.4f}")

        print(
            "Mean best IoU per predicted box:",
            f"{np.mean(best_ious):.4f}",
        )

    elif ground_truth_boxes:
        print("No predicted boxes to evaluate.")

    else:
        print("Evaluation unavailable without ground truth.")

    # --------------------------------------------------
    # 6. Visualize experimental results
    # --------------------------------------------------

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, 6),
    )

    axes[0].imshow(
        cam,
        cmap="jet",
        vmin=0,
        vmax=1,
    )

    axes[0].set_title("Original CAM Heatmap")
    axes[0].axis("off")

    axes[1].imshow(
        cam,
        cmap="jet",
        vmin=0,
        vmax=1,
    )

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

    for box in ground_truth_boxes:

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

    axes[1].set_title(
        "Watershed Boxes\n"
        "Red: Predicted | Green dashed: Ground truth"
    )

    axes[1].axis("off")

    fig.suptitle(
        f"Prediction: {prediction} | "
        f"P(Pneumonia): {float(probabilities[1]):.3f}"
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()