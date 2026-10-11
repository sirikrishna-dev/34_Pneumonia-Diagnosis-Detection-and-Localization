"""
Pneumonia Diagnosis and Localization Demo.

Pipeline:
1. Load chest X-ray
2. Classify Normal vs Pneumonia
3. Generate CAM heatmap
4. Threshold the heatmap
5. Cluster activated regions
6. Generate bounding boxes
7. Compare predictions with ground-truth boxes using IoU
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

# --------------------------------------------------
# Configure project imports
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

LOCALIZATION_DIR = Path(__file__).resolve().parent

if str(LOCALIZATION_DIR) not in sys.path:
    sys.path.insert(0, str(LOCALIZATION_DIR))

from cam import load_model, preprocess_image, generate_cam, load_grayscale_image
from heatmap import threshold_heatmap, threshold_heatmap_relative
from clustering import dfs_clusters
from bounding_boxes import clusters_to_boxes
from evaluation import load_ground_truth, calculate_iou, evaluate_localization


# --------------------------------------------------
# Main demo
# --------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Pneumonia classification and localization demo"
    )

    parser.add_argument(
        "--image",
        type=str,
        default=str(
            PROJECT_ROOT
            / "data"
            / "00436515-870c-4b36-a041-de91049b9ab4.dcm"
        ),
        help="Path to a DICOM or grayscale image",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(PROJECT_ROOT / "classification" / "cnn_best.pt"),
        help="Path to the trained model checkpoint",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Fixed CAM threshold between 0 and 1",
    )

    parser.add_argument(
        "--relative",
        action="store_true",
        help="Use a threshold relative to the maximum CAM value",
    )

    parser.add_argument(
        "--fraction",
        type=float,
        default=0.7,
        help="Relative threshold fraction between 0 and 1",
    )

    parser.add_argument(
        "--min-pixels",
        type=int,
        default=3,
        help="Minimum number of pixels in a cluster",
    )

    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.5,
        help="IoU threshold for localization evaluation",
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the visualization",
    )

    args = parser.parse_args()

    image_path = Path(args.image)
    checkpoint_path = Path(args.checkpoint)

    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    # --------------------------------------------------
    # 1. Load model
    # --------------------------------------------------

    model, mean, std, device = load_model(checkpoint_path)

    print("\n" + "=" * 65)
    print("PNEUMONIA DIAGNOSIS AND LOCALIZATION")
    print("=" * 65)
    print("Device:", device)
    print("Image:", image_path)

    # --------------------------------------------------
    # 2. Classification
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

    label = "Pneumonia" if predicted_class == 1 else "Normal"

    print("\nCLASSIFICATION")
    print("-" * 65)
    print("Prediction:", label)
    print(f"P(Normal): {float(probabilities[0]):.4f}")
    print(f"P(Pneumonia): {float(probabilities[1]):.4f}")

    # --------------------------------------------------
    # 3. Threshold CAM
    # --------------------------------------------------

    if args.relative:
        mask = threshold_heatmap_relative(
            cam,
            fraction=args.fraction,
        )
        threshold_description = (
            f"Relative threshold ({args.fraction:.2f})"
        )
    else:
        mask = threshold_heatmap(
            cam,
            threshold=args.threshold,
        )
        threshold_description = (
            f"Fixed threshold ({args.threshold:.2f})"
        )

    print("\nHEATMAP")
    print("-" * 65)
    print("CAM shape:", cam.shape)
    print(f"CAM range: {cam.min():.4f} to {cam.max():.4f}")
    print("Threshold method:", threshold_description)
    print("Activated pixels:", int(mask.sum()))

    # --------------------------------------------------
    # 4. Cluster activated pixels
    # --------------------------------------------------

    clusters = dfs_clusters(
        mask,
        min_pixels=args.min_pixels,
        connectivity=4,
    )

    print("\nCLUSTERING")
    print("-" * 65)
    print("Connected regions:", len(clusters))

    # --------------------------------------------------
    # 5. Convert clusters into bounding boxes
    # --------------------------------------------------

    predicted_boxes = clusters_to_boxes(clusters)

    print("\nPREDICTED BOUNDING BOXES")
    print("-" * 65)

    if predicted_boxes:
        for index, box in enumerate(predicted_boxes, start=1):
            print(f"Box {index}: {box}")
    else:
        print("No bounding boxes generated.")

    # --------------------------------------------------
    # 6. Load ground truth and evaluate
    # --------------------------------------------------

    ground_truth_boxes = load_ground_truth(
    PROJECT_ROOT / "data" / "stage_2_train_labels_actual.csv",
    image_path.stem,
    image_size=128,
)

    print("\nGROUND-TRUTH BOUNDING BOXES")
    print("-" * 65)

    if ground_truth_boxes:
        for index, box in enumerate(ground_truth_boxes, start=1):
            print(f"Ground truth {index}: {box}")
    else:
        print("No ground-truth bounding boxes found.")

    print("\nLOCALIZATION EVALUATION")
    print("-" * 65)

    if ground_truth_boxes:
        best_ious = []

        for index, pred_box in enumerate(predicted_boxes, start=1):
            overlaps = [
                calculate_iou(pred_box, gt_box)
                for gt_box in ground_truth_boxes
            ]
            
            print(f"  All IoUs with ground truth: {[round(x, 4) for x in overlaps]}")
            best_iou = max(overlaps, default=0.0)
            best_ious.append(best_iou)

            print(
                f"Predicted box {index}: "
                f"best IoU = {best_iou:.4f}"
            )

        if best_ious:
            print(
                "Mean best IoU (per predicted box): "
                f"{np.mean(best_ious):.4f}"
            )
        else:
            print("No predicted boxes to compare.")

        metrics = evaluate_localization(
            predicted_boxes,
            ground_truth_boxes,
            iou_threshold=args.iou_threshold,
        )

        print(
            f"\nMetrics at IoU >= {args.iou_threshold:.2f}"
        )

        if isinstance(metrics, dict):
            for name, value in metrics.items():
                if isinstance(value, (float, np.floating)):
                    print(f"{name}: {value:.4f}")
                else:
                    print(f"{name}: {value}")
        else:
            print(metrics)

    else:
        print(
            "Localization metrics unavailable because "
            "no ground-truth boxes were found."
        )

    # --------------------------------------------------
    # 7. Visualize results
    # --------------------------------------------------

    if args.no_show:
        return

    original_image = load_grayscale_image(image_path)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(17, 6),
    )

    axes[0].imshow(original_image, cmap="gray")
    axes[0].set_title("Original Chest X-ray")
    axes[0].axis("off")

    axes[1].imshow(original_image, cmap="gray")
    axes[1].imshow(
        cam,
        cmap="jet",
        alpha=0.45,
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("CAM Heatmap")
    axes[1].axis("off")

    axes[2].imshow(original_image, cmap="gray")
    axes[2].imshow(
        cam,
        cmap="jet",
        alpha=0.25,
        vmin=0,
        vmax=1,
    )

    for box in predicted_boxes:
        x1, y1, x2, y2 = box

        axes[2].add_patch(
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

        axes[2].add_patch(
            Rectangle(
                (x1, y1),
                x2 - x1,
                y2 - y1,
                fill=False,
                edgecolor="lime",
                linewidth=2,
                linestyle="--",
            )
        )

    axes[2].set_title(
        "Localization\nRed: Predicted | Green dashed: Ground truth"
    )
    axes[2].axis("off")

    fig.suptitle(
        f"Prediction: {label} | "
        f"P(Pneumonia): {float(probabilities[1]):.3f}"
    )

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()