"""
Pneumonia Diagnosis and Localization Demo.

Pipeline:
1. Load chest X-ray.
2. Classify Normal vs Pneumonia.
3. Generate CAM heatmap.
4. Localize activated regions using the selected method.
5. Generate predicted bounding boxes.
6. Compare predictions with ground-truth boxes using IoU.
7. Visualize classification and localization results.
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
LOCALIZATION_DIR = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if str(LOCALIZATION_DIR) not in sys.path:
    sys.path.insert(0, str(LOCALIZATION_DIR))

from cam import (
    load_model,
    preprocess_image,
    generate_cam,
    load_grayscale_image,
)
from heatmap import threshold_heatmap_relative
from clustering import dfs_clusters
from bounding_boxes import clusters_to_boxes
from evaluation import (
    load_ground_truth,
    calculate_iou,
    evaluate_localization,
)
from peak_localization import peak_based_boxes


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
            / "stage_2_train_images"
            / "00436515-870c-4b36-a041-de91049b9ab4.dcm"
        ),
        help="Path to a DICOM or grayscale image",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(
            PROJECT_ROOT / "classification" / "cnn_best.pt"
        ),
        help="Path to the trained model checkpoint",
    )

    parser.add_argument(
        "--method",
        choices=["peak", "original"],
        default="peak",
        help="Localization method: peak or original CAM",
    )

    parser.add_argument(
        "--fraction",
        type=float,
        default=0.70,
        help="Relative CAM threshold fraction",
    )

    parser.add_argument(
        "--peak-fraction",
        type=float,
        default=0.90,
        help="Peak threshold fraction for peak-based localization",
    )

    parser.add_argument(
        "--min-pixels",
        type=int,
        default=3,
        help="Minimum pixels per connected region for original CAM",
    )

    parser.add_argument(
        "--iou-threshold",
        type=float,
        default=0.50,
        help="IoU threshold for localization evaluation",
    )

    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not display the visualization",
    )

    parser.add_argument(
        "--save",
        type=str,
        default=None,
        help="Optional path to save the visualization image",
    )

    args = parser.parse_args()

    if not 0 < args.fraction <= 1:
        parser.error("--fraction must be between 0 and 1.")

    if not 0 < args.peak_fraction <= 1:
        parser.error("--peak-fraction must be between 0 and 1.")

    if args.min_pixels < 1:
        parser.error("--min-pixels must be at least 1.")

    if not 0 <= args.iou_threshold <= 1:
        parser.error("--iou-threshold must be between 0 and 1.")

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

    labels_path = (
        PROJECT_ROOT
        / "data"
        / "stage_2_train_labels_actual.csv"
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
    print("Localization method:", args.method)

    # --------------------------------------------------
    # 2. Classification and CAM generation
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

    label = (
        "Pneumonia"
        if predicted_class == 1
        else "Normal"
    )

    pneumonia_probability = float(probabilities[1])
    normal_probability = float(probabilities[0])

    print("\nCLASSIFICATION")
    print("-" * 65)
    print("Prediction:", label)
    print(f"P(Normal): {normal_probability:.4f}")
    print(f"P(Pneumonia): {pneumonia_probability:.4f}")

    # --------------------------------------------------
    # 3. Generate predicted bounding boxes
    # --------------------------------------------------

    print("\nLOCALIZATION")
    print("-" * 65)
    print("CAM shape:", cam.shape)
    print(
        f"CAM range: {cam.min():.4f} to {cam.max():.4f}"
    )

    peaks = []

    if args.method == "peak":

        predicted_boxes, peaks = peak_based_boxes(
            cam,
            fraction=args.fraction,
            peak_fraction=args.peak_fraction,
            min_distance=10,
            min_pixels=10,
        )

        print("Method: Peak-based CAM")
        print(f"Heatmap fraction: {args.fraction:.2f}")
        print(f"Peak fraction: {args.peak_fraction:.2f}")
        print("Detected peaks:", len(peaks))

    else:

        mask = threshold_heatmap_relative(
            cam,
            fraction=args.fraction,
        )

        clusters = dfs_clusters(
            mask,
            min_pixels=args.min_pixels,
            connectivity=4,
        )

        predicted_boxes = clusters_to_boxes(clusters)

        print("Method: Original CAM")
        print(f"Heatmap fraction: {args.fraction:.2f}")
        print("Activated pixels:", int(mask.sum()))
        print("Connected regions:", len(clusters))

    print("\nPREDICTED BOUNDING BOXES")
    print("-" * 65)

    if predicted_boxes:

        for index, box in enumerate(
            predicted_boxes,
            start=1,
        ):
            print(f"Box {index}: {box}")

    else:
        print("No bounding boxes generated.")

    # --------------------------------------------------
    # 4. Load ground-truth annotations
    # --------------------------------------------------

    ground_truth_boxes = []

    if labels_path.exists():

        ground_truth_boxes = load_ground_truth(
            labels_path,
            image_path.stem,
            image_size=128,
        )

    print("\nGROUND-TRUTH BOUNDING BOXES")
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
    # 5. Evaluate localization
    # --------------------------------------------------

    print("\nLOCALIZATION EVALUATION")
    print("-" * 65)

    metrics = None
    best_ious = []

    if ground_truth_boxes:

        for index, predicted_box in enumerate(
            predicted_boxes,
            start=1,
        ):

            overlaps = [
                calculate_iou(predicted_box, gt_box)
                for gt_box in ground_truth_boxes
            ]

            best_iou = max(overlaps, default=0.0)
            best_ious.append(best_iou)

            print(
                f"Predicted box {index}: "
                f"IoUs = {[round(v, 4) for v in overlaps]}, "
                f"best IoU = {best_iou:.4f}"
            )

        if best_ious:

            print(
                "Mean best IoU per predicted box:",
                f"{np.mean(best_ious):.4f}",
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

        for name, value in metrics.items():

            if isinstance(value, (float, np.floating)):
                print(f"{name}: {value:.4f}")

            else:
                print(f"{name}: {value}")

    else:

        print(
            "Localization metrics unavailable because "
            "no ground-truth boxes were found."
        )

    # --------------------------------------------------
    # 6. Visualization
    # --------------------------------------------------

    if args.no_show and not args.save:
        return

    original_image = load_grayscale_image(image_path)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(18, 7),
    )

    # Original image.
    axes[0].imshow(
        original_image,
        cmap="gray",
    )

    axes[0].set_title("Original Chest X-ray")
    axes[0].axis("off")

    # CAM heatmap.
    axes[1].imshow(
        original_image,
        cmap="gray",
    )

    axes[1].imshow(
        cam,
        cmap="jet",
        alpha=0.45,
        vmin=0,
        vmax=1,
    )

    axes[1].set_title("CAM Heatmap")
    axes[1].axis("off")

    # Localization overlay.
    axes[2].imshow(
        original_image,
        cmap="gray",
    )

    axes[2].imshow(
        cam,
        cmap="jet",
        alpha=0.20,
        vmin=0,
        vmax=1,
    )

    # Red: predicted boxes.
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
                label="Predicted region",
            )
        )

    # Green dashed: ground-truth boxes.
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
                label="Ground truth",
            )
        )

    axes[2].set_title(
        "Pneumonia Localization\n"
        "Red: Predicted | Green dashed: Ground truth"
    )

    axes[2].axis("off")

    # Avoid duplicate legend entries.
    handles, legend_labels = axes[2].get_legend_handles_labels()

    if handles:

        unique_entries = dict(zip(legend_labels, handles))

        axes[2].legend(
            unique_entries.values(),
            unique_entries.keys(),
            loc="lower right",
            fontsize=8,
        )

    # Main title.
    fig.suptitle(
        f"Prediction: {label} | "
        f"P(Pneumonia): {pneumonia_probability:.3f} | "
        f"Method: {args.method.upper()}",
        fontsize=13,
    )

    plt.tight_layout()

    if args.save:

        save_path = Path(args.save)
        save_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fig.savefig(
            save_path,
            dpi=200,
            bbox_inches="tight",
        )

        print("\nVisualization saved to:")
        print(save_path)

    if not args.no_show:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    main()