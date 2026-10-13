"""
Compare original CAM localization with peak-based localization.

Uses a separate validation sample, excluding patients from the
previous experiment. Saves validation results to a separate CSV.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOCALIZATION_DIR = Path(__file__).resolve().parent

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(LOCALIZATION_DIR))

from cam import load_model, preprocess_image, generate_cam
from heatmap import threshold_heatmap_relative
from clustering import dfs_clusters
from bounding_boxes import clusters_to_boxes
from evaluation import load_ground_truth, evaluate_localization
from peak_localization import peak_based_boxes


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate and compare CAM localization methods."
    )

    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--fraction", type=float, default=0.70)
    parser.add_argument("--peak-fraction", type=float, default=0.90)
    parser.add_argument("--iou-threshold", type=float, default=0.50)
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--output",
        type=str,
        default="batch_validation_results.csv",
    )

    args = parser.parse_args()

    if args.limit <= 0:
        parser.error("--limit must be greater than zero.")

    if not 0 < args.fraction <= 1:
        parser.error("--fraction must be in the interval (0, 1].")

    if not 0 < args.peak_fraction <= 1:
        parser.error("--peak-fraction must be in the interval (0, 1].")

    if not 0 <= args.iou_threshold <= 1:
        parser.error("--iou-threshold must be between 0 and 1.")

    # Dataset and model paths.
    data_dir = PROJECT_ROOT / "data"
    image_dir = data_dir / "stage_2_train_images"
    labels_path = data_dir / "stage_2_train_labels_actual.csv"
    checkpoint = PROJECT_ROOT / "classification" / "cnn_best.pt"

    previous_results_path = (
        LOCALIZATION_DIR / "batch_evaluation_results.csv"
    )

    # Validate required files.
    if not labels_path.exists():
        raise FileNotFoundError(
            f"Labels CSV not found: {labels_path}"
        )

    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {checkpoint}"
        )

    if not image_dir.exists():
        raise FileNotFoundError(
            f"Image directory not found: {image_dir}"
        )

    labels = pd.read_csv(labels_path)

    required_columns = {
        "patientId", "x", "y", "width", "height"
    }

    missing_columns = required_columns - set(labels.columns)

    if missing_columns:
        raise ValueError(
            f"Missing columns in labels CSV: {sorted(missing_columns)}"
        )

    # Select patients with annotated pneumonia bounding boxes.
    annotated = labels.dropna(
        subset=["x", "y", "width", "height"]
    )

    patient_ids = (
        annotated["patientId"]
        .astype(str)
        .drop_duplicates()
    )

    print("Total unique annotated patients:", len(patient_ids))

    # Exclude patients from the previous experiment.
    # This exclusion is applied when generating validation results.
    if args.output == "batch_validation_results.csv":

        if not previous_results_path.exists():
            raise FileNotFoundError(
                "Previous experiment results were not found:\n"
                f"{previous_results_path}\n"
                "Run the original experiment first."
            )

        previous_results = pd.read_csv(previous_results_path)

        if "patient_id" not in previous_results.columns:
            raise ValueError(
                "Previous results CSV does not contain 'patient_id'."
            )

        previous_ids = set(
            previous_results["patient_id"].astype(str)
        )

        before_count = len(patient_ids)

        patient_ids = patient_ids[
            ~patient_ids.isin(previous_ids)
        ]

        print("Previously evaluated patients excluded:", (
            before_count - len(patient_ids)
        ))

    if patient_ids.empty:
        print("No eligible annotated patients remain.")
        return

    # Choose a reproducible sample from the remaining patients.
    patient_ids = patient_ids.sample(
        n=min(args.limit, len(patient_ids)),
        random_state=args.seed,
    ).tolist()

    print("Patients selected for evaluation:", len(patient_ids))
    print("Random seed:", args.seed)
    print("IoU threshold:", args.iou_threshold)
    print("Peak threshold:", args.peak_fraction)

    # Load the model once.
    model, mean, std, device = load_model(checkpoint)

    rows = []
    skipped = 0

    # Evaluate both methods on exactly the same images.
    for index, patient_id in enumerate(patient_ids, start=1):

        image_path = image_dir / f"{patient_id}.dcm"

        if not image_path.exists():
            skipped += 1
            print(f"Skipped missing image: {patient_id}")
            continue

        try:
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

            ground_truth = load_ground_truth(
                labels_path,
                patient_id,
                image_size=128,
            )

            if not ground_truth:
                skipped += 1
                print(f"Skipped image without ground-truth boxes: {patient_id}")
                continue

            # -------------------------------------------------
            # Method 1: Original CAM localization
            # -------------------------------------------------
            mask = threshold_heatmap_relative(
                cam,
                fraction=args.fraction,
            )

            clusters = dfs_clusters(
                mask,
                min_pixels=args.min_pixels or 3,
                connectivity=4,
            )

            original_boxes = clusters_to_boxes(clusters)

            # -------------------------------------------------
            # Method 2: Experimental peak-based localization
            # -------------------------------------------------
            peak_boxes, peaks = peak_based_boxes(
                cam,
                fraction=args.fraction,
                peak_fraction=args.peak_fraction,
                min_distance=10,
                min_pixels=10,
            )

            # Evaluate both methods using one-to-one IoU matching.
            original = evaluate_localization(
                original_boxes,
                ground_truth,
                iou_threshold=args.iou_threshold,
            )

            peak = evaluate_localization(
                peak_boxes,
                ground_truth,
                iou_threshold=args.iou_threshold,
            )

            rows.append({
                "patient_id": patient_id,
                "prediction": (
                    "Pneumonia"
                    if int(predicted_class) == 1
                    else "Normal"
                ),
                "p_pneumonia": float(probabilities[1]),
                "gt_boxes": len(ground_truth),
                "original_boxes": len(original_boxes),
                "peak_boxes": len(peak_boxes),

                "original_TP": original["TP"],
                "original_FP": original["FP"],
                "original_FN": original["FN"],
                "original_precision": original["precision"],
                "original_recall": original["recall"],
                "original_F1": original["F1"],
                "original_mean_matched_IoU": (
                    original["mean_matched_iou"]
                ),

                "peak_TP": peak["TP"],
                "peak_FP": peak["FP"],
                "peak_FN": peak["FN"],
                "peak_precision": peak["precision"],
                "peak_recall": peak["recall"],
                "peak_F1": peak["F1"],
                "peak_mean_matched_IoU": (
                    peak["mean_matched_iou"]
                ),
            })

            print(
                f"[{index}/{len(patient_ids)}] {patient_id} | "
                f"Original F1={original['F1']:.3f} | "
                f"Peak F1={peak['F1']:.3f}"
            )

        except Exception as exc:
            skipped += 1
            print(f"Skipped {patient_id}: {exc}")

    if not rows:
        print("\nNo images were evaluated successfully.")
        print("Check the dataset paths and error messages.")
        return

    results = pd.DataFrame(rows)

    # Save to the requested output file without overwriting
    # the previous experiment's results during validation.
    output_path = LOCALIZATION_DIR / args.output

    if (
        args.output == "batch_evaluation_results.csv"
        and previous_results_path.exists()
    ):
        print(
            "\nWARNING: The output filename points to the previous "
            "results file. Choose a different filename to preserve it."
        )
        return

    results.to_csv(output_path, index=False)

    # ---------------------------------------------------------
    # Aggregate evaluation metrics
    # ---------------------------------------------------------
    print("\n" + "=" * 65)
    print("LOCALIZATION EVALUATION SUMMARY")
    print("=" * 65)

    print("Images evaluated:", len(results))
    print("Images skipped:", skipped)
    print("IoU threshold:", args.iou_threshold)
    print("Heatmap fraction:", args.fraction)
    print("Peak fraction:", args.peak_fraction)

    for method in ("original", "peak"):

        print(f"\n{method.upper()} CAM LOCALIZATION")
        print("-" * 45)

        tp = int(results[f"{method}_TP"].sum())
        fp = int(results[f"{method}_FP"].sum())
        fn = int(results[f"{method}_FN"].sum())

        precision = (
            tp / (tp + fp)
            if tp + fp
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn
            else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        print("True Positives (TP):", tp)
        print("False Positives (FP):", fp)
        print("False Negatives (FN):", fn)
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1-score:  {f1:.4f}")

        # This mean includes only images with a matched box.
        # It must not be interpreted as overall localization accuracy.
        matched_ious = results[
            f"{method}_mean_matched_IoU"
        ]

        matched_ious = matched_ious[matched_ious > 0]

        if len(matched_ious):
            print(
                "Mean matched IoU (matched images only):",
                f"{matched_ious.mean():.4f}",
            )
            print(
                "Images with matched boxes:",
                len(matched_ious),
            )
        else:
            print("Mean matched IoU (matched images only): 0.0000")
            print("Images with matched boxes: 0")

    print("\nValidation results saved to:")
    print(output_path)


if __name__ == "__main__":
    main()