"""
Convert DFS clusters into bounding boxes and filter
area outliers using the two-standard-deviation rule.
"""

import numpy as np


def cluster_to_box(cluster):
    """
    Convert cluster pixel coordinates into a bounding box.

    Cluster coordinates are (row, column).

    Returns:
        (x_min, y_min, x_max, y_max)
    """
    if len(cluster) == 0:
        raise ValueError("Cluster cannot be empty.")

    points = np.asarray(cluster, dtype=int)

    rows = points[:, 0]
    cols = points[:, 1]

    x_min = int(cols.min())
    y_min = int(rows.min())

    # Exclusive maximum coordinates.
    x_max = int(cols.max()) + 1
    y_max = int(rows.max()) + 1

    return x_min, y_min, x_max, y_max


def box_area(box):
    """Calculate bounding-box area."""
    x1, y1, x2, y2 = box

    width = max(0, x2 - x1)
    height = max(0, y2 - y1)

    return width * height


def filter_boxes_std(boxes, std_multiplier=2.0):
    """
    Keep boxes whose areas are within two standard
    deviations of the mean area.

    Args:
        boxes: Bounding boxes in (x1, y1, x2, y2) format.
        std_multiplier: Standard-deviation multiplier.

    Returns:
        List of filtered boxes.
    """
    if not boxes:
        return []

    if std_multiplier < 0:
        raise ValueError(
            "std_multiplier must be non-negative."
        )

    areas = np.asarray(
        [box_area(box) for box in boxes],
        dtype=np.float64,
    )

    mean_area = float(np.mean(areas))
    std_area = float(np.std(areas))

    lower = mean_area - std_multiplier * std_area
    upper = mean_area + std_multiplier * std_area

    filtered_boxes = [
        box
        for box, area in zip(boxes, areas)
        if lower <= area <= upper
    ]

    return filtered_boxes


def clusters_to_boxes(clusters, std_multiplier=2.0):
    """Convert clusters to boxes and filter area outliers."""

    boxes = [
        cluster_to_box(cluster)
        for cluster in clusters
        if len(cluster) > 0
    ]

    return filter_boxes_std(
        boxes,
        std_multiplier=std_multiplier,
    )