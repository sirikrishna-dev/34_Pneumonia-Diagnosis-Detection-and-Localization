import numpy as np


def threshold_heatmap(cam, threshold=0.5):
    """
    Threshold a normalized CAM to identify suspicious regions.

    Args:
        cam: 2D CAM with values between 0 and 1.
        threshold: Activation threshold between 0 and 1.

    Returns:
        Boolean mask.
    """
    cam = np.asarray(cam, dtype=np.float32)

    if cam.ndim != 2:
        raise ValueError("CAM must be a 2D array.")

    if not np.isfinite(cam).all():
        raise ValueError("CAM contains invalid values.")

    if not 0 <= threshold <= 1:
        raise ValueError("Threshold must be between 0 and 1.")

    if cam.min() < 0 or cam.max() > 1:
        raise ValueError("CAM must be normalized to [0, 1].")

    return cam >= threshold


def threshold_heatmap_relative(cam, fraction=0.7):
    """
    Threshold relative to the maximum CAM activation.

    Pixels are retained when their activation is at least
    `fraction` times the maximum activation.

    This is an experimental alternative to a fixed threshold.
    """
    cam = np.asarray(cam, dtype=np.float32)

    if cam.ndim != 2:
        raise ValueError("CAM must be a 2D array.")

    if not np.isfinite(cam).all():
        raise ValueError("CAM contains invalid values.")

    if not 0 <= fraction <= 1:
        raise ValueError("Fraction must be between 0 and 1.")

    if cam.min() < 0 or cam.max() > 1:
        raise ValueError("CAM must be normalized to [0, 1].")

    maximum = float(cam.max())

    if maximum <= 0:
        return np.zeros_like(cam, dtype=bool)

    return cam >= fraction * maximum