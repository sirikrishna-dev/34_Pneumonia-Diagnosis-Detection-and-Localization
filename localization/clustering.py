import numpy as np


def dfs_clusters(mask, min_pixels=3, connectivity=8):
    """
    Find connected regions in a binary heatmap using DFS.

    Args:
        mask: 2D boolean array from heatmap thresholding.
        min_pixels: Minimum region size to retain.
        connectivity: 4 or 8 neighbouring pixels.

    Returns:
        List of clusters containing (row, column) coordinates.
    """
    mask = np.asarray(mask, dtype=bool)

    if mask.ndim != 2:
        raise ValueError("Mask must be a 2D array.")

    if min_pixels < 1:
        raise ValueError("min_pixels must be at least 1.")

    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8.")

    height, width = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    clusters = []

    neighbours = [
        (-1, 0),
        (1, 0),
        (0, -1),
        (0, 1),
    ]

    if connectivity == 8:
        neighbours += [
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]

    for row in range(height):
        for col in range(width):

            if not mask[row, col] or visited[row, col]:
                continue

            stack = [(row, col)]
            visited[row, col] = True
            cluster = []

            while stack:
                current_row, current_col = stack.pop()
                cluster.append((current_row, current_col))

                for dr, dc in neighbours:
                    nr = current_row + dr
                    nc = current_col + dc

                    if (
                        0 <= nr < height
                        and 0 <= nc < width
                        and mask[nr, nc]
                        and not visited[nr, nc]
                    ):
                        visited[nr, nc] = True
                        stack.append((nr, nc))

            if len(cluster) >= min_pixels:
                clusters.append(cluster)

    return clusters