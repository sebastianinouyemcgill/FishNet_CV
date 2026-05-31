"""
Fish length estimation in pixels and conversion to millimeters.
"""

from __future__ import annotations

import logging
from enum import Enum
from typing import Callable

import cv2
import numpy as np
from sklearn.decomposition import PCA

from src.calibration import CalibrationResult
from src.masks import cleanup_mask, extract_contours, skeletonize_mask
from src.utils import get_logger

logger = get_logger(__name__)


class MeasurementMethod(str, Enum):
    """Supported length estimation methods."""

    BBOX = "bbox"
    PCA = "pca"
    SKELETON = "skeleton"


def pixels_to_mm(length_px: float, pixels_per_mm: float) -> float:
    """Convert a pixel length to millimeters."""
    if pixels_per_mm <= 0:
        logger.warning("Invalid pixels_per_mm=%s; returning NaN", pixels_per_mm)
        return float("nan")
    return length_px / pixels_per_mm


def measure_bbox_length(mask: np.ndarray) -> float:
    """
    Length as the diagonal of the axis-aligned bounding box of the mask.

    Baseline assignment method: sqrt(width² + height²) in pixels.
    """
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return 0.0
    width = float(xs.max() - xs.min() + 1)
    height = float(ys.max() - ys.min() + 1)
    return float(np.hypot(width, height))


def measure_pca_length(mask: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """
    Length along the first principal axis of mask foreground pixels.

    Returns
    -------
    length_px:
        Extent along PC1 (max - min projection).
    center:
        Mean point (2,).
    axis:
        Unit direction vector of PC1 (2,).
    """
    ys, xs = np.where(mask > 0)
    if len(xs) < 2:
        return 0.0, np.zeros(2), np.array([1.0, 0.0])
    points = np.column_stack([xs.astype(np.float64), ys.astype(np.float64)])
    pca = PCA(n_components=2)
    pca.fit(points)
    axis = pca.components_[0]
    axis = axis / (np.linalg.norm(axis) + 1e-12)
    projections = points @ axis
    length_px = float(projections.max() - projections.min())
    center = points.mean(axis=0)
    return length_px, center, axis


def measure_skeleton_length(mask: np.ndarray) -> float:
    """
    Geodesic length along the medial axis skeleton (graph path estimate).

    Uses contour of skeleton pixels and sums Euclidean steps along the
    ordered path. TODO: Replace with true geodesic distance on skeleton graph
    (e.g. longest path between endpoints).
    """
    skel = skeletonize_mask(mask)
    ys, xs = np.where(skel > 0)
    if len(xs) < 2:
        return 0.0
    points = np.column_stack([xs, ys])
    # Order points along x for a simple path proxy
    order = np.argsort(points[:, 0])
    ordered = points[order]
    diffs = np.diff(ordered.astype(np.float64), axis=0)
    length_px = float(np.linalg.norm(diffs, axis=1).sum())
    return length_px


def measure_fish_length(
    fish_mask: np.ndarray,
    method: str | MeasurementMethod = MeasurementMethod.PCA,
    preprocess: bool = True,
) -> float:
    """
    Estimate fish length in pixels using the selected method.

    Parameters
    ----------
    fish_mask:
        Binary uint8 mask (fish region).
    method:
        ``bbox``, ``pca``, or ``skeleton``.
    preprocess:
        If True, run ``cleanup_mask`` before measuring.
    """
    method = MeasurementMethod(method)
    mask = cleanup_mask(fish_mask) if preprocess else fish_mask
    if mask.max() == 0:
        logger.warning("Empty fish mask")
        return 0.0

    if method == MeasurementMethod.BBOX:
        length = measure_bbox_length(mask)
    elif method == MeasurementMethod.PCA:
        length, _, _ = measure_pca_length(mask)
    else:
        length = measure_skeleton_length(mask)

    logger.debug("Fish length (%s): %.2f px", method.value, length)
    return length


def estimate_length_mm(
    fish_mask: np.ndarray,
    calibration: CalibrationResult,
    method: str | MeasurementMethod = MeasurementMethod.PCA,
) -> float:
    """End-to-end: mask -> pixels -> millimeters."""
    length_px = measure_fish_length(fish_mask, method=method)
    return pixels_to_mm(length_px, calibration.pixels_per_mm)


# Registry for easy extension of new methods
_MEASUREMENT_REGISTRY: dict[str, Callable[[np.ndarray], float]] = {
    MeasurementMethod.BBOX.value: measure_bbox_length,
    MeasurementMethod.PCA.value: lambda m: measure_pca_length(m)[0],
    MeasurementMethod.SKELETON.value: measure_skeleton_length,
}


def register_measurement_method(name: str, func: Callable[[np.ndarray], float]) -> None:
    """Register a custom pixel-length function. TODO: document in README."""
    _MEASUREMENT_REGISTRY[name] = func
    logger.info("Registered measurement method: %s", name)


def get_contour_endpoints(mask: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """
    Farthest pair of points on the largest contour (approximate fish endpoints).

    TODO: Use for skeleton endpoint initialization.
    """
    contours = extract_contours(mask)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea).reshape(-1, 2)
    if len(cnt) < 2:
        return None
    # Brute-force diameter (fine for moderate contours)
    max_dist = 0.0
    pair = (0, 1)
    for i in range(len(cnt)):
        for j in range(i + 1, len(cnt)):
            d = np.linalg.norm(cnt[i] - cnt[j])
            if d > max_dist:
                max_dist = d
                pair = (i, j)
    p0 = tuple(cnt[pair[0]].astype(int))
    p1 = tuple(cnt[pair[1]].astype(int))
    return p0, p1
