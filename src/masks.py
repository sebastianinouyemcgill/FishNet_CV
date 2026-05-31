"""
Binary mask generation from polygons and morphological / skeleton utilities.
"""

from __future__ import annotations

import logging
from typing import Iterable

import cv2
import numpy as np
from skimage import morphology
from skimage.measure import label, regionprops

from src.dataset import PolygonAnnotation
from src.utils import get_logger

logger = get_logger(__name__)


def polygon_to_mask(
    polygon_pixels: np.ndarray,
    height: int,
    width: int,
    fill_value: int = 255,
) -> np.ndarray:
    """
    Rasterize a closed polygon to a binary mask (uint8, 0 or 255).

    Parameters
    ----------
    polygon_pixels:
        Shape ``(N, 2)`` vertex coordinates in pixel space.
    """
    mask = np.zeros((height, width), dtype=np.uint8)
    if polygon_pixels is None or len(polygon_pixels) < 3:
        return mask
    pts = polygon_pixels.astype(np.int32).reshape(-1, 1, 2)
    cv2.fillPoly(mask, [pts], fill_value)
    return mask


def annotations_to_mask(
    annotations: Iterable[PolygonAnnotation],
    height: int,
    width: int,
    class_filter: set[int] | None = None,
) -> np.ndarray:
    """Union multiple polygon annotations into one binary mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    for ann in annotations:
        if class_filter is not None and ann.class_id not in class_filter:
            continue
        if ann.coords_pixels is None:
            logger.warning("Skipping annotation without pixel coords: %s", ann.class_name)
            continue
        m = polygon_to_mask(ann.coords_pixels, height, width)
        mask = np.maximum(mask, m)
    return mask


def largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected foreground component.

    Expects mask with foreground > 0.
    """
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return mask
    labeled = label(binary)
    if labeled.max() == 0:
        return mask
    regions = regionprops(labeled)
    largest = max(regions, key=lambda r: r.area)
    cleaned = np.zeros_like(mask)
    cleaned[labeled == largest.label] = 255
    return cleaned


def extract_contours(mask: np.ndarray) -> list[np.ndarray]:
    """
    Extract external contours from a binary mask.

    Returns
    -------
    list of contours, each shape ``(M, 1, 2)`` OpenCV format.
    """
    contours, _ = cv2.findContours(
        (mask > 0).astype(np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    return list(contours)


def cleanup_mask(
    mask: np.ndarray,
    kernel_size: int = 5,
    opening_iters: int = 1,
    closing_iters: int = 1,
    keep_largest: bool = True,
) -> np.ndarray:
    """
    Morphological opening/closing and optional largest-component filter.

    TODO: Tune kernel size per image resolution and noise level.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    cleaned = mask.copy()
    if opening_iters > 0:
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=opening_iters)
    if closing_iters > 0:
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel, iterations=closing_iters)
    if keep_largest:
        cleaned = largest_connected_component(cleaned)
    return cleaned


def skeletonize_mask(mask: np.ndarray) -> np.ndarray:
    """
    Compute a 1-pixel-wide skeleton of the foreground.

    Returns boolean array; use ``.astype(np.uint8) * 255`` for visualization.
    """
    binary = mask > 0
    if not binary.any():
        return np.zeros_like(mask, dtype=np.uint8)
    skel = morphology.skeletonize(binary)
    return (skel.astype(np.uint8) * 255)


def overlay_mask_on_image(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 0),
    alpha: float = 0.4,
) -> np.ndarray:
    """Blend a colored mask overlay onto a BGR image."""
    out = image_bgr.copy()
    if mask.max() == 0:
        return out
    colored = np.zeros_like(out)
    colored[mask > 0] = color
    cv2.addWeighted(colored, alpha, out, 1 - alpha, 0, dst=out)
    return out


def mask_from_class(
    sample_annotations: list[PolygonAnnotation],
    class_name: str,
    height: int,
    width: int,
) -> np.ndarray:
    """Build a mask for all instances of a given class name."""
    filtered = [a for a in sample_annotations if a.class_name == class_name]
    return annotations_to_mask(filtered, height, width)
