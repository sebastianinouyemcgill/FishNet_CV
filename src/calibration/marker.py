"""
Calibration from blue/yellow rectangles: scale (px/mm) and optional homography.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from src.config import CALIBRATION_RECT_MM, ProjectConfig, get_config
from src.dataset import DatasetSample, PolygonAnnotation
from src.masks import extract_contours, polygon_to_mask
from src.utils import get_logger, polygon_area

logger = get_logger(__name__)


@dataclass
class CalibrationResult:
    """Outputs of marker-based calibration for one image."""

    pixels_per_mm: float
    blue_mask: np.ndarray | None = None
    yellow_mask: np.ndarray | None = None
    homography: np.ndarray | None = None  # 3x3 image -> rectified plane
    # TODO: store rectified image dimensions and corner correspondences


def _marker_length_px(polygon: np.ndarray) -> float:
    if len(polygon) < 3:
        return 0.0

    rect = cv2.minAreaRect(polygon.astype(np.float32))
    (_, (w, h), _) = rect
    return float(max(w, h))


def _polygon_centroid(pts: np.ndarray) -> np.ndarray:
    """Centroid of polygon vertices."""
    return pts.mean(axis=0)


def estimate_scale_from_markers(
    blue_annotations: list[PolygonAnnotation],
    yellow_annotations: list[PolygonAnnotation],
    image_width: int,
    image_height: int,
    physical_length_mm: float = CALIBRATION_RECT_MM,
) -> float:
    """
    Estimate pixels-per-mm from blue and yellow calibration rectangles.

    Averages scale from all available markers. Returns 0.0 if none found.
    """
    scales: list[float] = []
    for anns in (blue_annotations, yellow_annotations):
        for ann in anns:
            if ann.coords_pixels is None or len(ann.coords_pixels) < 3:
                continue
            length_px = _marker_length_px(ann.coords_pixels)
            area = polygon_area(ann.coords_pixels)
            if length_px < 1e-3:
                continue
            px_per_mm = length_px / physical_length_mm
            scales.append(px_per_mm)
            logger.debug(
                "Marker %s: length_px=%.1f area=%.0f px_per_mm=%.3f",
                ann.class_name,
                length_px,
                area,
                px_per_mm,
            )
    if not scales:
        logger.warning("No calibration markers found; cannot estimate scale")
        return 0.0
    pixels_per_mm = float(np.mean(scales))
    logger.info("Estimated scale: %.4f px/mm (n=%d markers, mean)", pixels_per_mm, len(scales))
    return pixels_per_mm


def compute_homography(
    src_points: np.ndarray,
    dst_points: np.ndarray,
) -> np.ndarray | None:
    """
    Compute homography mapping ``src_points`` -> ``dst_points``.

    Both arrays shape ``(N, 2)`` with ``N >= 4``.

    TODO: Use four corners of blue+yellow markers to define a planar
    reference frame aligned with the tank / ruler grid in assignment.pdf.
    """
    if src_points.shape[0] < 4 or dst_points.shape[0] < 4:
        logger.warning("Need at least 4 point pairs for homography; got %d", src_points.shape[0])
        return None
    H, status = cv2.findHomography(
        src_points.astype(np.float32),
        dst_points.astype(np.float32),
        method=cv2.RANSAC,
    )
    if H is None:
        logger.error("Homography estimation failed")
        return None
    inliers = int(status.sum()) if status is not None else 0
    logger.info("Homography estimated with %d inliers", inliers)
    return H


def rectify_image(image: np.ndarray, homography: np.ndarray) -> np.ndarray:
    """Warp image with a 3x3 homography; output size matches input."""
    h, w = image.shape[:2]
    return cv2.warpPerspective(image, homography, (w, h))


def _default_rectified_corners(
    blue_pts: np.ndarray | None,
    yellow_pts: np.ndarray | None,
    rect_side_px: float = 400.0,
) -> tuple[np.ndarray, np.ndarray] | None:
    """
    Build naive src/dst corner pairs from marker centroids.

    TODO: Replace with true quadrilateral corners from fitted rectangles.
    """
    centroids = []
    for pts in (blue_pts, yellow_pts):
        if pts is not None and len(pts) >= 3:
            centroids.append(_polygon_centroid(pts))
    if len(centroids) < 2:
        return None
    # Placeholders: map two centroids to axis-aligned square corners
    src = np.array(centroids[:4] if len(centroids) >= 4 else centroids * 2, dtype=np.float32)
    if len(src) < 4:
        c0, c1 = centroids[0], centroids[1]
        src = np.array([c0, c1, c1 + [50, 0], c0 + [50, 0]], dtype=np.float32)
    dst = np.array(
        [
            [0, 0],
            [rect_side_px, 0],
            [rect_side_px, rect_side_px],
            [0, rect_side_px],
        ],
        dtype=np.float32,
    )
    return src, dst


def calibrate_sample(
    sample: DatasetSample,
    cfg: ProjectConfig | None = None,
    physical_length_mm: float = CALIBRATION_RECT_MM,
) -> CalibrationResult:
    """
    Full calibration pipeline for one ``DatasetSample``.

    Optionally estimates homography when ``cfg.apply_perspective_correction`` is True.
    """
    cfg = cfg or get_config()
    if sample.image is None:
        raise ValueError("sample.image must be loaded before calibration")

    h, w = sample.height, sample.width
    pixels_per_mm = estimate_scale_from_markers(
        sample.blue_annotations(),
        sample.yellow_annotations(),
        w,
        h,
        physical_length_mm=physical_length_mm,
    )

    blue_mask = yellow_mask = None
    if sample.blue_annotations():
        ann = sample.blue_annotations()[0]
        if ann.coords_pixels is not None:
            blue_mask = polygon_to_mask(ann.coords_pixels, h, w)
    if sample.yellow_annotations():
        ann = sample.yellow_annotations()[0]
        if ann.coords_pixels is not None:
            yellow_mask = polygon_to_mask(ann.coords_pixels, h, w)

    homography = None
    if cfg.apply_perspective_correction:
        blue_pts = sample.blue_annotations()[0].coords_pixels if sample.blue_annotations() else None
        yellow_pts = (
            sample.yellow_annotations()[0].coords_pixels if sample.yellow_annotations() else None
        )
        pair = _default_rectified_corners(blue_pts, yellow_pts)
        if pair is not None:
            src, dst = pair
            homography = compute_homography(src, dst)
        else:
            logger.warning("Perspective correction requested but markers insufficient")

    return CalibrationResult(
        pixels_per_mm=pixels_per_mm,
        blue_mask=blue_mask,
        yellow_mask=yellow_mask,
        homography=homography,
    )
