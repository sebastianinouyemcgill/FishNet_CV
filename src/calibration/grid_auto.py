"""
Automatic tank-grid calibration without fiduciary markers.

Uses Hough line detection, angle clustering, and intersection spacing to estimate
``pixels_per_grid_square``, then converts to ``pixels_per_mm`` via a configured
physical grid cell size.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from src.config import ProjectConfig, get_config
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class GridCalibrationResult:
    """Outputs of grid-based automatic calibration for one image."""

    pixels_per_grid_square: float
    pixels_per_mm: float
    grid_square_mm: float
    success: bool
    n_horizontal_lines: int = 0
    n_vertical_lines: int = 0
    depth_scale_mm_per_unit: float = 1.0


def _normalize_angle_deg(theta_deg: float) -> float:
    """Map angle to [0, 180)."""
    a = theta_deg % 180.0
    return a


def _detect_line_segments(
    gray: np.ndarray,
    *,
    canny_low: int = 50,
    canny_high: int = 150,
    hough_threshold: int = 80,
    min_line_length: int = 80,
    max_line_gap: int = 15,
) -> list[tuple[float, float]]:
    """
    Return line segments as (angle_deg, midpoint_distance_from_origin).

    Angle is in degrees [0, 180); distance is perpendicular distance to origin.
    """
    edges = cv2.Canny(gray, canny_low, canny_high, apertureSize=3)
    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=hough_threshold,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap,
    )
    if lines is None:
        return []
    segments: list[tuple[float, float]] = []
    for x1, y1, x2, y2 in lines[:, 0]:
        dx, dy = float(x2 - x1), float(y2 - y1)
        length = np.hypot(dx, dy)
        if length < 1e-3:
            continue
        angle = _normalize_angle_deg(np.degrees(np.arctan2(dy, dx)))
        mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
        # Perpendicular distance: |x cos + y sin| for line normal
        rad = np.radians(angle)
        dist = abs(mx * np.cos(rad) + my * np.sin(rad))
        segments.append((angle, dist))
    return segments


def _cluster_line_distances(
    segments: list[tuple[float, float]],
    *,
    angle_target: float,
    angle_tol: float = 12.0,
) -> list[float]:
    """Collect perpendicular distances for segments near ``angle_target`` degrees."""
    distances: list[float] = []
    for angle, dist in segments:
        delta = min(abs(angle - angle_target), abs(angle - angle_target + 180))
        if delta <= angle_tol:
            distances.append(dist)
    return distances


def _median_spacing(distances: list[float], bin_px: float = 8.0) -> float:
    """
    Estimate median spacing between parallel grid lines from sorted distances.

    Clusters nearby distances and returns the median of cluster gaps.
    """
    if len(distances) < 3:
        return 0.0
    sorted_d = np.sort(np.asarray(distances, dtype=np.float64))
    # Quantize to reduce duplicate detections on the same line
    quantized = np.round(sorted_d / bin_px) * bin_px
    unique = np.unique(quantized)
    if len(unique) < 2:
        return 0.0
    gaps = np.diff(unique)
    gaps = gaps[gaps > bin_px * 0.5]
    if len(gaps) == 0:
        return 0.0
    return float(np.median(gaps))


def _spacing_from_intersections(gray: np.ndarray) -> float:
    """
    Secondary estimate from Harris corners: median nearest-neighbor distance.
    """
    corners = cv2.goodFeaturesToTrack(
        gray,
        maxCorners=800,
        qualityLevel=0.01,
        minDistance=12,
        blockSize=5,
    )
    if corners is None or len(corners) < 10:
        return 0.0
    pts = corners.reshape(-1, 2)
    # Subsample for speed
    if len(pts) > 200:
        idx = np.linspace(0, len(pts) - 1, 200, dtype=int)
        pts = pts[idx]
    dists: list[float] = []
    for i, p in enumerate(pts):
        others = np.delete(pts, i, axis=0)
        nn = np.min(np.linalg.norm(others - p, axis=1))
        if 8.0 < nn < 200.0:
            dists.append(float(nn))
    if len(dists) < 5:
        return 0.0
    return float(np.median(dists))


def estimate_pixels_per_grid_square(
    image_bgr: np.ndarray,
    *,
    min_spacing_px: float = 15.0,
    max_spacing_px: float = 250.0,
) -> tuple[float, int, int]:
    """
    Estimate average pixel spacing between adjacent grid lines.

    Returns (spacing_px, n_horizontal, n_vertical).
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    h, w = gray.shape[:2]
    scale = 1.0
    if max(h, w) > 1200:
        scale = 1200.0 / max(h, w)
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

    segments = _detect_line_segments(gray)
    horiz_dists = _cluster_line_distances(segments, angle_target=0.0)
    vert_dists = _cluster_line_distances(segments, angle_target=90.0)

    sp_h = _median_spacing(horiz_dists)
    sp_v = _median_spacing(vert_dists)
    sp_corner = _spacing_from_intersections(gray)

    candidates = [s for s in (sp_h, sp_v, sp_corner) if min_spacing_px <= s <= max_spacing_px]
    if not candidates:
        return 0.0, len(horiz_dists), len(vert_dists)

    spacing = float(np.median(candidates))
    if scale != 1.0:
        spacing /= scale
    return spacing, len(horiz_dists), len(vert_dists)


def estimate_depth_metric_scale(
    depth_map: np.ndarray,
    pixels_per_grid_square: float,
    grid_square_mm: float,
) -> float:
    """
    Map relative depth units to millimeters using grid spacing as reference.

    Uses robust depth gradient magnitude near the image center as a proxy for
    depth change across one grid cell.
    """
    if pixels_per_grid_square <= 0 or grid_square_mm <= 0:
        return 1.0
    d = depth_map.astype(np.float64)
    if d.size == 0 or not np.isfinite(d).any():
        return 1.0
    gy, gx = np.gradient(d)
    grad = np.sqrt(gx * gx + gy * gy)
    h, w = grad.shape
    cy, cx = h // 2, w // 2
    r = min(h, w) // 4
    roi = grad[max(0, cy - r) : cy + r, max(0, cx - r) : cx + r]
    median_grad = float(np.median(roi[np.isfinite(roi)])) if roi.size else 0.0
    if median_grad < 1e-8:
        return 1.0
    # Depth change across ~one cell / cell size in mm
    depth_delta_per_cell = median_grad * pixels_per_grid_square
    return grid_square_mm / max(depth_delta_per_cell, 1e-8)


def estimate_grid_calibration(
    image_bgr: np.ndarray,
    cfg: ProjectConfig | None = None,
) -> GridCalibrationResult:
    """
    Full grid calibration for one RGB image (no marker polygons).
    """
    cfg = cfg or get_config()
    grid_square_mm = cfg.grid_square_mm
    spacing, n_h, n_v = estimate_pixels_per_grid_square(image_bgr)
    success = spacing > 0
    pixels_per_mm = spacing / grid_square_mm if success else 0.0

    if success:
        logger.info(
            "Grid calibration: %.2f px/square (%.4f px/mm), lines H=%d V=%d",
            spacing,
            pixels_per_mm,
            n_h,
            n_v,
        )
    else:
        logger.warning("Grid calibration failed; could not detect grid spacing")

    return GridCalibrationResult(
        pixels_per_grid_square=spacing,
        pixels_per_mm=pixels_per_mm,
        grid_square_mm=grid_square_mm,
        success=success,
        n_horizontal_lines=n_h,
        n_vertical_lines=n_v,
    )


def grid_result_to_marker_calibration(grid: GridCalibrationResult):
    """Build a ``CalibrationResult``-compatible object for metric conversion."""
    from src.calibration.marker import CalibrationResult

    return CalibrationResult(pixels_per_mm=grid.pixels_per_mm)
