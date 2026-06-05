"""Marker-based scale (production). Facade over ``src.calibration.marker``."""

from src.calibration.marker import (
    CalibrationResult,
    calibrate_sample,
    compute_homography,
    estimate_scale_from_markers,
    rectify_image,
)

__all__ = [
    "CalibrationResult",
    "calibrate_sample",
    "compute_homography",
    "estimate_scale_from_markers",
    "rectify_image",
]
