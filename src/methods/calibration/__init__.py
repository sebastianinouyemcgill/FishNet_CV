"""
Calibration backends injected into the length pipeline.

- **marker** — blue/yellow 100 mm rectangles (default, production)
- **grid** — automatic tank grid (experimental, disabled by default)
"""

from src.methods.calibration.grid import estimate_grid_calibration, GridCalibrationResult
from src.methods.calibration.marker import (
    CalibrationResult,
    calibrate_sample,
    compute_homography,
    estimate_scale_from_markers,
    rectify_image,
)

__all__ = [
    "CalibrationResult",
    "GridCalibrationResult",
    "calibrate_sample",
    "compute_homography",
    "estimate_grid_calibration",
    "estimate_scale_from_markers",
    "rectify_image",
]
