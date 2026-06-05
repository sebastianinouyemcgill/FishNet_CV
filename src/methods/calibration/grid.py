"""
Automatic grid calibration (EXPERIMENTAL).

Disabled by default via ``use_grid_calibration=False``. Preserved for analysis;
not part of the recommended skeleton + optional regression pipeline.
"""

from src.calibration.grid_auto import GridCalibrationResult, estimate_grid_calibration

__all__ = ["GridCalibrationResult", "estimate_grid_calibration"]
