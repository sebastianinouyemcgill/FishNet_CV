"""
Baseline geometry methods: bounding-box diagonal, PCA axis, skeleton path.

Primary production baseline: **skeleton**. Implementations live in
``src.measurement.core`` (unchanged for metric compatibility).
"""

from src.measurement.core import (
    MeasurementMethod,
    estimate_length_mm,
    measure_bbox_length,
    measure_fish_length,
    measure_pca_length,
    measure_skeleton_length,
    pixels_to_mm,
)

__all__ = [
    "MeasurementMethod",
    "estimate_length_mm",
    "measure_bbox_length",
    "measure_fish_length",
    "measure_pca_length",
    "measure_skeleton_length",
    "pixels_to_mm",
]
