"""Tests for 3D skeleton arc length."""

from __future__ import annotations

import numpy as np

from src.calibration import CalibrationResult
from src.measurement.skeleton3d import measure_skeleton_3d_arc_length


def test_3d_arc_length_horizontal_line() -> None:
    """Straight skeleton with flat depth → length ≈ span in mm."""
    h, w = 100, 200
    mask = np.zeros((h, w), dtype=np.uint8)
    # Thick bar so skeletonize yields a connected 1px-wide path with endpoints
    mask[48:53, 40:160] = 255
    depth = np.ones((h, w), dtype=np.float32) * 5.0
    pixels_per_mm = 10.0
    length_mm = measure_skeleton_3d_arc_length(
        mask,
        depth,
        pixels_per_mm=pixels_per_mm,
        depth_scale_mm_per_unit=0.0,
        preprocess=False,
    )
    # ~120 px / 10 px/mm = 12 mm (ordered path may be slightly shorter)
    assert 8.0 <= length_mm <= 14.0


def test_estimate_skeleton_3d_nan_without_scale() -> None:
    from src.measurement import estimate_skeleton_3d_length_mm

    mask = np.zeros((50, 50), dtype=np.uint8)
    mask[25, 10:40] = 255
    depth = np.ones((50, 50), dtype=np.float32)
    calib = CalibrationResult(pixels_per_mm=0.0)
    assert np.isnan(estimate_skeleton_3d_length_mm(mask, depth, calib))
