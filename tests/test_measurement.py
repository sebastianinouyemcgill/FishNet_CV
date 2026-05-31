"""Tests for length measurement helpers."""

from __future__ import annotations

import numpy as np

from src.measurement import measure_bbox_length, measure_pca_length, pixels_to_mm


def test_measure_bbox_horizontal_bar() -> None:
    mask = np.zeros((50, 100), dtype=np.uint8)
    mask[20:30, :] = 255
    assert measure_bbox_length(mask) == np.hypot(100.0, 10.0)


def test_pixels_to_mm() -> None:
    assert pixels_to_mm(100.0, 10.0) == 10.0


def test_measure_pca_on_diagonal_mask() -> None:
    mask = np.zeros((60, 60), dtype=np.uint8)
    for i in range(10, 50):
        mask[i, i] = 255
        mask[i, i + 1] = 255
    length, center, axis = measure_pca_length(mask)
    assert length > 0
    assert center.shape == (2,)
    assert np.isclose(np.linalg.norm(axis), 1.0)
