"""Tests for advanced calibration guardrails."""

from __future__ import annotations

import numpy as np

from src.config import ProjectConfig
from src.calibration.grid_auto import GridCalibrationResult
from src.pipelines.advanced_inference import _choose_calibration


class _FakeSample:
    def __init__(self, image_id: str = "test") -> None:
        self.image_id = image_id
        self.image = np.zeros((100, 100, 3), dtype=np.uint8)
        self.height = 100
        self.width = 100
        self.annotations = []

    def blue_annotations(self):
        return []

    def yellow_annotations(self):
        return []


def test_grid_rejected_when_ratio_out_of_range(monkeypatch) -> None:
    cfg = ProjectConfig(
        use_grid_auto_calibration=True,
        grid_ppm_ratio_min=0.85,
        grid_ppm_ratio_max=1.15,
    )
    sample = _FakeSample()
    grid = GridCalibrationResult(
        pixels_per_grid_square=100.0,
        pixels_per_mm=10.0,
        grid_square_mm=10.0,
        success=True,
    )

    monkeypatch.setattr(
        "src.pipelines.advanced_inference.calibrate_sample",
        lambda *a, **k: type("C", (), {"pixels_per_mm": 2.0})(),
    )

    calib, source, _, _, _ = _choose_calibration(sample, cfg, sample.image, grid=grid)
    assert source == "marker_fallback"
    assert calib.pixels_per_mm == 2.0


def test_grid_accepted_when_ratio_in_range(monkeypatch) -> None:
    cfg = ProjectConfig(use_grid_auto_calibration=True)
    sample = _FakeSample()
    grid = GridCalibrationResult(
        pixels_per_grid_square=20.0,
        pixels_per_mm=2.0,
        grid_square_mm=10.0,
        success=True,
    )

    monkeypatch.setattr(
        "src.pipelines.advanced_inference.calibrate_sample",
        lambda *a, **k: type("C", (), {"pixels_per_mm": 2.0})(),
    )

    calib, source, _, _, _ = _choose_calibration(sample, cfg, sample.image, grid=grid)
    assert source == "grid"
    assert calib.pixels_per_mm == 2.0
