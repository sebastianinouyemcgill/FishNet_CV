"""Smoke tests for src.methods facades (no numeric drift)."""

from src.methods import FishLengthEstimator, LengthEstimate
from src.methods.calibration import calibrate_sample, estimate_grid_calibration
from src.methods.geometric import measure_skeleton_length
from src.methods.regression import LengthRegressionModel, extract_length_features


def test_facade_imports() -> None:
    assert FishLengthEstimator is not None
    assert LengthRegressionModel is not None
    assert callable(measure_skeleton_length)
    assert callable(extract_length_features)
    assert callable(calibrate_sample)
    assert callable(estimate_grid_calibration)
