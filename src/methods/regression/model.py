"""Sklearn length regressor. Facade over ``src.models.length_regression``."""

from src.models.length_regression import LengthRegressionModel, TARGET_COLUMN

__all__ = ["LengthRegressionModel", "TARGET_COLUMN"]
