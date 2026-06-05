"""
Fish length estimation methods — modular, presentation-ready layout.

Implementation remains in legacy modules; this package provides stable
facades and :class:`FishLengthEstimator` without changing numerics.
"""

from src.methods.estimator import FishLengthEstimator, LengthEstimate

__all__ = ["FishLengthEstimator", "LengthEstimate"]
