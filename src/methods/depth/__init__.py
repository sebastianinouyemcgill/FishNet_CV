"""
Monocular depth estimation (EXPERIMENTAL — isolated).

Default OFF. Does not affect baseline or regression when disabled.
Requires ``FISHNET_ALLOW_EXPERIMENTAL=1`` for 3D / depth experiment paths.
"""

from src.depth import DepthEstimator, get_depth_estimator
from src.depth.cache import depth_cache_path, load_cached_depth
from src.measurement.skeleton3d import estimate_skeleton_3d_length_mm

__all__ = [
    "DepthEstimator",
    "depth_cache_path",
    "estimate_skeleton_3d_length_mm",
    "get_depth_estimator",
    "load_cached_depth",
]
