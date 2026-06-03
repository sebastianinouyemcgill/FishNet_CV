"""
3D fish length from 2D skeleton pixels and a monocular depth map.

Computes arc length along the ordered skeleton polyline in metric (x, y, z) space.
"""

from __future__ import annotations

import logging
from collections import deque

import numpy as np
from scipy.ndimage import convolve

from src.calibration import CalibrationResult
from src.masks import cleanup_mask, skeletonize_mask
from src.measurement.core import pixels_to_mm
from src.utils import get_logger

logger = get_logger(__name__)


def _skeleton_graph_neighbors(skel: np.ndarray) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """8-connected adjacency for skeleton pixels."""
    ys, xs = np.where(skel > 0)
    points = set(zip(xs.tolist(), ys.tolist(), strict=True))
    graph: dict[tuple[int, int], list[tuple[int, int]]] = {p: [] for p in points}
    for x, y in points:
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                n = (x + dx, y + dy)
                if n in points:
                    graph[(x, y)].append(n)
    return graph


def _find_endpoints(skel: np.ndarray) -> list[tuple[int, int]]:
    """Skeleton pixels with exactly one neighbor (path endpoints)."""
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.int32)
    neighbors = convolve((skel > 0).astype(np.int32), kernel, mode="constant")
    endpoint_mask = (skel > 0) & (neighbors == 11)
    ey, ex = np.where(endpoint_mask)
    return list(zip(ex.tolist(), ey.tolist(), strict=True))


def _longest_path_on_skeleton(graph: dict[tuple[int, int], list[tuple[int, int]]]) -> list[tuple[int, int]]:
    """
    Approximate longest simple path between two degree-1 nodes via double BFS.
    """
    if not graph:
        return []
    endpoints = [p for p, nbs in graph.items() if len(nbs) == 1]
    if len(endpoints) < 2:
        # Closed loop or star: pick arbitrary start
        start = next(iter(graph))
        endpoints = [start]

    def bfs_farthest(start: tuple[int, int]) -> tuple[tuple[int, int], dict[tuple[int, int], tuple[int, int] | None]]:
        dist = {start: 0}
        parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
        q: deque[tuple[int, int]] = deque([start])
        far = start
        while q:
            cur = q.popleft()
            for nb in graph.get(cur, []):
                if nb in dist:
                    continue
                dist[nb] = dist[cur] + 1
                parent[nb] = cur
                q.append(nb)
                if dist[nb] > dist[far]:
                    far = nb
        return far, parent

    a, _ = bfs_farthest(endpoints[0])
    b, parent = bfs_farthest(a)
    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = b
    while cur is not None:
        path.append(cur)
        cur = parent.get(cur)
    path.reverse()
    return path


def _sample_depth(depth_map: np.ndarray, x: int, y: int) -> float:
    h, w = depth_map.shape[:2]
    xi = int(np.clip(x, 0, w - 1))
    yi = int(np.clip(y, 0, h - 1))
    val = float(depth_map[yi, xi])
    return val if np.isfinite(val) else 0.0


def get_skeleton_path_pixels(
    fish_mask: np.ndarray,
    *,
    preprocess: bool = True,
) -> list[tuple[int, int]]:
    """Ordered skeleton pixel path (longest path between endpoints)."""
    mask = cleanup_mask(fish_mask) if preprocess else fish_mask
    skel = skeletonize_mask(mask)
    graph = _skeleton_graph_neighbors(skel)
    return _longest_path_on_skeleton(graph)


def measure_skeleton_3d_arc_length(
    fish_mask: np.ndarray,
    depth_map: np.ndarray,
    *,
    pixels_per_mm: float,
    depth_scale_mm_per_unit: float = 1.0,
    preprocess: bool = True,
) -> float:
    """
    3D arc length along the fish skeleton in millimeters.

    Parameters
    ----------
    fish_mask:
        Binary uint8 mask.
    depth_map:
        H×W relative depth (aligned with mask image).
    pixels_per_mm:
        Scale from grid (or marker) calibration.
    depth_scale_mm_per_unit:
        Millimeters per depth map unit (from grid + depth gradient).
    """
    mask = cleanup_mask(fish_mask) if preprocess else fish_mask
    skel = skeletonize_mask(mask)
    graph = _skeleton_graph_neighbors(skel)
    path = _longest_path_on_skeleton(graph)
    if len(path) < 2:
        logger.warning("3D skeleton: insufficient path points")
        return 0.0

    points_mm: list[np.ndarray] = []
    for x, y in path:
        z_mm = _sample_depth(depth_map, x, y) * depth_scale_mm_per_unit
        points_mm.append(
            np.array([x / pixels_per_mm, y / pixels_per_mm, z_mm], dtype=np.float64)
        )

    arc = 0.0
    for i in range(1, len(points_mm)):
        arc += float(np.linalg.norm(points_mm[i] - points_mm[i - 1]))
    logger.debug("3D skeleton arc length: %.2f mm (%d points)", arc, len(path))
    return arc


def estimate_skeleton_3d_length_mm(
    fish_mask: np.ndarray,
    depth_map: np.ndarray,
    calibration: CalibrationResult,
    *,
    depth_scale_mm_per_unit: float = 1.0,
) -> float:
    """Mask + depth + calibration → fish length in mm."""
    if calibration.pixels_per_mm <= 0:
        return float("nan")
    return measure_skeleton_3d_arc_length(
        fish_mask,
        depth_map,
        pixels_per_mm=calibration.pixels_per_mm,
        depth_scale_mm_per_unit=depth_scale_mm_per_unit,
    )
