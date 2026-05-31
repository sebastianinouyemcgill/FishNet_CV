"""
Shared utilities: logging setup, path helpers, and small geometry functions.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Iterable

import numpy as np

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: int = logging.INFO, log_file: Path | None = None) -> None:
    """
    Configure root logging for CLI and notebooks.

    Parameters
    ----------
    level:
        Logging level (e.g. ``logging.DEBUG``).
    log_file:
        Optional file handler path.
    """
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=DATE_FORMAT,
        handlers=handlers,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Return a module-level logger."""
    return logging.getLogger(name)


def normalize_polygon(
    coords: Iterable[float], image_width: int, image_height: int
) -> np.ndarray:
    """
    Convert normalized YOLO polygon (x,y,...) to pixel coordinates.

    Returns
    -------
    ndarray
        Shape ``(N, 2)`` with columns ``[x, y]`` in pixel space.
    """
    arr = np.asarray(list(coords), dtype=np.float64)
    if arr.size % 2 != 0:
        raise ValueError(f"Polygon must have even length; got {arr.size} values")
    pts = arr.reshape(-1, 2)
    pts[:, 0] *= image_width
    pts[:, 1] *= image_height
    return pts


def polygon_area(pts: np.ndarray) -> float:
    """Shoelace formula for polygon area in pixel²."""
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def clamp_points(pts: np.ndarray, width: int, height: int) -> np.ndarray:
    """Clip polygon vertices to image bounds."""
    out = pts.copy()
    out[:, 0] = np.clip(out[:, 0], 0, width - 1)
    out[:, 1] = np.clip(out[:, 1], 0, height - 1)
    return out


def stem_matches_label(image_path: Path, label_path: Path) -> bool:
    """True if label stem matches image stem (ignoring extension)."""
    return image_path.stem == label_path.stem
