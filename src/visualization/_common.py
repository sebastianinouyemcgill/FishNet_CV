"""Shared helpers for visualization modules."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.utils import get_logger

logger = get_logger(__name__)

CLASS_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "fish": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
}

ALL_PANEL_NAMES: tuple[str, ...] = (
    "rgb",
    "mask",
    "skeleton",
    "pca",
    "calibration",
    "grid",
    "perspective",
    "depth",
    "skeleton3d",
    "measurement",
)


def bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def save_figure(fig: plt.Figure, path: Path, dpi: int = 150) -> Path:
    """Save matplotlib figure and close."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)
    return path


def resolve_output_dir(
    run_dir: Path | None,
    output_dir: Path | None,
    cfg,
    *,
    subdir: str = "figures",
) -> Path:
    """Pick save directory: explicit output_dir > run_dir/subdir > cfg.outputs_figures."""
    if output_dir is not None:
        return output_dir
    if run_dir is not None:
        return Path(run_dir) / subdir
    return cfg.outputs_figures
