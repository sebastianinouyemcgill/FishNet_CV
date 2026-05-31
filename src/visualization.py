"""
Plotting and figure export for masks, PCA axes, homography, and reports.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.calibration import CalibrationResult
from src.config import ProjectConfig, get_config
from src.dataset import DatasetSample
from src.masks import mask_from_class, overlay_mask_on_image
from src.measurement import measure_pca_length
from src.utils import get_logger

logger = get_logger(__name__)

# BGR colors for class overlays
CLASS_COLORS_BGR: dict[str, tuple[int, int, int]] = {
    "fish": (0, 255, 0),
    "blue": (255, 0, 0),
    "yellow": (0, 255, 255),
}


def _bgr_to_rgb(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def draw_polygon_annotations(
    sample: DatasetSample,
    image_bgr: np.ndarray | None = None,
    thickness: int = 2,
) -> np.ndarray:
    """
    Draw YOLO polygon outlines on a copy of the sample image.

    Parameters
    ----------
    sample:
        Loaded sample with ``coords_pixels`` on annotations.
    image_bgr:
        Optional BGR image; defaults to ``sample.image``.
    """
    if image_bgr is None:
        if sample.image is None:
            raise ValueError("sample.image is not loaded")
        image_bgr = sample.image
    vis = image_bgr.copy()
    for ann in sample.annotations:
        if ann.coords_pixels is None or len(ann.coords_pixels) < 3:
            continue
        color = CLASS_COLORS_BGR.get(ann.class_name, (255, 255, 255))
        pts = ann.coords_pixels.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(vis, [pts], isClosed=True, color=color, thickness=thickness)
        # Class label at first vertex
        v0 = tuple(ann.coords_pixels[0].astype(int))
        cv2.putText(
            vis,
            ann.class_name,
            v0,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            1,
            cv2.LINE_AA,
        )
    return vis


def overlay_masks(
    sample: DatasetSample,
    image_bgr: np.ndarray | None = None,
    alpha: float = 0.4,
) -> np.ndarray:
    """
    Overlay fish, blue, and yellow masks on the image (fish=green, blue=red, yellow=cyan).
    """
    if image_bgr is None:
        if sample.image is None:
            raise ValueError("sample.image is not loaded")
        image_bgr = sample.image
    h, w = sample.height, sample.width
    vis = image_bgr.copy()
    for class_name, color in CLASS_COLORS_BGR.items():
        mask = mask_from_class(sample.annotations, class_name, h, w)
        if mask.max() > 0:
            vis = overlay_mask_on_image(vis, mask, color=color, alpha=alpha)
    return vis


def display_sample(
    sample: DatasetSample,
    show_polygons: bool = True,
    show_masks: bool = False,
    title: str | None = None,
) -> plt.Figure:
    """
    Matplotlib figure for one sample (polygons and/or mask overlays).

    Parameters
    ----------
    show_polygons:
        Draw polygon outlines.
    show_masks:
        If True, show mask overlay instead of polygons.
    """
    if sample.image is None:
        raise ValueError("sample.image is not loaded")

    if show_masks:
        vis = overlay_masks(sample)
        mode = "masks"
    elif show_polygons:
        vis = draw_polygon_annotations(sample)
        mode = "polygons"
    else:
        vis = sample.image.copy()
        mode = "image"

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.imshow(_bgr_to_rgb(vis))
    ax.set_title(title or f"{sample.image_id} ({mode})")
    ax.axis("off")
    return fig


def save_figure(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    """Save matplotlib figure and close."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved figure: %s", path)


def plot_mask_overlay(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    title: str = "",
    color: tuple[int, int, int] = (0, 255, 0),
) -> plt.Figure:
    """Create figure with mask overlay on the RGB image."""
    overlay = overlay_mask_on_image(image_bgr, mask, color=color)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(_bgr_to_rgb(overlay))
    ax.set_title(title or "Mask overlay")
    ax.axis("off")
    return fig


def plot_contours(
    image_bgr: np.ndarray,
    contours: list[np.ndarray],
    title: str = "Contours",
) -> plt.Figure:
    """Draw OpenCV contours on the image."""
    vis = image_bgr.copy()
    cv2.drawContours(vis, contours, -1, (255, 0, 0), 2)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(_bgr_to_rgb(vis))
    ax.set_title(title)
    ax.axis("off")
    return fig


def plot_pca_axis(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    title: str = "PCA major axis",
) -> plt.Figure:
    """Visualize PCA length axis on the fish mask."""
    length_px, center, axis = measure_pca_length(mask)
    vis = image_bgr.copy()
    half = length_px / 2.0
    p0 = center - axis * half
    p1 = center + axis * half
    cv2.line(
        vis,
        tuple(p0.astype(int)),
        tuple(p1.astype(int)),
        (0, 0, 255),
        2,
    )
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(_bgr_to_rgb(vis))
    ax.set_title(f"{title} ({length_px:.1f} px)")
    ax.axis("off")
    return fig


def plot_homography_comparison(
    original_bgr: np.ndarray,
    rectified_bgr: np.ndarray,
    title: str = "Perspective correction",
) -> plt.Figure:
    """Side-by-side original vs rectified image."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(_bgr_to_rgb(original_bgr))
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(_bgr_to_rgb(rectified_bgr))
    axes[1].set_title("Rectified")
    axes[1].axis("off")
    fig.suptitle(title)
    return fig


def visualize_sample(
    image_bgr: np.ndarray,
    fish_mask: np.ndarray,
    calibration: CalibrationResult | None = None,
    cfg: ProjectConfig | None = None,
    image_id: str = "sample",
    save: bool = True,
) -> list[Path]:
    """
    Generate and optionally save a standard figure set for one image.

    Returns list of saved paths.
    """
    cfg = cfg or get_config()
    cfg.ensure_output_dirs()
    saved: list[Path] = []

    fig = plot_mask_overlay(image_bgr, fish_mask, title=f"{image_id} — fish mask")
    if save:
        p = cfg.outputs_figures / f"{image_id}_mask.png"
        save_figure(fig, p)
        saved.append(p)
    else:
        plt.show()

    fig = plot_pca_axis(image_bgr, fish_mask, title=f"{image_id} — PCA axis")
    if save:
        p = cfg.outputs_figures / f"{image_id}_pca.png"
        save_figure(fig, p)
        saved.append(p)
    else:
        plt.show()

    if calibration is not None and calibration.homography is not None:
        rectified = cv2.warpPerspective(
            image_bgr,
            calibration.homography,
            (image_bgr.shape[1], image_bgr.shape[0]),
        )
        fig = plot_homography_comparison(image_bgr, rectified)
        if save:
            p = cfg.outputs_figures / f"{image_id}_homography.png"
            save_figure(fig, p)
            saved.append(p)
        else:
            plt.show()

    return saved
