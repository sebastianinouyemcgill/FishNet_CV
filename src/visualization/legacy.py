"""
Original plotting helpers (masks, PCA, homography).

Kept for backward compatibility; new code should prefer ``framework`` helpers.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

from src.calibration import CalibrationResult
from src.config import ProjectConfig, get_config
from src.dataset import DatasetSample
from src.masks import mask_from_class, overlay_mask_on_image
from src.measurement import measure_pca_length
from src.visualization._common import CLASS_COLORS_BGR, bgr_to_rgb, save_figure


def draw_polygon_annotations(
    sample: DatasetSample,
    image_bgr: np.ndarray | None = None,
    thickness: int = 2,
) -> np.ndarray:
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
    ax.imshow(bgr_to_rgb(vis))
    ax.set_title(title or f"{sample.image_id} ({mode})")
    ax.axis("off")
    return fig


def plot_mask_overlay(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    title: str = "",
    color: tuple[int, int, int] = (0, 255, 0),
) -> plt.Figure:
    overlay = overlay_mask_on_image(image_bgr, mask, color=color)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(bgr_to_rgb(overlay))
    ax.set_title(title or "Mask overlay")
    ax.axis("off")
    return fig


def plot_contours(
    image_bgr: np.ndarray,
    contours: list[np.ndarray],
    title: str = "Contours",
) -> plt.Figure:
    vis = image_bgr.copy()
    cv2.drawContours(vis, contours, -1, (255, 0, 0), 2)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.imshow(bgr_to_rgb(vis))
    ax.set_title(title)
    ax.axis("off")
    return fig


def plot_pca_axis(
    image_bgr: np.ndarray,
    mask: np.ndarray,
    title: str = "PCA major axis",
) -> plt.Figure:
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
    ax.imshow(bgr_to_rgb(vis))
    ax.set_title(f"{title} ({length_px:.1f} px)")
    ax.axis("off")
    return fig


def plot_homography_comparison(
    original_bgr: np.ndarray,
    rectified_bgr: np.ndarray,
    title: str = "Perspective correction",
) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(bgr_to_rgb(original_bgr))
    axes[0].set_title("Original")
    axes[0].axis("off")
    axes[1].imshow(bgr_to_rgb(rectified_bgr))
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
