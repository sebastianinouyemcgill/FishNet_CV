"""
Plotting and figure export for masks, PCA axes, homography, and debug panels.

Notebook helpers::

    from src.visualization import visualize_image, visualize_depth, build_image_context

    ctx = build_image_context("10224", run_dir="outputs/runs/baseline_pca_v1")
    visualize_image(ctx, debug=True)
"""

from __future__ import annotations

from src.visualization.context import ImageInspectionContext, build_image_context, load_run_config
from src.visualization.framework import (
    save_run_visualizations,
    visualize_batch,
    visualize_comparison,
    visualize_depth,
    visualize_image,
    visualize_measurement,
)
from src.visualization.legacy import (
    display_sample,
    draw_polygon_annotations,
    overlay_masks,
    plot_contours,
    plot_homography_comparison,
    plot_mask_overlay,
    plot_pca_axis,
    save_figure,
    visualize_sample,
)
from src.visualization._common import ALL_PANEL_NAMES

__all__ = [
    "ALL_PANEL_NAMES",
    "ImageInspectionContext",
    "build_image_context",
    "load_run_config",
    "visualize_image",
    "visualize_depth",
    "visualize_measurement",
    "visualize_comparison",
    "visualize_batch",
    "save_run_visualizations",
    "display_sample",
    "draw_polygon_annotations",
    "overlay_masks",
    "plot_contours",
    "plot_homography_comparison",
    "plot_mask_overlay",
    "plot_pca_axis",
    "save_figure",
    "visualize_sample",
]
