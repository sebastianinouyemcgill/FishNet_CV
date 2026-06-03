"""Debug text panel for visualization figures."""

from __future__ import annotations

import math

import matplotlib.pyplot as plt

from src.visualization.context import ImageInspectionContext


def _fmt(value, unit: str = "", precision: int = 2) -> str:
    if value is None:
        return "n/a"
    try:
        if math.isnan(float(value)):
            return "n/a"
    except (TypeError, ValueError):
        return str(value)
    if unit:
        return f"{float(value):.{precision}f} {unit}"
    return f"{float(value):.{precision}f}"


def _px_to_mm(length_px: float, pixels_per_mm: float) -> float:
    if pixels_per_mm <= 0:
        return float("nan")
    return length_px / pixels_per_mm


def debug_panel_lines(ctx: ImageInspectionContext) -> list[str]:
    """Return debug metrics as text lines."""
    lines = [
        f"image_id: {ctx.image_id}",
        f"pipeline: {ctx.pipeline} / {ctx.method}",
        f"split: {ctx.split}",
        "",
        f"ground truth: {_fmt(ctx.ground_truth_mm, 'mm')}",
        f"predicted: {_fmt(ctx.predicted_mm, 'mm')}",
        f"abs error: {_fmt(ctx.abs_error_mm, 'mm')}",
        "",
        f"calibration scale: {_fmt(ctx.calibration.pixels_per_mm, 'px/mm', 4)}",
        f"scale source: {ctx.scale_source}",
        f"marker ppm: {_fmt(ctx.marker_ppm, 'px/mm', 4)}",
        f"grid ppm: {_fmt(ctx.grid_ppm, 'px/mm', 4)}",
        "",
        f"skeleton length: {_fmt(ctx.skeleton_px, 'px', 1)} "
        f"({_fmt(_px_to_mm(ctx.skeleton_px, ctx.calibration.pixels_per_mm), 'mm')})",
        f"2D length ({ctx.method}): {_fmt(ctx.length_2d_mm, 'mm')}",
        f"3D length: {_fmt(ctx.length_3d_mm, 'mm')}",
        "",
        "depth statistics:",
        f"  min: {_fmt(ctx.depth_stats.get('min'), '', 4)}",
        f"  max: {_fmt(ctx.depth_stats.get('max'), '', 4)}",
        f"  median: {_fmt(ctx.depth_stats.get('median'), '', 4)}",
        f"depth scale: {_fmt(ctx.depth_scale, 'mm/unit', 4)}",
    ]
    if ctx.grid is not None:
        lines.extend(
            [
                "",
                f"grid lines H/V: {ctx.grid.n_horizontal_lines}/{ctx.grid.n_vertical_lines}",
                f"grid px/square: {_fmt(ctx.grid.pixels_per_grid_square, 'px', 1)}",
            ]
        )
    return lines


def add_debug_panel(fig: plt.Figure, ctx: ImageInspectionContext) -> None:
    """Add a right-hand text panel to an existing figure."""
    fig.text(
        0.99,
        0.98,
        "\n".join(debug_panel_lines(ctx)),
        transform=fig.transFigure,
        fontsize=9,
        verticalalignment="top",
        horizontalalignment="right",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="#cccccc"),
    )


def create_debug_figure(ctx: ImageInspectionContext) -> plt.Figure:
    """Standalone debug metrics figure."""
    fig, ax = plt.subplots(figsize=(6, 8))
    ax.axis("off")
    ax.text(
        0.02,
        0.98,
        "\n".join(debug_panel_lines(ctx)),
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        family="monospace",
    )
    ax.set_title(f"Debug — {ctx.image_id}")
    return fig
