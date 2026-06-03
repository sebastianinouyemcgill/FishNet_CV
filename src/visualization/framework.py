"""
Notebook-friendly visualization API for baseline and advanced pipelines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np

from src.config import ProjectConfig, get_config
from src.visualization._common import ALL_PANEL_NAMES, bgr_to_rgb, resolve_output_dir, save_figure
from src.visualization.context import ImageInspectionContext, build_image_context
from src.visualization.debug import add_debug_panel
from src.visualization.panels import PANEL_TITLES, render_panel


def _resolve_context(
    image_id: str | ImageInspectionContext,
    *,
    cfg: ProjectConfig | None = None,
    run_dir: Path | str | None = None,
    **kwargs,
) -> ImageInspectionContext:
    if isinstance(image_id, ImageInspectionContext):
        return image_id
    return build_image_context(image_id, cfg=cfg, run_dir=run_dir, **kwargs)


def _figure_from_panels(
    ctx: ImageInspectionContext,
    panels: Sequence[str],
    *,
    debug: bool = False,
    suptitle: str | None = None,
    ncols: int = 3,
) -> plt.Figure:
    n = len(panels)
    nrows = int(np.ceil(n / ncols))
    fig_w = 5 * ncols + (3 if debug else 0)
    fig_h = 4 * nrows
    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h))
    axes_flat = np.atleast_1d(axes).flatten()

    for ax, panel in zip(axes_flat, panels, strict=False):
        img = render_panel(ctx, panel)
        ax.imshow(bgr_to_rgb(img))
        ax.set_title(PANEL_TITLES.get(panel, panel))
        ax.axis("off")

    for ax in axes_flat[len(panels) :]:
        ax.axis("off")

    title = suptitle or f"{ctx.image_id} — {ctx.pipeline}/{ctx.method}"
    fig.suptitle(title, fontsize=14, y=1.02)
    if debug:
        add_debug_panel(fig, ctx)
    fig.tight_layout()
    return fig


def _maybe_save(
    fig: plt.Figure,
    ctx: ImageInspectionContext,
    *,
    save: bool,
    output_dir: Path | None,
    run_dir: Path | str | None,
    filename: str,
) -> Path | None:
    if not save:
        return None
    out_dir = resolve_output_dir(
        Path(run_dir) if run_dir else None,
        output_dir,
        ctx.cfg,
    )
    path = out_dir / filename
    save_figure(fig, path)
    return path


def visualize_image(
    image_id: str | ImageInspectionContext,
    *,
    cfg: ProjectConfig | None = None,
    run_dir: Path | str | None = None,
    panels: Sequence[str] | None = None,
    debug: bool = True,
    save: bool = False,
    output_dir: Path | None = None,
    show: bool = True,
    **context_kwargs,
) -> plt.Figure:
    """
    Full multi-panel visualization for one image.

    Parameters
    ----------
    panels:
        Subset of panel names; default is all ten visualization types.
    run_dir:
        Experiment run directory — loads config and predictions automatically.
    save:
        Write PNG to ``run_dir/figures/`` or ``output_dir``.
    """
    ctx = _resolve_context(image_id, cfg=cfg, run_dir=run_dir, **context_kwargs)
    panel_list = list(panels) if panels else list(ALL_PANEL_NAMES)
    fig = _figure_from_panels(ctx, panel_list, debug=debug, suptitle=f"{ctx.image_id} — overview")
    _maybe_save(
        fig,
        ctx,
        save=save,
        output_dir=output_dir,
        run_dir=run_dir,
        filename=f"{ctx.image_id}_overview.png",
    )
    if show and not save:
        plt.show()
    return fig


def visualize_depth(
    image_id: str | ImageInspectionContext,
    *,
    cfg: ProjectConfig | None = None,
    run_dir: Path | str | None = None,
    debug: bool = True,
    save: bool = False,
    output_dir: Path | None = None,
    show: bool = True,
    **context_kwargs,
) -> plt.Figure:
    """Depth map panel with optional RGB and 3D skeleton context."""
    ctx = _resolve_context(image_id, cfg=cfg, run_dir=run_dir, **context_kwargs)
    fig = _figure_from_panels(
        ctx,
        ["rgb", "depth", "skeleton3d"],
        debug=debug,
        suptitle=f"{ctx.image_id} — depth",
    )
    _maybe_save(fig, ctx, save=save, output_dir=output_dir, run_dir=run_dir, filename=f"{ctx.image_id}_depth.png")
    if show and not save:
        plt.show()
    return fig


def visualize_measurement(
    image_id: str | ImageInspectionContext,
    *,
    cfg: ProjectConfig | None = None,
    run_dir: Path | str | None = None,
    debug: bool = True,
    save: bool = False,
    output_dir: Path | None = None,
    show: bool = True,
    **context_kwargs,
) -> plt.Figure:
    """Measurement overlay with calibration and method-specific geometry."""
    ctx = _resolve_context(image_id, cfg=cfg, run_dir=run_dir, **context_kwargs)
    extra = ["pca"] if ctx.method == "pca" else ["skeleton"] if ctx.method == "skeleton" else []
    panels = ["rgb", "calibration", "measurement"] + extra
    fig = _figure_from_panels(
        ctx,
        panels,
        debug=debug,
        suptitle=f"{ctx.image_id} — measurement ({ctx.method})",
        ncols=2,
    )
    _maybe_save(
        fig,
        ctx,
        save=save,
        output_dir=output_dir,
        run_dir=run_dir,
        filename=f"{ctx.image_id}_measurement.png",
    )
    if show and not save:
        plt.show()
    return fig


def visualize_comparison(
    image_id: str | ImageInspectionContext,
    *,
    cfg: ProjectConfig | None = None,
    run_dir: Path | str | None = None,
    compare_run_dir: Path | str | None = None,
    debug: bool = True,
    save: bool = False,
    output_dir: Path | None = None,
    show: bool = True,
    **context_kwargs,
) -> plt.Figure:
    """
    Ground truth vs predicted measurement for one image.

    When ``compare_run_dir`` is set, overlays predictions from both runs.
    """
    ctx = _resolve_context(image_id, cfg=cfg, run_dir=run_dir, **context_kwargs)

    compare_pred: float | None = None
    compare_label = ""
    if compare_run_dir is not None:
        compare_ctx = build_image_context(
            ctx.image_id,
            cfg=cfg,
            run_dir=compare_run_dir,
            split=ctx.split,
            load_depth=ctx.depth_map is not None,
        )
        compare_pred = compare_ctx.predicted_mm
        compare_label = Path(compare_run_dir).name

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    meas = render_panel(ctx, "measurement")
    axes[0].imshow(bgr_to_rgb(meas))
    axes[0].set_title("Measurement overlay")
    axes[0].axis("off")

    axes[1].axis("off")
    lines = [
        f"image_id: {ctx.image_id}",
        "",
        f"ground truth: {ctx.ground_truth_mm:.1f} mm"
        if ctx.ground_truth_mm is not None
        else "ground truth: n/a",
        f"predicted ({ctx.pipeline}): {ctx.predicted_mm:.1f} mm"
        if ctx.predicted_mm is not None
        else "predicted: n/a",
    ]
    if ctx.abs_error_mm is not None:
        lines.append(f"abs error: {ctx.abs_error_mm:.1f} mm")
    if compare_pred is not None:
        lines.extend(
            [
                "",
                f"compare run ({compare_label}): {compare_pred:.1f} mm",
                f"delta: {compare_pred - ctx.predicted_mm:.1f} mm"
                if ctx.predicted_mm is not None
                else "",
            ]
        )
    axes[1].text(
        0.05,
        0.95,
        "\n".join(lines),
        transform=axes[1].transAxes,
        fontsize=12,
        verticalalignment="top",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="#f5f5f5"),
    )
    axes[1].set_title("Comparison")

    fig.suptitle(f"{ctx.image_id} — GT vs prediction", fontsize=14)
    if debug:
        add_debug_panel(fig, ctx)
    fig.tight_layout()

    _maybe_save(
        fig,
        ctx,
        save=save,
        output_dir=output_dir,
        run_dir=run_dir,
        filename=f"{ctx.image_id}_comparison.png",
    )
    if show and not save:
        plt.show()
    return fig


def visualize_batch(
    image_ids: Sequence[str],
    *,
    cfg: ProjectConfig | None = None,
    run_dir: Path | str | None = None,
    panels: Sequence[str] | None = None,
    debug: bool = False,
    save: bool = True,
    output_dir: Path | None = None,
    show: bool = False,
    **context_kwargs,
) -> list[Path]:
    """
    Generate visualizations for multiple image IDs.

    Returns list of saved paths (empty entries skipped when save=False).
    """
    cfg = cfg or get_config()
    saved: list[Path] = []
    for image_id in image_ids:
        fig = visualize_image(
            image_id,
            cfg=cfg,
            run_dir=run_dir,
            panels=panels,
            debug=debug,
            save=save,
            output_dir=output_dir,
            show=show,
            **context_kwargs,
        )
        if save:
            out_dir = resolve_output_dir(
                Path(run_dir) if run_dir else None,
                output_dir,
                cfg,
            )
            saved.append(out_dir / f"{image_id}_overview.png")
        else:
            plt.close(fig)
    return saved


def save_run_visualizations(
    image_ids: Sequence[str],
    run_dir: Path | str,
    *,
    cfg: ProjectConfig | None = None,
    include_comparison: bool = False,
    compare_run_dir: Path | str | None = None,
) -> list[Path]:
    """
    Save full visualization set into an experiment run directory.

    Writes to ``<run_dir>/figures/`` and ``<run_dir>/debug/`` (debug panel PNGs).
    """
    run_dir = Path(run_dir)
    cfg = cfg or get_config()
    saved: list[Path] = []

    for image_id in image_ids:
        ctx = build_image_context(image_id, cfg=cfg, run_dir=run_dir)
        visualize_image(ctx, run_dir=run_dir, debug=True, save=True, show=False)
        saved.append(run_dir / "figures" / f"{image_id}_overview.png")

        visualize_depth(ctx, run_dir=run_dir, debug=True, save=True, show=False)
        saved.append(run_dir / "figures" / f"{image_id}_depth.png")

        visualize_measurement(ctx, run_dir=run_dir, debug=True, save=True, show=False)
        saved.append(run_dir / "figures" / f"{image_id}_measurement.png")

        from src.visualization.debug import create_debug_figure

        debug_fig = create_debug_figure(ctx)
        debug_path = run_dir / "debug" / f"{image_id}_debug.png"
        save_figure(debug_fig, debug_path)
        saved.append(debug_path)

        if include_comparison:
            visualize_comparison(
                ctx,
                run_dir=run_dir,
                compare_run_dir=compare_run_dir,
                debug=True,
                save=True,
                show=False,
            )
            saved.append(run_dir / "figures" / f"{image_id}_comparison.png")

    return saved
