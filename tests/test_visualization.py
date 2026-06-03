"""Tests for visualization framework."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.visualization._common import ALL_PANEL_NAMES
from src.visualization.context import ImageInspectionContext
from src.visualization.debug import debug_panel_lines
from src.visualization.panels import PANEL_RENDERERS, render_panel


def _mock_context(**overrides) -> ImageInspectionContext:
    h, w = 100, 120
    image = np.zeros((h, w, 3), dtype=np.uint8)
    fish_mask = np.zeros((h, w), dtype=np.uint8)
    fish_mask[30:70, 20:100] = 255
    skeleton = np.zeros((h, w), dtype=np.uint8)
    skeleton[50, 20:100] = 255

    defaults = dict(
        image_id="test123",
        sample=MagicMock(image_id="test123", annotations=[]),
        image_bgr=image,
        original_bgr=image.copy(),
        fish_mask=fish_mask,
        calibration=MagicMock(pixels_per_mm=5.0),
        grid=None,
        scale_source="marker",
        marker_ppm=5.0,
        grid_ppm=0.0,
        depth_map=None,
        depth_scale=0.0,
        method="pca",
        skeleton=skeleton,
        skeleton_path=[(20, 50), (60, 50), (99, 50)],
        skeleton_px=79.0,
        pca_length_px=80.0,
        pca_center=np.array([60.0, 50.0]),
        pca_axis=np.array([1.0, 0.0]),
        bbox_length_px=85.0,
        length_2d_mm=16.0,
        length_3d_mm=None,
        predicted_mm=16.0,
        ground_truth_mm=15.0,
        abs_error_mm=1.0,
        depth_stats={"min": float("nan"), "max": float("nan"), "median": float("nan")},
        rectified_bgr=None,
        split="valid",
        pipeline="baseline",
        cfg=MagicMock(outputs_figures=Path("/tmp/figures")),
    )
    defaults.update(overrides)
    return ImageInspectionContext(**defaults)


def test_all_panels_render():
    ctx = _mock_context()
    for name in ALL_PANEL_NAMES:
        img = render_panel(ctx, name)
        assert img.shape == ctx.image_bgr.shape
        assert img.dtype == np.uint8


def test_debug_panel_includes_required_fields():
    ctx = _mock_context()
    text = "\n".join(debug_panel_lines(ctx))
    for token in (
        "image_id",
        "ground truth",
        "predicted",
        "abs error",
        "calibration scale",
        "skeleton length",
        "3D length",
        "depth statistics",
    ):
        assert token in text


def test_panel_registry_matches_all_panel_names():
    assert set(PANEL_RENDERERS.keys()) == set(ALL_PANEL_NAMES)


@patch("src.visualization.context.load_sample")
@patch("src.visualization.context.find_image_path")
@patch("src.visualization.context._choose_calibration")
def test_build_image_context_from_run_dir(
    mock_choose,
    mock_find,
    mock_load,
    tmp_path,
):
    from src.visualization.context import build_image_context

    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    (run_dir / "config.json").write_text(
        '{"pipeline":"baseline","method":"pca","split":"valid","perspective":false}',
        encoding="utf-8",
    )
    (run_dir / "comparison.csv").write_text(
        "image_id,length_mm,predicted_length_mm,error_mm,abs_error_mm\n"
        "42,100.0,95.0,-5.0,5.0\n",
        encoding="utf-8",
    )

    h, w = 50, 60
    image = np.zeros((h, w, 3), dtype=np.uint8)
    sample = MagicMock(
        image_id="42",
        image=image,
        height=h,
        width=w,
        annotations=[],
    )
    mock_find.return_value = Path("/fake/42.jpg")
    mock_load.return_value = sample
    calib = MagicMock(pixels_per_mm=4.0, homography=None)
    mock_choose.return_value = (calib, "marker", 4.0, 0.0, None)

    ctx = build_image_context("42", run_dir=run_dir)
    assert ctx.image_id == "42"
    assert ctx.method == "pca"
    assert ctx.predicted_mm == 95.0
    assert ctx.ground_truth_mm == 100.0
    assert ctx.abs_error_mm == 5.0
