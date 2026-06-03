"""
Assemble per-image inspection context for visualization and debugging.

Recomputes pipeline intermediates (masks, calibration, depth, measurements)
using the same logic as inference, optionally merged with run CSV outputs.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.calibration import CalibrationResult, rectify_image
from src.calibration.grid_auto import (
    GridCalibrationResult,
    estimate_grid_calibration,
)
from src.config import ProjectConfig, get_config
from src.dataset import DatasetSample, find_image_path, load_sample
from src.depth.cache import depth_cache_path, load_cached_depth
from src.depth.depth_anything import _model_tag_from_name
from src.evaluation import load_ground_truth_csv
from src.masks import cleanup_mask, mask_from_class, skeletonize_mask
from src.measurement.core import (
    measure_bbox_length,
    measure_fish_length,
    measure_pca_length,
    measure_skeleton_length,
    pixels_to_mm,
)
from src.measurement.skeleton3d import (
    estimate_skeleton_3d_length_mm,
    get_skeleton_path_pixels,
)
from src.pipelines.advanced_inference import _choose_calibration, _depth_scale_for_sample
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class ImageInspectionContext:
    """All intermediates needed to render debug visualizations for one image."""

    image_id: str
    sample: DatasetSample
    image_bgr: np.ndarray
    original_bgr: np.ndarray
    fish_mask: np.ndarray
    calibration: CalibrationResult
    grid: GridCalibrationResult | None
    scale_source: str
    marker_ppm: float
    grid_ppm: float
    depth_map: np.ndarray | None
    depth_scale: float
    method: str
    skeleton: np.ndarray
    skeleton_path: list[tuple[int, int]]
    skeleton_px: float
    pca_length_px: float
    pca_center: np.ndarray
    pca_axis: np.ndarray
    bbox_length_px: float
    length_2d_mm: float
    length_3d_mm: float | None
    predicted_mm: float | None
    ground_truth_mm: float | None
    abs_error_mm: float | None
    depth_stats: dict[str, float]
    rectified_bgr: np.ndarray | None
    split: str
    pipeline: str
    cfg: ProjectConfig
    extra: dict[str, Any] = field(default_factory=dict)


def load_run_config(run_dir: Path | str) -> dict[str, Any]:
    """Load ``config.json`` from an experiment run directory."""
    path = Path(run_dir) / "config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Run config not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _apply_run_config(cfg: ProjectConfig, run_config: dict[str, Any]) -> ProjectConfig:
    """Return a copy of cfg with pipeline flags from a saved run."""
    out = copy.copy(cfg)
    out.default_split = run_config.get("split", out.default_split)
    out.measurement_method = run_config.get("method", out.measurement_method)
    out.apply_perspective_correction = bool(run_config.get("perspective", False))
    out.use_grid_auto_calibration = bool(run_config.get("use_grid_auto_calibration", False))
    out.use_depth_estimation = bool(run_config.get("use_depth_estimation", False))
    out.use_3d_measurement = bool(run_config.get("use_3d_measurement", False))
    return out


def _load_run_predictions(run_dir: Path) -> pd.DataFrame | None:
    path = run_dir / "predictions.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    df["image_id"] = df["image_id"].astype(str)
    return df


def _load_run_comparison(run_dir: Path) -> pd.DataFrame | None:
    path = run_dir / "comparison.csv"
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    df["image_id"] = df["image_id"].astype(str)
    return df


def _depth_stats(depth_map: np.ndarray | None) -> dict[str, float]:
    if depth_map is None:
        return {"min": float("nan"), "max": float("nan"), "median": float("nan")}
    finite = depth_map[np.isfinite(depth_map)]
    if finite.size == 0:
        return {"min": float("nan"), "max": float("nan"), "median": float("nan")}
    return {
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "median": float(np.median(finite)),
    }


def build_image_context(
    image_id: str,
    *,
    cfg: ProjectConfig | None = None,
    run_dir: Path | str | None = None,
    split: str | None = None,
    method: str | None = None,
    pipeline: str | None = None,
    load_depth: bool = True,
    predicted_mm: float | None = None,
    ground_truth_mm: float | None = None,
) -> ImageInspectionContext:
    """
    Load sample data and recompute pipeline intermediates for visualization.

    When ``run_dir`` is provided, pipeline flags and predictions are read from
    the saved experiment (``config.json``, ``comparison.csv``).
    """
    cfg = cfg or get_config()
    run_config: dict[str, Any] | None = None
    if run_dir is not None:
        run_dir = Path(run_dir)
        run_config = load_run_config(run_dir)
        cfg = _apply_run_config(cfg, run_config)
        pipeline = pipeline or run_config.get("pipeline", "baseline")
        split = split or run_config.get("split", cfg.default_split)
        method = method or run_config.get("method", cfg.measurement_method)

        comp = _load_run_comparison(run_dir)
        if comp is not None and image_id in comp["image_id"].values:
            row = comp.loc[comp["image_id"] == image_id].iloc[0]
            if predicted_mm is None and "predicted_length_mm" in row:
                predicted_mm = float(row["predicted_length_mm"])
            if ground_truth_mm is None and "length_mm" in row:
                ground_truth_mm = float(row["length_mm"])
            if "abs_error_mm" in row:
                abs_err = float(row["abs_error_mm"])
            else:
                abs_err = None
        else:
            abs_err = None
            preds = _load_run_predictions(run_dir)
            if preds is not None and image_id in preds["image_id"].values:
                if predicted_mm is None:
                    predicted_mm = float(
                        preds.loc[preds["image_id"] == image_id, "predicted_length_mm"].iloc[0]
                    )
    else:
        abs_err = None
        pipeline = pipeline or "baseline"
        split = split or cfg.default_split
        method = method or cfg.measurement_method

    if ground_truth_mm is None:
        from src.experiments import default_ground_truth_path

        gt_path = default_ground_truth_path(cfg)
        if gt_path is not None and gt_path.is_file():
            gt_df = load_ground_truth_csv(gt_path)
            if image_id in gt_df["image_id"].values:
                ground_truth_mm = float(gt_df.loc[gt_df["image_id"] == image_id, "length_mm"].iloc[0])

    if abs_err is None and predicted_mm is not None and ground_truth_mm is not None:
        abs_err = abs(predicted_mm - ground_truth_mm)

    sample_path = find_image_path(cfg.images_dir(split), image_id)
    if sample_path is None:
        raise FileNotFoundError(f"No image found for image_id={image_id} in split={split}")
    sample = load_sample(sample_path, split=split, load_image_array=True)

    original_bgr = sample.image.copy()
    image = original_bgr
    rectified_bgr: np.ndarray | None = None

    grid: GridCalibrationResult | None = None
    if cfg.use_grid_auto_calibration:
        grid = estimate_grid_calibration(image, cfg=cfg)

    calib, scale_source, marker_ppm, grid_ppm, grid = _choose_calibration(
        sample, cfg, image, grid=grid
    )

    if cfg.apply_perspective_correction and calib.homography is not None:
        rectified_bgr = rectify_image(image, calib.homography)
        image = rectified_bgr

    fish_mask = mask_from_class(
        sample.annotations,
        class_name="fish",
        height=image.shape[0],
        width=image.shape[1],
    )
    clean_mask = cleanup_mask(fish_mask)
    skeleton = skeletonize_mask(clean_mask)
    skeleton_path = get_skeleton_path_pixels(clean_mask)

    skeleton_px = measure_skeleton_length(clean_mask)
    pca_length_px, pca_center, pca_axis = measure_pca_length(clean_mask)
    bbox_length_px = measure_bbox_length(clean_mask)

    depth_map: np.ndarray | None = None
    depth_scale = 0.0
    if load_depth and (cfg.use_depth_estimation or cfg.use_3d_measurement):
        model_tag = _model_tag_from_name(cfg.depth_model_name)
        cache_path = depth_cache_path(image_id, split, model_tag=model_tag, cfg=cfg)
        depth_map = load_cached_depth(cache_path)
        depth_scale = _depth_scale_for_sample(cfg, depth_map, grid)

    length_px = measure_fish_length(clean_mask, method=method)
    length_2d_mm = pixels_to_mm(length_px, calib.pixels_per_mm)

    length_3d_mm: float | None = None
    if cfg.use_3d_measurement and depth_map is not None:
        if depth_scale > 0:
            length_3d_mm = estimate_skeleton_3d_length_mm(
                clean_mask,
                depth_map,
                calib,
                depth_scale_mm_per_unit=depth_scale,
            )
        else:
            length_3d_mm = pixels_to_mm(skeleton_px, calib.pixels_per_mm)

    if predicted_mm is None:
        if cfg.use_3d_measurement and cfg.use_depth_estimation and depth_map is not None:
            predicted_mm = length_3d_mm if length_3d_mm is not None else length_2d_mm
        else:
            predicted_mm = length_2d_mm

    return ImageInspectionContext(
        image_id=image_id,
        sample=sample,
        image_bgr=image,
        original_bgr=original_bgr,
        fish_mask=clean_mask,
        calibration=calib,
        grid=grid,
        scale_source=scale_source,
        marker_ppm=marker_ppm,
        grid_ppm=grid_ppm,
        depth_map=depth_map,
        depth_scale=depth_scale,
        method=method,
        skeleton=skeleton,
        skeleton_path=skeleton_path,
        skeleton_px=skeleton_px,
        pca_length_px=pca_length_px,
        pca_center=pca_center,
        pca_axis=pca_axis,
        bbox_length_px=bbox_length_px,
        length_2d_mm=length_2d_mm,
        length_3d_mm=length_3d_mm,
        predicted_mm=predicted_mm,
        ground_truth_mm=ground_truth_mm,
        abs_error_mm=abs_err,
        depth_stats=_depth_stats(depth_map),
        rectified_bgr=rectified_bgr,
        split=split,
        pipeline=pipeline or "baseline",
        cfg=cfg,
        extra={"run_config": run_config} if run_config else {},
    )
