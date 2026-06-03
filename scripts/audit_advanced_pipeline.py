#!/usr/bin/env python3
"""Diagnostic audit: baseline vs advanced intermediates (no pipeline changes)."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

from src.calibration import calibrate_sample
from src.calibration.grid_auto import (
    estimate_depth_metric_scale,
    estimate_grid_calibration,
)
from src.calibration.marker import estimate_scale_from_markers
from src.config import get_config
from src.dataset import iterate_image_ids
from src.depth.cache import depth_cache_path, load_cached_depth
from src.masks import mask_from_class
from src.measurement.core import measure_skeleton_length, pixels_to_mm
from src.measurement.skeleton3d import measure_skeleton_3d_arc_length

LOG_PATH = Path(__file__).resolve().parents[1] / ".cursor" / "debug-95b075.log"


def _log(hypothesis_id: str, location: str, message: str, data: dict) -> None:
    # #region agent log
    payload = {
        "sessionId": "95b075",
        "hypothesisId": hypothesis_id,
        "location": location,
        "message": message,
        "data": data,
        "timestamp": int(pd.Timestamp.utcnow().timestamp() * 1000),
        "runId": "audit",
    }
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload) + "\n")
    # #endregion


def main() -> None:
    cfg = get_config()
    repo = cfg.repo_root
    gt_path = repo / "outputs/runs/baseline_skeleton_v1/comparison.csv"
    if not gt_path.is_file():
        gt_path = repo / "data/annotations/validation_lengths.csv"
    comp = pd.read_csv(gt_path)
    comp["image_id"] = comp["image_id"].astype(str)
    image_ids = comp["image_id"].tolist()[:12]

    rows = []
    for sample in iterate_image_ids(cfg, split="valid", image_ids=image_ids, load_images=True):
        img = sample.image
        fish_mask = mask_from_class(
            sample.annotations, "fish", img.shape[0], img.shape[1]
        )
        marker_ppm = estimate_scale_from_markers(
            sample.blue_annotations(),
            sample.yellow_annotations(),
            sample.width,
            sample.height,
        )
        grid = estimate_grid_calibration(img, cfg=cfg)
        calib = calibrate_sample(sample, cfg=cfg)
        skel_px = measure_skeleton_length(fish_mask)
        baseline_mm = pixels_to_mm(skel_px, marker_ppm)
        grid_mm_2d = pixels_to_mm(skel_px, grid.pixels_per_mm) if grid.success else float("nan")

        depth_path = depth_cache_path(
            sample.image_id, "valid", model_tag="depth-anything_DA3-SMALL", cfg=cfg
        )
        depth = load_cached_depth(depth_path)
        depth_scale = 1.0
        arc3d_mm = float("nan")
        d_min = d_max = d_med = float("nan")
        if depth is not None:
            d_min, d_max, d_med = float(np.nanmin(depth)), float(np.nanmax(depth)), float(
                np.nanmedian(depth)
            )
            if grid.success:
                depth_scale = estimate_depth_metric_scale(
                    depth, grid.pixels_per_grid_square, grid.grid_square_mm
                )
            arc3d_mm = measure_skeleton_3d_arc_length(
                fish_mask,
                depth,
                pixels_per_mm=grid.pixels_per_mm if grid.success else marker_ppm,
                depth_scale_mm_per_unit=depth_scale,
            )

        ppm_ratio = grid.pixels_per_mm / marker_ppm if marker_ppm > 0 and grid.success else float(
            "nan"
        )
        row = {
            "image_id": sample.image_id,
            "gt_mm": float(comp.loc[comp.image_id == sample.image_id, "length_mm"].iloc[0]),
            "marker_ppm": marker_ppm,
            "grid_px_per_sq": grid.pixels_per_grid_square,
            "grid_ppm": grid.pixels_per_mm,
            "ppm_ratio_grid_over_marker": ppm_ratio,
            "skeleton_px": skel_px,
            "baseline_mm": baseline_mm,
            "grid_2d_skeleton_mm": grid_mm_2d,
            "depth_min": d_min,
            "depth_max": d_max,
            "depth_median": d_med,
            "depth_scale_mm_per_unit": depth_scale,
            "advanced_3d_arc_mm": arc3d_mm,
            "grid_n_h": grid.n_horizontal_lines,
            "grid_n_v": grid.n_vertical_lines,
        }
        rows.append(row)
        _log(
            "H1",
            "audit:per_image",
            "intermediates",
            {**row, "marker_calib_ppm": calib.pixels_per_mm},
        )

    df = pd.DataFrame(rows)
    out_csv = repo / "outputs" / "audit_advanced_intermediates.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(df.to_string())
    print(f"\nWrote {out_csv}")
    _log("H2", "audit:summary", "ppm_ratio_stats", {
        "ppm_ratio_median": float(df["ppm_ratio_grid_over_marker"].median()),
        "ppm_ratio_mean": float(df["ppm_ratio_grid_over_marker"].mean()),
    })


if __name__ == "__main__":
    main()
