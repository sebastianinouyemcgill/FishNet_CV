"""
2×2 experiment grid: ground truth source × {skeleton baseline, regression calibration}.
"""

from __future__ import annotations

import json
import logging
from dataclasses import replace
from pathlib import Path

import pandas as pd

from src.config import ProjectConfig, get_config
from src.experiments.ground_truth import GROUND_TRUTH_SOURCES, resolve_ground_truth_path
from src.experiments.regression import run_regression_experiment
from src.experiments.results import ExperimentResult

logger = logging.getLogger(__name__)

GT_SOURCES = ("validation_lengths", "validation_lengths2")


def _run_name(ground_truth_source: str, use_regression: bool) -> str:
    prefix = "regression" if use_regression else "baseline"
    return f"{prefix}_skeleton_{ground_truth_source}"


def run_four_way_comparison(
    cfg: ProjectConfig | None = None,
    *,
    overwrite: bool = False,
    eval_split: str = "valid",
    train_split: str | None = None,
    model_type: str = "random_forest",
) -> pd.DataFrame:
    """
    Run all four combinations and write a summary table under ``cfg.runs_root``.

    Grid (4 cells)
    --------------
    - GT: ``validation_lengths`` | ``validation_lengths2``
    - Method: skeleton baseline | regression calibrated

    Returns
    -------
    DataFrame
        One row per cell with MAE/RMSE and run directory paths.
    """
    from src.experiments import run_experiment

    cfg = cfg or get_config()
    train_split = train_split or eval_split
    rows: list[dict] = []
    results: list[ExperimentResult] = []

    for gt_source in GT_SOURCES:
        gt_cfg = replace(cfg, ground_truth_source=gt_source)
        gt_path = resolve_ground_truth_path(gt_cfg)
        if gt_path is None:
            raise FileNotFoundError(
                f"Missing {GROUND_TRUTH_SOURCES[gt_source]} under {gt_cfg.data_annotations}"
            )

        baseline_name = _run_name(gt_source, use_regression=False)
        logger.info("=== baseline skeleton | GT=%s ===", gt_source)
        baseline = run_experiment(
            cfg=gt_cfg,
            pipeline="baseline",
            method="skeleton",
            split=eval_split,
            run_name=baseline_name,
            ground_truth_path=gt_path,
            validation_set_only=True,
            overwrite=overwrite,
            evaluate=True,
        )
        results.append(baseline)
        rows.append(_row_from_result(baseline, gt_source, "skeleton_baseline", gt_path))

        reg_name = _run_name(gt_source, use_regression=True)
        logger.info("=== regression calibrated | GT=%s ===", gt_source)
        reg = run_regression_experiment(
            cfg=gt_cfg,
            train_split=train_split,
            eval_split=eval_split,
            run_name=reg_name,
            ground_truth_path=gt_path,
            validation_set_only=True,
            overwrite=overwrite,
            model_type=model_type,
            evaluate=True,
        )
        results.append(reg)
        rows.append(_row_from_result(reg, gt_source, "regression_calibrated", gt_path))

        if reg.metrics_path and reg.metrics_path.is_file():
            extra = json.loads(reg.metrics_path.read_text())
            rows[-1]["skeleton_mae_mm"] = extra.get("skeleton_baseline_mae_mm")
            rows[-1]["skeleton_rmse_mm"] = extra.get("skeleton_baseline_rmse_mm")

    summary = pd.DataFrame(rows)
    out_path = cfg.runs_root / "comparison_grid_summary.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    logger.info("Wrote 4-way summary to %s", out_path)
    return summary


def _row_from_result(
    result: ExperimentResult,
    gt_source: str,
    method_label: str,
    gt_path: Path,
) -> dict:
    mae_mm = rmse_mm = None
    if result.metrics:
        mae_mm = result.metrics.mae_mm
        rmse_mm = result.metrics.rmse_mm
    model_path = result.run_dir / "regression_model.joblib"
    return {
        "ground_truth_source": gt_source,
        "method": method_label,
        "run_name": result.run_name,
        "run_dir": str(result.run_dir),
        "predictions_path": str(result.predictions_path),
        "comparison_path": str(result.comparison_path) if result.comparison_path else None,
        "metrics_path": str(result.metrics_path) if result.metrics_path else None,
        "model_path": str(model_path) if model_path.is_file() else None,
        "ground_truth_csv": str(gt_path),
        "n_predictions": result.n_predictions,
        "mae_mm": mae_mm,
        "rmse_mm": rmse_mm,
    }


def format_comparison_table(summary: pd.DataFrame) -> str:
    """Pretty 2×2 MAE table for logging or notebooks."""
    lines = ["", "=== 2×2 comparison (MAE mm) ===", ""]
    for gt in GT_SOURCES:
        lines.append(f"Ground truth: {gt}")
        sub = summary[summary["ground_truth_source"] == gt]
        for _, r in sub.iterrows():
            lines.append(
                f"  {r['method']:22s}  MAE={r['mae_mm']:.2f}  RMSE={r['rmse_mm']:.2f}  → {r['run_name']}"
            )
        lines.append("")
    return "\n".join(lines)
