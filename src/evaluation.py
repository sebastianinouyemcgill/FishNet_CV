"""
Evaluation metrics, ground-truth loading, and experiment logging.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import ProjectConfig, get_config
from src.utils import get_logger

logger = get_logger(__name__)


@dataclass
class MetricResult:
    """Summary metrics for one experiment run."""

    mae_mm: float
    rmse_mm: float
    n_samples: int
    method: str = ""
    split: str = ""
    notes: str = ""


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true[mask] - y_pred[mask])))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root mean squared error."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if mask.sum() == 0:
        return float("nan")
    return float(np.sqrt(np.mean((y_true[mask] - y_pred[mask]) ** 2)))


def load_ground_truth_csv(path: Path) -> pd.DataFrame:
    """
    Load manual length annotations.

    Expected columns (flexible naming):
    - ``image_id`` (or ``id``, ``filename``)
    - ``length_mm`` (or ``true_length_mm``, ``ground_truth_mm``)

    TODO: Align column names with the CSV format specified in assignment.pdf.
    """
    if not path.is_file():
        raise FileNotFoundError(f"Ground truth CSV not found: {path}")
    df = pd.read_csv(path)
    col_map = {
        "id": "image_id",
        "filename": "image_id",
        "true_length_mm": "length_mm",
        "ground_truth_mm": "length_mm",
    }
    from_true_length_mm = "true_length_mm" in df.columns
    df = df.rename(columns={k: v for k, v in col_map.items() if k in df.columns})
    if "image_id" not in df.columns:
        raise ValueError(f"CSV must contain image_id column; got {list(df.columns)}")
    if "length_mm" not in df.columns:
        raise ValueError(f"CSV must contain length_mm column; got {list(df.columns)}")
    df["image_id"] = df["image_id"].astype(str).str.replace(r"\.[^.]+$", "", regex=True)
    df = _maybe_convert_length_units(df, path, from_true_length_mm=from_true_length_mm)
    logger.info("Loaded %d ground-truth rows from %s", len(df), path)
    return df


def _maybe_convert_length_units(
    df: pd.DataFrame,
    path: Path,
    *,
    from_true_length_mm: bool = False,
) -> pd.DataFrame:
    """
    Convert manual validation lengths to millimeters when values look like centimeters.

    Applied only when the source column was ``true_length_mm`` (notebook 02 export)
    **and** all values are small (max < 250), indicating cm stored under a mm column.
    Files already in millimeters (e.g. after manual ×10 fix) are left unchanged.
    """
    if not from_true_length_mm or "length_mm" not in df.columns or df.empty:
        return df
    max_val = float(df["length_mm"].max())
    if max_val >= 250:
        return df
    logger.info(
        "Ground truth in %s looks like centimeters (max=%.1f); converting to mm (×10)",
        path.name,
        max_val,
    )
    out = df.copy()
    out["length_mm"] = out["length_mm"] * 10.0
    return out


def load_predictions_csv(path: Path) -> pd.DataFrame:
    """Load ``predictions.csv`` with columns ``image_id``, ``predicted_length_mm``."""
    df = pd.read_csv(path)
    required = {"image_id", "predicted_length_mm"}
    if not required.issubset(df.columns):
        raise ValueError(f"Predictions must have columns {required}; got {list(df.columns)}")
    df["image_id"] = df["image_id"].astype(str)
    return df


def compare_predictions(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge ground truth and predictions; add error columns.

    Returns a DataFrame with ``length_mm``, ``predicted_length_mm``, ``abs_error_mm``.
    """
    merged = ground_truth.merge(predictions, on="image_id", how="inner")
    if merged.empty:
        logger.warning("No overlapping image_ids between GT and predictions")
    merged["error_mm"] = merged["predicted_length_mm"] - merged["length_mm"]
    merged["abs_error_mm"] = merged["error_mm"].abs()
    return merged


def compute_metrics(
    ground_truth: pd.DataFrame,
    predictions: pd.DataFrame,
    method: str = "",
    split: str = "",
) -> MetricResult:
    """Compute MAE and RMSE in millimeters."""
    merged = compare_predictions(ground_truth, predictions)
    result = MetricResult(
        mae_mm=mae(merged["length_mm"].values, merged["predicted_length_mm"].values),
        rmse_mm=rmse(merged["length_mm"].values, merged["predicted_length_mm"].values),
        n_samples=len(merged),
        method=method,
        split=split,
    )
    logger.info(
        "Metrics (n=%d): MAE=%.3f mm, RMSE=%.3f mm",
        result.n_samples,
        result.mae_mm,
        result.rmse_mm,
    )
    return result


def metrics_to_table(results: list[MetricResult]) -> pd.DataFrame:
    """Build a comparison table from multiple ``MetricResult`` objects."""
    return pd.DataFrame([asdict(r) for r in results])


def log_experiment(
    cfg: ProjectConfig,
    result: MetricResult,
    predictions_path: Path,
    extra: dict | None = None,
) -> Path:
    """
    Append JSON experiment record under ``outputs/metrics/``.

    Returns path to the log file written.
    """
    cfg.ensure_output_dirs()
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "metrics": asdict(result),
        "predictions_path": str(predictions_path),
        **(extra or {}),
    }
    log_path = cfg.outputs_metrics / "experiments.jsonl"
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")
    logger.info("Logged experiment to %s", log_path)
    return log_path


def evaluate_run(
    ground_truth_path: Path,
    predictions_path: Path,
    cfg: ProjectConfig | None = None,
    method: str = "",
    split: str = "",
    output_dir: Path | None = None,
    pipeline: str = "",
    perspective: bool = False,
    run_name: str = "",
) -> tuple[MetricResult, pd.DataFrame]:
    """
    Load CSVs, compute metrics, and save comparison tables.

    Parameters
    ----------
    output_dir:
        If provided, write ``comparison.csv`` and ``metrics.json`` into this
        directory (typical for managed experiment runs). If omitted, uses legacy
        paths under ``cfg.outputs_metrics/``.
    """
    cfg = cfg or get_config()
    cfg.ensure_output_dirs()
    gt = load_ground_truth_csv(ground_truth_path)
    pred = load_predictions_csv(predictions_path)
    merged = compare_predictions(gt, pred)
    metrics = compute_metrics(gt, pred, method=method, split=split)

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        out_table = output_dir / "comparison.csv"
        metrics_path = output_dir / "metrics.json"
    else:
        out_table = cfg.outputs_metrics / "comparison.csv"
        metrics_path = None

    merged.to_csv(out_table, index=False)
    logger.info("Wrote comparison table to %s", out_table)

    if metrics_path is not None:
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump(asdict(metrics), f, indent=2)
        logger.info("Wrote metrics to %s", metrics_path)

    extra = {
        "pipeline": pipeline,
        "perspective": perspective,
        "run_name": run_name,
        "comparison_path": str(out_table),
    }
    if output_dir is not None:
        extra["run_dir"] = str(output_dir)
    log_experiment(cfg, metrics, predictions_path, extra=extra)
    return metrics, merged
