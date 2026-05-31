"""
Structured results from experiment runs for notebooks and analysis.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.evaluation import MetricResult


@dataclass
class ExperimentResult:
    """Summary of one completed experiment run."""

    run_name: str
    run_dir: Path
    pipeline: str
    method: str
    split: str
    perspective: bool
    predictions_path: Path
    comparison_path: Path | None = None
    metrics_path: Path | None = None
    config_path: Path | None = None
    metrics: MetricResult | None = None
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat() + "Z")
    n_predictions: int = 0
    evaluated: bool = False

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = {
            "run_name": self.run_name,
            "run_dir": str(self.run_dir),
            "pipeline": self.pipeline,
            "method": self.method,
            "split": self.split,
            "perspective": self.perspective,
            "predictions_path": str(self.predictions_path),
            "comparison_path": str(self.comparison_path) if self.comparison_path else "",
            "metrics_path": str(self.metrics_path) if self.metrics_path else "",
            "config_path": str(self.config_path) if self.config_path else "",
            "timestamp": self.timestamp,
            "n_predictions": self.n_predictions,
            "evaluated": self.evaluated,
            "mae_mm": self.metrics.mae_mm if self.metrics else float("nan"),
            "rmse_mm": self.metrics.rmse_mm if self.metrics else float("nan"),
            "n_samples": self.metrics.n_samples if self.metrics else 0,
        }
        return d


def results_to_dataframe(results: list[ExperimentResult]) -> pd.DataFrame:
    """Convert experiment results to a pandas DataFrame for notebook display."""
    if not results:
        return pd.DataFrame(
            columns=[
                "run_name",
                "pipeline",
                "method",
                "split",
                "perspective",
                "mae_mm",
                "rmse_mm",
                "n_samples",
                "predictions_path",
                "comparison_path",
                "run_dir",
                "timestamp",
            ]
        )
    return pd.DataFrame([r.to_dict() for r in results])


def load_registry(registry_path: Path) -> pd.DataFrame:
    """Load the master ``experiments.csv`` registry."""
    if not registry_path.is_file():
        return pd.DataFrame()
    return pd.read_csv(registry_path)


def load_run_metrics(run_dir: Path) -> dict[str, Any]:
    """Load ``metrics.json`` from a run directory."""
    import json

    path = run_dir / "metrics.json"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def load_all_runs(runs_root: Path) -> pd.DataFrame:
    """
    Scan ``outputs/runs/`` and build a summary table from config + metrics files.

    Complements the append-only registry when runs were created manually.
    """
    rows: list[dict[str, Any]] = []
    if not runs_root.is_dir():
        return pd.DataFrame(rows)

    import json

    for run_dir in sorted(p for p in runs_root.iterdir() if p.is_dir()):
        config_path = run_dir / "config.json"
        metrics_path = run_dir / "metrics.json"
        if not config_path.is_file():
            continue
        with config_path.open(encoding="utf-8") as f:
            config = json.load(f)
        metrics: dict[str, Any] = {}
        if metrics_path.is_file():
            with metrics_path.open(encoding="utf-8") as f:
                metrics = json.load(f)
        rows.append(
            {
                "run_name": run_dir.name,
                "run_dir": str(run_dir),
                "pipeline": config.get("pipeline", ""),
                "method": config.get("method", ""),
                "split": config.get("split", ""),
                "perspective": config.get("perspective", False),
                "mae_mm": metrics.get("mae_mm", float("nan")),
                "rmse_mm": metrics.get("rmse_mm", float("nan")),
                "n_samples": metrics.get("n_samples", 0),
                "predictions_path": str(run_dir / "predictions.csv"),
                "comparison_path": str(run_dir / "comparison.csv"),
                "timestamp": config.get("timestamp", ""),
            }
        )
    return pd.DataFrame(rows)
