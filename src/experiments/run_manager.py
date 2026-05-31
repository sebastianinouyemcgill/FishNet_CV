"""
Run directory management and experiment registry.

Each run lives under ``outputs/runs/<run_name>/`` with isolated artifacts.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import pandas as pd

from src.config import ProjectConfig, get_config
from src.evaluation import MetricResult

logger = logging.getLogger(__name__)

REGISTRY_COLUMNS = [
    "run_name",
    "timestamp",
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
]


class RunExistsError(FileExistsError):
    """Raised when a run directory already exists and overwrite is not allowed."""


class RunManager:
    """
    Create and manage isolated experiment run directories.

    Layout::

        outputs/runs/<run_name>/
            predictions.csv
            metrics.json
            comparison.csv   (after evaluation)
            config.json
            figures/         (optional)
            debug/           (optional)
    """

    def __init__(self, cfg: ProjectConfig | None = None) -> None:
        self.cfg = cfg or get_config()
        self.runs_root = self.cfg.repo_root / "outputs" / "runs"
        self.registry_path = self.runs_root / "experiments.csv"
        self.runs_root.mkdir(parents=True, exist_ok=True)

    def resolve_run_name(
        self,
        run_name: str | None,
        *,
        pipeline: str,
        method: str,
    ) -> str:
        """
        Use explicit ``run_name`` or auto-generate ``{pipeline}_{method}_{timestamp}``.
        """
        if run_name:
            self._validate_run_name(run_name)
            return run_name
        ts = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        return f"{pipeline}_{method}_{ts}"

    @staticmethod
    def _validate_run_name(name: str) -> None:
        if not re.match(r"^[a-zA-Z0-9._-]+$", name):
            raise ValueError(
                f"Invalid run_name {name!r}. Use letters, digits, '.', '_', '-' only."
            )

    def run_dir(self, run_name: str) -> Path:
        return self.runs_root / run_name

    def prepare_run(
        self,
        run_name: str,
        *,
        overwrite: bool = False,
    ) -> Path:
        """
        Resolve run directory; raise ``RunExistsError`` if it exists and not overwriting.
        """
        run_dir = self.run_dir(run_name)
        if run_dir.exists() and any(run_dir.iterdir()):
            if not overwrite:
                raise RunExistsError(
                    f"Run directory already exists: {run_dir}. "
                    "Pass overwrite=True to replace it."
                )
            logger.warning("Overwriting existing run directory: %s", run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "figures").mkdir(exist_ok=True)
        (run_dir / "debug").mkdir(exist_ok=True)
        return run_dir

    def save_config(
        self,
        run_dir: Path,
        *,
        run_name: str,
        pipeline: str,
        method: str,
        split: str,
        perspective: bool,
        limit: int | None = None,
        ground_truth_path: str | None = None,
        n_image_ids: int | None = None,
        extra: dict | None = None,
    ) -> Path:
        """Write ``config.json`` capturing run parameters for reproducibility."""
        config = {
            "run_name": run_name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "pipeline": pipeline,
            "method": method,
            "split": split,
            "perspective": perspective,
            "limit": limit,
            "n_image_ids": n_image_ids,
            "ground_truth_path": ground_truth_path,
            **(extra or {}),
        }
        path = run_dir / "config.json"
        with path.open("w", encoding="utf-8") as f:
            json.dump(config, f, indent=2)
        logger.info("Saved run config: %s", path)
        return path

    def save_metrics(self, run_dir: Path, metrics: MetricResult) -> Path:
        """Write ``metrics.json`` inside the run directory."""
        path = run_dir / "metrics.json"
        payload = asdict(metrics)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info("Saved run metrics: %s", path)
        return path

    def append_registry(
        self,
        *,
        run_name: str,
        pipeline: str,
        method: str,
        split: str,
        perspective: bool,
        predictions_path: Path,
        comparison_path: Path | None,
        metrics: MetricResult | None,
        timestamp: str,
    ) -> None:
        """
        Append one row to ``outputs/runs/experiments.csv`` (never deletes prior rows).
        """
        row = {
            "run_name": run_name,
            "timestamp": timestamp,
            "pipeline": pipeline,
            "method": method,
            "split": split,
            "perspective": perspective,
            "mae_mm": metrics.mae_mm if metrics else float("nan"),
            "rmse_mm": metrics.rmse_mm if metrics else float("nan"),
            "n_samples": metrics.n_samples if metrics else 0,
            "predictions_path": str(predictions_path),
            "comparison_path": str(comparison_path) if comparison_path else "",
            "run_dir": str(predictions_path.parent),
        }
        df = pd.DataFrame([row])
        write_header = not self.registry_path.is_file()
        df.to_csv(self.registry_path, mode="a", header=write_header, index=False)
        logger.info("Appended run to registry: %s", self.registry_path)
