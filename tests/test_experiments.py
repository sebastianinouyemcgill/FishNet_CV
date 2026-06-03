"""Tests for RunManager and experiment API."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from src.config import ProjectConfig
from src.evaluation import evaluate_run
from src.experiments.run_manager import RunExistsError, RunManager
from src.experiments import run_experiment


def test_run_manager_auto_name() -> None:
    mgr = RunManager(ProjectConfig.with_repo_root(Path("/tmp/fake")))
    name = mgr.resolve_run_name(None, pipeline="baseline", method="bbox")
    assert name.startswith("baseline_bbox_")


def test_run_manager_exists_raises(tmp_path: Path) -> None:
    cfg = ProjectConfig.with_repo_root(tmp_path)
    mgr = RunManager(cfg)
    run_dir = mgr.prepare_run("test_run", overwrite=False)
    (run_dir / "predictions.csv").write_text("x")
    with pytest.raises(RunExistsError):
        mgr.prepare_run("test_run", overwrite=False)


def test_run_manager_overwrite(tmp_path: Path) -> None:
    cfg = ProjectConfig.with_repo_root(tmp_path)
    mgr = RunManager(cfg)
    run_dir = mgr.prepare_run("test_run", overwrite=False)
    (run_dir / "old.txt").write_text("old")
    mgr.prepare_run("test_run", overwrite=True)
    assert run_dir.is_dir()


def test_evaluate_run_output_dir(tmp_path: Path) -> None:
    gt = tmp_path / "gt.csv"
    pred = tmp_path / "pred.csv"
    out_dir = tmp_path / "run"
    gt.write_text("image_id,length_mm\na,100.0\n")
    pred.write_text("image_id,predicted_length_mm\na,110.0\n")

    metrics, merged = evaluate_run(gt, pred, output_dir=out_dir, method="bbox", split="test")
    assert (out_dir / "comparison.csv").is_file()
    assert (out_dir / "metrics.json").is_file()
    assert metrics.mae_mm == pytest.approx(10.0)
    assert len(merged) == 1


def test_append_registry(tmp_path: Path) -> None:
    cfg = ProjectConfig.with_repo_root(tmp_path)
    mgr = RunManager(cfg)
    from src.evaluation import MetricResult

    metrics = MetricResult(mae_mm=1.0, rmse_mm=2.0, n_samples=3, method="bbox", split="valid")
    pred_path = tmp_path / "outputs" / "runs" / "r1" / "predictions.csv"
    pred_path.parent.mkdir(parents=True)
    pred_path.write_text("image_id,predicted_length_mm\n")
    mgr.append_registry(
        run_name="r1",
        pipeline="baseline",
        method="bbox",
        split="valid",
        perspective=False,
        predictions_path=pred_path,
        comparison_path=pred_path.parent / "comparison.csv",
        metrics=metrics,
        timestamp="2026-01-01T00:00:00Z",
    )
    df = pd.read_csv(mgr.registry_path)
    assert len(df) == 1
    assert df.iloc[0]["run_name"] == "r1"
