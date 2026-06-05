"""Tests for notebook configuration helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.experiments.notebook_helpers import (
    AnalyzeResultsConfig,
    RunExperimentsConfig,
    build_experiment_specs,
    filter_summary,
    preview_experiment_specs,
    project_config_for_experiments,
    top_k_errors,
)


def test_build_experiment_grid():
    cfg = RunExperimentsConfig(
        pipelines=["baseline"],
        methods=["bbox", "pca"],
        splits=["valid"],
        validation_set_only=True,
    )
    specs = build_experiment_specs(cfg)
    assert len(specs) == 2
    assert specs[0]["pipeline"] == "baseline"
    assert specs[0]["method"] in ("bbox", "pca")
    assert specs[0]["validation_set_only"] is True


def test_explicit_experiments_override_grid():
    cfg = RunExperimentsConfig(
        pipelines=["baseline"],
        methods=["bbox"],
        experiments=[{"pipeline": "advanced", "method": "skeleton", "run_name": "adv_v1"}],
    )
    specs = build_experiment_specs(cfg)
    assert len(specs) == 1
    assert specs[0]["run_name"] == "adv_v1"


def test_advanced_flags_in_spec():
    cfg = RunExperimentsConfig(
        pipelines=["advanced"],
        methods=["skeleton"],
        splits=["valid"],
        use_depth_estimation=True,
        use_grid_auto_calibration=True,
    )
    spec = build_experiment_specs(cfg)[0]
    assert spec["use_depth_estimation"] is True
    assert spec["use_grid_auto_calibration"] is True


def test_preview_dataframe():
    cfg = RunExperimentsConfig(methods=["bbox"], pipelines=["baseline"], splits=["valid"])
    df = preview_experiment_specs(build_experiment_specs(cfg))
    assert "run_name" in df.columns
    assert len(df) == 1


def test_preview_regression_calibration_flag():
    cfg = RunExperimentsConfig(
        pipelines=["advanced"],
        methods=["skeleton"],
        splits=["valid"],
        run_regression_calibration=True,
    )
    df = preview_experiment_specs(build_experiment_specs(cfg), cfg)
    assert len(df) == 2
    reg = df[df["pipeline"] == "regression"].iloc[0]
    adv = df[df["pipeline"] == "advanced"].iloc[0]
    assert bool(reg["train_regression"])
    assert not bool(adv["train_regression"])
    assert not bool(adv["apply_saved_model"])


def test_filter_summary_latest_per_run():
    df = pd.DataFrame(
        [
            {"run_name": "a", "timestamp": "2026-01-01", "mae_mm": 10},
            {"run_name": "a", "timestamp": "2026-01-02", "mae_mm": 5},
            {"run_name": "b", "timestamp": "2026-01-01", "mae_mm": 8},
        ]
    )
    out = filter_summary(df, run_names=["a"])
    assert len(out) == 1
    assert out.iloc[0]["mae_mm"] == 5


def test_run_experiments_config_storage_flags_default_true():
    cfg = RunExperimentsConfig()
    assert cfg.cache_results is True
    assert cfg.save_predictions_to_drive is True
    assert project_config_for_experiments(cfg).cache_depth_maps is True


def test_top_k_errors():
    df = pd.DataFrame({"image_id": ["1", "2", "3"], "abs_error_mm": [1.0, 5.0, 2.0]})
    worst = top_k_errors(df, k=2, worst=True)
    assert worst.iloc[0]["image_id"] == "2"
    best = top_k_errors(df, k=1, worst=False)
    assert best.iloc[0]["image_id"] == "1"
