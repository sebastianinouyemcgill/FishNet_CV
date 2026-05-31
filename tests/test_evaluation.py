"""Tests for evaluation metrics."""

from __future__ import annotations

import pandas as pd
import pytest

from src.evaluation import compare_predictions, compute_metrics, load_ground_truth_csv, mae, rmse


def test_mae_rmse() -> None:
    y = [10.0, 20.0, 30.0]
    p = [12.0, 18.0, 30.0]

    assert mae(y, p) == pytest.approx(4 / 3)
    assert rmse(y, p) == pytest.approx((8 / 3) ** 0.5)


def test_compare_and_compute_metrics() -> None:
    gt = pd.DataFrame({"image_id": ["a", "b"], "length_mm": [100.0, 200.0]})
    pred = pd.DataFrame({"image_id": ["a", "b"], "predicted_length_mm": [110.0, 190.0]})
    merged = compare_predictions(gt, pred)
    assert "abs_error_mm" in merged.columns
    metrics = compute_metrics(gt, pred)
    assert metrics.n_samples == 2
    assert metrics.mae_mm == pytest.approx(10.0)


def test_load_ground_truth_converts_cm_to_mm(tmp_path) -> None:
    from pathlib import Path

    csv_path = tmp_path / "validation_lengths.csv"
    pd.DataFrame({"image_id": ["1"], "true_length_mm": [50.0]}).to_csv(csv_path, index=False)
    df = load_ground_truth_csv(csv_path)
    assert df.iloc[0]["length_mm"] == pytest.approx(500.0)
