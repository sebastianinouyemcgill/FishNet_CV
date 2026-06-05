"""
Guard stored benchmark metrics against silent numerical drift.

Compares against ``outputs/runs/comparison_grid_summary.csv`` and per-run
``metrics.json`` when present. Full re-runs are optional (slow).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.config import ProjectConfig

REPO = Path(__file__).resolve().parents[1]
SUMMARY = REPO / "outputs" / "runs" / "comparison_grid_summary.csv"

# Reference MAE (mm) from validated 2×2 grid (valid split, n=30)
REFERENCE_MAE = {
    ("validation_lengths", False): 51.7,
    ("validation_lengths", True): 22.81,
    ("validation_lengths2", False): 31.24,
    ("validation_lengths2", True): 23.91,
}

TOLERANCE_MM = 0.15


@pytest.mark.skipif(not SUMMARY.is_file(), reason="comparison_grid_summary.csv not found")
def test_comparison_grid_summary_mae_unchanged() -> None:
    import pandas as pd

    df = pd.read_csv(SUMMARY)
    for _, row in df.iterrows():
        key = (row["ground_truth"], bool(row["regression"]))
        expected = REFERENCE_MAE[key]
        assert row["mae_mm"] == pytest.approx(expected, abs=TOLERANCE_MM), (
            f"{row['run_name']}: expected MAE {expected}, got {row['mae_mm']}"
        )


@pytest.mark.parametrize(
    "run_name,expected_mae",
    [
        ("baseline_skeleton_validation_lengths2", 31.24),
        ("regression_skeleton_validation_lengths2", 23.91),
        ("regression_skeleton_validation_lengths", 22.81),
    ],
)
def test_stored_metrics_json_mae(run_name: str, expected_mae: float) -> None:
    path = REPO / "outputs" / "runs" / run_name / "metrics.json"
    if not path.is_file():
        pytest.skip(f"{path} not found")
    data = json.loads(path.read_text())
    mae = data.get("regression_mae_mm", data.get("mae_mm"))
    assert mae == pytest.approx(expected_mae, abs=TOLERANCE_MM)


@pytest.mark.slow
def test_skeleton_baseline_matches_run_inference(tmp_path: Path) -> None:
    """Estimator batch must match legacy run_inference on tiny fixture."""
    import cv2
    import numpy as np
    import pandas as pd

    from src.methods.estimator import FishLengthEstimator
    from src.pipelines.base import run_inference

    root = tmp_path / "data" / "fishnet"
    valid_img = root / "images" / "valid"
    valid_lbl = root / "labels" / "valid"
    valid_img.mkdir(parents=True)
    valid_lbl.mkdir(parents=True)
    cv2.imwrite(str(valid_img / "fish1.JPG"), np.zeros((40, 60, 3), dtype=np.uint8))
    (valid_lbl / "fish1.txt").write_text(
        "0 0.1 0.1 0.2 0.1 0.2 0.2 0.1 0.2\n"
        "1 0.7 0.7 0.8 0.7 0.8 0.8 0.7 0.8\n"
        "2 0.3 0.3 0.7 0.3 0.7 0.6 0.35 0.55\n"
    )
    cfg = ProjectConfig.with_repo_root(tmp_path)
    cfg.measurement_method = "skeleton"
    cfg.use_regression_model = False

    legacy_out = tmp_path / "legacy.csv"
    run_inference(cfg, split="valid", method="skeleton", predictions_path=legacy_out)
    est_out = FishLengthEstimator(cfg).run_batch(split="valid")

    legacy = pd.read_csv(legacy_out)
    est = est_out[["image_id", "predicted_length_mm"]]
    pd.testing.assert_frame_equal(
        legacy.sort_values("image_id").reset_index(drop=True),
        est.sort_values("image_id").reset_index(drop=True),
        check_exact=False,
        rtol=0,
        atol=1e-9,
    )
