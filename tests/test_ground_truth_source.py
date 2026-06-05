"""Tests for configurable ground-truth CSV selection."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import ProjectConfig
from src.experiments.ground_truth import (
    GROUND_TRUTH_SOURCES,
    load_validation_image_ids,
    resolve_ground_truth_path,
)


def test_resolve_validation_lengths2(tmp_path: Path) -> None:
    ann = tmp_path / "data" / "annotations"
    ann.mkdir(parents=True)
    pd.DataFrame({"image_id": ["a"], "true_length_mm": [100.0]}).to_csv(
        ann / "validation_lengths2.csv", index=False
    )
    cfg = ProjectConfig(repo_root=tmp_path, data_annotations=ann, ground_truth_source="validation_lengths2")
    path = resolve_ground_truth_path(cfg)
    assert path == ann / "validation_lengths2.csv"
    assert load_validation_image_ids(cfg) == ["a"]


def test_ground_truth_sources_keys() -> None:
    assert "validation_lengths2" in GROUND_TRUTH_SOURCES
