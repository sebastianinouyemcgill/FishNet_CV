"""Tests for validation-set image ID filtering."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from src.config import ProjectConfig
from src.dataset import find_image_path, iterate_image_ids
from src.experiments import load_validation_image_ids


def test_find_image_path_uppercase_jpg(tmp_path: Path) -> None:
    images = tmp_path / "images" / "valid"
    images.mkdir(parents=True)
    (images / "12345.JPG").write_bytes(b"\xff\xd8\xff")
    assert find_image_path(images, "12345") == images / "12345.JPG"


def test_load_validation_image_ids(tmp_path: Path) -> None:
    ann = tmp_path / "data" / "annotations"
    ann.mkdir(parents=True)
    pd.DataFrame({"image_id": ["a", "b"], "true_length_mm": [100.0, 200.0]}).to_csv(
        ann / "validation_lengths.csv", index=False
    )
    cfg = ProjectConfig(repo_root=tmp_path, data_annotations=ann)
    assert load_validation_image_ids(cfg) == ["a", "b"]


def test_iterate_image_ids_skips_missing(tmp_path: Path) -> None:
    import cv2
    import numpy as np

    root = tmp_path / "data" / "fishnet"
    valid_img = root / "images" / "valid"
    valid_lbl = root / "labels" / "valid"
    valid_img.mkdir(parents=True)
    valid_lbl.mkdir(parents=True)
    img_path = valid_img / "exists.JPG"
    cv2.imwrite(str(img_path), np.zeros((10, 10, 3), dtype=np.uint8))
    (valid_lbl / "exists.txt").write_text(
        "2 0.1 0.2 0.3 0.2 0.35 0.4 0.15 0.45\n"
    )
    cfg = ProjectConfig(repo_root=tmp_path)
    samples = list(iterate_image_ids(cfg, "valid", ["exists", "missing"]))
    assert len(samples) == 1
    assert samples[0].image_id == "exists"
