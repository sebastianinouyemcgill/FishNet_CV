"""Tests for YOLO polygon parsing and sample loading."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.config import ProjectConfig
from src.dataset import discover_image_paths, normalized_image_suffixes, parse_yolo_polygon_line


def test_parse_yolo_polygon_line_normalized() -> None:
    line = "2 0.1 0.2 0.3 0.2 0.35 0.4 0.15 0.45"
    ann = parse_yolo_polygon_line(line, image_width=100, image_height=200)
    assert ann is not None
    assert ann.class_id == 2
    assert ann.class_name == "fish"
    assert ann.coords_pixels is not None
    assert ann.coords_pixels.shape == (4, 2)
    np.testing.assert_allclose(ann.coords_pixels[0], [10.0, 40.0])


def test_parse_empty_line() -> None:
    assert parse_yolo_polygon_line("") is None
    assert parse_yolo_polygon_line("# comment") is None


def test_normalized_image_suffixes() -> None:
    assert normalized_image_suffixes((".JPG", ".jpg", "png")) == frozenset(
        {".jpg", ".png"}
    )


def test_discover_finds_uppercase_jpg(tmp_path: Path) -> None:
    """FishNet files use .JPG; Path.glob('*.jpg') must not be relied on."""
    root = tmp_path / "data" / "fishnet"
    train_images = root / "images" / "train"
    train_labels = root / "labels" / "train"
    train_images.mkdir(parents=True)
    train_labels.mkdir(parents=True)
    (train_images / "fish_upper.JPG").write_bytes(b"\xff\xd8\xff")
    (train_labels / "fish_upper.txt").write_text(
        "2 0.1 0.2 0.3 0.2 0.35 0.4 0.15 0.45\n"
    )

    # Documents the underlying glob pitfall (case-sensitive pattern).
    assert list(train_images.glob("*.jpg")) == []
    assert list(train_images.glob("*.JPG")) == [train_images / "fish_upper.JPG"]

    cfg = ProjectConfig(repo_root=tmp_path)
    paths = discover_image_paths(cfg, "train", recursive=False)
    assert len(paths) == 1
    assert paths[0].name == "fish_upper.JPG"
