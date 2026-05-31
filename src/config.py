"""
Central configuration for paths, class IDs, and calibration constants.

Adjust ``ProjectConfig`` fields or override via environment variables where noted.
See ``assignment.pdf`` for dataset-specific conventions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

# Repository root (parent of ``src/``)
REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

# Default YOLO class IDs (see data/fishnet.yaml when present)
CLASS_BLUE: Final[int] = 0
CLASS_YELLOW: Final[int] = 1
CLASS_FISH: Final[int] = 2

CLASS_NAMES: Final[dict[int, str]] = {
    CLASS_BLUE: "blue",
    CLASS_YELLOW: "yellow",
    CLASS_FISH: "fish",
}

# Physical size of calibration rectangles (mm), per assignment spec
CALIBRATION_RECT_MM: Final[float] = 100.0


@dataclass
class ProjectConfig:
    """
    Runtime configuration for dataset I/O, calibration, and outputs.

    Attributes
    ----------
    repo_root:
        Project root directory.
    data_raw:
        Raw images and YOLO labels (e.g. extracted ``fishnet/`` archive).
    data_processed:
        Cached masks, rectified images, etc.
    data_annotations:
        Ground-truth CSVs and manual measurements.
    outputs_figures:
        Saved plots for reports.
    outputs_predictions:
        ``predictions.csv`` and per-run outputs.
    outputs_metrics:
        Evaluation tables and experiment logs.
    calibration_rect_mm:
        Known physical length of blue/yellow calibration rectangles.
    default_split:
        Default dataset split for training scripts.
    measurement_method:
        Default length method: ``bbox``, ``pca``, or ``skeleton``.
    apply_perspective_correction:
        Whether to rectify images using homography from markers.
    image_extensions:
        Filename suffixes treated as images when scanning directories.
    """

    repo_root: Path = field(default_factory=lambda: REPO_ROOT)

    # Prefer ``data/raw/fishnet`` if you symlink/copy the archive there;
    # falls back to ``data/fishnet`` for the bundled layout.
    data_raw: Path = field(default_factory=lambda: REPO_ROOT / "data" / "raw")
    data_processed: Path = field(default_factory=lambda: REPO_ROOT / "data" / "processed")
    data_annotations: Path = field(default_factory=lambda: REPO_ROOT / "data" / "annotations")

    outputs_figures: Path = field(default_factory=lambda: REPO_ROOT / "outputs" / "figures")
    outputs_predictions: Path = field(default_factory=lambda: REPO_ROOT / "outputs" / "predictions")
    outputs_metrics: Path = field(default_factory=lambda: REPO_ROOT / "outputs" / "metrics")

    calibration_rect_mm: float = CALIBRATION_RECT_MM
    default_split: str = "test"
    measurement_method: str = "bbox"
    apply_perspective_correction: bool = False

    # Matched case-insensitively in dataset.discover_image_paths (FishNet uses ".JPG").
    image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    def resolve_dataset_root(self) -> Path:
        """
        Return the directory containing ``images/`` and ``labels/`` subtrees.

        Checks, in order:
        1. ``data/raw/fishnet``
        2. ``data/fishnet`` (legacy / in-repo extract)
        3. ``data/raw``
        """
        candidates = [
            self.data_raw / "fishnet",
            self.repo_root / "data" / "fishnet",
            self.data_raw,
        ]
        for path in candidates:
            if (path / "images").is_dir() and (path / "labels").is_dir():
                return path
        # TODO: Raise a clearer error listing expected layout once data is required
        return candidates[0]

    def images_dir(self, split: str) -> Path:
        """Path to images for a split (``train``, ``valid``, ``test``)."""
        root = self.resolve_dataset_root()
        # YOLO layouts sometimes use ``val`` instead of ``valid``
        split_aliases = {"valid": ("valid", "val"), "val": ("val", "valid")}
        names = split_aliases.get(split, (split,))
        for name in names:
            candidate = root / "images" / name
            if candidate.is_dir():
                return candidate
        return root / "images" / split

    def labels_dir(self, split: str) -> Path:
        """Path to YOLO label ``.txt`` files for a split."""
        root = self.resolve_dataset_root()
        split_aliases = {"valid": ("valid", "val"), "val": ("val", "valid")}
        names = split_aliases.get(split, (split,))
        for name in names:
            candidate = root / "labels" / name
            if candidate.is_dir():
                return candidate
        return root / "labels" / split

    def ensure_output_dirs(self) -> None:
        """Create output directories if they do not exist."""
        for path in (
            self.data_processed,
            self.data_annotations,
            self.outputs_figures,
            self.outputs_predictions,
            self.outputs_metrics,
        ):
            path.mkdir(parents=True, exist_ok=True)


def get_config() -> ProjectConfig:
    """
    Build configuration, optionally overriding split/method from environment.

    Environment variables:
    - ``FISHNET_SPLIT``: train | valid | test
    - ``FISHNET_METHOD``: bbox | pca | skeleton
    - ``FISHNET_PERSPECTIVE``: 1 / true to enable homography rectification
    """
    cfg = ProjectConfig()
    if split := os.environ.get("FISHNET_SPLIT"):
        cfg.default_split = split
    if method := os.environ.get("FISHNET_METHOD"):
        cfg.measurement_method = method
    if os.environ.get("FISHNET_PERSPECTIVE", "").lower() in ("1", "true", "yes"):
        cfg.apply_perspective_correction = True
    return cfg
