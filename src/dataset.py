"""
Dataset loading: YOLO polygon labels, splits, and sample iteration.

Expected layout (under dataset root from ``config.resolve_dataset_root()``)::

    images/{train,valid,test}/*.jpg
    labels/{train,valid,test}/*.txt

Each label line: ``class_id x1 y1 x2 y2 ...`` with normalized polygon coordinates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np

from src.config import CLASS_NAMES, ProjectConfig, get_config
from src.utils import get_logger, normalize_polygon

logger = get_logger(__name__)


def normalized_image_suffixes(extensions: tuple[str, ...]) -> frozenset[str]:
    """
    Canonical lowercase suffixes (with leading dot) for case-insensitive matching.

    ``Path.glob("*.jpg")`` does not match ``file.JPG``; discovery compares
    ``path.suffix.lower()`` against this set instead.
    """
    out: set[str] = set()
    for ext in extensions:
        e = ext.lower()
        if not e.startswith("."):
            e = f".{e}"
        out.add(e)
    return frozenset(out)


def _is_image_file(path: Path, allowed_suffixes: frozenset[str]) -> bool:
    return path.is_file() and path.suffix.lower() in allowed_suffixes


@dataclass(frozen=True)
class PolygonAnnotation:
    """Single YOLO polygon instance."""

    class_id: int
    class_name: str
    coords_normalized: tuple[float, ...]  # x1,y1,x2,y2,...
    coords_pixels: np.ndarray | None = None  # (N, 2), filled when image size known

    @property
    def num_vertices(self) -> int:
        return len(self.coords_normalized) // 2


@dataclass
class DatasetSample:
    """
    One image and its parsed annotations.

    Attributes
    ----------
    image_id:
        Filename stem (unique within a split).
    image_path:
        Absolute path to the RGB image.
    label_path:
        Path to the corresponding ``.txt`` label file.
    split:
        Dataset split name.
    image:
        BGR image array, loaded on demand via ``load_sample``.
    height, width:
        Image dimensions in pixels.
    annotations:
        All polygon instances in the label file.
    """

    image_id: str
    image_path: Path
    label_path: Path
    split: str
    height: int = 0
    width: int = 0
    image: np.ndarray | None = None
    annotations: list[PolygonAnnotation] = field(default_factory=list)

    def fish_annotations(self) -> list[PolygonAnnotation]:
        """Return annotations with class ``fish``."""
        return [a for a in self.annotations if a.class_name == "fish"]

    def blue_annotations(self) -> list[PolygonAnnotation]:
        return [a for a in self.annotations if a.class_name == "blue"]

    def yellow_annotations(self) -> list[PolygonAnnotation]:
        return [a for a in self.annotations if a.class_name == "yellow"]


def _class_name(class_id: int) -> str:
    return CLASS_NAMES.get(class_id, f"class_{class_id}")


def parse_yolo_polygon_line(
    line: str,
    image_width: int | None = None,
    image_height: int | None = None,
) -> PolygonAnnotation | None:
    """
    Parse one line from a YOLO segmentation label file.

    Returns ``None`` for empty or comment lines.
    """
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    parts = line.split()
    if len(parts) < 7:  # class + at least 3 vertices (6 coords)
        logger.warning("Skipping short label line: %s", line[:80])
        return None
    class_id = int(float(parts[0]))
    coords = tuple(float(x) for x in parts[1:])
    coords_pixels = None

    if image_width is not None and image_height is not None:
        coords_pixels = normalize_polygon(coords, image_width, image_height)

    ann = PolygonAnnotation(
        class_id=class_id,
        class_name=_class_name(class_id),
        coords_normalized=coords,
        coords_pixels=coords_pixels,
    )

    return ann


def load_image(path: Path) -> np.ndarray:
    """Load a BGR image with OpenCV; raises ``FileNotFoundError`` if missing."""
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Failed to decode image: {path}")
    return image


def load_labels(
    label_path: Path,
    image_width: int | None = None,
    image_height: int | None = None,
) -> list[PolygonAnnotation]:
    """Load all polygon annotations from a ``.txt`` file."""
    if not label_path.is_file():
        logger.warning("Label file missing: %s", label_path)
        return []
    annotations: list[PolygonAnnotation] = []
    with label_path.open(encoding="utf-8") as f:
        for line in f:
            ann = parse_yolo_polygon_line(line, image_width, image_height)
            if ann is not None:
                annotations.append(ann)
    return annotations


def load_sample(
    image_path: Path,
    label_path: Path | None = None,
    split: str = "test",
    load_image_array: bool = True,
) -> DatasetSample:
    """
    Load one dataset sample (image + labels).

    If ``label_path`` is omitted, looks for ``labels/<split>/<stem>.txt``
    beside the dataset root inferred from ``image_path``.
    """
    image_id = image_path.stem
    if label_path is None:
        # Heuristic: .../images/<split>/file.jpg -> .../labels/<split>/file.txt
        label_path = image_path.parent.parent.parent / "labels" / image_path.parent.name / (
            f"{image_id}.txt"
        )

    sample = DatasetSample(
        image_id=image_id,
        image_path=image_path,
        label_path=label_path,
        split=split,
    )

    if load_image_array:
        sample.image = load_image(image_path)
        sample.height, sample.width = sample.image.shape[:2]
        sample.annotations = load_labels(label_path, sample.width, sample.height)
    else:
        sample.annotations = load_labels(label_path)

    return sample


def find_image_path(images_dir: Path, image_id: str) -> Path | None:
    """
    Resolve an image file under ``images_dir`` for ``image_id`` (any configured extension).

    Matching is case-insensitive on the suffix (FishNet uses ``.JPG``).
    """
    if not images_dir.is_dir():
        return None
    for path in images_dir.iterdir():
        if path.is_file() and path.stem == image_id:
            return path
    # Fallback: glob stem match
    matches = [p for p in images_dir.glob(f"{image_id}.*") if p.is_file()]
    return matches[0] if matches else None


def iterate_image_ids(
    cfg: ProjectConfig,
    split: str,
    image_ids: list[str],
    load_images: bool = True,
) -> Iterator[DatasetSample]:
    """
    Yield samples for an explicit list of ``image_id`` values (e.g. validation CSV).

    Skips IDs with no matching image file and logs a warning.
    """
    images_dir = cfg.images_dir(split)
    labels_dir = cfg.labels_dir(split)
    for image_id in image_ids:
        image_path = find_image_path(images_dir, str(image_id))
        if image_path is None:
            logger.warning("No image for image_id=%s in %s", image_id, images_dir)
            continue
        label_path = labels_dir / f"{image_path.stem}.txt"
        yield load_sample(
            image_path,
            label_path=label_path,
            split=split,
            load_image_array=load_images,
        )


def discover_image_paths(
    cfg: ProjectConfig,
    split: str,
    recursive: bool = True,
) -> list[Path]:
    """
    List image paths for a split, optionally searching subfolders.

    Matching is **case-insensitive** on the file suffix (e.g. ``.JPG`` matches
    configured ``.jpg``). Uses directory iteration, not extension-specific globs.
    """
    dataset_root = cfg.resolve_dataset_root()
    images_dir = cfg.images_dir(split)
    allowed = normalized_image_suffixes(cfg.image_extensions)

    logger.info("Dataset discovery for split=%r", split)
    logger.info("  resolved dataset root: %s", dataset_root)
    logger.info("  images_dir: %s (exists=%s)", images_dir, images_dir.is_dir())
    logger.info("  allowed suffixes (case-insensitive): %s", sorted(allowed))
    logger.info("  recursive scan: %s", recursive)

    if not images_dir.is_dir():
        logger.error("Images directory does not exist: %s", images_dir)
        return []

    per_suffix: dict[str, int] = {s: 0 for s in allowed}
    paths: list[Path] = []

    if recursive:
        candidates = images_dir.rglob("*")
    else:
        candidates = images_dir.iterdir()

    for path in candidates:
        if not _is_image_file(path, allowed):
            continue
        paths.append(path)
        per_suffix[path.suffix.lower()] = per_suffix.get(path.suffix.lower(), 0) + 1

    for suffix in sorted(allowed):
        logger.info("  matches for suffix %s: %d", suffix, per_suffix.get(suffix, 0))

    paths = sorted(set(paths))
    logger.info(
        "Found %d images in %s (%s)",
        len(paths),
        images_dir,
        split,
    )
    return paths


def iterate_dataset(
    cfg: ProjectConfig | None = None,
    split: str | None = None,
    load_images: bool = True,
    limit: int | None = None,
) -> Iterator[DatasetSample]:
    """
    Yield ``DatasetSample`` objects for all images in a split.

    Parameters
    ----------
    cfg:
        Project configuration; uses ``get_config()`` if None.
    split:
        ``train``, ``valid``, or ``test``.
    load_images:
        If False, only metadata and normalized labels are loaded.
    limit:
        Optional cap for debugging.
    """
    cfg = cfg or get_config()
    split = split or cfg.default_split
    labels_dir = cfg.labels_dir(split)

    for i, image_path in enumerate(discover_image_paths(cfg, split)):
        if limit is not None and i >= limit:
            break
        label_path = labels_dir / f"{image_path.stem}.txt"
        yield load_sample(
            image_path,
            label_path=label_path,
            split=split,
            load_image_array=load_images,
        )


def build_split_index(cfg: ProjectConfig | None = None) -> dict[str, list[str]]:
    """
  Build a mapping ``split -> [image_id, ...]`` for train/valid/test.

    TODO: Persist split manifest to ``data/processed/splits.json`` if you
    need reproducible custom splits beyond the provided folders.
    """
    cfg = cfg or get_config()
    index: dict[str, list[str]] = {}
    for split in ("train", "valid", "test"):
        index[split] = [p.stem for p in discover_image_paths(cfg, split, recursive=True)]
    return index
