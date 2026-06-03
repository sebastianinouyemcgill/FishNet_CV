"""Disk cache for monocular depth maps."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from src.config import ProjectConfig, get_config


def depth_cache_dir(cfg: ProjectConfig | None = None) -> Path:
    cfg = cfg or get_config()
    return cfg.depth_cache_dir()


def depth_cache_path(
    image_id: str,
    split: str,
    *,
    model_tag: str,
    cfg: ProjectConfig | None = None,
) -> Path:
    """``data/processed/depth/<split>/<model_tag>/<image_id>.npy``."""
    base = depth_cache_dir(cfg) / split / model_tag
    base.mkdir(parents=True, exist_ok=True)
    safe_id = image_id.replace("/", "_")
    return base / f"{safe_id}.npy"


def cache_key_for_image(image_path: Path) -> str:
    """Short hash from path + mtime for optional invalidation metadata."""
    stat = image_path.stat()
    payload = f"{image_path.resolve()}:{stat.st_mtime_ns}:{stat.st_size}"
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def load_cached_depth(path: Path) -> np.ndarray | None:
    if not path.is_file():
        return None
    return np.load(path)


def save_cached_depth(path: Path, depth: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, depth.astype(np.float32))
