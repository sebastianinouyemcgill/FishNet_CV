"""
Unified storage paths for local development and Google Colab + Drive.

Local execution keeps the existing ``outputs/`` and ``data/processed/`` layout.
Colab uses ``/content/drive/MyDrive/UH_CV/`` with the Drive folder structure.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Final

# Repository root (parent of ``src/``) — code and tests stay here.
PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

# Default Google Drive project folder name.
DRIVE_FOLDER_NAME: Final[str] = "UH_CV"

# Colab default mount path for Drive-backed storage.
COLAB_DRIVE_ROOT: Final[Path] = Path("/content/drive/MyDrive") / DRIVE_FOLDER_NAME


def is_colab() -> bool:
    """Return True when running inside a Google Colab runtime."""
    if os.environ.get("FISHNET_FORCE_COLAB", "").lower() in ("1", "true", "yes"):
        return True
    try:
        import google.colab  # noqa: F401

        return True
    except ImportError:
        return False


def resolve_drive_root() -> Path:
    """
    Root for persistent data, runs, figures, cache, and exports.

    Colab: ``/content/drive/MyDrive/UH_CV`` (after Drive is mounted).
    Local: ``PROJECT_ROOT`` with legacy subpaths (``outputs/runs``, etc.).
    Override with ``FISHNET_DRIVE_ROOT`` or ``FISHNET_STORAGE_ROOT``.
    """
    for key in ("FISHNET_DRIVE_ROOT", "FISHNET_STORAGE_ROOT"):
        if raw := os.environ.get(key):
            return Path(raw).expanduser().resolve()
    if is_colab():
        return COLAB_DRIVE_ROOT
    return PROJECT_ROOT


@dataclass(frozen=True)
class StoragePaths:
    """Resolved storage directories for the active environment."""

    storage_root: Path
    colab: bool

    @property
    def data_root(self) -> Path:
        return self.storage_root / "data"

    @property
    def data_raw(self) -> Path:
        return self.data_root / "raw"

    @property
    def data_annotations(self) -> Path:
        return self.data_root / "annotations"

    @property
    def data_fishnet(self) -> Path:
        return self.data_root / "fishnet"

    @property
    def runs_root(self) -> Path:
        if self.colab:
            return self.storage_root / "runs"
        return self.storage_root / "outputs" / "runs"

    @property
    def figures_root(self) -> Path:
        if self.colab:
            return self.storage_root / "figures"
        return self.storage_root / "outputs" / "figures"

    @property
    def figures_debug(self) -> Path:
        return self.figures_root / "debug"

    @property
    def figures_analysis(self) -> Path:
        return self.figures_root / "analysis"

    @property
    def cache_root(self) -> Path:
        if self.colab:
            return self.storage_root / "cache"
        return self.storage_root / "data" / "processed"

    @property
    def depth_cache_root(self) -> Path:
        if self.colab:
            return self.cache_root / "depth_maps"
        return self.cache_root / "depth"

    @property
    def intermediate_cache_root(self) -> Path:
        if self.colab:
            return self.cache_root / "intermediate_outputs"
        return self.cache_root

    @property
    def logs_root(self) -> Path:
        if self.colab:
            return self.storage_root / "logs"
        return self.storage_root / "outputs" / "metrics"

    @property
    def exports_root(self) -> Path:
        return self.storage_root / "exports"

    @property
    def exports_predictions(self) -> Path:
        return self.exports_root / "predictions"

    @property
    def exports_final_results(self) -> Path:
        return self.exports_root / "final_results"

    @property
    def legacy_outputs_predictions(self) -> Path:
        """Flat legacy predictions path (local: ``outputs/predictions``)."""
        if self.colab:
            return self.exports_predictions
        return self.storage_root / "outputs" / "predictions"

    def ephemeral_runs_root(self) -> Path:
        """Colab runtime-only runs (not synced to Drive)."""
        return Path(tempfile.gettempdir()) / "fishnet_cv" / "runs"

    def ephemeral_cache_root(self) -> Path:
        return Path(tempfile.gettempdir()) / "fishnet_cv" / "cache"


@lru_cache(maxsize=1)
def get_storage_paths() -> StoragePaths:
    """Cached storage layout for the current process."""
    root = resolve_drive_root()
    return StoragePaths(storage_root=root, colab=is_colab())


def ensure_storage_dirs(paths: StoragePaths | None = None) -> None:
    """Create standard storage directories if missing."""
    p = paths or get_storage_paths()
    for directory in (
        p.data_root,
        p.data_raw,
        p.data_annotations,
        p.data_fishnet,
        p.runs_root,
        p.figures_root,
        p.figures_debug,
        p.figures_analysis,
        p.depth_cache_root,
        p.intermediate_cache_root,
        p.logs_root,
        p.exports_predictions,
        p.exports_final_results,
        p.legacy_outputs_predictions,
    ):
        directory.mkdir(parents=True, exist_ok=True)


# Backwards-compatible aliases used in docs and notebooks.
DRIVE_ROOT = COLAB_DRIVE_ROOT
REPO_ROOT = PROJECT_ROOT
