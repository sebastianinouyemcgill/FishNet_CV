"""
Helpers for Google Colab notebooks: mount Drive and configure project paths.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from src.paths import PROJECT_ROOT, ensure_storage_dirs, get_storage_paths, is_colab


def mount_google_drive(force_remount: bool = False) -> Path:
    """
    Mount Google Drive in Colab. No-op locally.

    Returns the mounted ``MyDrive`` path (``/content/drive/MyDrive`` in Colab).
    """
    if not is_colab():
        return Path.home()
    from google.colab import drive

    drive.mount("/content/drive", force_remount=force_remount)
    return Path("/content/drive/MyDrive")


def setup_notebook_environment(
    *,
    repo_path: Path | str | None = None,
    mount_drive: bool = True,
    force_remount: bool = False,
) -> tuple[Path, Path]:
    """
    Standard notebook startup: optional Drive mount, ``sys.path``, storage dirs.

    Parameters
    ----------
    repo_path:
        Project root containing ``src/``. When ``None``, inferred from ``Path.cwd()``.
    mount_drive:
        Mount Google Drive when in Colab.
    force_remount:
        Passed to ``drive.mount`` when mounting.

    Returns
    -------
    (repo_root, storage_root)
    """
    if mount_drive and is_colab():
        mount_google_drive(force_remount=force_remount)

    cwd = Path.cwd()
    if repo_path is not None:
        repo = Path(repo_path).expanduser().resolve()
    else:
        repo = cwd if (cwd / "src").is_dir() else cwd.parent

    repo_str = str(repo)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)

    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

    storage = get_storage_paths()
    ensure_storage_dirs(storage)
    return repo, storage.storage_root


def colab_startup_cell_source() -> str:
    """Example Colab cell body (for docs / copy-paste)."""
    return '''\
from src.colab_bootstrap import mount_google_drive, setup_notebook_environment
from src.config import get_config
from src.utils import setup_logging

mount_google_drive()
REPO, STORAGE = setup_notebook_environment()
setup_logging()
cfg = get_config()
print("Colab:", cfg.is_colab, "| storage:", STORAGE)
'''
