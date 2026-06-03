"""Tests for environment-aware path resolution."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config import ProjectConfig, apply_storage_preferences, get_config
from src.paths import PROJECT_ROOT, get_storage_paths, is_colab, resolve_drive_root


def test_project_root_is_repo():
    assert (PROJECT_ROOT / "src").is_dir()


def test_local_storage_uses_legacy_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FISHNET_DRIVE_ROOT", raising=False)
    monkeypatch.delenv("FISHNET_STORAGE_ROOT", raising=False)
    monkeypatch.delenv("FISHNET_FORCE_COLAB", raising=False)
    get_storage_paths.cache_clear()
    sp = get_storage_paths()
    assert not sp.colab
    assert sp.runs_root == PROJECT_ROOT / "outputs" / "runs"
    assert sp.depth_cache_root == PROJECT_ROOT / "data" / "processed" / "depth"


def test_colab_storage_layout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISHNET_FORCE_COLAB", "1")
    get_storage_paths.cache_clear()
    sp = get_storage_paths()
    assert sp.colab
    assert sp.runs_root.name == "runs"
    assert sp.depth_cache_root.name == "depth_maps"
    get_storage_paths.cache_clear()
    monkeypatch.delenv("FISHNET_FORCE_COLAB", raising=False)


def test_drive_root_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("FISHNET_DRIVE_ROOT", str(tmp_path))
    get_storage_paths.cache_clear()
    assert resolve_drive_root() == tmp_path.resolve()
    get_storage_paths.cache_clear()
    monkeypatch.delenv("FISHNET_DRIVE_ROOT", raising=False)


def test_apply_storage_preferences_ephemeral_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FISHNET_FORCE_COLAB", "1")
    get_storage_paths.cache_clear()
    cfg = get_config()
    out = apply_storage_preferences(cfg, save_predictions_to_drive=False)
    assert "fishnet_cv" in str(out.runs_root)
    get_storage_paths.cache_clear()
    monkeypatch.delenv("FISHNET_FORCE_COLAB", raising=False)


def test_with_repo_root_paths(tmp_path: Path) -> None:
    cfg = ProjectConfig.with_repo_root(tmp_path)
    assert cfg.runs_root == tmp_path / "outputs" / "runs"
