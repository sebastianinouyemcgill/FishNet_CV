"""Pipeline registry for baseline and advanced runners."""

from __future__ import annotations

from typing import Protocol

from src.pipelines.advanced import AdvancedPipeline
from src.pipelines.baseline import BaselinePipeline

_PIPELINES = {
    "baseline": BaselinePipeline(),
    "advanced": AdvancedPipeline(),
}


class Pipeline(Protocol):
    name: str


def get_pipeline(name: str) -> BaselinePipeline | AdvancedPipeline:
    """Return a pipeline instance by name."""
    key = name.lower().strip()
    if key not in _PIPELINES:
        raise ValueError(
            f"Unknown pipeline {name!r}. Choose from: {list(_PIPELINES.keys())}"
        )
    return _PIPELINES[key]


def list_pipelines() -> list[str]:
    """Registered pipeline names."""
    return list(_PIPELINES.keys())
