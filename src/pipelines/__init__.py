"""Pipeline abstractions for baseline and advanced fish-length estimation."""

from src.pipelines.advanced import AdvancedPipeline
from src.pipelines.baseline import BaselinePipeline
from src.pipelines.registry import get_pipeline, list_pipelines

__all__ = [
    "AdvancedPipeline",
    "BaselinePipeline",
    "get_pipeline",
    "list_pipelines",
]
