"""
Advanced pipeline: baseline components plus optional extensions.

Extensions include perspective correction and future ML-based approaches.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from src.config import ProjectConfig, get_config
from src.pipelines.base import run_inference

logger = logging.getLogger(__name__)


@dataclass
class AdvancedPipeline:
    """
    Extensions beyond the assignment baseline.

    Reuses the same inference loop but allows ``perspective=True`` and future hooks.
    """

    name: str = "advanced"

    def configure(
        self,
        cfg: ProjectConfig,
        *,
        method: str,
        split: str,
        perspective: bool = False,
    ) -> ProjectConfig:
        cfg.default_split = split
        cfg.measurement_method = method
        cfg.apply_perspective_correction = perspective
        return cfg

    def run(
        self,
        cfg: ProjectConfig | None = None,
        *,
        method: str = "pca",
        split: str = "test",
        perspective: bool = False,
        predictions_path: Path,
        limit: int | None = None,
        image_ids: list[str] | None = None,
        visualize: bool = False,
        figures_dir: Path | None = None,
    ) -> Path:
        cfg = self.configure(
            cfg or get_config(),
            method=method,
            split=split,
            perspective=perspective,
        )
        logger.info(
            "AdvancedPipeline: split=%s method=%s perspective=%s n_images=%s",
            split,
            method,
            perspective,
            len(image_ids) if image_ids else f"all(limit={limit})",
        )
        return run_inference(
            cfg,
            split=split,
            method=method,
            predictions_path=predictions_path,
            limit=limit,
            image_ids=image_ids,
            visualize=visualize,
            figures_dir=figures_dir,
        )
