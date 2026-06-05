"""
Unified fish length estimation API (presentation / notebook friendly).

Delegates to existing pipeline code paths so baseline numerics stay identical.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from src.calibration import CalibrationResult, calibrate_sample, rectify_image
from src.config import ProjectConfig, get_config
from src.dataset import DatasetSample, iterate_dataset, iterate_image_ids
from src.masks import mask_from_class
from src.measurement import estimate_length_mm
from src.methods.regression import LengthRegressionModel, extract_length_features
from src.pipelines.advanced_inference import _choose_calibration
from src.pipelines.routing import uses_advanced_inference_path

logger = logging.getLogger(__name__)


@dataclass
class LengthEstimate:
    """Single-image length result."""

    image_id: str
    length_mm: float
    method: str
    calibration_source: str = "marker"
    skeleton_length_mm: float | None = None
    features: dict[str, float] | None = None


@dataclass
class FishLengthEstimator:
    """
    Config-driven length estimator.

    Toggles (via ``ProjectConfig``):
    - ``measurement_method`` / ``use_method``: ``skeleton`` | ``pca`` | ``bbox``
    - ``use_grid_calibration``: marker (default) vs grid scale (experimental)
    - ``use_regression_model``: optional sklearn correction
    - ``use_depth_estimation``: experimental (ignored unless enabled + allowed)
    """

    config: ProjectConfig = field(default_factory=get_config)
    _regression_model: LengthRegressionModel | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.config.use_regression_model and self.config.regression_model_path:
            path = Path(self.config.regression_model_path)
            if path.is_file():
                self._regression_model = LengthRegressionModel.load(path)

    @property
    def use_method(self) -> str:
        return self.config.measurement_method

    @property
    def use_calibration(self) -> str:
        return "grid" if self.config.use_grid_auto_calibration else "marker"

    def run(
        self,
        image: np.ndarray,
        fish_mask: np.ndarray,
        *,
        sample: DatasetSample | None = None,
        image_id: str = "",
        calibration: CalibrationResult | None = None,
    ) -> LengthEstimate:
        """
        Estimate length for one image + fish mask.

        If ``sample`` is provided, marker/grid calibration is resolved from annotations.
        """
        calib, scale_source = self._resolve_calibration(image, sample, calibration)
        work_image = image
        work_mask = fish_mask
        if self.config.apply_perspective_correction and calib.homography is not None:
            work_image = rectify_image(image, calib.homography)
            work_mask = mask_from_class(
                sample.annotations if sample else [],
                class_name="fish",
                height=work_image.shape[0],
                width=work_image.shape[1],
            ) if sample else fish_mask

        method = self.use_method
        skeleton_mm = estimate_length_mm(work_mask, calib, method="skeleton")
        length_mm = estimate_length_mm(work_mask, calib, method=method)

        features_dict: dict[str, float] | None = None
        if self._regression_model is not None:
            feat = extract_length_features(work_mask, calib)
            features_dict = feat.as_dict()
            length_mm = float(self._regression_model.predict(feat.as_array().reshape(1, -1))[0])
            method = "regression"

        return LengthEstimate(
            image_id=image_id or (sample.image_id if sample else ""),
            length_mm=length_mm,
            method=method,
            calibration_source=scale_source,
            skeleton_length_mm=skeleton_mm,
            features=features_dict,
        )

    def run_batch(
        self,
        *,
        split: str | None = None,
        image_ids: list[str] | None = None,
        limit: int | None = None,
    ) -> pd.DataFrame:
        """Run on a dataset split; returns predictions DataFrame."""
        split = split or self.config.default_split
        rows: list[dict[str, Any]] = []

        if image_ids is not None:
            iterator: Iterator[DatasetSample] = iterate_image_ids(
                self.config, split=split, image_ids=image_ids, load_images=True
            )
        else:
            iterator = iterate_dataset(self.config, split=split, load_images=True, limit=limit)

        for sample in iterator:
            if sample.image is None:
                continue
            fish_mask = mask_from_class(
                sample.annotations,
                class_name="fish",
                height=sample.image.shape[0],
                width=sample.image.shape[1],
            )
            est = self.run(sample.image, fish_mask, sample=sample, image_id=sample.image_id)
            row: dict[str, Any] = {
                "image_id": est.image_id,
                "predicted_length_mm": est.length_mm,
            }
            if est.skeleton_length_mm is not None:
                row["skeleton_length_mm"] = est.skeleton_length_mm
            if est.features:
                row.update(est.features)
            rows.append(row)

        return pd.DataFrame(rows)

    def _resolve_calibration(
        self,
        image: np.ndarray,
        sample: DatasetSample | None,
        calibration: CalibrationResult | None,
    ) -> tuple[CalibrationResult, str]:
        if calibration is not None:
            return calibration, "provided"

        if sample is None:
            raise ValueError("Either sample or calibration must be provided")

        if uses_advanced_inference_path(self.config) and self.config.use_grid_auto_calibration:
            grid = None
            from src.methods.calibration.grid import estimate_grid_calibration

            grid = estimate_grid_calibration(image, cfg=self.config)
            calib, source, _, _, _ = _choose_calibration(sample, self.config, image, grid=grid)
            return calib, source

        calib = calibrate_sample(sample, cfg=self.config)
        return calib, "marker"
