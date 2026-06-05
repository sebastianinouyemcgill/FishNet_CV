"""
Assignment submission CSV export.

Format (test set, no ground truth):
    image_id,predicted_length_mm
    10224.JPG,322.4

``image_id`` is the filename **with extension**, as required for grading.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from src.calibration import calibrate_sample
from src.config import ProjectConfig, get_config
from src.dataset import iterate_dataset
from src.masks import mask_from_class
from src.measurement.features import extract_length_features
from src.models.length_regression import LengthRegressionModel

logger = logging.getLogger(__name__)

SUBMISSION_COLUMNS = ("image_id", "predicted_length_mm")


def export_regression_test_predictions(
    *,
    output_path: Path | str,
    model_path: Path | str,
    cfg: ProjectConfig | None = None,
    split: str = "test",
) -> Path:
    """
    Run skeleton features + saved regression model on a split; write submission CSV.

    Parameters
    ----------
    output_path:
        Destination file (typically repo-root ``predictions.csv``).
    model_path:
        Trained ``regression_model.joblib`` from a regression experiment run.
    """
    cfg = cfg or get_config()
    model_path = Path(model_path)
    output_path = Path(output_path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Regression model not found: {model_path}")

    model = LengthRegressionModel.load(model_path)
    rows: list[dict[str, str | float]] = []

    for sample in tqdm(
        iterate_dataset(cfg, split=split, load_images=True),
        desc=f"Test predictions ({split})",
        unit="img",
    ):
        if sample.image is None:
            logger.warning("Skipping %s: no image", sample.image_path.name)
            continue

        calib = calibrate_sample(sample, cfg=cfg)
        fish_mask = mask_from_class(
            sample.annotations,
            class_name="fish",
            height=sample.image.shape[0],
            width=sample.image.shape[1],
        )
        features = extract_length_features(fish_mask, calib)
        length_mm = float(model.predict(features.as_array().reshape(1, -1))[0])

        rows.append(
            {
                "image_id": sample.image_path.name,
                "predicted_length_mm": length_mm,
            }
        )

    df = pd.DataFrame(rows, columns=list(SUBMISSION_COLUMNS))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Wrote %d submission rows to %s", len(df), output_path)
    return output_path
