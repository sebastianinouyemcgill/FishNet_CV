#!/usr/bin/env python3
"""Export assignment-format predictions.csv for the test split (regression on skeleton)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.config import get_config
from src.submission import export_regression_test_predictions
from src.utils import setup_logging

DEFAULT_MODEL = (
    "outputs/runs/regression_skeleton_validation_lengths2/regression_model.joblib"
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Export test-set predictions.csv")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(DEFAULT_MODEL),
        help="Path to regression_model.joblib",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("predictions.csv"),
        help="Output CSV (image_id with extension, predicted_length_mm)",
    )
    parser.add_argument("--split", default="test", choices=("train", "valid", "test"))
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    setup_logging(level=getattr(logging, args.log_level))
    cfg = get_config()
    path = export_regression_test_predictions(
        output_path=args.output,
        model_path=args.model,
        cfg=cfg,
        split=args.split,
    )
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
