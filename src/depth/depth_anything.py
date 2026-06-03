"""
Depth Anything V3 integration with on-disk caching.

Requires optional install::

    pip install torch torchvision
    pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git

See README for full setup.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

# macOS: pycolmap/torch may load duplicate OpenMP runtimes
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np

from src.config import ProjectConfig, get_config
from src.depth.cache import depth_cache_path, load_cached_depth, save_cached_depth
from src.utils import get_logger

logger = get_logger(__name__)

_MODEL: object | None = None
_MODEL_TAG: str | None = None


def _model_tag_from_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace("\\", "_")


class DepthEstimator:
    """
    Lazy-loaded Depth Anything V3 wrapper with per-image NPZ/NPY cache.
    """

    def __init__(self, cfg: ProjectConfig | None = None) -> None:
        self.cfg = cfg or get_config()
        self.model_name = self.cfg.depth_model_name
        self.tag = _model_tag_from_name(self.model_name)
        self._model = None

    def _load_model(self) -> object:
        global _MODEL, _MODEL_TAG
        if _MODEL is not None and _MODEL_TAG == self.tag:
            self._model = _MODEL
            return _MODEL
        try:
            import torch
            from depth_anything_3.api import DepthAnything3
        except ImportError as exc:
            raise ImportError(
                "Depth Anything 3 is not installed. Install PyTorch and the DA3 package:\n"
                "  pip install torch torchvision\n"
                "  pip install git+https://github.com/ByteDance-Seed/Depth-Anything-3.git\n"
                "See README.md section 'Advanced pipeline (depth)'."
            ) from exc

        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
        logger.info("Loading Depth Anything 3 model %s on %s", self.model_name, device)
        model = DepthAnything3.from_pretrained(self.model_name)
        model = model.to(device=device)
        model.eval()
        _MODEL = model
        _MODEL_TAG = self.tag
        self._model = model
        return model

    def predict_depth(
        self,
        image_bgr: np.ndarray,
        *,
        image_id: str,
        split: str,
        use_cache: bool = True,
    ) -> np.ndarray:
        """
        Return H×W float32 depth map aligned with ``image_bgr``.

        Loads from cache when present; otherwise runs DA3 and saves cache.
        """
        cache_path = depth_cache_path(image_id, split, model_tag=self.tag, cfg=self.cfg)
        if use_cache:
            cached = load_cached_depth(cache_path)
            if cached is not None:
                if cached.shape[:2] == image_bgr.shape[:2]:
                    logger.debug("Depth cache hit: %s", cache_path.name)
                    return cached
                cached = cv2.resize(
                    cached,
                    (image_bgr.shape[1], image_bgr.shape[0]),
                    interpolation=cv2.INTER_LINEAR,
                )
                logger.debug("Depth cache hit (resized): %s", cache_path.name)
                return cached.astype(np.float32)

        depth = self._run_inference(image_bgr)
        save_cached_depth(cache_path, depth)
        logger.info("Cached depth map: %s", cache_path)
        return depth

    def _run_inference(self, image_bgr: np.ndarray) -> np.ndarray:
        import torch

        model = self._load_model()
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]

        # DA3 accepts paths, PIL, or numpy; pass list of one RGB image
        with torch.inference_mode():
            prediction = model.inference([rgb])

        depth = np.asarray(prediction.depth, dtype=np.float32)
        if depth.ndim == 3:
            depth = depth[0]
        if depth.shape[:2] != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_LINEAR)
        return depth


def get_depth_estimator(cfg: ProjectConfig | None = None) -> DepthEstimator:
    return DepthEstimator(cfg=cfg)
