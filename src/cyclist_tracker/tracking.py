"""DeepSORT multi-object tracking with a choice of appearance feature extractors."""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass

import cv2
import numpy as np

from .config import AnalysisSettings
from .detection import Detection

log = logging.getLogger(__name__)

#: Feature extractors ("embedders") for DeepSORT, with a short explanation for the GUI.
EMBEDDERS = {
    "mobilenet": "MobileNetV2 - fast, included (recommended)",
    "torchreid": "OSNet (torchreid) - best tracking in the paper, needs the 'torchreid' extra",
    "clip_ViT-B/32": "CLIP ViT-B/32 - needs the 'clip' extra",
    "histogram": "Colour histogram - no neural network, very fast, lower quality",
}

_EMBEDDER_MODULES = {"torchreid": "torchreid", "clip_ViT-B/32": "clip"}


def embedder_available(name: str) -> bool:
    module = _EMBEDDER_MODULES.get(name)
    return module is None or importlib.util.find_spec(module) is not None


def histogram_embedding(crop_bgr: np.ndarray, bins: int = 8) -> np.ndarray:
    """A normalised 3-D HSV colour histogram, used as a cheap appearance feature."""
    if crop_bgr.size == 0:
        return np.full(bins**3, 1.0 / np.sqrt(bins**3), dtype=np.float32)
    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [bins] * 3, [0, 180, 0, 256, 0, 256]).flatten()
    norm = np.linalg.norm(hist)
    return (hist / norm if norm > 0 else hist).astype(np.float32)


@dataclass
class Track:
    track_id: int
    ltrb: tuple[float, float, float, float]
    confidence: float


class CyclistTracker:
    """Thin wrapper around ``deep_sort_realtime`` returning confirmed tracks only."""

    def __init__(self, settings: AnalysisSettings):
        from deep_sort_realtime.deepsort_tracker import DeepSort

        name = settings.embedder
        if name not in EMBEDDERS:
            raise ValueError(f"Unknown feature extractor {name!r}; choose one of {', '.join(EMBEDDERS)}")
        if not embedder_available(name):
            raise RuntimeError(
                f"The feature extractor {name!r} needs an optional package. Install it with:\n"
                f"    uv sync --extra {_EMBEDDER_MODULES[name]}"
            )
        self._own_embeddings = name == "histogram"
        import torch

        gpu = settings.use_gpu and torch.cuda.is_available()
        params = dict(
            max_age=settings.max_age,
            n_init=settings.n_init,
            max_cosine_distance=settings.max_cosine_distance,
            nn_budget=None,
            embedder=None if self._own_embeddings else name,
            half=settings.half_precision and gpu,
            bgr=True,
            embedder_gpu=gpu,
        )
        if settings.embedder_weights and not self._own_embeddings:
            params["embedder_wts"] = settings.embedder_weights
        self._tracker = DeepSort(**params)
        log.info("DeepSORT tracker ready (feature extractor: %s, GPU: %s)", name, gpu)

    def update(self, detections: list[Detection], frame_bgr: np.ndarray) -> list[Track]:
        raw = [([x1, y1, x2 - x1, y2 - y1], d.score, "cyclist") for d in detections
               for x1, y1, x2, y2 in [d.ltrb]]
        embeds = None
        if self._own_embeddings:
            h, w = frame_bgr.shape[:2]
            embeds = []
            for d in detections:
                x1, y1, x2, y2 = (int(round(v)) for v in d.ltrb)
                embeds.append(histogram_embedding(frame_bgr[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]))
        tracks = self._tracker.update_tracks(raw, embeds=embeds, frame=frame_bgr)
        result = []
        for t in tracks:
            if not t.is_confirmed():
                continue
            conf = t.get_det_conf()
            result.append(Track(int(t.track_id), tuple(float(v) for v in t.to_ltrb()),
                                float(conf) if conf is not None else 1.0))
        return result
