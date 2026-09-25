"""Cyclist detectors.

Three detectors share one interface (``detect(image_rgb, frame_name) -> list[Detection]``):

``DetrDetector``
    A DETR model fine-tuned on cyclists (a Lightning ``.ckpt`` file from training).
``CocoDetrDetector``
    The public COCO-trained DETR without fine-tuning. A cyclist is a *bicycle*
    box merged with the *person* riding it. Useful to try the tool on your own
    video before you have annotated and trained anything.
``AnnotationDetector``
    Uses the boxes from a COCO annotation file as "perfect" detections. This
    isolates tracking/counting/speed errors from detector errors, and makes it
    possible to test the whole pipeline without a GPU or a trained model.
"""

from __future__ import annotations

import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)

BASE_MODEL = "facebook/detr-resnet-50"
COCO_PERSON, COCO_BICYCLE = 1, 2


@dataclass
class Detection:
    ltrb: tuple[float, float, float, float]
    score: float


class Detector(Protocol):
    def detect(self, image_rgb: np.ndarray, frame_name: str) -> list[Detection]: ...


def resolve_device(prefer_gpu: bool = True):
    import torch

    if prefer_gpu and torch.cuda.is_available():
        return torch.device("cuda")
    if prefer_gpu and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# --------------------------------------------------------------------------- checkpoints
_RENAMES = [
    # transformers < 4.40 wrapped the backbone in a "conv_encoder"
    (re.compile(r"\.backbone\.conv_encoder\.model\."), ".backbone.model."),
    # transformers 5 renamed attention output projections and moved the MLP
    (re.compile(r"\.(self_attn|encoder_attn)\.out_proj\."), r".\1.o_proj."),
    (re.compile(r"(\.layers\.\d+)\.fc([12])\."), r"\1.mlp.fc\2."),
]
_REVERSE_RENAMES = [
    (re.compile(r"\.backbone\.model\."), ".backbone.conv_encoder.model."),
    (re.compile(r"\.(self_attn|encoder_attn)\.o_proj\."), r".\1.out_proj."),
    (re.compile(r"(\.layers\.\d+)\.mlp\.fc([12])\."), r"\1.fc\2."),
]


def adapt_state_dict(state_dict: dict, expected_keys: set[str]) -> dict:
    """Rename checkpoint keys so that checkpoints from other ``transformers`` versions load.

    Keys that already match are kept; others are renamed in whichever direction
    makes them match ``expected_keys``.
    """
    adapted = {}
    for key, value in state_dict.items():
        new_key = key
        if key not in expected_keys:
            for renames in (_RENAMES, _REVERSE_RENAMES):
                candidate = key
                for pattern, replacement in renames:
                    candidate = pattern.sub(replacement, candidate)
                if candidate in expected_keys:
                    new_key = candidate
                    break
        adapted[new_key] = value
    return adapted


def default_cyclist_label(num_labels: int) -> int:
    """Class index of "cyclist" in a fine-tuned model.

    Models trained with a single class use index 0. The model of the paper was
    trained with two classes, where cyclists have index 1.
    """
    return 0 if num_labels == 1 else 1


def load_detr_checkpoint(checkpoint_path: str | Path, device=None):
    """Load a fine-tuned DETR from a Lightning checkpoint, independent of the Lightning class.

    Returns ``(model, num_labels)``.
    """
    import torch
    from transformers import DetrConfig, DetrForObjectDetection

    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    state = checkpoint.get("state_dict", checkpoint)
    state = {k.removeprefix("model."): v for k, v in state.items()}  # strip the Lightning wrapper
    classifier = state.get("class_labels_classifier.weight")
    if classifier is None:
        raise ValueError(f"{checkpoint_path} does not look like a DETR checkpoint (no classifier weights).")
    num_labels = classifier.shape[0] - 1  # the last output is "no object"

    config = DetrConfig.from_pretrained(BASE_MODEL, num_labels=num_labels)
    if hasattr(config, "use_pretrained_backbone"):
        config.use_pretrained_backbone = False  # all weights come from the checkpoint
    model = DetrForObjectDetection(config)
    state = adapt_state_dict(state, set(model.state_dict()))
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"Checkpoint does not match the DETR architecture: {len(missing)} missing and "
            f"{len(unexpected)} unexpected weights (e.g. {(missing or unexpected)[:3]})."
        )
    model.to(device or resolve_device()).eval()
    return model, num_labels


# --------------------------------------------------------------------------- detectors
class _DetrBase:
    def __init__(self, model, device, threshold: float):
        from transformers import DetrImageProcessor

        self.model = model
        self.device = device
        self.threshold = threshold
        self.processor = DetrImageProcessor.from_pretrained(BASE_MODEL)

    def _predict(self, image_rgb: np.ndarray):
        import torch

        inputs = self.processor(images=image_rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        height, width = image_rgb.shape[:2]
        return self.processor.post_process_object_detection(
            outputs, target_sizes=torch.tensor([[height, width]], device=self.device), threshold=self.threshold
        )[0]


class DetrDetector(_DetrBase):
    """Fine-tuned DETR checkpoint."""

    def __init__(self, checkpoint: str | Path, threshold: float = 0.7, cyclist_label: int | None = None,
                 prefer_gpu: bool = True):
        device = resolve_device(prefer_gpu)
        model, num_labels = load_detr_checkpoint(checkpoint, device)
        super().__init__(model, device, threshold)
        self.cyclist_label = default_cyclist_label(num_labels) if cyclist_label is None else cyclist_label
        log.info("Loaded DETR checkpoint %s (%d class(es), cyclist = class %d) on %s",
                 checkpoint, num_labels, self.cyclist_label, device)

    def detect(self, image_rgb: np.ndarray, frame_name: str = "") -> list[Detection]:
        result = self._predict(image_rgb)
        return [
            Detection(tuple(box.tolist()), float(score))
            for score, label, box in zip(result["scores"], result["labels"], result["boxes"], strict=True)
            if int(label) == self.cyclist_label
        ]


def merge_riders(persons: list[Detection], bicycles: list[Detection], min_overlap: float = 0.3) -> list[Detection]:
    """Turn bicycle boxes into cyclist boxes by joining each with the person riding it.

    A person rides a bicycle when their boxes overlap by at least ``min_overlap``
    of the bicycle area. Bicycles without a rider are kept as they are.
    """
    cyclists = []
    for bike in bicycles:
        bx1, by1, bx2, by2 = bike.ltrb
        bike_area = max(1e-6, (bx2 - bx1) * (by2 - by1))
        best, best_overlap = None, min_overlap
        for person in persons:
            px1, py1, px2, py2 = person.ltrb
            inter = max(0.0, min(bx2, px2) - max(bx1, px1)) * max(0.0, min(by2, py2) - max(by1, py1))
            if inter / bike_area >= best_overlap:
                best, best_overlap = person, inter / bike_area
        if best is None:
            cyclists.append(bike)
        else:
            px1, py1, px2, py2 = best.ltrb
            cyclists.append(Detection((min(bx1, px1), min(by1, py1), max(bx2, px2), max(by2, py2)),
                                      max(bike.score, best.score)))
    return cyclists


class CocoDetrDetector(_DetrBase):
    """Off-the-shelf COCO DETR (no training needed): bicycle + rider = cyclist."""

    def __init__(self, threshold: float = 0.7, prefer_gpu: bool = True):
        from transformers import DetrForObjectDetection

        device = resolve_device(prefer_gpu)
        model = DetrForObjectDetection.from_pretrained(BASE_MODEL).to(device).eval()
        super().__init__(model, device, threshold)
        log.info("Loaded COCO-pretrained DETR (%s) on %s", BASE_MODEL, device)

    def detect(self, image_rgb: np.ndarray, frame_name: str = "") -> list[Detection]:
        result = self._predict(image_rgb)
        persons, bicycles = [], []
        for score, label, box in zip(result["scores"], result["labels"], result["boxes"], strict=True):
            det = Detection(tuple(box.tolist()), float(score))
            if int(label) == COCO_PERSON:
                persons.append(det)
            elif int(label) == COCO_BICYCLE:
                bicycles.append(det)
        return merge_riders(persons, bicycles)


class AnnotationDetector:
    """Returns the annotated boxes of a frame (COCO file) as detections with score 1."""

    def __init__(self, coco_json: str | Path):
        data = json.loads(Path(coco_json).read_text())
        names = {img["id"]: Path(img["file_name"]).name for img in data.get("images", [])}
        self.boxes: dict[str, list[Detection]] = defaultdict(list)
        for ann in data.get("annotations", []):
            x, y, w, h = ann["bbox"]
            if w > 0 and h > 0 and ann["image_id"] in names:
                self.boxes[names[ann["image_id"]]].append(Detection((x, y, x + w, y + h), 1.0))
        log.info("Loaded %d annotated boxes for %d frames from %s",
                 sum(map(len, self.boxes.values())), len(self.boxes), coco_json)

    def detect(self, image_rgb: np.ndarray, frame_name: str) -> list[Detection]:
        return list(self.boxes.get(Path(frame_name).name, []))


DETECTOR_KINDS = {
    "checkpoint": "Fine-tuned DETR checkpoint (.ckpt)",
    "coco": "Pre-trained COCO DETR (no training needed)",
    "annotations": "Annotated boxes from a COCO file (perfect detector)",
}


def create_detector(kind: str, *, checkpoint: str | Path | None = None, annotations: str | Path | None = None,
                    threshold: float = 0.7, prefer_gpu: bool = True) -> Detector:
    if kind == "checkpoint":
        if not checkpoint:
            raise ValueError("Choose a model checkpoint (.ckpt) file.")
        return DetrDetector(checkpoint, threshold=threshold, prefer_gpu=prefer_gpu)
    if kind == "coco":
        return CocoDetrDetector(threshold=threshold, prefer_gpu=prefer_gpu)
    if kind == "annotations":
        if not annotations:
            raise ValueError("Choose a COCO annotation (.json) file.")
        return AnnotationDetector(annotations)
    raise ValueError(f"Unknown detector {kind!r}; choose one of {', '.join(DETECTOR_KINDS)}")
