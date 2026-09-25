"""Dataset layout, validation and preparation tools.

Expected layout (see ``docs/DATASET.md`` for the full description)::

    my_dataset/
    ├── train/  val/  test/         image frames (.jpg/.png)
    ├── annotations/
    │   ├── train.json  val.json  test.json      COCO boxes (+ track_id attribute)
    ├── ground_truth/                optional, for evaluation only
    │   ├── tracking_test.txt  speeds_test.csv  counts_test.csv
    └── scene.json                   optional camera configuration

Older file names from the original experiments (``training_set_f.json`` ...)
are recognised as well.
"""

from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import cv2

from .geometry import Point, parse_points
from .io_formats import IMAGE_EXTENSIONS, natural_key

log = logging.getLogger(__name__)

SPLITS = ("train", "val", "test")
ANNOTATION_NAMES = {
    "train": ("train.json", "training_set.json", "training_set_f.json"),
    "val": ("val.json", "validation_set.json", "validation_set_f.json"),
    "test": ("test.json", "testing_set.json", "testing_set_f.json"),
}
GROUND_TRUTH_FILES = {
    "tracking": ("tracking_test.txt", "Boxes with track IDs (tracking evaluation)"),
    "speeds": ("speeds_test.csv", "Reference speeds (speed evaluation)"),
    "counts": ("counts_test.csv", "Reference crossings (counting evaluation)"),
}

OK, INFO, WARNING, ERROR = "ok", "info", "warning", "error"


@dataclass
class Check:
    level: str
    topic: str
    message: str


@dataclass
class DatasetReport:
    root: Path
    checks: list[Check] = field(default_factory=list)
    images: dict[str, int] = field(default_factory=dict)
    annotations: dict[str, Path] = field(default_factory=dict)
    ground_truth: dict[str, Path] = field(default_factory=dict)

    def add(self, level: str, topic: str, message: str) -> None:
        self.checks.append(Check(level, topic, message))

    @property
    def errors(self) -> list[Check]:
        return [c for c in self.checks if c.level == ERROR]

    @property
    def ready_for_training(self) -> bool:
        return not self.errors and all(s in self.annotations and self.images.get(s) for s in ("train", "val"))

    @property
    def ready_for_analysis(self) -> bool:
        return not self.errors and bool(self.images.get("test"))

    def as_text(self) -> str:
        symbols = {OK: "[OK]  ", INFO: "[i]   ", WARNING: "[!]   ", ERROR: "[X]   "}
        lines = [f"Dataset check for {self.root}"]
        lines += [f"{symbols[c.level]}{c.topic}: {c.message}" for c in self.checks]
        lines.append(f"Ready for training: {'yes' if self.ready_for_training else 'no'}; "
                     f"ready for analysis: {'yes' if self.ready_for_analysis else 'no'}")
        return "\n".join(lines)


def find_annotation_file(root: Path, split: str) -> Path | None:
    for name in ANNOTATION_NAMES[split]:
        if (root / "annotations" / name).is_file():
            return root / "annotations" / name
    return None


def resolve_image(root: Path, split: str, file_name: str) -> Path:
    """Where the image of a COCO entry lives: ``root/file_name`` or ``root/split/<basename>``."""
    direct = root / file_name
    return direct if direct.is_file() else root / split / Path(file_name).name


def list_images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted((p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS), key=lambda p: natural_key(p.name))


def validate_dataset(root: str | Path) -> DatasetReport:
    """Check a dataset folder and explain in plain language what is missing or wrong."""
    root = Path(root)
    report = DatasetReport(root)
    if not root.is_dir():
        report.add(ERROR, "Folder", "The dataset folder does not exist.")
        return report

    for split in SPLITS:
        images = list_images(root / split)
        report.images[split] = len(images)
        purpose = "analysis and evaluation" if split == "test" else "training"
        if images:
            report.add(OK, f"{split}/", f"{len(images)} images found.")
        elif (root / split).is_dir():
            report.add(WARNING, f"{split}/", f"Folder exists but has no images (needed for {purpose}).")
        else:
            report.add(WARNING, f"{split}/", f"Folder is missing (needed for {purpose}).")

        ann = find_annotation_file(root, split)
        if ann is None:
            level = INFO if split == "test" else WARNING
            report.add(level, f"annotations/{split}.json", "Not found (needed for training and for "
                       "creating tracking ground truth)." if split != "test" else
                       "Not found (only needed for evaluation or the 'annotations' detector).")
            continue
        report.annotations[split] = ann
        _check_coco(report, root, split, ann)

    gt_dir = root / "ground_truth"
    for key, (name, meaning) in GROUND_TRUTH_FILES.items():
        path = gt_dir / name
        if path.is_file():
            report.ground_truth[key] = path
            report.add(OK, f"ground_truth/{name}", f"Found: {meaning}.")
        else:
            report.add(INFO, f"ground_truth/{name}", f"Not found (optional: {meaning}).")

    if (root / "scene.json").is_file():
        from .config import SceneConfig

        try:
            problems = SceneConfig.load(root / "scene.json").problems()
            if problems:
                report.add(WARNING, "scene.json", " ".join(problems))
            else:
                report.add(OK, "scene.json", "Camera scene is complete.")
        except (ValueError, json.JSONDecodeError) as exc:
            report.add(ERROR, "scene.json", f"Cannot be read: {exc}")
    else:
        report.add(INFO, "scene.json", "Not found (create one in the 'Camera scene' tab).")
    return report


def _check_coco(report: DatasetReport, root: Path, split: str, path: Path) -> None:
    topic = f"annotations/{path.name}"
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        report.add(ERROR, topic, f"Not a valid JSON file: {exc}")
        return
    missing_keys = [k for k in ("images", "annotations") if k not in data]
    if missing_keys:
        report.add(ERROR, topic, f"Missing the COCO section(s): {', '.join(missing_keys)}.")
        return
    images, anns = data["images"], data["annotations"]
    ids = {img.get("id") for img in images}
    bad_boxes = sum(1 for a in anns if len(a.get("bbox", [])) != 4 or a["bbox"][2] <= 0 or a["bbox"][3] <= 0)
    orphan = sum(1 for a in anns if a.get("image_id") not in ids)
    missing_files = sum(1 for img in images if not resolve_image(root, split, img.get("file_name", "")).is_file())
    with_boxes = len({a.get("image_id") for a in anns})
    with_track = sum(1 for a in anns if "track_id" in a.get("attributes", {}) or "track_id" in a)

    report.add(OK, topic, f"{len(images)} images, {len(anns)} boxes, {with_boxes} images contain cyclists.")
    if bad_boxes:
        report.add(WARNING, topic, f"{bad_boxes} boxes have no size or a wrong format and will be ignored.")
    if orphan:
        report.add(ERROR, topic, f"{orphan} boxes refer to an image id that does not exist.")
    if missing_files:
        level = ERROR if missing_files == len(images) else WARNING
        report.add(level, topic, f"{missing_files} of {len(images)} listed images were not found in "
                   f"'{split}/' (file_name must be '{split}/<image>' or '<image>').")
    if anns and with_track < len(anns):
        report.add(INFO, topic, f"{len(anns) - with_track} boxes have no track_id attribute "
                   "(only needed to create tracking ground truth).")


# ----------------------------------------------------------------------------- preparation tools
def create_layout(root: str | Path) -> list[Path]:
    """Create the empty folder structure; returns the folders that were created."""
    root = Path(root)
    created = []
    for sub in (*SPLITS, "annotations", "ground_truth"):
        folder = root / sub
        if not folder.exists():
            folder.mkdir(parents=True)
            created.append(folder)
    return created


def extract_frames(video: str | Path, output_dir: str | Path, every_n: int = 5,
                   progress: Callable[[int, int], None] | None = None,
                   should_stop: Callable[[], bool] | None = None) -> int:
    """Save every ``every_n``-th frame of ``video`` as ``<video>_frame_<n>.jpg``. Returns the number saved."""
    video, output_dir = Path(video), Path(output_dir)
    if every_n < 1:
        raise ValueError("every_n must be 1 or larger")
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"Could not open video {video}")
    output_dir.mkdir(parents=True, exist_ok=True)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    number = saved = 0
    try:
        while True:
            if should_stop and should_stop():
                break
            ok, frame = cap.read()
            if not ok:
                break
            if number % every_n == 0:
                cv2.imwrite(str(output_dir / f"{video.stem}_frame_{number:06d}.jpg"), frame)
                saved += 1
            number += 1
            if progress and number % 25 == 0:
                progress(number, total)
    finally:
        cap.release()
    if progress:
        progress(total, total)
    log.info("Saved %d frames of %s to %s", saved, video.name, output_dir)
    return saved


def write_frame_list(coco_json: str | Path, output_txt: str | Path, only_annotated: bool = True) -> int:
    """Write the file names of (annotated) images in a COCO file, one per line, in natural order."""
    data = json.loads(Path(coco_json).read_text())
    annotated = {a["image_id"] for a in data.get("annotations", [])}
    names = [img["file_name"] for img in data.get("images", []) if not only_annotated or img["id"] in annotated]
    names.sort(key=lambda n: natural_key(Path(n).name))
    Path(output_txt).write_text("".join(f"{n}\n" for n in names))
    return len(names)


def coco_to_mot(coco_json: str | Path, output_txt: str | Path) -> int:
    """Create a tracking ground-truth file from a COCO export with ``track_id`` attributes (e.g. CVAT).

    Track ids are shifted by one so that they start at 1 (MOT convention). Returns the number of boxes.
    """
    data = json.loads(Path(coco_json).read_text())
    names = {img["id"]: img["file_name"] for img in data.get("images", [])}
    rows = []
    for ann in data.get("annotations", []):
        track = ann.get("attributes", {}).get("track_id", ann.get("track_id"))
        if track is None:
            raise ValueError(f"Annotation {ann.get('id')} has no track_id; export the tracks from CVAT "
                             "as 'COCO 1.0' with attributes.")
        x, y, w, h = ann["bbox"]
        rows.append((natural_key(Path(names[ann["image_id"]]).name), names[ann["image_id"]], int(track) + 1, x, y, w, h))
    rows.sort(key=lambda r: (r[0], r[2]))
    with open(output_txt, "w") as f:
        for _, name, track, x, y, w, h in rows:
            f.write(f"{name},{track},{x:.2f},{y:.2f},{w:.2f},{h:.2f},1.0,1,1\n")
    return len(rows)


def flatten_file_names(coco_json: str | Path, output_json: str | Path, split: str) -> int:
    """Rewrite ``file_name`` entries (e.g. ``test/2022/08/frames/x.jpg``) to ``<split>/x.jpg``."""
    data = json.loads(Path(coco_json).read_text())
    for img in data.get("images", []):
        img["file_name"] = f"{split}/{Path(img['file_name']).name}"
    Path(output_json).write_text(json.dumps(data, indent=2))
    return len(data.get("images", []))


def read_cvat_polygons(xml_file: str | Path) -> list[tuple[str, list[Point]]]:
    """Polygons (label, points) from a CVAT 'for images 1.1' XML export, e.g. the calibration area."""
    tree = ET.parse(xml_file)
    return [(poly.get("label", ""), parse_points(poly.get("points", "")))
            for poly in tree.getroot().iter("polygon")]
