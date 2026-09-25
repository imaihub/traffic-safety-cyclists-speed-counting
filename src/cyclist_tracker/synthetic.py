"""Generate a small synthetic demo dataset with known cyclist speeds.

The real traffic videos are private, so this module draws a cycle track seen by
a tilted camera with simple cyclists riding in both directions at known speeds.
The result has exactly the layout described in ``docs/DATASET.md``, including
annotations, ground truth and a camera scene, so every function of the tool can
be tried (and tested) without any private data.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import AnalysisSettings, SceneConfig
from .counting import SPEED_CSV_HEADERS, CrossingCounter
from .dataset import coco_to_mot, create_layout, write_frame_list

log = logging.getLogger(__name__)

WIDTH, HEIGHT = 960, 540
TRACK_WIDTH_M, TRACK_LENGTH_M = 4.0, 30.0
#: Road corners (metres) and where they appear in the image (pixels).
WORLD_CORNERS = np.float64([[0, 0], [TRACK_WIDTH_M, 0], [TRACK_WIDTH_M, TRACK_LENGTH_M], [0, TRACK_LENGTH_M]])
IMAGE_CORNERS = np.float64([[440, 90], [540, 90], [800, 530], [180, 530]])
LINE_A_Y, LINE_B_Y = 24.0, 10.0  # line A is near the camera, line B further away
DIRECTION_AWAY, DIRECTION_TOWARDS = "away", "towards"
SPLIT_SIZES = {"train": (160, 7, 11), "val": (80, 4, 12), "test": (220, 10, 13)}  # frames, cyclists, seed


def _world_to_image() -> np.ndarray:
    return cv2.getPerspectiveTransform(WORLD_CORNERS.astype(np.float32), IMAGE_CORNERS.astype(np.float32))


def _project(h: np.ndarray, x: float, y: float) -> tuple[float, float]:
    p = cv2.perspectiveTransform(np.float64([[[x, y]]]), h)[0][0]
    return float(p[0]), float(p[1])


def demo_scene() -> SceneConfig:
    h = _world_to_image()
    corners = [(0, LINE_B_Y), (TRACK_WIDTH_M, LINE_B_Y), (TRACK_WIDTH_M, LINE_A_Y), (0, LINE_A_Y)]
    margin = 0.6
    roi = [(-margin, 0), (TRACK_WIDTH_M + margin, 0), (TRACK_WIDTH_M + margin, TRACK_LENGTH_M),
           (-margin, TRACK_LENGTH_M)]
    return SceneConfig(
        name="Synthetic demo camera",
        roi=[_project(h, *p) for p in roi],
        line_a=[_project(h, -margin, LINE_A_Y), _project(h, TRACK_WIDTH_M + margin, LINE_A_Y)],
        line_b=[_project(h, -margin, LINE_B_Y), _project(h, TRACK_WIDTH_M + margin, LINE_B_Y)],
        direction_a_to_b=DIRECTION_AWAY,
        direction_b_to_a=DIRECTION_TOWARDS,
        calibration_image_points=[_project(h, *p) for p in corners],
        calibration_world_points=[(float(x), float(y)) for x, y in corners],
        reference_distance_m=LINE_A_Y - LINE_B_Y,
    )


@dataclass
class _Rider:
    track: int
    lane_x: float
    speed_kph: float
    away: bool
    start_frame: int
    colour: tuple[int, int, int]

    def world_y(self, frame: int, fps: float) -> float:
        travelled = (frame - self.start_frame) / fps * self.speed_kph / 3.6
        return TRACK_LENGTH_M + 1 - travelled if self.away else -1 + travelled


def _background(h: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    img = np.zeros((HEIGHT, WIDTH, 3), np.uint8)
    img[:] = (70, 120, 60)  # grass (BGR)
    img[:70] = (200, 190, 170)  # sky/background band
    noise = rng.integers(-12, 12, img.shape, dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    track = np.int32([_project(h, x, y) for x, y in WORLD_CORNERS])
    cv2.fillPoly(img, [track], (95, 95, 100))
    for y in np.arange(0.0, TRACK_LENGTH_M, 1.0):  # 0.5 m painted blocks with 0.5 m gaps, as on the real road
        block = np.int32([_project(h, x, yy) for x, yy in [(3.6, y), (4.0, y), (4.0, y + 0.5), (3.6, y + 0.5)]])
        cv2.fillPoly(img, [block], (235, 235, 235))
    for y in np.arange(0.0, TRACK_LENGTH_M, 3.0):  # dashed centre line
        cv2.line(img, tuple(map(int, _project(h, 2.0, y))), tuple(map(int, _project(h, 2.0, y + 1.5))),
                 (220, 220, 220), 2)
    return img


def _draw_rider(img: np.ndarray, h: np.ndarray, rider: _Rider, y: float) -> tuple[float, float, float, float]:
    """Draw a cyclist whose wheels touch the road at (lane_x, y); returns its box (l, t, r, b)."""
    fx, fy = _project(h, rider.lane_x, y)
    scale = abs(_project(h, rider.lane_x + 0.5, y)[0] - _project(h, rider.lane_x - 0.5, y)[0])  # px per metre
    wheel = max(2, int(0.33 * scale))
    half_len = 0.55 * scale * 0.45  # bicycle seen almost from behind/front: short in the image
    body_h, body_w = 0.75 * scale, 0.42 * scale
    top = fy - 1.75 * scale
    cv2.circle(img, (int(fx), int(fy - wheel)), wheel, (30, 30, 30), max(1, int(scale * 0.08)))
    cv2.line(img, (int(fx - half_len), int(fy - wheel)), (int(fx + half_len), int(fy - wheel)), (40, 40, 160), 2)
    cv2.ellipse(img, (int(fx), int(fy - wheel * 2 - body_h / 2)), (max(1, int(body_w / 2)), max(1, int(body_h / 2))),
                0, 0, 360, rider.colour, -1)
    cv2.circle(img, (int(fx), int(top + 0.14 * scale)), max(2, int(0.14 * scale)), (150, 180, 220), -1)
    left, right = fx - max(body_w / 2, wheel), fx + max(body_w / 2, wheel)
    return left, top, right, fy


def _simulate(split: str, frames: int, riders: int, seed: int, fps: float) -> list[_Rider]:
    rng = np.random.default_rng(seed)
    starts = np.linspace(0, max(1, frames - 45), riders).astype(int) + rng.integers(0, 6, riders)
    palette = [(40, 40, 220), (220, 60, 40), (40, 200, 230), (200, 40, 200), (30, 160, 30), (240, 240, 240),
               (20, 20, 20), (0, 120, 255), (160, 100, 40), (120, 200, 120)]
    return [
        _Rider(track=i, lane_x=1.0 if away else 3.0, speed_kph=float(rng.uniform(12, 30)), away=bool(away),
               start_frame=int(start), colour=palette[i % len(palette)])
        for i, (start, away) in enumerate(zip(starts, rng.integers(0, 2, riders), strict=True))
    ]


def generate_demo_dataset(root: str | Path, fps: float = 5.0, make_video: bool = True,
                          progress: Callable[[int, int], None] | None = None) -> Path:
    """Create the demo dataset in ``root`` and return its path."""
    root = Path(root)
    create_layout(root)
    h = _world_to_image()
    scene = demo_scene()
    scene.save(root / "scene.json")
    total = sum(v[0] for v in SPLIT_SIZES.values())
    done = 0
    truth: list[dict] = []

    for split, (frames, n_riders, seed) in SPLIT_SIZES.items():
        rng = np.random.default_rng(seed + 100)
        background = _background(h, rng)
        riders = _simulate(split, frames, n_riders, seed, fps)
        images, annotations = [], []
        for f in range(frames):
            img = background.copy()
            visible = []
            for r in riders:
                y = r.world_y(f, fps)
                if -1 <= y <= TRACK_LENGTH_M + 1 and f >= r.start_frame:
                    visible.append((y, r))
            for y, r in sorted(visible, key=lambda item: item[0]):  # far riders first
                l, t, rr, b = _draw_rider(img, h, r, y)
                l, t, rr, b = max(0.0, l), max(0.0, t), min(WIDTH - 1.0, rr), min(HEIGHT - 1.0, b)
                if rr - l > 2 and b - t > 2:
                    annotations.append({
                        "id": len(annotations) + 1, "image_id": f + 1, "category_id": 0,
                        "bbox": [round(l, 2), round(t, 2), round(rr - l, 2), round(b - t, 2)],
                        "area": round((rr - l) * (b - t), 2), "iscrowd": 0,
                        "attributes": {"track_id": r.track, "occluded": False},
                    })
                    if split == "test":
                        truth.append({"frame": f, "track": r.track, "box": (l, t, rr, b)})
            img = cv2.GaussianBlur(img, (3, 3), 0)
            name = f"demo_{split}_frame_{f:05d}.jpg"
            cv2.imwrite(str(root / split / name), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            images.append({"id": f + 1, "file_name": f"{split}/{name}", "width": WIDTH, "height": HEIGHT})
            done += 1
            if progress:
                progress(done, total)
        coco = {"images": images, "annotations": annotations,
                "categories": [{"id": 0, "name": "cyclist", "supercategory": ""}]}
        (root / "annotations" / f"{split}.json").write_text(json.dumps(coco))
        if split == "test":
            test_riders = riders

    coco_to_mot(root / "annotations" / "test.json", root / "ground_truth" / "tracking_test.txt")
    write_frame_list(root / "annotations" / "test.json", root / "test_frames.txt", only_annotated=False)
    _write_reference_crossings(root, scene, fps, truth, test_riders)
    if make_video:
        _write_video(root / "test", root / "demo_test_video.mp4", fps)
    (root / "README.txt").write_text(
        "Synthetic demo dataset generated by cyclist-tracker.\n"
        "Frames: train/ val/ test/, COCO boxes: annotations/, reference data: ground_truth/,\n"
        "camera configuration: scene.json. True speeds are in ground_truth/speeds_test.csv.\n")
    log.info("Demo dataset written to %s", root)
    return root


def _write_reference_crossings(root: Path, scene: SceneConfig, fps: float, truth: list[dict],
                               riders: list[_Rider]) -> None:
    """Reference crossings: entry/exit frames from the true boxes (bottom-centre), speed = the simulated speed."""
    counter = CrossingCounter(scene, AnalysisSettings(fps=fps, speed_mode="fixed_distance"))
    for item in sorted(truth, key=lambda i: (i["frame"], i["track"])):
        l, t, r, b = item["box"]
        counter.update(item["track"], ((l + r) / 2, b), item["frame"])
    speed_of = {r.track: r.speed_kph for r in riders}
    rows = sorted(counter.crossings, key=lambda c: c.entry_frame)
    with open(root / "ground_truth" / "speeds_test.csv", "w") as f:
        f.write(",".join(SPEED_CSV_HEADERS) + "\n")
        for c in rows:
            f.write(f"{c.track_id + 1},{speed_of[c.track_id]:.3f},{c.direction},{c.frames_taken / fps:.3f},"
                    f"{c.frames_taken},{c.entry_frame},{c.exit_frame}\n")
    with open(root / "ground_truth" / "counts_test.csv", "w") as f:
        f.write("track_id,direction,frames_taken,entry_frame,exit_frame\n")
        for c in rows:
            f.write(f"{c.track_id + 1},{c.direction},{c.frames_taken},{c.entry_frame},{c.exit_frame}\n")


def _write_video(frames_dir: Path, output: Path, fps: float) -> None:
    frames = sorted(frames_dir.glob("*.jpg"))
    writer = cv2.VideoWriter(str(output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (WIDTH, HEIGHT))
    for path in frames:
        writer.write(cv2.imread(str(path)))
    writer.release()

