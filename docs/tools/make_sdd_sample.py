"""Build the sample dataset in ``sample_data/stanford_drone_gates`` from the Stanford Drone Dataset.

    uv run python docs/tools/make_sdd_sample.py --sdd path/to/stanford_campus_dataset

``--sdd`` is the extracted dataset (folders ``videos/`` and ``annotations/``); only the three videos
``gates/video4``, ``gates/video5`` and ``gates/video7`` and their annotations are needed
(``docs/tools/download_sdd_videos.py`` fetches just those files).

What the script does:

* test  = gates/video4, every 6th frame (29.97 fps -> 5 fps), with ground truth
* train = gates/video7 and val = gates/video5, one frame per second (for detector training)
* frames are cropped to the lower road and the roundabout and resized to reduce the size
* only "Biker" boxes that are in view are kept, with their SDD track id as ``track_id``
* reference speeds and crossings are computed from the *full-rate* human annotations and the metric
  scale of the video (OpenTraj estimated_scales.yaml), independently of the counting algorithm
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from cyclist_tracker.config import AnalysisSettings, SceneConfig
from cyclist_tracker.counting import SPEED_CSV_HEADERS
from cyclist_tracker.dataset import coco_to_mot, create_layout, write_frame_list

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "sample_data" / "stanford_drone_gates"

VIDEO_FPS = 29.97
TEST_STRIDE = 6  # -> 4.995 fps, as the 5 fps of the paper
TRAIN_STRIDE = 30  # one frame per second
CROP = (200, 702, 1300, 1982)  # x1, y1, x2, y2 in original pixels: lower road and roundabout
SCALE = 0.5  # resize factor after cropping
JPEG_QUALITY = 78
# metres per original pixel (OpenTraj, datasets/SDD/estimated_scales.yaml)
METRES_PER_PIXEL = {"video4": 0.04412268, "video5": 0.0342392, "video7": 0.04540353}
SPLITS = {"test": ("video4", TEST_STRIDE), "train": ("video7", TRAIN_STRIDE), "val": ("video5", TRAIN_STRIDE)}

# Scene in *original* pixels of gates/video4 (converted to sample pixels below)
LINE_A_Y, LINE_B_Y = 1800.0, 1500.0  # line A near the bottom edge, line B towards the roundabout
LINE_X = (600.0, 1020.0)
# The region covers the whole cropped view: every biker is on a road here, so all of them are tracked
# (and can be compared with the tracking ground truth); only the lower road is counted.
ROI = [(CROP[0], CROP[1]), (CROP[2], CROP[1]), (CROP[2], CROP[3]), (CROP[0], CROP[3])]
DIRECTION_A_TO_B, DIRECTION_B_TO_A = "to roundabout", "from roundabout"


def to_sample(x: float, y: float) -> tuple[float, float]:
    return round((x - CROP[0]) * SCALE, 2), round((y - CROP[1]) * SCALE, 2)


def read_annotations(path: Path) -> dict[int, list[tuple]]:
    """frame -> [(track, x1, y1, x2, y2, occluded, generated)] for visible bikers."""
    frames = defaultdict(list)
    for line in path.read_text().splitlines():
        parts = line.split()
        track, x1, y1, x2, y2, frame, lost, occluded, generated = map(int, parts[:9])
        if parts[9].strip('"') == "Biker" and not lost:
            frames[frame].append((track, x1, y1, x2, y2, occluded, generated))
    return frames


def convert_split(sdd: Path, split: str, video: str, stride: int) -> list[str]:
    ann = read_annotations(sdd / "annotations" / "gates" / video / "annotations.txt")
    cap = cv2.VideoCapture(str(sdd / "videos" / "gates" / video / "video.mov"))
    images, annotations, names = [], [], []
    number = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if number % stride == 0:
            crop = frame[CROP[1]:CROP[3], CROP[0]:CROP[2]]
            crop = cv2.resize(crop, None, fx=SCALE, fy=SCALE, interpolation=cv2.INTER_AREA)
            name = f"sdd_gates_{video}_frame_{number:05d}.jpg"
            cv2.imwrite(str(OUT / split / name), crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            image_id = len(images) + 1
            h, w = crop.shape[:2]
            images.append({"id": image_id, "file_name": f"{split}/{name}", "width": w, "height": h})
            names.append(name)
            for track, x1, y1, x2, y2, occluded, generated in ann.get(number, []):
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                if not (CROP[0] <= cx < CROP[2] and CROP[1] <= cy < CROP[3]):
                    continue
                (a, b), (c, d) = to_sample(x1, y1), to_sample(x2, y2)
                a, b, c, d = max(0, a), max(0, b), min(w - 1, c), min(h - 1, d)
                if c - a < 2 or d - b < 2:
                    continue
                annotations.append({
                    "id": len(annotations) + 1, "image_id": image_id, "category_id": 0,
                    "bbox": [a, b, round(c - a, 2), round(d - b, 2)], "area": round((c - a) * (d - b), 2),
                    "iscrowd": 0,
                    "attributes": {"track_id": track, "occluded": bool(occluded), "generated": bool(generated)},
                })
        number += 1
    cap.release()
    coco = {"info": {"description": f"Stanford Drone Dataset, gates/{video}, bikers only (converted)",
                     "url": "https://cvgl.stanford.edu/projects/uav_data/"},
            "licenses": [{"id": 1, "name": "CC BY-NC-SA 3.0",
                          "url": "https://creativecommons.org/licenses/by-nc-sa/3.0/"}],
            "images": images, "annotations": annotations,
            "categories": [{"id": 0, "name": "cyclist", "supercategory": ""}]}
    (OUT / "annotations" / f"{split}.json").write_text(json.dumps(coco))
    print(f"{split}: {len(images)} frames, {len(annotations)} boxes from gates/{video}")
    return names


def reference_crossings(sdd: Path) -> list[dict]:
    """Crossings of line A and B from the full-rate annotations (box centre, top-down view)."""
    ann = read_annotations(sdd / "annotations" / "gates" / "video4" / "annotations.txt")
    tracks = defaultdict(list)
    for frame, boxes in ann.items():
        for track, x1, y1, x2, y2, *_ in boxes:
            tracks[track].append((frame, (x1 + x2) / 2, (y1 + y2) / 2))

    def crossing(points, line_y):
        for (f0, x0, y0), (f1, x1, y1) in zip(points, points[1:], strict=False):
            if (y0 - line_y) * (y1 - line_y) <= 0 and y0 != y1:
                t = (line_y - y0) / (y1 - y0)
                x = x0 + t * (x1 - x0)
                if LINE_X[0] <= x <= LINE_X[1]:
                    return f0 + t * (f1 - f0), x
        return None

    rows = []
    m_per_px = METRES_PER_PIXEL["video4"]
    for track, points in tracks.items():
        points.sort()
        a, b = crossing(points, LINE_A_Y), crossing(points, LINE_B_Y)
        if a is None or b is None:
            continue
        (fa, xa), (fb, xb) = a, b
        first, second, direction = (a, b, DIRECTION_A_TO_B) if fa < fb else (b, a, DIRECTION_B_TO_A)
        seconds = abs(fb - fa) / VIDEO_FPS
        metres = float(np.hypot(xb - xa, LINE_B_Y - LINE_A_Y)) * m_per_px
        entry, exit_ = round(first[0] / TEST_STRIDE), round(second[0] / TEST_STRIDE)
        rows.append({"track_id": track, "speed_kph": metres / seconds * 3.6, "direction": direction,
                     "crossing_time_sec": seconds, "frames_taken": exit_ - entry, "entry_frame": entry,
                     "exit_frame": exit_})
    return sorted(rows, key=lambda r: r["entry_frame"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--sdd", required=True, type=Path, help="extracted Stanford Drone Dataset folder")
    args = parser.parse_args()

    keep = {name: (OUT / name).read_bytes() for name in ("README.md", "LICENSE.md") if (OUT / name).is_file()}
    shutil.rmtree(OUT, ignore_errors=True)
    create_layout(OUT)
    for name, data in keep.items():
        (OUT / name).write_bytes(data)
    for split, (video, stride) in SPLITS.items():
        convert_split(args.sdd, split, video, stride)

    coco_to_mot(OUT / "annotations" / "test.json", OUT / "ground_truth" / "tracking_test.txt")
    write_frame_list(OUT / "annotations" / "test.json", OUT / "test_frames.txt", only_annotated=False)

    rows = reference_crossings(args.sdd)
    with open(OUT / "ground_truth" / "speeds_test.csv", "w") as f:
        f.write(",".join(SPEED_CSV_HEADERS) + "\n")
        for r in rows:
            f.write(f"{r['track_id']},{r['speed_kph']:.3f},{r['direction']},{r['crossing_time_sec']:.3f},"
                    f"{r['frames_taken']},{r['entry_frame']},{r['exit_frame']}\n")
    with open(OUT / "ground_truth" / "counts_test.csv", "w") as f:
        f.write("track_id,direction,frames_taken,entry_frame,exit_frame\n")
        for r in rows:
            f.write(f"{r['track_id']},{r['direction']},{r['frames_taken']},{r['entry_frame']},{r['exit_frame']}\n")
    print(f"reference crossings: {len(rows)}")

    # Camera scene: a top-down view, so the calibration is a pure scale (metres per pixel).
    m_per_sample_px = METRES_PER_PIXEL["video4"] / SCALE
    corners = [(0.0, 0.0), ((CROP[2] - CROP[0]) * SCALE, 0.0),
               ((CROP[2] - CROP[0]) * SCALE, (CROP[3] - CROP[1]) * SCALE), (0.0, (CROP[3] - CROP[1]) * SCALE)]
    scene = SceneConfig(
        name="Stanford Drone Dataset - gates/video4 (lower road)",
        roi=[to_sample(*p) for p in ROI],
        line_a=[to_sample(LINE_X[0], LINE_A_Y), to_sample(LINE_X[1], LINE_A_Y)],
        line_b=[to_sample(LINE_X[0], LINE_B_Y), to_sample(LINE_X[1], LINE_B_Y)],
        direction_a_to_b=DIRECTION_A_TO_B,
        direction_b_to_a=DIRECTION_B_TO_A,
        calibration_image_points=corners,
        calibration_world_points=[(round(x * m_per_sample_px, 4), round(y * m_per_sample_px, 4)) for x, y in corners],
        reference_distance_m=round((LINE_A_Y - LINE_B_Y) * METRES_PER_PIXEL["video4"], 3),
    )
    scene.save(OUT / "scene.json")
    # Top-down camera: the box centre is the position on the ground.
    AnalysisSettings(fps=round(VIDEO_FPS / TEST_STRIDE, 3), tracking_point="center").save(OUT / "settings.json")

    frames = sorted((OUT / "test").glob("*.jpg"))
    video = OUT / "gates_video4_5fps.mp4"
    if shutil.which("ffmpeg"):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-framerate", str(VIDEO_FPS / TEST_STRIDE), "-pattern_type",
                        "glob", "-i", str(OUT / "test" / "*.jpg"), "-c:v", "libx264", "-pix_fmt", "yuv420p",
                        "-crf", "30", str(video)], check=True)
    print(f"{len(frames)} test frames; sample written to {OUT}")


if __name__ == "__main__":
    main()
