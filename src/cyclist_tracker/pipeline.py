"""The analysis pipeline: frames -> detections -> tracks -> counts and speeds.

Frames can come from a folder of images (optionally restricted to a list of
file names) or directly from a video file. Results are written while the
frames are processed, so memory use does not grow with the length of the video:

* ``<name>_tracks.txt``   - every tracked box (MOT-style, see :mod:`io_formats`)
* ``<name>_speeds.csv``   - one row per counted cyclist with direction and speed
* ``<name>_summary.json`` - counts, average speeds and the settings used
* ``<name>.mp4``          - the video with overlays (optional)
"""

from __future__ import annotations

import csv
import json
import logging
import math
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from . import __version__
from .config import AnalysisSettings, SceneConfig
from .counting import SPEED_CSV_HEADERS, Crossing, CrossingCounter
from .detection import Detector
from .geometry import bbox_center, point_in_polygon, polygon_array, reference_point
from .io_formats import IMAGE_EXTENSIONS, load_frame_list, mot_row, natural_key
from .tracking import CyclistTracker, Track
from .visualization import draw_counts, draw_scene, draw_track

log = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".mpg", ".mpeg", ".wmv"}


@dataclass
class Frame:
    index: int
    name: str
    image_bgr: np.ndarray


class ImageFolderSource:
    """Images from a folder, in natural (or, for old results, alphabetical) order."""

    def __init__(self, folder: str | Path, frame_list: str | Path | None = None, natural_sort: bool = True):
        self.folder = Path(folder)
        if not self.folder.is_dir():
            raise FileNotFoundError(f"Frame folder not found: {self.folder}")
        available = {p.name: p for p in self.folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS}
        if frame_list:
            wanted = [Path(name).name for name in load_frame_list(frame_list)]
            missing = [n for n in wanted if n not in available]
            if missing:
                log.warning("%d of %d frames in the list are not in %s (e.g. %s)",
                            len(missing), len(wanted), self.folder, missing[0])
            names = [n for n in dict.fromkeys(wanted) if n in available]
        else:
            names = list(available)
        names.sort(key=natural_key if natural_sort else None)
        self.paths = [available[n] for n in names]
        self.fps: float | None = None

    def __len__(self) -> int:
        return len(self.paths)

    def __iter__(self) -> Iterator[Frame]:
        for index, path in enumerate(self.paths):
            image = cv2.imread(str(path), cv2.IMREAD_COLOR)
            if image is None:
                log.warning("Could not read image %s - skipped", path)
                continue
            yield Frame(index, path.name, image)


class VideoSource:
    """Every ``stride``-th frame of a video file (stride 5 turns 25 fps into 5 fps)."""

    def __init__(self, path: str | Path, stride: int = 1):
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(f"Video not found: {self.path}")
        if stride < 1:
            raise ValueError("Stride must be 1 or larger.")
        self.stride = stride
        cap = cv2.VideoCapture(str(self.path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video {self.path}")
        self.video_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        self._frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        self.fps = self.video_fps / stride

    def __len__(self) -> int:
        return math.ceil(self._frame_count / self.stride) if self._frame_count > 0 else 0

    def __iter__(self) -> Iterator[Frame]:
        cap = cv2.VideoCapture(str(self.path))
        number, index = 0, 0
        try:
            while True:
                ok, image = cap.read()
                if not ok:
                    break
                if number % self.stride == 0:
                    yield Frame(index, f"{self.path.stem}_frame_{number:06d}.jpg", image)
                    index += 1
                number += 1
        finally:
            cap.release()


def open_source(path: str | Path, frame_list: str | Path | None = None, stride: int = 1,
                natural_sort: bool = True) -> ImageFolderSource | VideoSource:
    path = Path(path)
    if path.is_dir():
        return ImageFolderSource(path, frame_list, natural_sort)
    if path.suffix.lower() in VIDEO_EXTENSIONS:
        return VideoSource(path, stride)
    raise ValueError(f"{path} is neither a folder of images nor a supported video file.")


@dataclass
class AnalysisResult:
    frames_processed: int
    counts: dict[str, int]
    crossings: list[Crossing]
    output_files: dict[str, str]
    elapsed_s: float
    stopped: bool = False
    fps: float = 5.0
    speeds_by_direction: dict[str, float] = field(default_factory=dict)

    def summary(self) -> dict:
        return {
            "frames_processed": self.frames_processed,
            "counts": self.counts,
            "total": sum(self.counts.values()),
            "average_speed_kph": self.speeds_by_direction,
            "elapsed_seconds": round(self.elapsed_s, 1),
            "stopped_early": self.stopped,
            "output_files": self.output_files,
        }


class Analyzer:
    """Processes frames one by one; keeps tracker and counter state between frames."""

    def __init__(self, detector: Detector, tracker: CyclistTracker, scene: SceneConfig, settings: AnalysisSettings):
        self.detector = detector
        self.tracker = tracker
        self.scene = scene
        self.settings = settings
        self.counter = CrossingCounter(scene, settings)
        self._roi = polygon_array(scene.roi)
        self._speed_labels: dict[int, str] = {}

    def process(self, frame: Frame, draw: bool = True) -> tuple[list[Track], list[Crossing], np.ndarray | None]:
        image_rgb = cv2.cvtColor(frame.image_bgr, cv2.COLOR_BGR2RGB)
        detections = [
            d for d in self.detector.detect(image_rgb, frame.name)
            if point_in_polygon(bbox_center(d.ltrb), self._roi)
        ]
        tracks = self.tracker.update(detections, frame.image_bgr)
        new_crossings = []
        for track in tracks:
            point = reference_point(track.ltrb, self.settings.tracking_point)
            crossing = self.counter.update(track.track_id, point, frame.index)
            if crossing is not None:
                new_crossings.append(crossing)
                self._speed_labels[track.track_id] = f"{crossing.speed_kph:.1f} km/h"
        inside = [t for t in tracks if point_in_polygon(bbox_center(t.ltrb), self._roi)]
        canvas = self._draw(frame, inside) if draw else None
        return inside, new_crossings, canvas

    def _draw(self, frame: Frame, tracks: list[Track]) -> np.ndarray:
        canvas = draw_scene(frame.image_bgr.copy(), self.scene)
        for t in tracks:
            note = self._speed_labels.get(t.track_id, "")
            if not note and (progress := self.counter.progress(t.track_id)):
                note = f"{progress[0]} {progress[1] * 100:.0f}%"
            draw_track(canvas, t.track_id, t.ltrb, note, counted=t.track_id in self._speed_labels)
        draw_counts(canvas, self.counter.counts, f"frame {frame.index}")
        return canvas


def run_analysis(
    source: ImageFolderSource | VideoSource,
    detector: Detector,
    tracker: CyclistTracker,
    scene: SceneConfig,
    settings: AnalysisSettings,
    output_dir: str | Path,
    name: str = "analysis",
    save_video: bool = True,
    progress: Callable[[int, int, dict[str, int]], None] | None = None,
    preview: Callable[[np.ndarray], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    on_crossing: Callable[[Crossing], None] | None = None,
) -> AnalysisResult:
    """Run the whole pipeline over ``source`` and write the result files to ``output_dir``."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "tracks": str(output_dir / f"{name}_tracks.txt"),
        "speeds": str(output_dir / f"{name}_speeds.csv"),
        "summary": str(output_dir / f"{name}_summary.json"),
    }
    if save_video:
        files["video"] = str(output_dir / f"{name}.mp4")

    analyzer = Analyzer(detector, tracker, scene, settings)
    total = len(source)
    start = time.perf_counter()
    writer = None
    processed = 0
    stopped = False
    log.info("Analysing %d frames -> %s", total, output_dir)

    with open(files["tracks"], "w", newline="") as mot_file, open(files["speeds"], "w", newline="") as speed_file:
        speed_writer = csv.writer(speed_file)
        speed_writer.writerow(SPEED_CSV_HEADERS)
        try:
            for frame in source:
                if should_stop and should_stop():
                    stopped = True
                    log.info("Stopped by the user after %d frames", processed)
                    break
                draw = save_video or preview is not None
                tracks, crossings, canvas = analyzer.process(frame, draw=draw)
                for t in tracks:
                    mot_file.write(",".join(map(str, mot_row(frame.name, t.track_id, t.ltrb, t.confidence))) + "\n")
                for c in crossings:
                    speed_writer.writerow(c.as_row(settings.fps))
                    speed_file.flush()
                    log.info("Cyclist %d counted: %s, %.1f km/h (frames %d-%d)",
                             c.track_id, c.direction, c.speed_kph, c.entry_frame, c.exit_frame)
                    if on_crossing:
                        on_crossing(c)
                if canvas is not None and save_video:
                    if writer is None:
                        video_size = (canvas.shape[1], canvas.shape[0])
                        writer = cv2.VideoWriter(files["video"], cv2.VideoWriter_fourcc(*"mp4v"), settings.fps,
                                                 video_size)
                    if (canvas.shape[1], canvas.shape[0]) != video_size:
                        canvas = cv2.resize(canvas, video_size)
                    writer.write(canvas)
                if preview is not None and canvas is not None:
                    preview(canvas)
                processed += 1
                if progress:
                    progress(processed, total, analyzer.counter.counts)
        finally:
            if writer is not None:
                writer.release()

    crossings = analyzer.counter.crossings
    speeds = {}
    for direction in analyzer.counter.counts:
        values = [c.speed_kph for c in crossings if c.direction == direction]
        if values:
            speeds[direction] = round(float(np.mean(values)), 2)
    result = AnalysisResult(processed, analyzer.counter.counts, crossings, files, time.perf_counter() - start,
                            stopped, settings.fps, speeds)
    summary = result.summary() | {
        "software_version": __version__,
        "scene": scene.to_dict(),
        "settings": settings.to_dict(),
    }
    Path(files["summary"]).write_text(json.dumps(summary, indent=2))
    log.info("Done: %d frames, counts %s, %.1f s", processed, result.counts, result.elapsed_s)
    return result
