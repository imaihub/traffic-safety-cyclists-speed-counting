"""Bidirectional counting and speed estimation for tracked cyclists.

A cyclist is counted when its track first comes close to one counting line
(entry) and later reaches the other line (exit). The time between entry and
exit, together with the distance travelled, gives the speed. The logic is the
same as in the original experiments (paper, Sec. 3.4-3.5):

* entry when the track centre is closer than ``entry_threshold_px`` to line A or B;
* exit when the centre is closer than ``exit_threshold_px`` to the *other* line,
  or closer than twice that threshold while still approaching it;
* the crossing is rejected when the track moved fewer than ``min_pixel_distance``
  pixels, took fewer/more frames than allowed, or gives an implausible speed.

Two ways to obtain the distance are supported:

``homography``
    Entry and exit positions are projected on the road with the calibration
    homography and the metric distance between them is used (the prediction).
``fixed_distance``
    The known distance between the lines (``reference_distance_m``) is used.
    This is how the reference ("ground truth") speeds were produced.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .config import AnalysisSettings, SceneConfig
from .geometry import Point, compute_homography, point_to_line_distance, world_distance

SPEED_CSV_HEADERS = [
    "track_id",
    "speed_kph",
    "direction",
    "crossing_time_sec",
    "frames_taken",
    "entry_frame",
    "exit_frame",
]


@dataclass
class Crossing:
    """One counted cyclist."""

    track_id: int
    direction: str
    entry_frame: int
    exit_frame: int
    speed_kph: float
    distance_m: float

    @property
    def frames_taken(self) -> int:
        return self.exit_frame - self.entry_frame

    def crossing_time(self, fps: float) -> float:
        return self.frames_taken / fps

    def as_row(self, fps: float) -> list:
        return [
            self.track_id,
            round(self.speed_kph, 3),
            self.direction,
            round(self.crossing_time(fps), 3),
            self.frames_taken,
            self.entry_frame,
            self.exit_frame,
        ]


@dataclass
class _TrackState:
    entry_line: str | None = None  # "A" or "B"
    entry_frame: int | None = None
    entry_position: Point | None = None
    last_position: Point | None = None
    done: bool = False
    history: list[Point] = field(default_factory=list)


class CrossingCounter:
    """Keeps per-track state and turns track positions into counted crossings."""

    def __init__(self, scene: SceneConfig, settings: AnalysisSettings):
        issues = scene.problems() + settings.problems()
        if issues:
            raise ValueError("Invalid configuration: " + " ".join(issues))
        if settings.speed_mode == "homography" and not scene.has_calibration:
            raise ValueError("Speed mode 'homography' needs calibration points in the scene.")
        if settings.speed_mode == "fixed_distance" and scene.reference_distance_m <= 0:
            raise ValueError("Speed mode 'fixed_distance' needs a positive reference distance in the scene.")
        self.scene = scene
        self.settings = settings
        self.homography = (
            compute_homography(scene.calibration_image_points, scene.calibration_world_points)
            if scene.has_calibration
            else None
        )
        self.crossings: list[Crossing] = []
        self._tracks: dict[int, _TrackState] = {}

    # ----------------------------------------------------------------- queries
    @property
    def counts(self) -> dict[str, int]:
        result = {self.scene.direction_a_to_b: 0, self.scene.direction_b_to_a: 0}
        for c in self.crossings:
            result[c.direction] += 1
        return result

    @property
    def total(self) -> int:
        return len(self.crossings)

    def progress(self, track_id: int) -> tuple[str, float] | None:
        """For a track between the lines: (direction, fraction of the way done)."""
        state = self._tracks.get(track_id)
        if state is None or state.entry_line is None or state.done or state.last_position is None:
            return None
        source, target = (
            (self.scene.line_a, self.scene.line_b) if state.entry_line == "A" else (self.scene.line_b, self.scene.line_a)
        )
        d_source = point_to_line_distance(state.last_position, source)
        d_target = point_to_line_distance(state.last_position, target)
        direction = self.scene.direction_a_to_b if state.entry_line == "A" else self.scene.direction_b_to_a
        total = d_source + d_target
        return direction, (d_source / total if total > 0 else 0.0)

    # ------------------------------------------------------------------ update
    def update(self, track_id: int, point: Point, frame_idx: int) -> Crossing | None:
        """Feed the centre of a confirmed track; returns a :class:`Crossing` when one completes."""
        s = self.settings
        state = self._tracks.setdefault(track_id, _TrackState())
        state.history.append(point)
        del state.history[:-50]
        crossing = None

        if state.entry_line is None:
            if point_to_line_distance(point, self.scene.line_a) < s.entry_threshold_px:
                state.entry_line = "A"
            elif point_to_line_distance(point, self.scene.line_b) < s.entry_threshold_px:
                state.entry_line = "B"
            if state.entry_line is not None:
                state.entry_frame = frame_idx
                state.entry_position = point
        elif not state.done:
            crossing = self._check_exit(track_id, state, point, frame_idx)

        state.last_position = point
        return crossing

    def _check_exit(self, track_id: int, state: _TrackState, point: Point, frame_idx: int) -> Crossing | None:
        s = self.settings
        if state.entry_line == "A":
            target, direction = self.scene.line_b, self.scene.direction_a_to_b
        else:
            target, direction = self.scene.line_a, self.scene.direction_b_to_a

        current = point_to_line_distance(point, target)
        previous = point_to_line_distance(state.last_position, target) if state.last_position else float("inf")
        near = current < s.exit_threshold_px or (current < 2 * s.exit_threshold_px and current < previous)
        if not near:
            return None
        if np.hypot(point[0] - state.entry_position[0], point[1] - state.entry_position[1]) < s.min_pixel_distance:
            return None

        frames_taken = frame_idx - state.entry_frame
        if not s.min_crossing_frames <= frames_taken <= s.max_crossing_frames:
            return None

        if s.speed_mode == "fixed_distance":
            distance = self.scene.reference_distance_m
        else:
            distance = world_distance(state.entry_position, point, self.homography)
        speed_kph = distance / (frames_taken / s.fps) * 3.6
        if not s.min_speed_kph <= speed_kph <= s.max_speed_kph:
            return None

        state.done = True
        crossing = Crossing(track_id, direction, state.entry_frame, frame_idx, speed_kph, distance)
        self.crossings.append(crossing)
        return crossing
