"""Configuration objects for a camera scene and for the analysis settings.

A *scene* describes one fixed camera view: where the cycle track is (region of
interest), the two virtual lines that a cyclist must cross to be counted, and
the calibration points that relate image pixels to metres on the road.

*Analysis settings* hold the tunable parameters of detection, tracking and the
counting/speed logic. Both are stored as plain JSON files so that they can be
edited by hand, shared, or created with the "Camera scene" tab of the GUI.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any

Point = tuple[float, float]


def _points(value: Any, expected: int | None = None, name: str = "points") -> list[Point]:
    pts = [(float(p[0]), float(p[1])) for p in value]
    if expected is not None and len(pts) != expected:
        raise ValueError(f"{name} needs exactly {expected} points, got {len(pts)}")
    return pts


@dataclass
class SceneConfig:
    """Geometry of one camera view (all image coordinates are in pixels)."""

    name: str = "Unnamed scene"
    #: Polygon around the cycle track. Detections whose centre is outside are ignored.
    roi: list[Point] = field(default_factory=list)
    #: First counting line ("line A"), two points.
    line_a: list[Point] = field(default_factory=list)
    #: Second counting line ("line B"), two points.
    line_b: list[Point] = field(default_factory=list)
    #: Label written for cyclists that cross line A first and then line B.
    direction_a_to_b: str = "down"
    #: Label written for cyclists that cross line B first and then line A.
    direction_b_to_a: str = "up"
    #: Four or more pixel positions of points whose road position is known.
    calibration_image_points: list[Point] = field(default_factory=list)
    #: The same points in metres on the road surface (x across, y along the track).
    calibration_world_points: list[Point] = field(default_factory=list)
    #: Real distance in metres between line A and line B (used by the fixed-distance speed mode).
    reference_distance_m: float = 0.0

    def __post_init__(self) -> None:
        self.roi = _points(self.roi, name="roi")
        self.line_a = _points(self.line_a, name="line_a")
        self.line_b = _points(self.line_b, name="line_b")
        self.calibration_image_points = _points(self.calibration_image_points)
        self.calibration_world_points = _points(self.calibration_world_points)
        self.reference_distance_m = float(self.reference_distance_m)

    # ------------------------------------------------------------------ checks
    def problems(self) -> list[str]:
        """Return human-readable problems; an empty list means the scene is usable."""
        issues = []
        if len(self.roi) < 3:
            issues.append("The region of interest needs at least 3 points.")
        if len(self.line_a) != 2:
            issues.append("Line A needs exactly 2 points.")
        if len(self.line_b) != 2:
            issues.append("Line B needs exactly 2 points.")
        if self.direction_a_to_b.strip() == "" or self.direction_b_to_a.strip() == "":
            issues.append("Both direction names must be filled in.")
        if self.direction_a_to_b == self.direction_b_to_a:
            issues.append("The two direction names must be different.")
        n_img, n_world = len(self.calibration_image_points), len(self.calibration_world_points)
        if n_img != n_world:
            issues.append(f"Calibration: {n_img} image points but {n_world} road points.")
        elif 0 < n_img < 4:
            issues.append("Calibration needs at least 4 point pairs.")
        return issues

    @property
    def has_calibration(self) -> bool:
        return len(self.calibration_image_points) >= 4 and len(self.calibration_image_points) == len(
            self.calibration_world_points
        )

    # ---------------------------------------------------------------- file I/O
    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        for key in ("roi", "line_a", "line_b", "calibration_image_points", "calibration_world_points"):
            data[key] = [list(p) for p in data[key]]
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SceneConfig:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown scene field(s): {', '.join(sorted(unknown))}")
        return cls(**data)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> SceneConfig:
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data.get("scene", data))


@dataclass
class AnalysisSettings:
    """Tunable parameters. Defaults reproduce the experiments in the paper."""

    # Detection
    confidence_threshold: float = 0.7
    # Timing: frame rate of the *processed* sequence (25 fps video, every 5th frame -> 5)
    fps: float = 5.0
    # Counting / speed logic (pixels, frames, km/h)
    entry_threshold_px: float = 15.0
    exit_threshold_px: float = 15.0
    min_crossing_frames: int = 5
    max_crossing_frames: int = 2000
    min_pixel_distance: float = 50.0
    min_speed_kph: float = 3.0
    max_speed_kph: float = 50.0
    #: "homography" (distance measured with the calibration) or "fixed_distance"
    #: (distance = scene.reference_distance_m, used to create reference speeds).
    speed_mode: str = "homography"
    #: Point of the box that is followed: "bottom" (bottom-centre, where the wheels touch the
    #: road - matches the road-plane calibration) or "center" (box centre, as in the paper).
    tracking_point: str = "bottom"
    # DeepSORT tracker
    max_age: int = 6
    n_init: int = 2
    max_cosine_distance: float = 0.3
    embedder: str = "mobilenet"
    embedder_weights: str | None = None
    use_gpu: bool = True
    half_precision: bool = True

    SPEED_MODES = ("homography", "fixed_distance")
    TRACKING_POINTS = ("bottom", "center")

    @classmethod
    def paper(cls) -> AnalysisSettings:
        """The exact settings of the original experiments (box centre as reference point)."""
        return cls(tracking_point="center")

    def problems(self) -> list[str]:
        issues = []
        if not 0 < self.confidence_threshold < 1:
            issues.append("Confidence threshold must be between 0 and 1.")
        if self.fps <= 0:
            issues.append("Frame rate must be positive.")
        if self.min_crossing_frames > self.max_crossing_frames:
            issues.append("Minimum crossing frames is larger than the maximum.")
        if self.min_speed_kph > self.max_speed_kph:
            issues.append("Minimum speed is larger than the maximum speed.")
        if self.speed_mode not in self.SPEED_MODES:
            issues.append(f"Speed mode must be one of {', '.join(self.SPEED_MODES)}.")
        if self.tracking_point not in self.TRACKING_POINTS:
            issues.append(f"Tracking point must be one of {', '.join(self.TRACKING_POINTS)}.")
        return issues

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AnalysisSettings:
        known = {f.name for f in fields(cls)}
        unknown = set(data) - known
        if unknown:
            raise ValueError(f"Unknown setting(s): {', '.join(sorted(unknown))}")
        return cls(**data)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> AnalysisSettings:
        return cls.from_dict(json.loads(Path(path).read_text()))


def utrecht_scene() -> SceneConfig:
    """The camera PSFV-01-N413-Zuid (Utrecht) used in the original paper."""
    # Road markings: 14 blocks of 0.5 m with 13 gaps of 0.5 m between line A and B.
    return SceneConfig(
        name="Utrecht PSFV-01-N413-Zuid (paper)",
        roi=[(937, 390), (1100, 388), (1003, 1080), (469, 1079)],
        line_a=[(832, 544), (1085, 548)],
        line_b=[(930, 387), (1103, 387)],
        direction_a_to_b="down",
        direction_b_to_a="up",
        calibration_image_points=[(937.30, 390.20), (1100.30, 388.70), (1084.60, 506.20), (862.40, 504.00)],
        calibration_world_points=[(0, 0), (4, 0), (4, 10.5), (0, 10.5)],
        reference_distance_m=14 * 0.5 + 13 * 0.5,
    )
