"""Drawing of overlays (region, lines, tracks, counts) on BGR frames."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

from .config import SceneConfig
from .geometry import Point

# BGR colours
ROI_COLOUR = (80, 220, 80)
LINE_A_COLOUR = (0, 140, 255)  # orange, as in the GUI
LINE_B_COLOUR = (255, 144, 30)  # blue
TRACK_COLOUR = (0, 220, 255)
COUNTED_COLOUR = (60, 200, 60)
CALIBRATION_COLOUR = (255, 0, 255)


def _pt(p: Point) -> tuple[int, int]:
    return int(round(p[0])), int(round(p[1]))


def draw_scene(frame: np.ndarray, scene: SceneConfig, calibration: bool = False) -> np.ndarray:
    if len(scene.roi) >= 3:
        cv2.polylines(frame, [np.asarray(scene.roi, dtype=np.int32)], True, ROI_COLOUR, 2)
    for line, colour, label in ((scene.line_a, LINE_A_COLOUR, "A"), (scene.line_b, LINE_B_COLOUR, "B")):
        if len(line) == 2:
            cv2.line(frame, _pt(line[0]), _pt(line[1]), colour, 3)
            cv2.putText(frame, label, _pt(line[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.9, colour, 2)
    if calibration:
        for i, p in enumerate(scene.calibration_image_points):
            cv2.circle(frame, _pt(p), 6, CALIBRATION_COLOUR, -1)
            cv2.putText(frame, str(i + 1), _pt((p[0] + 8, p[1] - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        CALIBRATION_COLOUR, 2)
    return frame


def draw_track(frame: np.ndarray, track_id: int, ltrb: Sequence[float], note: str = "",
               counted: bool = False) -> None:
    colour = COUNTED_COLOUR if counted else TRACK_COLOUR
    x1, y1, x2, y2 = (int(round(v)) for v in ltrb)
    cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    cv2.circle(frame, (cx, cy), 4, colour, -1)
    label = f"ID {track_id}" + (f"  {note}" if note else "")
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
    cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), colour, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)


def draw_counts(frame: np.ndarray, counts: dict[str, int], frame_label: str = "") -> None:
    lines = [f"{name}: {value}" for name, value in counts.items()]
    lines.append(f"Total: {sum(counts.values())}")
    if frame_label:
        lines.append(frame_label)
    width = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0][0] for t in lines) + 20
    height = 30 * len(lines) + 10
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + width, 10 + height), (255, 255, 255), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, dst=frame)
    for i, text in enumerate(lines):
        cv2.putText(frame, text, (20, 38 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
