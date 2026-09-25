"""Small geometric helpers: points, lines, polygons, boxes and homographies."""

from __future__ import annotations

import math
from collections.abc import Sequence

import cv2
import numpy as np

Point = tuple[float, float]


def parse_points(text: str) -> list[Point]:
    """Parse ``"x1,y1,x2,y2,..."`` into ``[(x1, y1), (x2, y2), ...]``."""
    values = [v for v in text.replace(";", ",").replace(" ", ",").split(",") if v != ""]
    if len(values) % 2:
        raise ValueError(f"Expected an even number of coordinates, got {len(values)}: {text!r}")
    numbers = [float(v) for v in values]
    return list(zip(numbers[0::2], numbers[1::2], strict=True))


def format_points(points: Sequence[Point]) -> str:
    return ",".join(f"{x:g},{y:g}" for x, y in points)


def point_to_line_distance(point: Point, line: Sequence[Point]) -> float:
    """Perpendicular distance from ``point`` to the infinite line through ``line[0]`` and ``line[1]``."""
    (x, y), (x1, y1), (x2, y2) = point, line[0], line[1]
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length == 0:
        return math.hypot(x - x1, y - y1)
    return abs(dy * x - dx * y + x2 * y1 - y2 * x1) / length


def polygon_array(polygon: Sequence[Point]) -> np.ndarray:
    return np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)


def point_in_polygon(point: Point, polygon: Sequence[Point] | np.ndarray) -> bool:
    """True when ``point`` is inside or on the border of ``polygon``."""
    poly = polygon if isinstance(polygon, np.ndarray) else polygon_array(polygon)
    return cv2.pointPolygonTest(poly, (float(point[0]), float(point[1])), False) >= 0


def bbox_center(ltrb: Sequence[float]) -> Point:
    x1, y1, x2, y2 = ltrb
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0


def reference_point(ltrb: Sequence[float], mode: str = "bottom") -> Point:
    """The point of a box that is followed: its centre or its bottom-centre (road contact)."""
    x1, y1, x2, y2 = ltrb
    return ((x1 + x2) / 2.0, y2 if mode == "bottom" else (y1 + y2) / 2.0)


def xywh_to_ltrb(box: Sequence[float]) -> tuple[float, float, float, float]:
    x, y, w, h = box
    return x, y, x + w, y + h


def ltrb_to_xywh(box: Sequence[float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return x1, y1, x2 - x1, y2 - y1


def bbox_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Intersection over union of two boxes in ``[x, y, w, h]`` format."""
    ax1, ay1, ax2, ay2 = xywh_to_ltrb(box_a)
    bx1, by1, bx2, by2 = xywh_to_ltrb(box_b)
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    union = box_a[2] * box_a[3] + box_b[2] * box_b[3] - inter
    return inter / union if union > 0 else 0.0


def compute_homography(image_points: Sequence[Point], world_points: Sequence[Point]) -> np.ndarray:
    """Homography that maps image pixels onto road coordinates in metres.

    Needs at least four point pairs; with more pairs a least-squares (DLT) fit is used.
    """
    src = np.asarray(image_points, dtype=np.float64)
    dst = np.asarray(world_points, dtype=np.float64)
    if len(src) < 4 or len(src) != len(dst):
        raise ValueError("A homography needs at least 4 matching image/world point pairs.")
    matrix, _ = cv2.findHomography(src, dst, 0)
    if matrix is None:
        raise ValueError("Could not compute a homography: are three or more points on one line?")
    return matrix


def pixel_to_world(point: Point, homography: np.ndarray) -> np.ndarray:
    """Map a pixel position onto the road plane (metres)."""
    src = np.asarray([[point]], dtype=np.float64)
    return cv2.perspectiveTransform(src, homography)[0][0]


def world_to_pixel(point: Point, homography: np.ndarray) -> np.ndarray:
    return pixel_to_world(point, np.linalg.inv(homography))


def world_distance(p1: Point, p2: Point, homography: np.ndarray) -> float:
    """Distance in metres on the road between two pixel positions."""
    return float(np.linalg.norm(pixel_to_world(p2, homography) - pixel_to_world(p1, homography)))


def calibration_error(image_points: Sequence[Point], world_points: Sequence[Point], homography: np.ndarray) -> float:
    """Mean re-projection error (metres) of the calibration points; 0 for an exact fit."""
    errors = [
        np.linalg.norm(pixel_to_world(p, homography) - np.asarray(w, dtype=np.float64))
        for p, w in zip(image_points, world_points, strict=True)
    ]
    return float(np.mean(errors)) if errors else 0.0
