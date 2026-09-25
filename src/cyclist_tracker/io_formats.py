"""Reading and writing the plain-text file formats used by the project.

Tracking file (MOT-style, one box per line, no header)::

    frame_name,track_id,x,y,width,height,confidence,class_id,visibility

Speed / crossing file (CSV with header)::

    track_id,speed_kph,direction,crossing_time_sec,frames_taken,entry_frame,exit_frame
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from pathlib import Path

import pandas as pd

from .counting import SPEED_CSV_HEADERS, Crossing

TrackBox = tuple[int, list[float]]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def natural_key(text: str) -> list:
    """Sort key that orders ``frame_9995`` before ``frame_10000`` (plain sorting does not)."""
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", text)]


def load_frame_list(path: str | Path) -> list[str]:
    """Read a text file with one image name/path per line (empty lines ignored)."""
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def write_mot(path: str | Path, rows: Iterable[Sequence]) -> None:
    with open(path, "w", newline="") as f:
        for row in rows:
            f.write(",".join(str(v) for v in row) + "\n")


def mot_row(frame_name: str, track_id: int, ltrb: Sequence[float], confidence: float) -> list:
    x1, y1, x2, y2 = ltrb
    return [frame_name, track_id, f"{x1:.2f}", f"{y1:.2f}", f"{x2 - x1:.2f}", f"{y2 - y1:.2f}", f"{confidence:.4f}", 1, 1]


def read_mot(path: str | Path) -> dict[str, list[TrackBox]]:
    """Read a tracking file into ``{frame_file_name: [(track_id, [x, y, w, h]), ...]}``.

    Only the file name of the first column is kept, so ``test/img_001.jpg`` and
    ``img_001.jpg`` refer to the same frame. Malformed lines raise ``ValueError``.
    """
    boxes: dict[str, list[TrackBox]] = defaultdict(list)
    with open(path) as f:
        for number, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split(",")
            if len(parts) < 6:
                raise ValueError(f"{path}:{number}: expected at least 6 comma-separated values")
            try:
                track_id = int(float(parts[1]))
                box = [float(v) for v in parts[2:6]]
            except ValueError as exc:
                raise ValueError(f"{path}:{number}: {exc}") from exc
            boxes[Path(parts[0]).name].append((track_id, box))
    return dict(boxes)


def write_speed_csv(path: str | Path, crossings: Iterable[Crossing], fps: float) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(SPEED_CSV_HEADERS)
        for c in crossings:
            writer.writerow(c.as_row(fps))


def read_speed_csv(path: str | Path, required: Sequence[str] = ("direction", "entry_frame", "exit_frame")) -> pd.DataFrame:
    """Read a speed or counting CSV and check that the required columns are present."""
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing column(s) {', '.join(missing)}; found {', '.join(df.columns)}")
    return df
