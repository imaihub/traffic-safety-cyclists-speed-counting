"""Evaluation of tracking (MOT metrics), speed estimation and counting against ground truth."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.figure import Figure

from .geometry import bbox_iou
from .io_formats import read_mot, read_speed_csv

log = logging.getLogger(__name__)

#: Tracking metrics shown to users, with a plain-language explanation.
MOT_METRICS = {
    "idf1": "IDF1 - how well each cyclist keeps the same ID (higher is better)",
    "mota": "MOTA - overall accuracy: misses, false boxes and ID switches (higher is better)",
    "precision": "Precision - share of tracked boxes that are real cyclists",
    "recall": "Recall - share of real cyclists that were tracked",
    "num_switches": "ID switches - times a cyclist got a new ID (lower is better)",
    "num_false_positives": "False positives - boxes without a real cyclist",
    "num_misses": "Misses - real cyclists without a box",
    "mostly_tracked": "Mostly tracked - cyclists followed for at least 80% of their time",
    "motp": "MOTP - average box misalignment, 1 - IoU (lower is better)",
}


# ------------------------------------------------------------------------------------ tracking
def evaluate_tracking(gt_file: str | Path, pred_file: str | Path, iou_threshold: float = 0.5) -> pd.DataFrame:
    """CLEAR-MOT and identity metrics (motmetrics) for one prediction file.

    Frames are matched by file name. Returns a one-row DataFrame indexed by ``"tracking"``.
    """
    import motmetrics as mm

    if not 0 < iou_threshold <= 1:
        raise ValueError("The IoU threshold must be larger than 0 and at most 1.")
    gt, pred = read_mot(gt_file), read_mot(pred_file)
    if not gt:
        raise ValueError(f"The ground-truth file {gt_file} contains no boxes.")
    common = set(gt) & set(pred)
    if pred and not common:
        raise ValueError("No frame names are shared between ground truth and predictions; "
                         "check that both files refer to the same images.")
    acc = mm.MOTAccumulator(auto_id=True)
    for frame in sorted(set(gt) | set(pred)):
        g, p = gt.get(frame, []), pred.get(frame, [])
        dist = np.full((len(g), len(p)), np.nan)
        for i, (_, gbox) in enumerate(g):
            for j, (_, pbox) in enumerate(p):
                iou = bbox_iou(gbox, pbox)
                if iou >= iou_threshold:
                    dist[i, j] = 1 - iou
        acc.update([t for t, _ in g], [t for t, _ in p], dist)
    metrics = mm.metrics.create()
    return metrics.compute(acc, metrics=mm.metrics.motchallenge_metrics, name="tracking")


# --------------------------------------------------------------------------------------- speed
def match_speeds(gt: pd.DataFrame, pred: pd.DataFrame, frame_tolerance: int = 5) -> pd.DataFrame:
    """Pair each reference crossing with the closest predicted crossing.

    A prediction qualifies when it has the same direction and both entry and exit
    frame are within ``frame_tolerance`` frames. This is the matching used for the
    results in the paper.
    """
    rows = []
    for _, g in gt.iterrows():
        cand = pred[
            (pred["direction"] == g["direction"])
            & ((pred["entry_frame"] - g["entry_frame"]).abs() <= frame_tolerance)
            & ((pred["exit_frame"] - g["exit_frame"]).abs() <= frame_tolerance)
        ]
        if cand.empty:
            continue
        diff = (cand["entry_frame"] - g["entry_frame"]).abs() + (cand["exit_frame"] - g["exit_frame"]).abs()
        best = cand.loc[diff.idxmin()]
        rows.append({
            "direction": g["direction"],
            "entry_frame_gt": int(g["entry_frame"]),
            "exit_frame_gt": int(g["exit_frame"]),
            "speed_kph_gt": float(g["speed_kph"]),
            "entry_frame_pred": int(best["entry_frame"]),
            "exit_frame_pred": int(best["exit_frame"]),
            "speed_kph_pred": float(best["speed_kph"]),
        })
    columns = ["direction", "entry_frame_gt", "exit_frame_gt", "speed_kph_gt", "entry_frame_pred",
               "exit_frame_pred", "speed_kph_pred"]
    return pd.DataFrame(rows, columns=columns)


def speed_metrics(matched: pd.DataFrame) -> dict[str, float]:
    if matched.empty:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "Within_2kmh_%": np.nan, "Matches": 0}
    y, y_hat = matched["speed_kph_gt"].to_numpy(float), matched["speed_kph_pred"].to_numpy(float)
    err = y_hat - y
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err**2))),
        "R2": 1 - float(np.sum(err**2)) / ss_tot if ss_tot > 0 else np.nan,
        "Within_2kmh_%": float(np.mean(np.abs(err) <= 2) * 100),
        "Matches": len(matched),
    }


def evaluate_speeds(gt_csv: str | Path, pred_csvs: Sequence[str | Path], names: Sequence[str] | None = None,
                    frame_tolerance: int = 5) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Compare one or more prediction files with the reference speeds.

    Returns a table with one row per model and the matched pairs per model.
    """
    names = list(names) if names else [Path(p).stem for p in pred_csvs]
    if len(names) != len(pred_csvs):
        raise ValueError("Give exactly one name per prediction file.")
    required = ("direction", "entry_frame", "exit_frame", "speed_kph")
    gt = read_speed_csv(gt_csv, required)
    table, matches = [], {}
    for name, path in zip(names, pred_csvs, strict=True):
        matched = match_speeds(gt, read_speed_csv(path, required), frame_tolerance)
        matches[name] = matched
        table.append({"Model": name, "Reference crossings": len(gt)} | speed_metrics(matched))
    return pd.DataFrame(table), matches


def speed_scatter_figure(matches: dict[str, pd.DataFrame], fig: Figure | None = None) -> Figure:
    """Predicted vs. reference speed, one panel per model, with the ±2 km/h band.

    Draws into ``fig`` when given (e.g. a figure embedded in the GUI), otherwise into a new figure.
    """
    n = max(1, len(matches))
    if fig is None:
        fig = Figure(figsize=(4.2 * n, 4.2), layout="constrained")
    fig.clear()
    for i, (name, df) in enumerate(matches.items() or {"(no data)": pd.DataFrame()}.items()):
        ax = fig.add_subplot(1, n, i + 1)
        top = 50.0
        if not df.empty:
            top = float(max(df["speed_kph_gt"].max(), df["speed_kph_pred"].max())) + 5
            for direction, part in df.groupby("direction"):
                ax.scatter(part["speed_kph_gt"], part["speed_kph_pred"], s=22, alpha=0.8, label=str(direction))
            m = speed_metrics(df)
            ax.set_title(f"{name}\nMAE {m['MAE']:.2f} km/h, R² {m['R2']:.3f}", fontsize=10)
            ax.legend(title="Direction", fontsize=8)
        else:
            ax.set_title(f"{name}\nno matched crossings", fontsize=10)
        ax.fill_between([0, top], [-2, top - 2], [2, top + 2], color="grey", alpha=0.15, label="±2 km/h")
        ax.plot([0, top], [0, top], "--", color="black", linewidth=1)
        ax.set(xlim=(0, top), ylim=(0, top), xlabel="Reference speed (km/h)", ylabel="Estimated speed (km/h)")
        ax.grid(alpha=0.3)
    return fig


# ------------------------------------------------------------------------------------ counting
def evaluate_counts(gt_csv: str | Path, pred_csv: str | Path, frame_tolerance: int = 10) -> pd.DataFrame:
    """Compare counted crossings with the reference crossings, per direction and in total.

    Each reference crossing can be matched to at most one predicted crossing with
    the same direction whose entry and exit frame are within ``frame_tolerance``.
    Unmatched predictions are false positives (e.g. a moped counted as cyclist),
    unmatched references are misses.
    """
    required = ("direction", "entry_frame", "exit_frame")
    gt, pred = read_speed_csv(gt_csv, required), read_speed_csv(pred_csv, required)
    rows = []
    for direction in sorted(set(gt["direction"]) | set(pred["direction"]), key=str):
        g = gt[gt["direction"] == direction]
        p = pred[pred["direction"] == direction]
        pairs = []
        for gi, grow in g.iterrows():
            for pi, prow in p.iterrows():
                de = abs(grow["entry_frame"] - prow["entry_frame"])
                dx = abs(grow["exit_frame"] - prow["exit_frame"])
                if de <= frame_tolerance and dx <= frame_tolerance:
                    pairs.append((de + dx, gi, pi))
        used_g, used_p = set(), set()
        for _, gi, pi in sorted(pairs):
            if gi not in used_g and pi not in used_p:
                used_g.add(gi)
                used_p.add(pi)
        rows.append(_count_row(str(direction), len(g), len(p), len(used_g)))
    rows.append(_count_row("Total", sum(r["Reference"] for r in rows), sum(r["Counted"] for r in rows),
                           sum(r["TP"] for r in rows)))
    return pd.DataFrame(rows)


def _count_row(direction: str, n_gt: int, n_pred: int, tp: int) -> dict:
    precision = tp / n_pred if n_pred else np.nan
    recall = tp / n_gt if n_gt else np.nan
    f1 = 2 * tp / (n_gt + n_pred) if (n_gt + n_pred) else np.nan
    return {"Direction": direction, "Reference": n_gt, "Counted": n_pred, "TP": tp, "FP": n_pred - tp,
            "FN": n_gt - tp, "Precision": precision, "Recall": recall, "F1": f1}
