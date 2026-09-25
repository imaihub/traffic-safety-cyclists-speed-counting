import numpy as np
import pandas as pd
import pytest

from cyclist_tracker import evaluation as ev
from cyclist_tracker.io_formats import write_mot

pytestmark = pytest.mark.unit

HEADER = "track_id,speed_kph,direction,crossing_time_sec,frames_taken,entry_frame,exit_frame\n"


def _tracks(path, shift=0.0, swap_at=None):
    rows = []
    for f in range(10):
        for tid, x in ((1, 10.0), (2, 200.0)):
            out_id = (3 - tid) if swap_at is not None and f >= swap_at else tid
            rows.append([f"f{f:03d}.jpg", out_id, x + shift + f, 50, 40, 80, 1.0, 1, 1])
    write_mot(path, rows)
    return path


def test_perfect_tracking_scores_one(tmp_path):
    gt = _tracks(tmp_path / "gt.txt")
    summary = ev.evaluate_tracking(gt, gt)
    assert summary.loc["tracking", "mota"] == pytest.approx(1.0)
    assert summary.loc["tracking", "idf1"] == pytest.approx(1.0)
    assert summary.loc["tracking", "num_switches"] == 0


def test_identity_switch_is_detected(tmp_path):
    gt = _tracks(tmp_path / "gt.txt")
    pred = _tracks(tmp_path / "pred.txt", swap_at=5)
    summary = ev.evaluate_tracking(gt, pred)
    assert summary.loc["tracking", "num_switches"] == 2
    assert summary.loc["tracking", "idf1"] < 1.0


def test_boxes_below_iou_threshold_are_misses(tmp_path):
    gt = _tracks(tmp_path / "gt.txt")
    pred = _tracks(tmp_path / "pred.txt", shift=35)  # IoU (5/75) is far below 0.5
    summary = ev.evaluate_tracking(gt, pred, iou_threshold=0.5)
    assert summary.loc["tracking", "recall"] == pytest.approx(0.0)


def test_tracking_needs_shared_frames(tmp_path):
    gt = _tracks(tmp_path / "gt.txt")
    other = tmp_path / "other.txt"
    write_mot(other, [["x.jpg", 1, 0, 0, 1, 1, 1, 1, 1]])
    with pytest.raises(ValueError, match="No frame names are shared"):
        ev.evaluate_tracking(gt, other)
    with pytest.raises(ValueError, match="IoU"):
        ev.evaluate_tracking(gt, gt, iou_threshold=0)


def test_speed_metrics_known_values():
    matched = pd.DataFrame({"speed_kph_gt": [10.0, 20.0, 30.0], "speed_kph_pred": [11.0, 18.0, 30.0]})
    m = ev.speed_metrics(matched)
    assert m["MAE"] == pytest.approx(1.0)
    assert m["RMSE"] == pytest.approx(np.sqrt(5 / 3))
    assert m["R2"] == pytest.approx(1 - 5 / 200)
    assert m["Within_2kmh_%"] == pytest.approx(100.0)
    assert np.isnan(ev.speed_metrics(pd.DataFrame())["MAE"])


def test_match_speeds_uses_direction_and_tolerance():
    gt = pd.DataFrame({"direction": ["up", "down"], "entry_frame": [10, 50], "exit_frame": [20, 60],
                       "speed_kph": [15.0, 25.0]})
    pred = pd.DataFrame({"direction": ["up", "up", "down"], "entry_frame": [12, 11, 70], "exit_frame": [22, 20, 80],
                         "speed_kph": [14.0, 16.0, 25.0]})
    matched = ev.match_speeds(gt, pred, frame_tolerance=3)
    assert len(matched) == 1
    assert matched.iloc[0]["speed_kph_pred"] == 16.0  # closest in frames


def test_evaluate_speeds_and_figure(tmp_path):
    (tmp_path / "gt.csv").write_text(HEADER + "1,15,up,2,10,10,20\n2,25,down,2,10,50,60\n")
    (tmp_path / "a.csv").write_text(HEADER + "7,16,up,2,10,11,21\n8,24,down,2,10,50,61\n")
    table, matches = ev.evaluate_speeds(tmp_path / "gt.csv", [tmp_path / "a.csv"], ["A"])
    assert table.loc[0, "Model"] == "A"
    assert table.loc[0, "MAE"] == pytest.approx(1.0)
    assert table.loc[0, "Matches"] == 2
    fig = ev.speed_scatter_figure(matches)
    assert len(fig.axes) == 1
    with pytest.raises(ValueError, match="one name per"):
        ev.evaluate_speeds(tmp_path / "gt.csv", [tmp_path / "a.csv"], ["A", "B"])


def test_evaluate_counts_true_false_positives(tmp_path):
    (tmp_path / "gt.csv").write_text("track_id,direction,frames_taken,entry_frame,exit_frame\n"
                                     "1,up,10,10,20\n2,up,10,100,110\n3,down,5,50,55\n")
    (tmp_path / "pred.csv").write_text(HEADER + "5,20,up,2,10,12,21\n6,20,up,2,10,300,310\n7,20,down,1,5,51,56\n")
    table = ev.evaluate_counts(tmp_path / "gt.csv", tmp_path / "pred.csv", frame_tolerance=5).set_index("Direction")
    assert table.loc["up", ["Reference", "Counted", "TP", "FP", "FN"]].tolist() == [2, 2, 1, 1, 1]
    assert table.loc["down", "TP"] == 1
    assert table.loc["Total", "F1"] == pytest.approx(2 * 2 / 6)
