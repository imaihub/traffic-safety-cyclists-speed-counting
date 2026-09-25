import pytest

from cyclist_tracker.counting import Crossing
from cyclist_tracker.io_formats import (
    load_frame_list,
    mot_row,
    natural_key,
    read_mot,
    read_speed_csv,
    write_mot,
    write_speed_csv,
)

pytestmark = pytest.mark.unit


def test_mot_round_trip_and_path_stripping(tmp_path):
    path = tmp_path / "tracks.txt"
    write_mot(path, [mot_row("test/a.jpg", 1, (10, 20, 30, 60), 0.9), mot_row("b.jpg", 2, (0, 0, 5, 5), 1.0)])
    boxes = read_mot(path)
    assert boxes == {"a.jpg": [(1, [10.0, 20.0, 20.0, 40.0])], "b.jpg": [(2, [0.0, 0.0, 5.0, 5.0])]}


def test_read_mot_reports_malformed_line(tmp_path):
    path = tmp_path / "bad.txt"
    path.write_text("a.jpg,1,2,3\n")
    with pytest.raises(ValueError, match=":1: expected at least 6"):
        read_mot(path)
    path.write_text("a.jpg,x,2,3,4,5\n")
    with pytest.raises(ValueError, match=":1:"):
        read_mot(path)


def test_speed_csv_round_trip(tmp_path):
    path = tmp_path / "speeds.csv"
    write_speed_csv(path, [Crossing(4, "up", 10, 25, 18.25, 7.6)], fps=5)
    df = read_speed_csv(path)
    assert list(df.columns) == ["track_id", "speed_kph", "direction", "crossing_time_sec", "frames_taken",
                                "entry_frame", "exit_frame"]
    assert df.iloc[0].to_dict() == {"track_id": 4, "speed_kph": 18.25, "direction": "up", "crossing_time_sec": 3.0,
                                    "frames_taken": 15, "entry_frame": 10, "exit_frame": 25}


def test_read_speed_csv_requires_columns(tmp_path):
    path = tmp_path / "x.csv"
    path.write_text("id,speed\n1,2\n")
    with pytest.raises(ValueError, match="missing column"):
        read_speed_csv(path)


def test_frame_list_ignores_empty_lines(tmp_path):
    path = tmp_path / "list.txt"
    path.write_text("a.jpg\n\n  b.jpg  \n")
    assert load_frame_list(path) == ["a.jpg", "b.jpg"]


def test_natural_sort_orders_frame_numbers():
    names = ["v_frame_10000.jpg", "v_frame_9995.jpg", "v_frame_0005.jpg"]
    assert sorted(names) == ["v_frame_0005.jpg", "v_frame_10000.jpg", "v_frame_9995.jpg"]  # the old bug
    assert sorted(names, key=natural_key) == ["v_frame_0005.jpg", "v_frame_9995.jpg", "v_frame_10000.jpg"]
