import pytest

from cyclist_tracker.config import AnalysisSettings, SceneConfig
from cyclist_tracker.counting import CrossingCounter

pytestmark = pytest.mark.unit


@pytest.fixture
def scene():
    """A top-down scene: 100 pixels = 1 metre, line A at y=800 px (8 m), line B at y=200 px (2 m)."""
    return SceneConfig(
        name="test", roi=[(0, 0), (400, 0), (400, 1000), (0, 1000)],
        line_a=[(0, 800), (400, 800)], line_b=[(0, 200), (400, 200)],
        direction_a_to_b="north", direction_b_to_a="south",
        calibration_image_points=[(0, 0), (400, 0), (400, 1000), (0, 1000)],
        calibration_world_points=[(0, 0), (4, 0), (4, 10), (0, 10)],
        reference_distance_m=6.0,
    )


def feed(counter, track_id, ys, x=200.0, start=0):
    found = []
    for i, y in enumerate(ys):
        c = counter.update(track_id, (x, y), start + i)
        if c:
            found.append(c)
    return found


def test_crossing_from_a_to_b_gives_direction_and_speed(scene):
    counter = CrossingCounter(scene, AnalysisSettings())
    crossings = feed(counter, 1, range(840, 150, -40))  # 40 px = 0.4 m per frame at 5 fps = 7.2 km/h
    assert len(crossings) == 1
    c = crossings[0]
    assert (c.direction, c.entry_frame, c.exit_frame, c.frames_taken) == ("north", 1, 16, 15)
    assert c.distance_m == pytest.approx(6.0)
    assert c.speed_kph == pytest.approx(7.2)
    assert c.crossing_time(5.0) == pytest.approx(3.0)
    assert counter.counts == {"north": 1, "south": 0}


def test_crossing_from_b_to_a(scene):
    counter = CrossingCounter(scene, AnalysisSettings())
    crossings = feed(counter, 7, range(160, 900, 40))
    assert [c.direction for c in crossings] == ["south"]
    assert counter.total == 1


def test_each_cyclist_is_counted_once(scene):
    counter = CrossingCounter(scene, AnalysisSettings())
    feed(counter, 1, list(range(840, 150, -40)) + list(range(160, 900, 40)))
    assert counter.total == 1


def test_two_cyclists_are_counted_separately(scene):
    counter = CrossingCounter(scene, AnalysisSettings())
    feed(counter, 1, range(840, 150, -40))
    feed(counter, 2, range(160, 900, 40), start=30)
    assert counter.counts == {"north": 1, "south": 1}


def test_fixed_distance_mode_uses_reference_distance(scene):
    scene.reference_distance_m = 12.0
    counter = CrossingCounter(scene, AnalysisSettings(speed_mode="fixed_distance"))
    (c,) = feed(counter, 1, range(840, 150, -40))
    assert c.speed_kph == pytest.approx(12.0 / 3.0 * 3.6)


def test_too_fast_crossing_is_rejected(scene):
    counter = CrossingCounter(scene, AnalysisSettings(min_crossing_frames=5))
    assert feed(counter, 1, [800, 500, 200]) == []  # 2 frames between the lines


def test_implausible_speed_is_rejected(scene):
    counter = CrossingCounter(scene, AnalysisSettings(max_speed_kph=5))
    assert feed(counter, 1, range(840, 150, -40)) == []  # 7.2 km/h > 5 km/h


def test_small_movement_is_not_an_exit(scene):
    # a track that jitters around line A never reaches line B
    counter = CrossingCounter(scene, AnalysisSettings())
    assert feed(counter, 1, [800, 805, 795, 800, 810] * 5) == []


def test_track_far_from_lines_is_ignored(scene):
    counter = CrossingCounter(scene, AnalysisSettings())
    assert feed(counter, 1, [500, 490, 480, 470]) == []
    assert counter.progress(1) is None


def test_progress_between_lines(scene):
    counter = CrossingCounter(scene, AnalysisSettings())
    feed(counter, 1, [800, 500])  # entered at A, now half way
    direction, fraction = counter.progress(1)
    assert direction == "north"
    assert fraction == pytest.approx(0.5)


def test_invalid_configuration_is_refused(scene):
    scene.calibration_image_points, scene.calibration_world_points = [], []
    with pytest.raises(ValueError, match="calibration"):
        CrossingCounter(scene, AnalysisSettings(speed_mode="homography"))
    scene.reference_distance_m = 0
    with pytest.raises(ValueError, match="reference distance"):
        CrossingCounter(scene, AnalysisSettings(speed_mode="fixed_distance"))
    with pytest.raises(ValueError, match="Invalid configuration"):
        CrossingCounter(SceneConfig(), AnalysisSettings())


def test_as_row_matches_csv_columns(scene):
    counter = CrossingCounter(scene, AnalysisSettings())
    (c,) = feed(counter, 3, range(840, 150, -40))
    assert c.as_row(5.0) == [3, 7.2, "north", 3.0, 15, 1, 16]
