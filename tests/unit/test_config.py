import json

import pytest

from cyclist_tracker.config import AnalysisSettings, SceneConfig, utrecht_scene

pytestmark = pytest.mark.unit


def test_paper_scene_is_complete():
    scene = utrecht_scene()
    assert scene.problems() == []
    assert scene.has_calibration
    assert scene.reference_distance_m == pytest.approx(13.5)  # 14 blocks + 13 gaps of 0.5 m


def test_scene_json_round_trip(tmp_path):
    scene = utrecht_scene()
    path = tmp_path / "scene.json"
    scene.save(path)
    assert SceneConfig.load(path) == scene
    assert json.loads(path.read_text())["line_a"] == [[832.0, 544.0], [1085.0, 548.0]]


def test_scene_rejects_unknown_fields():
    with pytest.raises(ValueError, match="Unknown scene field"):
        SceneConfig.from_dict({"name": "x", "colour": "red"})


def test_scene_problems_are_explained():
    scene = SceneConfig(roi=[(0, 0), (1, 1)], line_a=[(0, 0)], direction_a_to_b="up", direction_b_to_a="up",
                        calibration_image_points=[(0, 0)] * 3, calibration_world_points=[(0, 0)] * 2)
    problems = " ".join(scene.problems())
    for text in ("at least 3 points", "Line A", "Line B", "must be different", "3 image points but 2"):
        assert text in problems


def test_scene_with_too_few_calibration_points():
    scene = utrecht_scene()
    scene.calibration_image_points = scene.calibration_image_points[:3]
    scene.calibration_world_points = scene.calibration_world_points[:3]
    assert any("at least 4" in p for p in scene.problems())
    assert not scene.has_calibration


def test_settings_defaults_and_paper_preset():
    assert AnalysisSettings().problems() == []
    assert AnalysisSettings().tracking_point == "bottom"
    paper = AnalysisSettings.paper()
    assert paper.tracking_point == "center"
    assert (paper.confidence_threshold, paper.fps, paper.max_age, paper.n_init) == (0.7, 5.0, 6, 2)


@pytest.mark.parametrize(
    ("changes", "message"),
    [({"speed_mode": "magic"}, "Speed mode"), ({"min_speed_kph": 60}, "Minimum speed"),
     ({"fps": 0}, "Frame rate"), ({"confidence_threshold": 1.5}, "Confidence"),
     ({"min_crossing_frames": 10, "max_crossing_frames": 5}, "crossing frames"),
     ({"tracking_point": "top"}, "Tracking point")],
)
def test_settings_problems(changes, message):
    assert any(message in p for p in AnalysisSettings(**changes).problems())


def test_settings_json_round_trip(tmp_path):
    s = AnalysisSettings(embedder="histogram", fps=12.5)
    s.save(tmp_path / "s.json")
    assert AnalysisSettings.load(tmp_path / "s.json") == s
