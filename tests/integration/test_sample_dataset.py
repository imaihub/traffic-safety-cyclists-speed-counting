"""The real sample dataset (Stanford Drone Dataset excerpt) shipped in sample_data/."""

import pandas as pd
import pytest

from cyclist_tracker.config import AnalysisSettings, SceneConfig
from cyclist_tracker.dataset import validate_dataset
from cyclist_tracker.detection import AnnotationDetector
from cyclist_tracker.evaluation import evaluate_counts, evaluate_speeds, evaluate_tracking
from cyclist_tracker.pipeline import open_source, run_analysis
from cyclist_tracker.tracking import CyclistTracker

from ..conftest import SAMPLE

pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(not (SAMPLE / "scene.json").is_file(), reason="sample dataset not present")]


def test_sample_follows_the_documented_structure():
    report = validate_dataset(SAMPLE)
    assert not report.errors and not [c for c in report.checks if c.level == "warning"]
    assert report.ready_for_training and report.ready_for_analysis
    assert report.images == {"train": 74, "val": 70, "test": 367}
    assert (SAMPLE / "LICENSE.md").is_file() and "CC BY-NC-SA 3.0" in (SAMPLE / "LICENSE.md").read_text()
    assert SceneConfig.load(SAMPLE / "scene.json").problems() == []


def test_reference_speeds_are_plausible():
    speeds = pd.read_csv(SAMPLE / "ground_truth" / "speeds_test.csv")
    counts = pd.read_csv(SAMPLE / "ground_truth" / "counts_test.csv")
    assert len(speeds) == len(counts) == 20
    assert speeds["speed_kph"].between(5, 40).all()
    assert (speeds["exit_frame"] > speeds["entry_frame"]).all()


def test_analysis_of_the_sample_matches_its_ground_truth(tmp_path):
    settings = AnalysisSettings.load(SAMPLE / "settings.json")
    assert settings.tracking_point == "center"  # top-down camera
    settings.embedder, settings.use_gpu = "histogram", False
    scene = SceneConfig.load(SAMPLE / "scene.json")
    result = run_analysis(open_source(SAMPLE / "test"), AnnotationDetector(SAMPLE / "annotations" / "test.json"),
                          CyclistTracker(settings), scene, settings, tmp_path, save_video=False)
    gt = SAMPLE / "ground_truth"
    counts = evaluate_counts(gt / "counts_test.csv", result.output_files["speeds"]).iloc[-1]
    assert counts["F1"] >= 0.9
    speeds, _ = evaluate_speeds(gt / "speeds_test.csv", [result.output_files["speeds"]])
    assert speeds.loc[0, "Matches"] >= 18 and speeds.loc[0, "MAE"] < 1.5
    tracking = evaluate_tracking(gt / "tracking_test.txt", result.output_files["tracks"]).loc["tracking"]
    assert tracking["idf1"] > 0.75 and tracking["mota"] > 0.75


def test_sample_video_has_all_frames():
    source = open_source(SAMPLE / "gates_video4_5fps.mp4")
    assert len(source) == 367 and source.fps == pytest.approx(4.995, abs=0.01)
