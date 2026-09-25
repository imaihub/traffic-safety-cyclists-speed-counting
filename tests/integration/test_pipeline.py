"""The whole pipeline on the synthetic demo dataset (perfect detections, real DeepSORT tracker)."""

import json

import pandas as pd
import pytest

from cyclist_tracker.detection import AnnotationDetector
from cyclist_tracker.evaluation import evaluate_counts, evaluate_speeds, evaluate_tracking
from cyclist_tracker.pipeline import ImageFolderSource, VideoSource, open_source, run_analysis
from cyclist_tracker.tracking import CyclistTracker

pytestmark = pytest.mark.integration


def _run(demo_dataset, scene, settings, tmp_path, source=None, **kwargs):
    source = source or open_source(demo_dataset / "test")
    detector = AnnotationDetector(demo_dataset / "annotations" / "test.json")
    return run_analysis(source, detector, CyclistTracker(settings), scene, settings, tmp_path, **kwargs)


def test_demo_dataset_is_counted_and_measured_correctly(demo_dataset, demo_scene, fast_settings, tmp_path):
    result = _run(demo_dataset, demo_scene, fast_settings, tmp_path, name="run")
    reference = pd.read_csv(demo_dataset / "ground_truth" / "counts_test.csv")
    assert result.counts == reference["direction"].value_counts().to_dict()
    assert result.frames_processed == 220 and not result.stopped
    for key in ("tracks", "speeds", "summary", "video"):
        assert (tmp_path / result.output_files[key].split("/")[-1]).is_file()
    summary = json.loads((tmp_path / "run_summary.json").read_text())
    assert summary["total"] == sum(result.counts.values())
    assert summary["settings"]["embedder"] == "histogram"

    speeds, _ = evaluate_speeds(demo_dataset / "ground_truth" / "speeds_test.csv", [result.output_files["speeds"]])
    assert speeds.loc[0, "Matches"] == len(reference)
    assert speeds.loc[0, "MAE"] < 1.5  # km/h, against the true simulated speeds

    counts = evaluate_counts(demo_dataset / "ground_truth" / "counts_test.csv", result.output_files["speeds"])
    assert counts.iloc[-1]["F1"] == pytest.approx(1.0)

    tracking = evaluate_tracking(demo_dataset / "ground_truth" / "tracking_test.txt", result.output_files["tracks"])
    assert tracking.loc["tracking", "num_switches"] == 0
    assert tracking.loc["tracking", "precision"] > 0.9


class _VideoFrameDetector:
    """Annotated boxes for video frames: video frame number n is demo test frame n."""

    def __init__(self, coco):
        self.inner = AnnotationDetector(coco)

    def detect(self, image, name):
        number = int(name.rsplit("_", 1)[1].split(".")[0])
        return self.inner.detect(image, f"demo_test_frame_{number:05d}.jpg")


def test_video_input_gives_the_same_counts(demo_dataset, demo_scene, fast_settings, tmp_path):
    source = VideoSource(demo_dataset / "demo_test_video.mp4")
    assert len(source) == 220 and source.fps == pytest.approx(5.0)
    detector = _VideoFrameDetector(demo_dataset / "annotations" / "test.json")
    result = run_analysis(source, detector, CyclistTracker(fast_settings), demo_scene, fast_settings, tmp_path,
                          save_video=False)
    assert result.counts == {"away": 8, "towards": 2}
    halved = VideoSource(demo_dataset / "demo_test_video.mp4", stride=2)
    assert len(halved) == 110 and halved.fps == pytest.approx(2.5)


def test_stop_button_ends_the_run_early(demo_dataset, demo_scene, fast_settings, tmp_path):
    calls = []
    result = _run(demo_dataset, demo_scene, fast_settings, tmp_path, save_video=False,
                  should_stop=lambda: len(calls) >= 10, progress=lambda d, t, c: calls.append(d))
    assert result.stopped and result.frames_processed == 10
    assert (tmp_path / "analysis_summary.json").is_file()


def test_callbacks_report_progress_preview_and_crossings(demo_dataset, demo_scene, fast_settings, tmp_path):
    progress, previews, crossings = [], [], []
    result = _run(demo_dataset, demo_scene, fast_settings, tmp_path, save_video=False,
                  progress=lambda d, t, c: progress.append((d, t)), preview=previews.append,
                  on_crossing=crossings.append)
    assert progress[-1] == (220, 220)
    assert len(previews) == 220 and previews[0].shape == (540, 960, 3)
    assert crossings == result.crossings


def test_frame_list_restricts_frames(demo_dataset, tmp_path):
    names = sorted(p.name for p in (demo_dataset / "test").iterdir())[:15]
    (tmp_path / "list.txt").write_text("\n".join(f"test/{n}" for n in names + ["test/does_not_exist.jpg"]))
    source = ImageFolderSource(demo_dataset / "test", tmp_path / "list.txt")
    assert [f.name for f in source] == names


def test_legacy_alphabetical_sort_is_available(tmp_path):
    import cv2
    import numpy as np

    for n in (5, 9995, 10000):
        cv2.imwrite(str(tmp_path / f"v_frame_{n:04d}.jpg"), np.zeros((4, 4, 3), np.uint8))
    natural = [p.name for p in ImageFolderSource(tmp_path).paths]
    legacy = [p.name for p in ImageFolderSource(tmp_path, natural_sort=False).paths]
    assert natural == ["v_frame_0005.jpg", "v_frame_9995.jpg", "v_frame_10000.jpg"]
    assert legacy == ["v_frame_0005.jpg", "v_frame_10000.jpg", "v_frame_9995.jpg"]


def test_open_source_rejects_unknown_input(tmp_path):
    (tmp_path / "notes.txt").write_text("x")
    with pytest.raises(ValueError, match="neither a folder"):
        open_source(tmp_path / "notes.txt")
    with pytest.raises(FileNotFoundError):
        VideoSource(tmp_path / "missing.mp4")
