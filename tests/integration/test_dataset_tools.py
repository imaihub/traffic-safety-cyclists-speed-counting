import pytest

from cyclist_tracker.dataset import coco_to_mot, extract_frames, validate_dataset
from cyclist_tracker.io_formats import read_mot

pytestmark = pytest.mark.integration


def test_demo_dataset_passes_the_check(demo_dataset):
    report = validate_dataset(demo_dataset)
    assert not report.errors
    assert report.ready_for_training and report.ready_for_analysis
    assert set(report.ground_truth) == {"tracking", "speeds", "counts"}


def test_extract_frames_from_video(demo_dataset, tmp_path):
    progress = []
    saved = extract_frames(demo_dataset / "demo_test_video.mp4", tmp_path, every_n=5,
                           progress=lambda d, t: progress.append((d, t)))
    assert saved == 44
    assert len(list(tmp_path.glob("*.jpg"))) == 44
    assert progress[-1][0] == progress[-1][1]
    with pytest.raises(ValueError):
        extract_frames(demo_dataset / "demo_test_video.mp4", tmp_path, every_n=0)


def test_ground_truth_from_coco_matches_the_demo_file(demo_dataset, tmp_path):
    coco_to_mot(demo_dataset / "annotations" / "test.json", tmp_path / "gt.txt")
    assert read_mot(tmp_path / "gt.txt") == read_mot(demo_dataset / "ground_truth" / "tracking_test.txt")
