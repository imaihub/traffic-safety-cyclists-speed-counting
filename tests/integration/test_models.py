"""Tests with real neural networks. Marked ``slow``: they download pre-trained weights (~200 MB) once."""

import cv2
import numpy as np
import pytest

from ..conftest import PAPER_CHECKPOINT, REPO

pytestmark = [pytest.mark.integration, pytest.mark.slow]


def test_short_training_run_produces_a_loadable_checkpoint(demo_dataset, tmp_path):
    from cyclist_tracker.detection import DetrDetector
    from cyclist_tracker.training import TrainConfig, train_detector

    epochs = []
    best = train_detector(TrainConfig(str(demo_dataset), str(tmp_path), epochs=1, limit_batches=2, num_workers=0,
                                      accumulate_grad_batches=1), progress=lambda e, n, m: epochs.append((e, m)))
    assert best is not None and best.is_file()
    assert epochs and "val_loss" in epochs[0][1]
    detector = DetrDetector(best, threshold=0.0001)
    assert detector.cyclist_label == 0
    image = cv2.imread(str(next((demo_dataset / "test").glob("*.jpg"))))[:, :, ::-1].copy()
    assert isinstance(detector.detect(image), list)


@pytest.mark.skipif(not PAPER_CHECKPOINT.is_file(), reason="checkpoint of the paper not present")
def test_checkpoint_of_the_paper_finds_the_cyclist():
    from cyclist_tracker.detection import DetrDetector

    detector = DetrDetector(PAPER_CHECKPOINT)
    assert detector.cyclist_label == 1
    image = cv2.imread(str(REPO / "docs" / "speeds_homography_v2 - frame at 0m38s.jpg"))[:, :, ::-1].copy()
    (cyclist,) = detector.detect(image)
    x1, y1, x2, y2 = cyclist.ltrb
    assert 800 < (x1 + x2) / 2 < 1000 and cyclist.score > 0.9


def test_coco_detector_runs_without_training():
    from cyclist_tracker.detection import CocoDetrDetector

    detector = CocoDetrDetector()
    assert detector.detect(np.zeros((240, 320, 3), np.uint8)) == []
