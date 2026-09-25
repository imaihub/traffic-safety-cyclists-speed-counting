import numpy as np
import pytest

from cyclist_tracker.config import AnalysisSettings
from cyclist_tracker.detection import Detection
from cyclist_tracker.tracking import CyclistTracker, embedder_available, histogram_embedding

pytestmark = pytest.mark.unit


def test_histogram_embedding_is_normalised():
    crop = np.random.default_rng(0).integers(0, 255, (40, 20, 3), dtype=np.uint8)
    emb = histogram_embedding(crop)
    assert emb.shape == (512,)
    assert np.linalg.norm(emb) == pytest.approx(1.0, abs=1e-5)
    assert np.linalg.norm(histogram_embedding(np.zeros((0, 0, 3), np.uint8))) == pytest.approx(1.0, abs=1e-5)


def test_tracker_confirms_a_moving_box_and_keeps_its_id():
    tracker = CyclistTracker(AnalysisSettings(embedder="histogram", use_gpu=False))
    frame = np.full((300, 400, 3), 90, np.uint8)
    frame[:, :, 2] = 180
    ids = []
    for i in range(6):
        box = (50 + 10 * i, 100, 90 + 10 * i, 180)
        tracks = tracker.update([Detection(box, 0.95)], frame)
        ids.append([t.track_id for t in tracks])
    assert ids[0] == []  # not confirmed yet (n_init = 2)
    confirmed = [i for i in ids if i]
    assert confirmed and all(i == confirmed[0] for i in confirmed)


def test_unknown_feature_extractor_is_refused():
    with pytest.raises(ValueError, match="Unknown feature extractor"):
        CyclistTracker(AnalysisSettings(embedder="resnet"))


def test_builtin_extractors_are_available():
    assert embedder_available("mobilenet")
    assert embedder_available("histogram")
