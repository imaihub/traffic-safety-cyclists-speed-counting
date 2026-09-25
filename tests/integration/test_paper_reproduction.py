"""Regression test: the evaluation code reproduces the numbers of the paper.

Needs the (private) prediction files of the original experiments in ``dataset/artifacts``;
the test is skipped when they are not present.
"""

import pytest

from cyclist_tracker.evaluation import evaluate_speeds, evaluate_tracking

from ..conftest import PRIVATE_ARTIFACTS as A

pytestmark = [pytest.mark.integration,
              pytest.mark.skipif(not (A / "groundtruth_test_fullds.txt").is_file(), reason="private data not present")]


@pytest.mark.parametrize(
    ("file", "idf1", "mota"),
    [("predictions_clip_ViT-B32.txt", 71.9, 58.2), ("predictions_osnet_ain_x1_0.txt", 72.8, 58.6),
     ("predictions_mobilenet.txt", 71.9, 58.2)],
)
def test_tracking_table_of_the_paper(file, idf1, mota):
    s = evaluate_tracking(A / "groundtruth_test_fullds.txt", A / file).loc["tracking"]
    assert round(s["idf1"] * 100, 1) == idf1
    assert round(s["mota"] * 100, 1) == mota


def test_speed_table_of_the_paper():
    files = ["predicted_base_clip_speeds.csv", "predicted_base_mobilenetv3_speeds.csv", "predicted_base_osnet_speeds.csv"]
    table, _ = evaluate_speeds(A / "gt_speeds_fullds.csv", [A / f for f in files], ["CLIP", "MobileNetV3", "OSNet"],
                               frame_tolerance=3)
    table = table.set_index("Model")
    # Paper, Table 4. The saved MobileNet file differs marginally from the one used for the paper
    # (RMSE 2.1747 instead of 2.1752), hence the tolerance of 0.01 km/h.
    paper = {"CLIP": (1.47, 2.20, 0.885), "MobileNetV3": (1.44, 2.18, 0.888), "OSNet": (1.45, 2.17, 0.889)}
    for model, (mae, rmse, r2) in paper.items():
        assert table.loc[model, "MAE"] == pytest.approx(mae, abs=0.01)
        assert table.loc[model, "RMSE"] == pytest.approx(rmse, abs=0.01)
        assert table.loc[model, "R2"] == pytest.approx(r2, abs=0.002)
