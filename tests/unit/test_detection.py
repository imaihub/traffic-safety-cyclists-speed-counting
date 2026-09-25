import json

import pytest
import torch

from cyclist_tracker.detection import (
    AnnotationDetector,
    Detection,
    adapt_state_dict,
    create_detector,
    default_cyclist_label,
    merge_riders,
)

pytestmark = pytest.mark.unit


def test_old_checkpoint_keys_are_renamed_for_new_transformers():
    old = {
        "model.backbone.conv_encoder.model.conv1.weight": 1,
        "model.encoder.layers.0.self_attn.out_proj.weight": 2,
        "model.decoder.layers.3.encoder_attn.out_proj.bias": 3,
        "model.decoder.layers.3.fc1.weight": 4,
        "class_labels_classifier.weight": 5,
    }
    expected = {
        "model.backbone.model.conv1.weight", "model.encoder.layers.0.self_attn.o_proj.weight",
        "model.decoder.layers.3.encoder_attn.o_proj.bias", "model.decoder.layers.3.mlp.fc1.weight",
        "class_labels_classifier.weight",
    }
    assert set(adapt_state_dict(old, expected)) == expected


def test_new_checkpoint_keys_are_renamed_for_old_transformers():
    new = {"model.encoder.layers.0.mlp.fc2.bias": 1, "model.backbone.model.bn1.weight": 2}
    expected = {"model.encoder.layers.0.fc2.bias", "model.backbone.conv_encoder.model.bn1.weight"}
    assert set(adapt_state_dict(new, expected)) == expected


def test_cyclist_label_depends_on_number_of_classes():
    assert default_cyclist_label(1) == 0  # models trained with this software
    assert default_cyclist_label(2) == 1  # the model of the paper


def test_merge_riders_joins_person_and_bicycle():
    bike = Detection((100, 150, 140, 200), 0.9)
    rider = Detection((105, 80, 135, 180), 0.8)
    pedestrian = Detection((300, 80, 330, 180), 0.95)
    lonely_bike = Detection((500, 150, 540, 200), 0.7)
    merged = merge_riders([rider, pedestrian], [bike, lonely_bike])
    assert merged[0].ltrb == (100, 80, 140, 200)
    assert merged[0].score == 0.9
    assert merged[1] == lonely_bike
    assert len(merged) == 2  # the pedestrian is not a cyclist


def test_annotation_detector_matches_by_file_name(tmp_path):
    coco = {"images": [{"id": 1, "file_name": "test/a.jpg"}, {"id": 2, "file_name": "test/b.jpg"}],
            "annotations": [{"image_id": 1, "bbox": [10, 20, 30, 40]}, {"image_id": 1, "bbox": [0, 0, 0, 5]},
                            {"image_id": 2, "bbox": [1, 2, 3, 4]}]}
    path = tmp_path / "coco.json"
    path.write_text(json.dumps(coco))
    detector = AnnotationDetector(path)
    assert [d.ltrb for d in detector.detect(None, "a.jpg")] == [(10, 20, 40, 60)]
    assert detector.detect(None, "other/b.jpg")[0].ltrb == (1, 2, 4, 6)
    assert detector.detect(None, "missing.jpg") == []


def test_create_detector_explains_missing_input():
    with pytest.raises(ValueError, match="checkpoint"):
        create_detector("checkpoint")
    with pytest.raises(ValueError, match="COCO"):
        create_detector("annotations")
    with pytest.raises(ValueError, match="Unknown detector"):
        create_detector("yolo")


def test_non_detr_checkpoint_is_refused(tmp_path):
    from cyclist_tracker.detection import load_detr_checkpoint

    path = tmp_path / "other.ckpt"
    torch.save({"state_dict": {"model.layer.weight": torch.zeros(1)}}, path)
    with pytest.raises(ValueError, match="does not look like a DETR"):
        load_detr_checkpoint(path)
