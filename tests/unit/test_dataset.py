import json

import pytest

from cyclist_tracker import dataset as ds

pytestmark = pytest.mark.unit


def _coco(images, annotations):
    return {"images": images, "annotations": annotations, "categories": [{"id": 0, "name": "cyclist"}]}


def _make(root, split="test", n=3, with_track=True):
    (root / split).mkdir(parents=True, exist_ok=True)
    (root / "annotations").mkdir(exist_ok=True)
    images, anns = [], []
    for i in range(n):
        name = f"{split}/frame_{i:05d}.jpg"
        (root / name).write_bytes(b"not really a jpeg")
        images.append({"id": i + 1, "file_name": name})
        attrs = {"track_id": 0} if with_track else {}
        anns.append({"id": i + 1, "image_id": i + 1, "bbox": [10 + i, 20, 30, 40], "category_id": 0,
                     "attributes": attrs})
    (root / "annotations" / f"{split}.json").write_text(json.dumps(_coco(images, anns)))


def test_missing_folder_is_an_error(tmp_path):
    report = ds.validate_dataset(tmp_path / "nope")
    assert report.errors and not report.ready_for_analysis


def test_empty_folder_gives_warnings_not_errors(tmp_path):
    report = ds.validate_dataset(tmp_path)
    assert not report.errors
    assert not report.ready_for_training and not report.ready_for_analysis
    assert "Ready for training: no" in report.as_text()


def test_complete_split_is_accepted(tmp_path):
    for split in ("train", "val", "test"):
        _make(tmp_path, split)
    report = ds.validate_dataset(tmp_path)
    assert not report.errors
    assert report.ready_for_training and report.ready_for_analysis
    assert report.images == {"train": 3, "val": 3, "test": 3}


def test_old_annotation_names_are_recognised(tmp_path):
    _make(tmp_path, "test")
    (tmp_path / "annotations" / "test.json").rename(tmp_path / "annotations" / "testing_set_f.json")
    assert ds.find_annotation_file(tmp_path, "test").name == "testing_set_f.json"


def test_problems_in_coco_file_are_reported(tmp_path):
    _make(tmp_path, "test", with_track=False)
    data = json.loads((tmp_path / "annotations" / "test.json").read_text())
    data["annotations"].append({"id": 99, "image_id": 1234, "bbox": [0, 0, 5, 5]})
    data["annotations"].append({"id": 100, "image_id": 1, "bbox": [0, 0, 0, 5]})
    data["images"].append({"id": 50, "file_name": "test/missing.jpg"})
    (tmp_path / "annotations" / "test.json").write_text(json.dumps(data))
    messages = " ".join(c.message for c in ds.validate_dataset(tmp_path).checks)
    assert "refer to an image id that does not exist" in messages
    assert "no size" in messages
    assert "1 of 4 listed images were not found" in messages
    assert "no track_id" in messages


def test_invalid_json_is_an_error(tmp_path):
    (tmp_path / "annotations").mkdir()
    (tmp_path / "annotations" / "train.json").write_text("{broken")
    assert any("Not a valid JSON" in c.message for c in ds.validate_dataset(tmp_path).errors)


def test_create_layout_is_idempotent(tmp_path):
    assert len(ds.create_layout(tmp_path)) == 5
    assert ds.create_layout(tmp_path) == []


def test_frame_list_in_natural_order(tmp_path):
    images = [{"id": 1, "file_name": "test/v_frame_10000.jpg"}, {"id": 2, "file_name": "test/v_frame_9995.jpg"},
              {"id": 3, "file_name": "test/v_frame_0001.jpg"}]
    path = tmp_path / "c.json"
    path.write_text(json.dumps(_coco(images, [{"image_id": 1, "bbox": [0, 0, 1, 1]},
                                              {"image_id": 2, "bbox": [0, 0, 1, 1]}])))
    assert ds.write_frame_list(path, tmp_path / "l.txt") == 2
    assert (tmp_path / "l.txt").read_text().split() == ["test/v_frame_9995.jpg", "test/v_frame_10000.jpg"]
    assert ds.write_frame_list(path, tmp_path / "all.txt", only_annotated=False) == 3


def test_coco_to_mot_shifts_track_ids(tmp_path):
    _make(tmp_path, "test", n=2)
    n = ds.coco_to_mot(tmp_path / "annotations" / "test.json", tmp_path / "gt.txt")
    assert n == 2
    first = (tmp_path / "gt.txt").read_text().splitlines()[0]
    assert first == "test/frame_00000.jpg,1,10.00,20.00,30.00,40.00,1.0,1,1"


def test_coco_to_mot_needs_track_ids(tmp_path):
    _make(tmp_path, "test", with_track=False)
    with pytest.raises(ValueError, match="no track_id"):
        ds.coco_to_mot(tmp_path / "annotations" / "test.json", tmp_path / "gt.txt")


def test_flatten_file_names(tmp_path):
    path = tmp_path / "c.json"
    path.write_text(json.dumps(_coco([{"id": 1, "file_name": "test/2022/08/frames/x.jpg"}], [])))
    ds.flatten_file_names(path, tmp_path / "f.json", "val")
    assert json.loads((tmp_path / "f.json").read_text())["images"][0]["file_name"] == "val/x.jpg"


def test_read_cvat_polygon(tmp_path):
    xml = tmp_path / "a.xml"
    xml.write_text('<annotations><image id="0"><polygon label="road_crossing" '
                   'points="937.30,390.20;1100.30,388.70;1084.60,506.20;862.40,504.00"/></image></annotations>')
    ((label, points),) = ds.read_cvat_polygons(xml)
    assert label == "road_crossing"
    assert points[0] == (937.3, 390.2) and len(points) == 4
