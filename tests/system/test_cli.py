"""End-to-end tests of the command line, run as a separate process like a user would."""

import subprocess
import sys

import pytest

pytestmark = pytest.mark.system


def cli(*args, check=True):
    result = subprocess.run([sys.executable, "-m", "cyclist_tracker", *map(str, args)], capture_output=True,
                            text=True, timeout=600)
    if check and result.returncode != 0:
        raise AssertionError(f"exit {result.returncode}\n{result.stdout}\n{result.stderr}")
    return result


def test_version_and_help():
    assert "cyclist-tracker" in cli("--version").stdout
    assert "analyze" in cli("-h").stdout


def test_check_command_lists_pytorch():
    assert "PyTorch" in cli("check").stdout


def test_full_workflow_on_demo_data(tmp_path):
    demo = tmp_path / "demo"
    cli("demo", demo, "--no-video")
    assert "Ready for training: yes; ready for analysis: yes" in cli("dataset", "validate", demo).stdout
    assert "is complete" in cli("scene", "check", demo / "scene.json").stdout

    out = cli("analyze", demo / "test", "--scene", demo / "scene.json", "--detector", "annotations",
              "--annotations", demo / "annotations" / "test.json", "--embedder", "histogram", "--cpu",
              "--output-dir", tmp_path / "results", "--name", "demo", "--no-video").stdout
    assert "away" in out and "towards" in out
    speeds = tmp_path / "results" / "demo_speeds.csv"
    assert speeds.is_file()

    gt = demo / "ground_truth"
    assert "idf1" in cli("evaluate", "tracking", "--gt", gt / "tracking_test.txt", "--pred",
                         tmp_path / "results" / "demo_tracks.txt").stdout
    speed = cli("evaluate", "speed", "--gt", gt / "speeds_test.csv", "--pred", speeds, "--names", "demo",
                "--plot", tmp_path / "plot.png", "--output", tmp_path / "speed.csv").stdout
    assert "MAE" in speed and (tmp_path / "plot.png").is_file() and (tmp_path / "speed.csv").is_file()
    assert "Total" in cli("evaluate", "counting", "--gt", gt / "counts_test.csv", "--pred", speeds).stdout


def test_dataset_preparation_commands(tmp_path):
    demo = tmp_path / "demo"
    cli("demo", demo)
    cli("dataset", "extract-frames", demo / "demo_test_video.mp4", tmp_path / "frames", "--every", "10")
    assert len(list((tmp_path / "frames").glob("*.jpg"))) == 22
    cli("dataset", "coco-to-mot", demo / "annotations" / "test.json", tmp_path / "gt.txt")
    cli("dataset", "frame-list", demo / "annotations" / "test.json", tmp_path / "list.txt")
    cli("dataset", "init", tmp_path / "new")
    assert (tmp_path / "new" / "annotations").is_dir()
    cli("scene", "paper", tmp_path / "paper.json")
    assert "is complete" in cli("scene", "check", tmp_path / "paper.json").stdout


def test_friendly_errors(tmp_path):
    result = cli("analyze", tmp_path, "--scene", tmp_path / "missing.json", check=False)
    assert result.returncode == 2 and "Error:" in result.stderr
    (tmp_path / "bad.json").write_text("{}")
    result = cli("scene", "check", tmp_path / "bad.json", check=False)
    assert result.returncode == 1 and "region of interest" in result.stdout
