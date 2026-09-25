"""Shared fixtures. The synthetic demo dataset replaces the private traffic videos in all tests."""

import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

REPO = Path(__file__).resolve().parents[1]
PRIVATE_ARTIFACTS = REPO / "dataset" / "artifacts"
PAPER_CHECKPOINT = REPO / "dataset" / "models" / "epoch=30-step=18879.ckpt"


@pytest.fixture(scope="session")
def demo_dataset(tmp_path_factory) -> Path:
    from cyclist_tracker.synthetic import generate_demo_dataset

    return generate_demo_dataset(tmp_path_factory.mktemp("data") / "demo")


@pytest.fixture
def demo_scene(demo_dataset):
    from cyclist_tracker.config import SceneConfig

    return SceneConfig.load(demo_dataset / "scene.json")


@pytest.fixture
def fast_settings():
    """Settings that need no neural network for tracking (colour-histogram features, CPU)."""
    from cyclist_tracker.config import AnalysisSettings

    return AnalysisSettings(embedder="histogram", use_gpu=False)

SAMPLE = REPO / "sample_data" / "stanford_drone_gates"
