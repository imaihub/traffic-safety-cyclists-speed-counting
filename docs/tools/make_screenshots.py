"""Create the GUI screenshots used in the README and the user manual.

    QT_QPA_PLATFORM=offscreen uv run python docs/tools/make_screenshots.py [--train]

Uses the synthetic demo dataset. ``--train`` also runs a short training on the GPU for the
Train-tab screenshot. The Camera-scene screenshot with a real frame uses the image in docs/.
"""

from __future__ import annotations

import argparse
import sys
import tempfile
import time
from pathlib import Path

from PyQt6.QtWidgets import QMessageBox

DOCS = Path(__file__).resolve().parents[1]
OUT = DOCS / "images"
REAL_FRAME = DOCS / "speeds_homography_v2 - frame at 0m38s.jpg"
SAMPLE = DOCS.parent / "sample_data" / "stanford_drone_gates"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="run a short training for the Train screenshot")
    args = parser.parse_args()
    QMessageBox.information = staticmethod(lambda *a, **k: None)

    from cyclist_tracker.gui.app import MainWindow, create_app

    app = create_app([])
    window = MainWindow()
    window.resize(1400, 880)
    window.show()

    def pump(seconds: float) -> None:
        end = time.time() + seconds
        while time.time() < end:
            app.processEvents()
            time.sleep(0.02)

    def shot(name: str) -> None:
        pump(0.4)
        window.grab().save(str(OUT / f"gui_{name}.png"))
        print("saved", name)

    def wait_for(task, limit: float = 1800) -> None:
        start = time.time()
        while task.is_running() and time.time() - start < limit:
            pump(0.2)
        pump(0.5)

    OUT.mkdir(exist_ok=True)
    work = Path(tempfile.mkdtemp())
    shot("start")

    window.start.create_demo(work / "cyclist_tracker_demo")
    wait_for(window.start._task)
    window.messages.clear()
    window.tabs.setCurrentWidget(window.dataset)
    shot("dataset")

    window.tabs.setCurrentWidget(window.scene)
    window.scene._set_mode("calibration")
    shot("scene_demo")
    if REAL_FRAME.is_file():
        window.scene.load_paper_scene()
        window.scene.load_image(REAL_FRAME)
        window.scene._set_mode("line_a")
        shot("scene_paper")
        window.scene.load_scene(work / "cyclist_tracker_demo" / "scene.json")
        window.scene.load_image(sorted((work / "cyclist_tracker_demo" / "test").glob("*.jpg"))[20])

    window.tabs.setCurrentWidget(window.train)
    if args.train:
        window.train.epochs.setValue(12)
        window.train.output.set_path(work / "run")
        window.train.start()
        wait_for(window.train._task)
    if args.train or not (OUT / "gui_train.png").is_file():
        shot("train")

    if SAMPLE.is_dir():  # real footage for the analysis and evaluation screenshots
        window.load_sample()
        window.tabs.setCurrentWidget(window.scene)
        window.scene._set_mode("line_b")
        shot("scene_sample")
    tab = window.analyse
    window.tabs.setCurrentWidget(tab)
    tab.output_dir.set_path(work / "results")
    tab.start()
    start = time.time()
    while tab._task.is_running():
        pump(0.1)
        if time.time() - start > 2.0 and tab.progress.value() > 110:
            window.grab().save(str(OUT / "gui_analyse_running.png"))
            break
    wait_for(tab._task)
    tab.crossings.scrollToTop()
    shot("analyse")

    window.tabs.setCurrentWidget(window.evaluate)
    for name, page in (("tracking", window.evaluate.tracking), ("speed", window.evaluate.speed),
                       ("counting", window.evaluate.counting)):
        window.evaluate.pages.setCurrentWidget(page)
        page.run()
        shot(f"evaluate_{name}")

    window.tabs.setCurrentWidget(window.help)
    shot("help")
    window.close()


if __name__ == "__main__":
    sys.exit(main())
