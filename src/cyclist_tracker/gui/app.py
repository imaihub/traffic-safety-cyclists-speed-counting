"""Main window of the Cyclist Tracker desktop application."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QAction, QFont
from PyQt6.QtWidgets import (
    QApplication,
    QDockWidget,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QTabWidget,
)

from .. import __version__
from ..dataset import DatasetReport, list_images
from .common import QtLogHandler, open_path
from .tab_analyse import AnalyseTab
from .tab_dataset import DatasetTab
from .tab_evaluate import EvaluateTab
from .tab_scene import SceneTab
from .tab_start import HelpTab, StartTab
from .tab_train import TrainTab

REPO_URL = "https://github.com/imaihub/traffic-safety-cyclists-speed-counting"

STYLE = """
QWidget { font-size: 10.5pt; }
QLabel#Heading { font-size: 17pt; font-weight: 600; color: #1d3557; }
QLabel#SubHeading { color: #4a5563; }
QGroupBox { font-weight: 600; border: 1px solid #d5dae1; border-radius: 6px; margin-top: 12px; padding-top: 6px; }
QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 4px; color: #1d3557; }
QPushButton { padding: 6px 12px; border-radius: 5px; border: 1px solid #b8c0cc; background: #f7f8fa; }
QPushButton:hover { background: #eef2f7; }
QPushButton:disabled { color: #9aa3ae; }
QPushButton#Primary { background: #2a6fdb; color: white; border: none; font-weight: 600; }
QPushButton#Primary:hover { background: #245fc0; }
QPushButton#Primary:disabled { background: #9db8e6; }
QFrame#StatCard { background: #f3f6fb; border: 1px solid #d5dae1; border-radius: 8px; }
QLabel#StatValue { font-size: 22pt; font-weight: 700; color: #1d3557; }
QLabel#StatCaption { color: #4a5563; }
QLabel#StepBadge { background: #2a6fdb; color: white; border-radius: 17px; font-weight: 700; font-size: 12pt; }
QLabel#PipelineCard { background: #f3f6fb; border: 1px solid #d5dae1; border-radius: 8px; padding: 10px; }
QLabel#Arrow { font-size: 18pt; color: #2a6fdb; }
QTabBar::tab { padding: 8px 16px; }
QTabBar::tab:selected { font-weight: 600; }
"""


def sample_path() -> Path | None:
    for base in (Path(__file__).resolve().parents[3], Path.cwd()):
        candidate = base / "sample_data" / "stanford_drone_gates"
        if (candidate / "scene.json").is_file():
            return candidate
    return None


def manual_path() -> Path | None:
    for base in (Path(__file__).resolve().parents[3], Path.cwd()):
        candidate = base / "docs" / "user_manual.pdf"
        if candidate.is_file():
            return candidate
    return None


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"Cyclist Tracker {__version__} - counting and speed estimation")
        self.resize(1400, 900)

        self.start = StartTab()
        self.dataset = DatasetTab()
        self.scene = SceneTab()
        self.train = TrainTab()
        self.analyse = AnalyseTab()
        self.evaluate = EvaluateTab()
        self.help = HelpTab()

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        for widget, title in ((self.start, "Start"), (self.dataset, "1 · Dataset"), (self.scene, "2 · Camera scene"),
                              (self.train, "3 · Train"), (self.analyse, "4 · Analyse"),
                              (self.evaluate, "5 · Evaluate"), (self.help, "Help")):
            self.tabs.addTab(widget, title)
        self.setCentralWidget(self.tabs)

        self.messages = QPlainTextEdit()
        self.messages.setReadOnly(True)
        self.messages.setMaximumBlockCount(5000)
        mono = QFont("Monospace")
        mono.setStyleHint(QFont.StyleHint.TypeWriter)
        self.messages.setFont(mono)
        dock = QDockWidget("Messages", self)
        dock.setWidget(self.messages)
        dock.setFeatures(QDockWidget.DockWidgetFeature.DockWidgetMovable |
                         QDockWidget.DockWidgetFeature.DockWidgetFloatable)
        self.addDockWidget(Qt.DockWidgetArea.BottomDockWidgetArea, dock)
        self.resizeDocks([dock], [140], Qt.Orientation.Vertical)
        self._log_handler = QtLogHandler()
        self._log_handler.emitter.message.connect(self._append_message)
        logging.getLogger("cyclist_tracker").addHandler(self._log_handler)
        logging.getLogger("cyclist_tracker").setLevel(logging.INFO)

        self._build_menu()
        self._connect()
        self.statusBar().showMessage("Ready. New here? Press 'Try with demo data' on the Start page.")

    def _build_menu(self) -> None:
        file_menu = self.menuBar().addMenu("&File")
        open_scene = QAction("Open camera scene…", self)
        open_scene.triggered.connect(lambda: (self.tabs.setCurrentWidget(self.scene), self.scene.choose_scene()))
        quit_action = QAction("Quit", self)
        quit_action.triggered.connect(self.close)
        file_menu.addAction(open_scene)
        file_menu.addSeparator()
        file_menu.addAction(quit_action)
        help_menu = self.menuBar().addMenu("&Help")
        manual = QAction("User manual (PDF)", self)
        manual.triggered.connect(self.open_manual)
        about = QAction("About", self)
        about.triggered.connect(self.about)
        help_menu.addAction(manual)
        help_menu.addAction(about)

    def _connect(self) -> None:
        self.start.go_to_step.connect(self.tabs.setCurrentIndex)
        self.start.demo_created.connect(self.load_demo)
        self.start.sample_requested.connect(self.load_sample)
        self.dataset.dataset_checked.connect(self._dataset_checked)
        self.scene.scene_saved.connect(self.analyse.scene_file.set_path)
        self.train.model_trained.connect(self._use_model)
        self.analyse.analysis_finished.connect(lambda r: self.evaluate.set_results(r.output_files))
        self.help.open_manual.connect(self.open_manual)

    # ---------------------------------------------------------------- wiring
    def _append_message(self, text: str, level: int) -> None:
        self.messages.appendPlainText(text)
        if level >= logging.WARNING:
            self.statusBar().showMessage(text, 10000)

    def _dataset_checked(self, report: DatasetReport) -> None:
        root = report.root
        if report.ready_for_training:
            self.train.dataset.set_path(root)
        self.evaluate.set_ground_truth(report.ground_truth)
        if report.images.get("test") and not self.analyse.source.path():
            self.analyse.source.set_path(root / "test")
        if "test" in report.annotations and not self.analyse.annotations.path():
            self.analyse.annotations.set_path(report.annotations["test"])
        if (root / "scene.json").is_file():
            self.analyse.scene_file.set_path(root / "scene.json")

    def _use_model(self, checkpoint: str) -> None:
        self.analyse.checkpoint.set_path(checkpoint)
        self.analyse.set_combo(self.analyse.detector, "checkpoint")
        self.tabs.setCurrentWidget(self.analyse)

    def load_example(self, folder: str | Path, name: str = "demo") -> None:
        """Fill in every tab for an example dataset (the demo data or the sample dataset)."""
        root = Path(folder)
        self.dataset.set_folder(root)
        self.dataset.check()
        self.scene.load_scene(root / "scene.json")
        frames = list_images(root / "test")
        if frames:
            self.scene.load_image(frames[min(20, len(frames) - 1)])
        if (root / "settings.json").is_file():
            self.analyse.load_settings(root / "settings.json")
        self.analyse.source.set_path(root / "test")
        self.analyse.frame_list.set_path(None)
        self.analyse.scene_file.set_path(root / "scene.json")
        self.analyse.annotations.set_path(root / "annotations" / "test.json")
        self.analyse.set_combo(self.analyse.detector, "annotations")
        self.analyse.output_dir.set_path(root / "results")
        self.analyse.run_name.setText(name)
        self.train.dataset.set_path(root)
        self.tabs.setCurrentWidget(self.dataset)

    def load_demo(self, folder: str) -> None:
        self.load_example(folder, "demo")
        QMessageBox.information(
            self, "Demo data ready",
            f"The demo dataset was created in\n{folder}\n\nAll tabs are filled in. Look at the dataset check, the "
            "camera scene (Step 2), and then press 'Start analysis' in Step 4. The demo uses the annotated boxes as "
            "detections, so no trained model is needed.")

    def load_sample(self) -> None:
        path = sample_path()
        if path is None:
            QMessageBox.information(self, "Sample dataset", "The sample dataset was not found. It is in the folder "
                                    f"sample_data/stanford_drone_gates of\n{REPO_URL}")
            return
        self.load_example(path, "sample")
        QMessageBox.information(
            self, "Sample dataset opened",
            f"The sample dataset in\n{path}\nis loaded in all tabs. It is real drone footage (Stanford Drone "
            "Dataset, licence CC BY-NC-SA 3.0, non-commercial use only). Its README explains how it was made.\n\n"
            "Step 4 uses the hand-made boxes as detections. The pre-trained COCO detector does not work on these "
            "top-down images. The train and val folders show the dataset structure but are too small to train a "
            "good detector.")

    def open_manual(self) -> None:
        path = manual_path()
        if path:
            open_path(path)
        else:
            QMessageBox.information(self, "User manual", f"The manual is in docs/user_manual.pdf of\n{REPO_URL}")

    def about(self) -> None:
        QMessageBox.about(
            self, "About Cyclist Tracker",
            f"<b>Cyclist Tracker {__version__}</b><br>Cyclist detection (DETR), tracking (DeepSORT), counting and "
            "speed estimation (homography).<br><br>Research: Avishkar Borkar, NHL Stenden Lectoraat Computer "
            "Vision &amp; Data Science, within the BikeSafeAI project (Regieorgaan SIA).<br>"
            f"Software and interface: Koen Molenaar.<br><br><a href='{REPO_URL}'>{REPO_URL}</a><br>License: AGPL-3.0")

    def closeEvent(self, event) -> None:  # noqa: N802
        for tab in (self.analyse, self.train, self.dataset, self.start):
            task = getattr(tab, "_task", None)
            if task is not None and task.is_running():
                task.stop()
                task.wait(10000)
        logging.getLogger("cyclist_tracker").removeHandler(self._log_handler)
        super().closeEvent(event)


def create_app(argv: list[str] | None = None) -> QApplication:
    app = QApplication.instance() or QApplication(argv if argv is not None else sys.argv)
    app.setApplicationName("Cyclist Tracker")
    app.setStyle("Fusion")
    app.setStyleSheet(STYLE)
    return app


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    for noisy in ("transformers", "timm", "matplotlib", "PIL", "urllib3", "httpx", "huggingface_hub"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    app = create_app(argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
