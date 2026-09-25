"""Tab 4 - detect, track, count and measure speed on a video or folder of frames."""

from __future__ import annotations

import time
from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..config import AnalysisSettings, SceneConfig
from ..detection import DETECTOR_KINDS
from ..pipeline import AnalysisResult
from ..tracking import EMBEDDERS, embedder_available
from .common import ImageView, InfoBox, PathPicker, StatCard, Task, heading, open_path, show_error

SPEED_MODES = {
    "homography": "Measured with the calibration (recommended)",
    "fixed_distance": "Known distance between line A and B (reference speeds)",
}
TRACKING_POINTS = {
    "bottom": "Bottom of the box - where the wheels touch the road (recommended)",
    "center": "Centre of the box (as in the original paper)",
}


class AnalyseTab(QWidget):
    analysis_finished = pyqtSignal(object)  # AnalysisResult

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._task: Task | None = None
        self._start_time = 0.0
        self.result: AnalysisResult | None = None

        # ---------------------------------------------------------------- inputs
        source_box = QGroupBox("1. What do you want to analyse?")
        sf = QFormLayout(source_box)
        self.source = PathPicker("file", "Videos (*.mp4 *.avi *.mov *.mkv *.m4v);;All files (*)",
                                 "A video file, or use the folder button for a folder with frames")
        folder_button = QPushButton("Frames folder…")
        folder_button.setToolTip("Choose a folder with frames (images) instead of a video file")
        folder_button.clicked.connect(self._choose_folder)
        src_row = QVBoxLayout()
        src_row.addWidget(self.source)
        src_row.addWidget(folder_button)
        self.frame_list = PathPicker("file", "Text files (*.txt);;All files (*)", "optional: only these frames")
        self.stride = QSpinBox(minimum=1, maximum=100, value=1)
        self.stride.setToolTip("For videos: use every N-th frame. 25 fps with N = 5 matches the paper (5 fps).")
        self.fps = QDoubleSpinBox(minimum=0.1, maximum=240, value=5.0, decimals=3, suffix=" frames/s")
        self.fps.setToolTip("Frames per second of the processed images. For videos this is filled in "
                            "automatically (video frame rate ÷ N).")
        sf.addRow("Video or frames", src_row)
        sf.addRow("Frame list", self.frame_list)
        sf.addRow("Use every N-th video frame", self.stride)
        sf.addRow("Frame rate", self.fps)
        self.source.changed.connect(self._source_changed)
        self.stride.valueChanged.connect(lambda _: self._source_changed(self.source.path()))

        model_box = QGroupBox("2. How should cyclists be found?")
        mf = QFormLayout(model_box)
        self.detector = QComboBox()
        for key, text in DETECTOR_KINDS.items():
            self.detector.addItem(text, key)
        self.detector.currentIndexChanged.connect(self._detector_changed)
        self.checkpoint = PathPicker("file", "Model checkpoints (*.ckpt);;All files (*)", "trained model (.ckpt)")
        self.annotations = PathPicker("file", "COCO annotations (*.json)", "annotations/test.json")
        self.confidence = QDoubleSpinBox(minimum=0.05, maximum=0.99, value=0.7, singleStep=0.05, decimals=2)
        self.confidence.setToolTip("Only detections at least this certain are used (0.7 in the paper).")
        self.embedder = QComboBox()
        for key, text in EMBEDDERS.items():
            self.embedder.addItem(text, key)
            if not embedder_available(key):
                index = self.embedder.count() - 1
                self.embedder.model().item(index).setEnabled(False)
                self.embedder.setItemText(index, text + " (not installed)")
        self.scene_file = PathPicker("file", "Scene files (*.json)", "scene.json from Step 2")
        mf.addRow("Detector", self.detector)
        mf.addRow("Model file", self.checkpoint)
        mf.addRow("Annotation file", self.annotations)
        mf.addRow("Minimum confidence", self.confidence)
        mf.addRow("Tracking features", self.embedder)
        mf.addRow("Camera scene", self.scene_file)

        speed_box = QGroupBox("3. Speed measurement")
        spf = QFormLayout(speed_box)
        self.speed_mode = QComboBox()
        for key, text in SPEED_MODES.items():
            self.speed_mode.addItem(text, key)
        self.tracking_point = QComboBox()
        for key, text in TRACKING_POINTS.items():
            self.tracking_point.addItem(text, key)
        spf.addRow("Distance", self.speed_mode)
        spf.addRow("Follow point", self.tracking_point)

        self.advanced = QGroupBox("Advanced settings (the defaults are those of the paper)")
        self.advanced.setCheckable(True)
        self.advanced.setChecked(False)
        af = QGridLayout(self.advanced)
        defaults = AnalysisSettings()
        self._adv: dict[str, QSpinBox | QDoubleSpinBox] = {}
        specs = [
            ("entry_threshold_px", "Entry distance to line (px)", 0, 500, 1),
            ("exit_threshold_px", "Exit distance to line (px)", 0, 500, 1),
            ("min_pixel_distance", "Minimum movement (px)", 0, 5000, 1),
            ("min_crossing_frames", "Minimum frames A→B", 0, 10000, None),
            ("max_crossing_frames", "Maximum frames A→B", 1, 100000, None),
            ("min_speed_kph", "Minimum speed (km/h)", 0, 300, 1),
            ("max_speed_kph", "Maximum speed (km/h)", 1, 300, 1),
            ("max_age", "Keep lost track (frames)", 1, 1000, None),
            ("n_init", "Frames to confirm a track", 1, 100, None),
            ("max_cosine_distance", "Appearance match threshold", 0.01, 2, 2),
        ]
        for i, (key, label, lo, hi, decimals) in enumerate(specs):
            if decimals is None:
                w = QSpinBox(minimum=int(lo), maximum=int(hi), value=int(getattr(defaults, key)))
            else:
                w = QDoubleSpinBox(minimum=lo, maximum=hi, value=float(getattr(defaults, key)), decimals=decimals)
            self._adv[key] = w
            caption = QLabel(label)
            caption.setWordWrap(True)
            af.addWidget(caption, i // 2, (i % 2) * 2)
            af.addWidget(w, i // 2, (i % 2) * 2 + 1)
        self.use_gpu = QCheckBox("Use the GPU when available")
        self.use_gpu.setChecked(True)
        af.addWidget(self.use_gpu, len(specs) // 2 + 1, 0, 1, 2)
        paper = QPushButton("Paper settings")
        paper.setToolTip("Exactly the settings of the original experiments (box centre as follow point).")
        paper.clicked.connect(lambda: self.apply_settings(AnalysisSettings.paper()))
        reset = QPushButton("Recommended settings")
        reset.clicked.connect(lambda: self.apply_settings(AnalysisSettings()))
        af.addWidget(paper, len(specs) // 2 + 1, 2)
        af.addWidget(reset, len(specs) // 2 + 1, 3)
        load_settings = QPushButton("Load settings…")
        load_settings.setToolTip("Read all settings of this tab from a settings JSON file (e.g. settings.json of a dataset)")
        load_settings.clicked.connect(self._load_settings_dialog)
        save_settings = QPushButton("Save settings…")
        save_settings.clicked.connect(self._save_settings_dialog)
        af.addWidget(load_settings, len(specs) // 2 + 2, 2)
        af.addWidget(save_settings, len(specs) // 2 + 2, 3)

        out_box = QGroupBox("4. Where to save the results?")
        of = QFormLayout(out_box)
        self.output_dir = PathPicker("folder", placeholder="results folder")
        self.output_dir.set_path(Path.cwd() / "results")
        self.run_name = QLineEdit("analysis")
        self.save_video = QCheckBox("Save a video with the results drawn on it")
        self.save_video.setChecked(True)
        of.addRow("Folder", self.output_dir)
        of.addRow("Name", self.run_name)
        of.addRow("", self.save_video)

        self.start_button = QPushButton("▶  Start analysis")
        self.start_button.setObjectName("Primary")
        self.start_button.clicked.connect(self.start)
        self.stop_button = QPushButton("■  Stop")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self.stop)
        buttons = QHBoxLayout()
        buttons.addWidget(self.start_button, 2)
        buttons.addWidget(self.stop_button, 1)

        for combo in (self.detector, self.embedder, self.speed_mode, self.tracking_point):
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(20)
            combo.setToolTip(combo.currentText())
            combo.currentTextChanged.connect(combo.setToolTip)

        form_widget = QWidget()
        fl = QVBoxLayout(form_widget)
        for w in (source_box, model_box, speed_box, self.advanced, out_box):
            fl.addWidget(w)
        fl.addLayout(buttons)
        fl.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(form_widget)
        scroll.setMinimumWidth(560)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        # ---------------------------------------------------------------- outputs
        self.preview = ImageView("The video with detected cyclists appears here while the analysis runs.")
        self.card_a = StatCard("Direction A → B")
        self.card_b = StatCard("Direction B → A")
        self.card_total = StatCard("Total counted")
        self.card_speed = StatCard("Average speed", "–")
        cards = QHBoxLayout()
        for c in (self.card_a, self.card_b, self.card_total, self.card_speed):
            cards.addWidget(c)
        self.progress = QProgressBar()
        self.progress.setFormat("%v / %m frames")
        self.eta = QLabel("")
        self.status = InfoBox("Fill in steps 1-4 on the left and press <b>Start analysis</b>.")
        self.crossings = QTableWidget(0, 5)
        self.crossings.setHorizontalHeaderLabels(["Cyclist", "Direction", "Speed (km/h)", "Entry frame", "Exit frame"])
        self.crossings.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.crossings.verticalHeader().setVisible(False)
        self.crossings.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.open_folder = QPushButton("Open results folder")
        self.open_folder.setEnabled(False)
        self.open_folder.clicked.connect(lambda: open_path(self.output_dir.path()))

        right = QWidget()
        rl = QVBoxLayout(right)
        rl.addWidget(self.preview, 3)
        rl.addLayout(cards)
        prow = QHBoxLayout()
        prow.addWidget(self.progress, 1)
        prow.addWidget(self.eta)
        rl.addLayout(prow)
        rl.addWidget(self.status)
        rl.addWidget(self.crossings, 2)
        rl.addWidget(self.open_folder)

        splitter = QSplitter()
        splitter.addWidget(scroll)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)

        layout = QVBoxLayout(self)
        layout.addWidget(heading("Step 4 · Analyse",
                                 "Find cyclists, follow them, count them per direction and measure their speed."))
        layout.addWidget(splitter, 1)
        self._detector_changed()

    # ------------------------------------------------------------------ settings
    def _choose_folder(self) -> None:
        from PyQt6.QtWidgets import QFileDialog

        path = QFileDialog.getExistingDirectory(self, "Folder with frames", self.source.path())
        if path:
            self.source.set_path(path)

    def _source_changed(self, path: str) -> None:
        p = Path(path)
        is_video = p.is_file()
        self.stride.setEnabled(is_video)
        self.frame_list.setEnabled(not is_video)
        if is_video:
            import cv2

            cap = cv2.VideoCapture(str(p))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if fps and fps > 0:
                self.fps.setValue(fps / self.stride.value())

    def _detector_changed(self) -> None:
        kind = self.detector.currentData()
        self.checkpoint.setEnabled(kind == "checkpoint")
        self.annotations.setEnabled(kind == "annotations")
        self.confidence.setEnabled(kind != "annotations")

    def set_combo(self, combo: QComboBox, key: str) -> None:
        index = combo.findData(key)
        if index >= 0:
            combo.setCurrentIndex(index)

    def apply_settings(self, s: AnalysisSettings) -> None:
        self.confidence.setValue(s.confidence_threshold)
        self.fps.setValue(s.fps)
        self.set_combo(self.embedder, s.embedder)
        self.set_combo(self.speed_mode, s.speed_mode)
        self.set_combo(self.tracking_point, s.tracking_point)
        for key, widget in self._adv.items():
            widget.setValue(getattr(s, key))
        self.use_gpu.setChecked(s.use_gpu)

    def load_settings(self, path: str | Path) -> bool:
        try:
            self.apply_settings(AnalysisSettings.load(path))
        except (OSError, ValueError) as exc:
            show_error(self, "Cannot read settings", str(exc))
            return False
        self.status.show_message(f"Settings loaded from {path}", "info")
        return True

    def _load_settings_dialog(self) -> None:
        from PyQt6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(self, "Load settings", "", "Settings (*.json)")
        if path:
            self.load_settings(path)

    def _save_settings_dialog(self) -> None:
        from PyQt6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getSaveFileName(self, "Save settings", "settings.json", "Settings (*.json)")
        if path:
            self.settings().save(path)

    def settings(self) -> AnalysisSettings:
        s = AnalysisSettings(
            confidence_threshold=self.confidence.value(),
            fps=self.fps.value(),
            embedder=self.embedder.currentData(),
            speed_mode=self.speed_mode.currentData(),
            tracking_point=self.tracking_point.currentData(),
            use_gpu=self.use_gpu.isChecked(),
        )
        for key, widget in self._adv.items():
            setattr(s, key, widget.value())
        return s

    # ------------------------------------------------------------------ running
    def _collect(self) -> tuple[SceneConfig, AnalysisSettings] | None:
        problems = []
        if not self.source.path() or not Path(self.source.path()).exists():
            problems.append("Choose a video file or a folder with frames (section 1).")
        kind = self.detector.currentData()
        if kind == "checkpoint" and not Path(self.checkpoint.path()).is_file():
            problems.append("Choose the trained model file (.ckpt), or pick another detector (section 2).")
        if kind == "annotations" and not Path(self.annotations.path()).is_file():
            problems.append("Choose the COCO annotation file (section 2).")
        scene = None
        if not Path(self.scene_file.path()).is_file():
            problems.append("Choose the camera scene file made in Step 2 (section 2).")
        else:
            try:
                scene = SceneConfig.load(self.scene_file.path())
                problems += [f"Scene: {p}" for p in scene.problems()]
            except ValueError as exc:
                problems.append(f"The scene file cannot be read: {exc}")
        settings = self.settings()
        problems += settings.problems()
        if scene and settings.speed_mode == "homography" and not scene.has_calibration:
            problems.append("The scene has no calibration points; add them in Step 2 or choose "
                            "'Known distance' as speed measurement.")
        if scene and settings.speed_mode == "fixed_distance" and scene.reference_distance_m <= 0:
            problems.append("The scene has no real distance between line A and B (Step 2).")
        if problems:
            self.status.show_message("<b>Cannot start yet:</b><br>• " + "<br>• ".join(problems), "warning")
            return None
        return scene, settings

    def start(self) -> None:
        collected = self._collect()
        if collected is None:
            return
        scene, settings = collected
        source_path, frame_list = self.source.path(), self.frame_list.path() or None
        stride, kind = self.stride.value(), self.detector.currentData()
        checkpoint, annotations = self.checkpoint.path(), self.annotations.path()
        output_dir, name, save_video = self.output_dir.path(), self.run_name.text().strip() or "analysis", \
            self.save_video.isChecked()

        self.card_a.set("0", scene.direction_a_to_b)
        self.card_b.set("0", scene.direction_b_to_a)
        self.card_total.set("0")
        self.card_speed.set("–")
        self.crossings.setRowCount(0)
        self.progress.setValue(0)
        self.status.show_message("Loading the models… (the first time this can take a minute because the "
                                 "pre-trained weights are downloaded)", "info")
        self._scene = scene

        def work(task: Task):
            from ..detection import create_detector
            from ..pipeline import open_source, run_analysis
            from ..tracking import CyclistTracker

            source = open_source(source_path, None if Path(source_path).is_file() else frame_list, stride)
            detector = create_detector(kind, checkpoint=checkpoint, annotations=annotations,
                                       threshold=settings.confidence_threshold, prefer_gpu=settings.use_gpu)
            tracker = CyclistTracker(settings)
            task.report(0, len(source), None)
            return run_analysis(source, detector, tracker, scene, settings, output_dir, name, save_video,
                                progress=lambda d, t, c: task.report(d, t, c), preview=task.send_preview,
                                should_stop=task.should_stop, on_crossing=task.emit_item)

        self._task = Task(work)
        self._task.progress.connect(self._on_progress)
        self._task.preview.connect(self.preview.set_image)
        self._task.item.connect(self._on_crossing)
        self._task.finished.connect(self._on_finished)
        self._task.failed.connect(self._on_failed)
        self._set_running(True)
        self._start_time = time.monotonic()
        self._task.start()

    def stop(self) -> None:
        if self._task:
            self._task.stop()
            self.status.show_message("Stopping after the current frame… the results so far are saved.", "info")

    def _set_running(self, running: bool) -> None:
        self.start_button.setEnabled(not running)
        self.stop_button.setEnabled(running)

    def _on_progress(self, done: int, total: int, counts) -> None:
        self.progress.setMaximum(max(total, 1))
        self.progress.setValue(done)
        if done == 0:
            self.status.show_message("Analysing…", "info")
            self._start_time = time.monotonic()
            return
        elapsed = time.monotonic() - self._start_time
        remaining = elapsed / done * (total - done)
        self.eta.setText(f"{done / elapsed:.1f} frames/s · {int(remaining // 60)}:{int(remaining % 60):02d} left")
        if counts:
            a, b = self._scene.direction_a_to_b, self._scene.direction_b_to_a
            self.card_a.set(str(counts.get(a, 0)))
            self.card_b.set(str(counts.get(b, 0)))
            self.card_total.set(str(sum(counts.values())))

    def _on_crossing(self, crossing) -> None:
        row = self.crossings.rowCount()
        self.crossings.insertRow(row)
        for c, value in enumerate((crossing.track_id, crossing.direction, f"{crossing.speed_kph:.1f}",
                                   crossing.entry_frame, crossing.exit_frame)):
            item = QTableWidgetItem(str(value))
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            self.crossings.setItem(row, c, item)
        self.crossings.scrollToBottom()
        speeds = [float(self.crossings.item(r, 2).text()) for r in range(self.crossings.rowCount())]
        self.card_speed.set(f"{sum(speeds) / len(speeds):.1f} km/h")

    def _on_finished(self, result: AnalysisResult) -> None:
        self._set_running(False)
        self.result = result
        self.open_folder.setEnabled(True)
        state = "stopped early" if result.stopped else "finished"
        files = "<br>".join(f"{k}: {v}" for k, v in result.output_files.items())
        self.status.show_message(
            f"<b>Analysis {state}:</b> {result.frames_processed} frames in {result.elapsed_s:.0f} s, "
            f"{sum(result.counts.values())} cyclists counted.<br>{files}", "ok")
        self.analysis_finished.emit(result)

    def _on_failed(self, message: str) -> None:
        self._set_running(False)
        self.status.show_message(f"<b>The analysis stopped with an error:</b><br>{message}<br>"
                                 "See the Messages panel for details.", "error")
        show_error(self, "Analysis failed", message)
