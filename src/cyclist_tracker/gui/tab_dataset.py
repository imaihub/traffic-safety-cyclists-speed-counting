"""Tab 1 - check a dataset folder and prepare frames/ground truth."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QColor, QFont
from PyQt6.QtWidgets import (
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..dataset import (
    ERROR,
    INFO,
    OK,
    WARNING,
    DatasetReport,
    coco_to_mot,
    create_layout,
    extract_frames,
    validate_dataset,
    write_frame_list,
)
from .common import InfoBox, PathPicker, Task, heading, show_error

LAYOUT_TEXT = """my_dataset/
├── train/            training images   (e.g. 0001.jpg …)
├── val/              validation images
├── test/             images to analyse and evaluate
├── annotations/
│   ├── train.json    boxes around cyclists (COCO format)
│   ├── val.json
│   └── test.json
├── ground_truth/     optional – only for Step 5
│   ├── tracking_test.txt   boxes + cyclist IDs
│   ├── speeds_test.csv     reference speeds
│   └── counts_test.csv     reference crossings
└── scene.json        optional – made in Step 2"""

LEVEL_STYLE = {OK: ("✓", "#1e7a34"), INFO: ("i", "#1f5fa8"), WARNING: ("!", "#8a5a00"), ERROR: ("✗", "#a12222")}


class DatasetTab(QWidget):
    dataset_checked = pyqtSignal(object)  # DatasetReport

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.report: DatasetReport | None = None
        self._task: Task | None = None

        self.folder = PathPicker("folder", placeholder="Folder that contains train/, val/, test/ and annotations/")
        check = QPushButton("Check my dataset")
        check.setObjectName("Primary")
        check.clicked.connect(self.check)
        create = QPushButton("Create empty folder structure")
        create.clicked.connect(self.create_structure)
        row = QHBoxLayout()
        row.addWidget(QLabel("Dataset folder:"))
        row.addWidget(self.folder, 1)
        row.addWidget(check)
        row.addWidget(create)

        self.summary = InfoBox("Choose your dataset folder and press <b>Check my dataset</b>. "
                               "Don't have a dataset yet? Use <b>Try with demo data</b> on the Start page.")
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["", "Item", "Result"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setWordWrap(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)

        results = QWidget()
        rl = QVBoxLayout(results)
        rl.setContentsMargins(0, 0, 0, 0)
        rl.addWidget(self.summary)
        rl.addWidget(self.table, 1)

        # Explanation of the structure
        explain = QGroupBox("How a dataset must look")
        el = QVBoxLayout(explain)
        tree = QPlainTextEdit(LAYOUT_TEXT)
        tree.setReadOnly(True)
        mono = QFont("Monospace")
        mono.setStyleHint(QFont.StyleHint.TypeWriter)
        tree.setFont(mono)
        el.addWidget(tree, 1)
        el.addWidget(InfoBox(
            "Images are single video frames (the paper used every 5th frame of 25 fps video, i.e. 5 frames per "
            "second). Annotations are made with a tool such as CVAT and exported as <b>COCO 1.0</b>: one box per "
            "cyclist per frame, with a <b>track_id</b> attribute that stays the same for one cyclist. "
            "See <i>docs/DATASET.md</i> or the user manual for all details."))

        tools = self._build_tools()
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(explain, 1)
        right_layout.addWidget(tools)

        splitter = QSplitter()
        splitter.addWidget(results)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        layout = QVBoxLayout(self)
        layout.addWidget(heading("Step 1 · Dataset",
                                 "Check that your own frames and annotations are organised correctly, or prepare them "
                                 "from videos. Your data never leaves your computer."))
        layout.addLayout(row)
        layout.addWidget(splitter, 1)

    def _build_tools(self) -> QGroupBox:
        box = QGroupBox("Preparation tools")
        form = QFormLayout(box)
        self.video = PathPicker("file", "Videos (*.mp4 *.avi *.mov *.mkv);;All files (*)", "Video to cut into frames")
        self.every = QSpinBox(minimum=1, maximum=100, value=5)
        self.every.setToolTip("Keep every N-th frame. 25 fps video with N = 5 gives 5 frames per second.")
        self.frames_out = PathPicker("folder", placeholder="Where to save the frames (e.g. my_dataset/test)")
        extract = QPushButton("Extract frames")
        extract.clicked.connect(self.extract)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        form.addRow("Video", self.video)
        form.addRow("Keep every N-th frame", self.every)
        form.addRow("Save frames to", self.frames_out)
        form.addRow(extract, self.progress)

        self.coco = PathPicker("file", "COCO annotations (*.json)", "annotations/test.json")
        gt = QPushButton("Create tracking ground truth")
        gt.setToolTip("Writes ground_truth/tracking_test.txt from the boxes and track IDs in the COCO file.")
        gt.clicked.connect(self.make_ground_truth)
        lst = QPushButton("Create frame list")
        lst.setToolTip("Writes a text file with the names of all annotated frames (to analyse only those).")
        lst.clicked.connect(self.make_frame_list)
        buttons = QHBoxLayout()
        buttons.addWidget(gt)
        buttons.addWidget(lst)
        form.addRow("COCO file", self.coco)
        form.addRow("", buttons)
        return box

    # ------------------------------------------------------------------ actions
    def set_folder(self, path: str | Path) -> None:
        self.folder.set_path(path)

    def check(self) -> DatasetReport | None:
        if not self.folder.path():
            show_error(self, "No folder", "Choose the dataset folder first.")
            return None
        report = validate_dataset(self.folder.path())
        self.show_report(report)
        return report

    def show_report(self, report: DatasetReport) -> None:
        self.report = report
        self.table.setRowCount(len(report.checks))
        for r, check in enumerate(report.checks):
            symbol, colour = LEVEL_STYLE[check.level]
            for c, text in enumerate((symbol, check.topic, check.message)):
                item = QTableWidgetItem(text)
                if c == 0:
                    item.setForeground(QColor(colour))
                    font = item.font()
                    font.setBold(True)
                    item.setFont(font)
                self.table.setItem(r, c, item)
        self.table.resizeRowsToContents()
        train = "✓ ready for training" if report.ready_for_training else "✗ not ready for training"
        analysis = "✓ ready for analysis" if report.ready_for_analysis else "✗ not ready for analysis"
        kind = "error" if report.errors else ("ok" if report.ready_for_analysis else "warning")
        errors = f"<br><b>{len(report.errors)} problem(s) must be fixed</b> (red crosses below)." if report.errors else ""
        self.summary.show_message(f"<b>{train} · {analysis}</b>{errors}", kind)
        if (report.root / "annotations").is_dir() and not self.coco.path():
            test_json = report.annotations.get("test")
            if test_json:
                self.coco.set_path(test_json)
        self.dataset_checked.emit(report)

    def create_structure(self) -> None:
        if not self.folder.path():
            show_error(self, "No folder", "Choose (or type) the folder in which the structure must be created.")
            return
        created = create_layout(self.folder.path())
        QMessageBox.information(self, "Folders created",
                                f"Created {len(created)} folder(s). Put your images in train/, val/ and test/, "
                                "and the COCO files in annotations/.")
        self.check()

    def extract(self) -> None:
        video, out = self.video.path(), self.frames_out.path()
        if not video or not out:
            show_error(self, "Missing input", "Choose a video and a folder for the frames.")
            return
        every = self.every.value()
        self.progress.setVisible(True)
        self.progress.setValue(0)

        def work(task: Task):
            return extract_frames(video, out, every, progress=lambda d, t: task.report(d, max(t, 1)),
                                  should_stop=task.should_stop)

        self._task = Task(work)
        self._task.progress.connect(lambda d, t, _: (self.progress.setMaximum(t), self.progress.setValue(d)))
        self._task.finished.connect(lambda n: QMessageBox.information(self, "Frames saved", f"Saved {n} frames in {out}."))
        self._task.failed.connect(lambda msg: show_error(self, "Extraction failed", msg))
        self._task.start()

    def make_ground_truth(self) -> None:
        coco = self.coco.path()
        if not coco:
            show_error(self, "No COCO file", "Choose a COCO annotation file.")
            return
        root = Path(self.folder.path() or Path(coco).parent.parent)
        (root / "ground_truth").mkdir(parents=True, exist_ok=True)
        target = root / "ground_truth" / "tracking_test.txt"
        try:
            n = coco_to_mot(coco, target)
        except (ValueError, KeyError, OSError) as exc:
            show_error(self, "Cannot create ground truth", str(exc))
            return
        QMessageBox.information(self, "Ground truth created", f"Wrote {n} boxes to\n{target}")

    def make_frame_list(self) -> None:
        coco = self.coco.path()
        if not coco:
            show_error(self, "No COCO file", "Choose a COCO annotation file.")
            return
        target = Path(coco).with_name(Path(coco).stem + "_frames.txt")
        n = write_frame_list(coco, target)
        QMessageBox.information(self, "Frame list created", f"Wrote {n} frame names to\n{target}")
