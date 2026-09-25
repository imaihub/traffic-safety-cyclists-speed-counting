"""Start page (welcome, workflow, computer check, demo data) and Help page."""

from __future__ import annotations

from pathlib import Path

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from .. import __version__
from ..system import system_report
from .common import InfoBox, Task, heading, show_error

STEPS = [
    ("1", "Dataset", "Check that your frames and annotations are organised correctly, or cut a video into frames."),
    ("2", "Camera scene", "Draw the cycle track, two counting lines and the calibration points on a camera image."),
    ("3", "Train detector", "Optional: teach the detector what cyclists look like in your camera images."),
    ("4", "Analyse", "Count cyclists per direction and measure their speed. Watch it live, results are saved."),
    ("5", "Evaluate", "Optional: compare the results with hand-made ground truth."),
]


PIPELINE = [
    ("Detect", "DETR finds every cyclist in a frame"),
    ("Track", "DeepSORT follows each cyclist and gives it an ID"),
    ("Count", "passing line A and B counts it, with direction"),
    ("Speed", "calibration converts pixels to metres; metres per second → km/h"),
]


class StartTab(QWidget):
    go_to_step = pyqtSignal(int)
    demo_created = pyqtSignal(str)
    sample_requested = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._task: Task | None = None
        layout = QVBoxLayout(self)
        layout.addWidget(heading(
            "Cyclist counting and speed estimation",
            "This program finds cyclists in traffic-camera video, follows each one, counts how many pass in each "
            "direction and estimates their speed. It is based on research within the BikeSafeAI project "
            "(NHL Stenden, Computer Vision & Data Science)."))

        steps = QGroupBox("How it works - five steps, each with its own tab")
        grid = QGridLayout(steps)
        for row, (number, title, text) in enumerate(STEPS):
            badge = QLabel(number)
            badge.setObjectName("StepBadge")
            badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            badge.setFixedSize(34, 34)
            name = QLabel(f"<b>{title}</b>")
            desc = QLabel(text)
            desc.setWordWrap(True)
            go = QPushButton("Open")
            go.clicked.connect(lambda _, i=row + 1: self.go_to_step.emit(i))
            grid.addWidget(badge, row, 0)
            grid.addWidget(name, row, 1)
            grid.addWidget(desc, row, 2)
            grid.addWidget(go, row, 3)
        grid.setColumnStretch(2, 1)

        demo = QGroupBox("New here? Try it with example data")
        dl = QVBoxLayout(demo)
        demo_text = QLabel(
            "The original traffic videos are private. The demo creates a small artificial dataset (a cycle track "
            "with cyclists at known speeds) and fills in all tabs, so you can click through every step.")
        demo_text.setWordWrap(True)
        dl.addWidget(demo_text)
        row = QHBoxLayout()
        self.demo_button = QPushButton("Try with demo data")
        self.demo_button.setObjectName("Primary")
        self.demo_button.clicked.connect(self.choose_demo_folder)
        self.demo_progress = QProgressBar()
        self.demo_progress.setVisible(False)
        row.addWidget(self.demo_button)
        row.addWidget(self.demo_progress, 1)
        dl.addLayout(row)
        sample_text = QLabel(
            "Or open the <b>real sample dataset</b> that comes with the program: 73 seconds of drone video of a "
            "campus road with 53 hand-annotated cyclists (Stanford Drone Dataset, non-commercial licence).")
        sample_text.setWordWrap(True)
        dl.addWidget(sample_text)
        self.sample_button = QPushButton("Open sample dataset")
        self.sample_button.clicked.connect(self.sample_requested)
        dl.addWidget(self.sample_button)

        system = QGroupBox("Your computer")
        sl = QVBoxLayout(system)
        self.system_table = QTableWidget(0, 2)
        self.system_table.horizontalHeader().setVisible(False)
        self.system_table.verticalHeader().setVisible(False)
        self.system_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.system_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.system_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.system_table.setFrameShape(QFrame.Shape.NoFrame)
        sl.addWidget(self.system_table)
        sl.addStretch()

        method = QGroupBox("What happens inside")
        ml = QHBoxLayout(method)
        for i, (title, text) in enumerate(PIPELINE):
            if i:
                arrow = QLabel("→")
                arrow.setObjectName("Arrow")
                ml.addWidget(arrow)
            card = QLabel(f"<b>{title}</b><br>{text}")
            card.setObjectName("PipelineCard")
            card.setWordWrap(True)
            card.setAlignment(Qt.AlignmentFlag.AlignCenter)
            ml.addWidget(card, 1)

        bottom = QHBoxLayout()
        left = QVBoxLayout()
        left.addWidget(demo)
        left.addWidget(method)
        left.addStretch()
        bottom.addLayout(left, 1)
        bottom.addWidget(system, 1)

        layout.addWidget(steps)
        layout.addLayout(bottom, 1)
        self.refresh_system()

    def refresh_system(self) -> None:
        rows = system_report()
        self.system_table.setRowCount(len(rows))
        for r, (item, value, ok) in enumerate(rows):
            a = QTableWidgetItem(("✓  " if ok else "•  ") + item)
            a.setForeground(QColor("#1e7a34" if ok else "#8a5a00"))
            self.system_table.setItem(r, 0, a)
            self.system_table.setItem(r, 1, QTableWidgetItem(value))
        self.system_table.resizeRowsToContents()
        height = sum(self.system_table.rowHeight(r) for r in range(len(rows)))
        self.system_table.setFixedHeight(height + 2 * self.system_table.frameWidth() + 2)

    def choose_demo_folder(self) -> None:
        folder = QFileDialog.getExistingDirectory(self, "Where should the demo dataset be created?", str(Path.home()))
        if folder:
            self.create_demo(Path(folder) / "cyclist_tracker_demo")

    def create_demo(self, target: Path) -> None:
        self.demo_button.setEnabled(False)
        self.demo_progress.setVisible(True)

        def work(task: Task):
            from ..synthetic import generate_demo_dataset

            return str(generate_demo_dataset(target, progress=lambda d, t: task.report(d, t)))

        self._task = Task(work)
        self._task.progress.connect(lambda d, t, _: (self.demo_progress.setMaximum(t), self.demo_progress.setValue(d)))
        self._task.finished.connect(self._demo_done)
        self._task.failed.connect(lambda m: (self.demo_button.setEnabled(True), show_error(self, "Demo failed", m)))
        self._task.start()

    def _demo_done(self, path: str) -> None:
        self.demo_button.setEnabled(True)
        self.demo_progress.setVisible(False)
        self.demo_created.emit(path)


HELP_HTML = f"""
<h2>Help</h2>
<p>Version {__version__}. The full illustrated guide is the <b>user manual (PDF)</b> - use the button above.</p>
<h3>Words used in this program</h3>
<table cellpadding="4">
<tr><td><b>Frame</b></td><td>One still image from a video. The paper used 5 frames per second.</td></tr>
<tr><td><b>Annotation</b></td><td>A hand-drawn box around a cyclist in a frame, made with a tool such as CVAT.</td></tr>
<tr><td><b>Detector</b></td><td>The AI model (DETR) that finds cyclists in one frame.</td></tr>
<tr><td><b>Tracker</b></td><td>DeepSORT: links the boxes of the same cyclist in consecutive frames and gives it an ID.</td></tr>
<tr><td><b>Tracking features</b></td><td>A small network that describes what a cyclist looks like, so the tracker can recognise them after they were hidden for a moment.</td></tr>
<tr><td><b>Region of interest</b></td><td>The area around the cycle track. Everything outside is ignored.</td></tr>
<tr><td><b>Counting lines A and B</b></td><td>A cyclist is counted once it has passed both lines; the order gives the direction.</td></tr>
<tr><td><b>Calibration / homography</b></td><td>Four or more road points with known positions in metres. They let the program convert pixels into metres, even though the camera looks at the road at an angle.</td></tr>
<tr><td><b>Ground truth</b></td><td>Correct answers made by hand, needed only to measure the quality (Step 5).</td></tr>
<tr><td><b>IDF1 / MOTA</b></td><td>Tracking scores between 0 and 100 %; higher is better.</td></tr>
<tr><td><b>MAE</b></td><td>Mean absolute error: the average difference between measured and true speed, in km/h.</td></tr>
</table>
<h3>Common problems</h3>
<ul>
<li><b>Nothing is counted.</b> Check in Step 2 that both lines cross the whole cycle track and lie inside the region
of interest, and that cyclists really move from one line to the other.</li>
<li><b>Speeds are much too high or too low.</b> Check the calibration: the road positions (metres) must match the
order in which you clicked the points. The status box in Step 2 shows the measured distance between the lines.</li>
<li><b>Very slow.</b> Without a GPU each frame can take a second or more. Use a shorter video, or 'every N-th frame'.</li>
<li><b>"Out of memory" while training.</b> Lower the batch size or choose mixed 16-bit precision.</li>
<li><b>The first analysis waits a long time.</b> The pre-trained weights are downloaded once (about 200 MB).</li>
</ul>
<h3>Privacy</h3>
<p>Everything runs on your own computer; no images are uploaded. Camera images of people are personal data: make sure
you are allowed to record and process them, and only keep what you need.</p>
"""


class HelpTab(QWidget):
    open_manual = pyqtSignal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        button = QPushButton("Open the user manual (PDF)")
        button.setObjectName("Primary")
        button.clicked.connect(self.open_manual)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setHtml(HELP_HTML)
        layout = QVBoxLayout(self)
        row = QHBoxLayout()
        row.addWidget(button)
        row.addStretch()
        layout.addLayout(row)
        layout.addWidget(InfoBox("Hover over a setting to see a short explanation. Messages from the program appear in "
                                 "the <b>Messages</b> panel at the bottom of the window."))
        layout.addWidget(browser, 1)
