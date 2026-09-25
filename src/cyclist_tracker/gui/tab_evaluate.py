"""Tab 5 - compare results with ground truth: tracking, speed and counting."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QListWidget,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .. import evaluation as ev
from .common import InfoBox, PathPicker, heading, show_error

PCT = {"idf1", "mota", "precision", "recall", "Precision", "Recall", "F1"}


def fill_table(table: QTableWidget, df: pd.DataFrame, explanations: dict[str, str] | None = None) -> None:
    table.clear()
    table.setRowCount(len(df))
    table.setColumnCount(len(df.columns))
    table.setHorizontalHeaderLabels([str(c) for c in df.columns])
    for r, (_, row) in enumerate(df.iterrows()):
        for c, col in enumerate(df.columns):
            value = row[col]
            if isinstance(value, float | np.floating):
                if np.isnan(value):
                    text = "–"
                elif col in PCT:
                    text = f"{value * 100:.1f} %"
                elif str(col).endswith("%"):
                    text = f"{value:.1f} %"
                else:
                    text = f"{value:.3f}"
            else:
                text = str(value)
            item = QTableWidgetItem(text)
            item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
            table.setItem(r, c, item)
    table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
    table.resizeRowsToContents()


class _Page(QWidget):
    def __init__(self, intro: str):
        super().__init__()
        self.form = QFormLayout()
        self.table = QTableWidget()
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.status = InfoBox(intro)
        self.run_button = QPushButton("Evaluate")
        self.run_button.setObjectName("Primary")
        export = QPushButton("Save table as CSV…")
        export.clicked.connect(self.export)
        buttons = QHBoxLayout()
        buttons.addWidget(self.run_button)
        buttons.addWidget(export)
        buttons.addStretch()
        self.layout_ = QVBoxLayout(self)
        self.layout_.addLayout(self.form)
        self.layout_.addLayout(buttons)
        self.layout_.addWidget(self.status)
        self.layout_.addWidget(self.table, 1)
        self.result: pd.DataFrame | None = None

    def export(self) -> None:
        if self.result is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "Save table", "evaluation.csv", "CSV (*.csv)")
        if path:
            self.result.to_csv(path, index=False)


class TrackingPage(_Page):
    def __init__(self):
        super().__init__("Compares the tracked boxes with hand-made boxes (both in the tracking text format). "
                         "IDF1 and MOTA are the main scores used in the paper.")
        self.gt = PathPicker("file", "Tracking files (*.txt);;All files (*)", "ground_truth/tracking_test.txt")
        self.pred = PathPicker("file", "Tracking files (*.txt);;All files (*)", "results/analysis_tracks.txt")
        self.iou = QDoubleSpinBox(minimum=0.05, maximum=1.0, value=0.5, singleStep=0.05)
        self.iou.setToolTip("How much a tracked box must overlap a true box to count as correct.")
        self.form.addRow("Ground truth", self.gt)
        self.form.addRow("Tracking result", self.pred)
        self.form.addRow("Minimum overlap (IoU)", self.iou)
        self.run_button.clicked.connect(self.run)

    def run(self) -> None:
        try:
            summary = ev.evaluate_tracking(self.gt.path(), self.pred.path(), self.iou.value())
        except (OSError, ValueError) as exc:
            show_error(self, "Cannot evaluate", str(exc))
            return
        row = summary.iloc[0]
        self.result = pd.DataFrame([
            {"Metric": ev.MOT_METRICS[k].split(" - ")[0], "Value": float(row[k]),
             "Meaning": ev.MOT_METRICS[k].split(" - ")[1]} for k in ev.MOT_METRICS
        ])
        shown = self.result.copy()
        shown["Value"] = [f"{v * 100:.1f} %" if k in PCT else f"{v:.3f}" if k == "motp" else f"{v:.0f}"
                          for k, v in zip(ev.MOT_METRICS, self.result["Value"], strict=True)]
        fill_table(self.table, shown)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.status.show_message(f"IDF1 {row['idf1'] * 100:.1f} % · MOTA {row['mota'] * 100:.1f} % · "
                                 f"{int(row['num_switches'])} ID switches", "ok")


class SpeedPage(_Page):
    def __init__(self):
        super().__init__("Compares measured speeds with reference speeds. Crossings are paired when direction, "
                         "entry frame and exit frame agree (within the tolerance). You can compare several results.")
        self.gt = PathPicker("file", "CSV files (*.csv *.txt);;All files (*)", "ground_truth/speeds_test.csv")
        self.preds = QListWidget()
        self.preds.setMaximumHeight(90)
        add = QPushButton("Add result…")
        add.clicked.connect(self._add)
        remove = QPushButton("Remove")
        remove.clicked.connect(lambda: [self.preds.takeItem(self.preds.row(i)) for i in self.preds.selectedItems()])
        row = QHBoxLayout()
        row.addWidget(self.preds, 1)
        col = QVBoxLayout()
        col.addWidget(add)
        col.addWidget(remove)
        row.addLayout(col)
        self.tolerance = QSpinBox(minimum=0, maximum=1000, value=5)
        self.form.addRow("Reference speeds", self.gt)
        self.form.addRow("Speed results", row)
        self.form.addRow("Frame tolerance", self.tolerance)
        self.figure = Figure(figsize=(6, 3), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setMinimumHeight(260)
        self.layout_.addWidget(self.canvas, 2)
        save_plot = QPushButton("Save plot…")
        save_plot.clicked.connect(self._save_plot)
        self.layout_.addWidget(save_plot)
        self.run_button.clicked.connect(self.run)

    def add_prediction(self, path: str) -> None:
        if path and not self.preds.findItems(path, Qt.MatchFlag.MatchExactly):
            self.preds.addItem(path)

    def _add(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(self, "Speed results", "", "CSV files (*.csv);;All files (*)")
        for p in paths:
            self.add_prediction(p)

    def run(self) -> None:
        files = [self.preds.item(i).text() for i in range(self.preds.count())]
        if not files:
            show_error(self, "No results", "Add at least one speed result file (…_speeds.csv).")
            return
        try:
            table, matches = ev.evaluate_speeds(self.gt.path(), files, [Path(f).stem for f in files],
                                                self.tolerance.value())
        except (OSError, ValueError, KeyError) as exc:
            show_error(self, "Cannot evaluate", str(exc))
            return
        self.result = table
        fill_table(self.table, table)
        ev.speed_scatter_figure(matches, self.figure)
        self.canvas.draw_idle()
        best = table.sort_values("MAE").iloc[0]
        self.status.show_message(f"Lowest error: <b>{best['Model']}</b> with an average error (MAE) of "
                                 f"{best['MAE']:.2f} km/h over {int(best['Matches'])} cyclists.", "ok")

    def _save_plot(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save plot", "speed_comparison.png", "PNG (*.png)")
        if path:
            self.figure.savefig(path, dpi=150)


class CountingPage(_Page):
    def __init__(self):
        super().__init__("Compares counted cyclists with reference crossings, per direction. A counted cyclist is "
                         "correct (TP) when a reference crossing in the same direction has nearly the same entry "
                         "and exit frame.")
        self.gt = PathPicker("file", "CSV files (*.csv *.txt);;All files (*)", "ground_truth/counts_test.csv")
        self.pred = PathPicker("file", "CSV files (*.csv);;All files (*)", "results/analysis_speeds.csv")
        self.tolerance = QSpinBox(minimum=0, maximum=1000, value=10)
        self.form.addRow("Reference crossings", self.gt)
        self.form.addRow("Counting result", self.pred)
        self.form.addRow("Frame tolerance", self.tolerance)
        self.run_button.clicked.connect(self.run)

    def run(self) -> None:
        try:
            table = ev.evaluate_counts(self.gt.path(), self.pred.path(), self.tolerance.value())
        except (OSError, ValueError, KeyError) as exc:
            show_error(self, "Cannot evaluate", str(exc))
            return
        self.result = table
        fill_table(self.table, table)
        total = table.iloc[-1]
        self.status.show_message(f"{int(total['TP'])} of {int(total['Reference'])} cyclists counted correctly, "
                                 f"{int(total['FP'])} counted wrongly · F1 {total['F1'] * 100:.1f} %", "ok")


class EvaluateTab(QWidget):
    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.tracking, self.speed, self.counting = TrackingPage(), SpeedPage(), CountingPage()
        self.pages = QTabWidget()
        self.pages.addTab(self.tracking, "Tracking")
        self.pages.addTab(self.speed, "Speed")
        self.pages.addTab(self.counting, "Counting")
        layout = QVBoxLayout(self)
        layout.addWidget(heading("Step 5 · Evaluate (optional)",
                                 "How good are the results? This needs ground truth: hand-made boxes, reference "
                                 "speeds or reference counts for the same frames."))
        layout.addWidget(self.pages, 1)

    def set_ground_truth(self, files: dict[str, Path]) -> None:
        if "tracking" in files:
            self.tracking.gt.set_path(files["tracking"])
        if "speeds" in files:
            self.speed.gt.set_path(files["speeds"])
        if "counts" in files:
            self.counting.gt.set_path(files["counts"])

    def set_results(self, output_files: dict[str, str]) -> None:
        if "tracks" in output_files:
            self.tracking.pred.set_path(output_files["tracks"])
        if "speeds" in output_files:
            self.speed.add_prediction(output_files["speeds"])
            self.counting.pred.set_path(output_files["speeds"])
