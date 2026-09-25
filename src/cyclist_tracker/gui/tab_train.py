"""Tab 3 - fine-tune the DETR cyclist detector on the user's own annotated frames."""

from __future__ import annotations

from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .common import InfoBox, PathPicker, Task, heading, show_error


class TrainTab(QWidget):
    model_trained = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._task: Task | None = None
        self.best_checkpoint: str | None = None
        self.history: list[tuple[int, float | None, float | None]] = []

        data = QGroupBox("Data and output")
        df = QFormLayout(data)
        self.dataset = PathPicker("folder", placeholder="dataset folder with train/, val/ and annotations/")
        self.output = PathPicker("folder", placeholder="where the trained model is stored")
        self.output.set_path(Path.cwd() / "runs" / "detr")
        df.addRow("Dataset folder", self.dataset)
        df.addRow("Save model in", self.output)

        params = QGroupBox("Training settings (defaults are those of the paper)")
        pf = QFormLayout(params)
        self.epochs = QSpinBox(minimum=1, maximum=1000, value=30)
        self.epochs.setToolTip("How many times the model sees all training images.")
        self.batch = QSpinBox(minimum=1, maximum=64, value=2)
        self.batch.setToolTip("Images per step. Lower this if you get an 'out of memory' error.")
        self.lr = QDoubleSpinBox(decimals=6, minimum=1e-6, maximum=1e-2, singleStep=1e-5, value=1e-4)
        self.lr_backbone = QDoubleSpinBox(decimals=6, minimum=1e-7, maximum=1e-2, singleStep=1e-6, value=1e-5)
        self.precision = QComboBox()
        self.precision.addItem("Normal (32-bit)", "32-true")
        self.precision.addItem("Mixed 16-bit - less GPU memory, faster", "16-mixed")
        self.use_gpu = QCheckBox("Use the GPU (strongly recommended)")
        self.use_gpu.setChecked(True)
        pf.addRow("Epochs", self.epochs)
        pf.addRow("Batch size", self.batch)
        pf.addRow("Learning rate", self.lr)
        pf.addRow("Learning rate backbone", self.lr_backbone)
        pf.addRow("Precision", self.precision)
        pf.addRow("", self.use_gpu)

        self.start_button = QPushButton("▶  Start training")
        self.start_button.setObjectName("Primary")
        self.start_button.clicked.connect(self.start)
        self.stop_button = QPushButton("■  Stop after this step")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(lambda: self._task and self._task.stop())
        self.use_model = QPushButton("Use this model in Step 4")
        self.use_model.setEnabled(False)
        self.use_model.clicked.connect(lambda: self.best_checkpoint and self.model_trained.emit(self.best_checkpoint))
        buttons = QHBoxLayout()
        buttons.addWidget(self.start_button, 2)
        buttons.addWidget(self.stop_button, 1)

        left = QWidget()
        ll = QVBoxLayout(left)
        ll.addWidget(InfoBox(
            "Training teaches the detector what cyclists look like <b>in your camera images</b>. It needs "
            "annotated frames in <i>train/</i> and <i>val/</i> (check them in Step 1). Training takes hours on a "
            "GPU; you can skip this step and use the pre-trained COCO detector in Step 4 to try things out."))
        ll.addWidget(data)
        ll.addWidget(params)
        ll.addLayout(buttons)
        ll.addWidget(self.use_model)
        ll.addStretch()

        self.figure = Figure(figsize=(5, 3), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.progress = QProgressBar()
        self.progress.setFormat("epoch %v / %m")
        self.status = InfoBox("The training loss (how wrong the model still is) is plotted here after every epoch. "
                              "Both lines should go down.")
        right = QWidget()
        rl = QVBoxLayout(right)
        rl.addWidget(self.canvas, 1)
        rl.addWidget(self.progress)
        rl.addWidget(self.status)

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        layout = QVBoxLayout(self)
        layout.addWidget(heading("Step 3 · Train the detector (optional)",
                                 "Fine-tune the DETR model on your own annotated frames."))
        layout.addWidget(splitter, 1)
        self._plot()

    def start(self) -> None:
        from ..dataset import validate_dataset

        report = validate_dataset(self.dataset.path()) if self.dataset.path() else None
        if report is None or not report.ready_for_training:
            show_error(self, "Dataset not ready", "The dataset is not ready for training. Check it in Step 1: "
                       "train/, val/ and annotations/train.json and val.json are needed.")
            return
        from ..training import TrainConfig

        config = TrainConfig(dataset_dir=self.dataset.path(), output_dir=self.output.path(),
                             epochs=self.epochs.value(), batch_size=self.batch.value(), lr=self.lr.value(),
                             lr_backbone=self.lr_backbone.value(), precision=self.precision.currentData(),
                             use_gpu=self.use_gpu.isChecked())
        self.history = []
        self._plot()
        self.progress.setMaximum(config.epochs)
        self.progress.setValue(0)
        self.status.show_message("Preparing the training (loading images and the pre-trained model)…", "info")

        def work(task: Task):
            from ..training import train_detector

            return train_detector(config, progress=lambda e, n, m: task.report(e, n, m), should_stop=task.should_stop)

        self._task = Task(work)
        self._task.progress.connect(self._on_epoch)
        self._task.finished.connect(self._on_finished)
        self._task.failed.connect(self._on_failed)
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self._task.start()

    def _on_epoch(self, epoch: int, total: int, metrics: dict) -> None:
        self.progress.setMaximum(total)
        self.progress.setValue(epoch)
        self.history.append((epoch, metrics.get("train_loss"), metrics.get("val_loss")))
        self.status.show_message(f"Epoch {epoch} of {total} done. Validation loss: "
                                 f"{metrics.get('val_loss', float('nan')):.4f}", "info")
        self._plot()

    def _plot(self) -> None:
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        if self.history:
            epochs = [h[0] for h in self.history]
            for idx, label in ((1, "training loss"), (2, "validation loss")):
                values = [h[idx] for h in self.history]
                if any(v is not None for v in values):
                    ax.plot(epochs, values, marker="o", label=label)
            ax.legend()
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss (lower is better)")
        ax.grid(alpha=0.3)
        self.canvas.draw_idle()

    def _on_finished(self, best) -> None:
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        if best:
            self.best_checkpoint = str(best)
            self.use_model.setEnabled(True)
            self.status.show_message(f"<b>Training finished.</b> Best model:<br>{best}", "ok")
        else:
            self.status.show_message("Training stopped before a model was saved.", "warning")

    def _on_failed(self, message: str) -> None:
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        hint = ("<br>Tip: lower the batch size or choose Mixed 16-bit precision."
                if "out of memory" in message.lower() else "")
        self.status.show_message(f"<b>Training failed:</b> {message}{hint}", "error")
