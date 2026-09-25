"""Reusable widgets, background workers and helpers for the GUI."""

from __future__ import annotations

import logging
import threading
import time
import traceback
from collections.abc import Callable
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QObject, Qt, QThread, QUrl, pyqtSignal
from PyQt6.QtGui import QDesktopServices, QImage, QPixmap
from PyQt6.QtWidgets import (
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

log = logging.getLogger("cyclist_tracker.gui")


# ------------------------------------------------------------------------------------ helpers
def bgr_to_qimage(image: np.ndarray) -> QImage:
    rgb = np.ascontiguousarray(image[:, :, ::-1])
    h, w = rgb.shape[:2]
    return QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888).copy()


def open_path(path: str | Path) -> None:
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))


def show_error(parent: QWidget, title: str, message: str) -> None:
    QMessageBox.critical(parent, title, message)


# ------------------------------------------------------------------------------------ widgets
class PathPicker(QWidget):
    """A text field with a "Browse..." button for a file or folder."""

    changed = pyqtSignal(str)

    def __init__(self, mode: str = "file", file_filter: str = "All files (*)", placeholder: str = "",
                 save: bool = False, parent: QWidget | None = None):
        super().__init__(parent)
        self.mode, self.file_filter, self.save = mode, file_filter, save
        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        self.edit.textChanged.connect(self.changed)
        button = QPushButton("Browse…")
        button.clicked.connect(self._browse)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.edit, 1)
        layout.addWidget(button)

    def _browse(self) -> None:
        start = self.edit.text() or str(Path.home())
        if self.mode == "folder":
            path = QFileDialog.getExistingDirectory(self, "Choose a folder", start)
        elif self.save:
            path, _ = QFileDialog.getSaveFileName(self, "Save as", start, self.file_filter)
        else:
            path, _ = QFileDialog.getOpenFileName(self, "Choose a file", start, self.file_filter)
        if path:
            self.edit.setText(path)

    def path(self) -> str:
        return self.edit.text().strip()

    def set_path(self, path: str | Path | None) -> None:
        self.edit.setText(str(path) if path else "")


class InfoBox(QLabel):
    """A friendly, coloured explanation box."""

    STYLES = {
        "info": ("#e8f1fb", "#1f5fa8"),
        "ok": ("#e7f6ea", "#1e7a34"),
        "warning": ("#fff6e0", "#8a5a00"),
        "error": ("#fdeaea", "#a12222"),
    }

    def __init__(self, text: str = "", kind: str = "info", parent: QWidget | None = None):
        super().__init__(text, parent)
        self.setWordWrap(True)
        self.setTextFormat(Qt.TextFormat.RichText)
        self.setOpenExternalLinks(True)
        self.set_kind(kind)

    def set_kind(self, kind: str) -> None:
        bg, fg = self.STYLES[kind]
        self.setStyleSheet(f"QLabel {{ background: {bg}; color: {fg}; border-radius: 6px; padding: 8px; }}")

    def show_message(self, text: str, kind: str = "info") -> None:
        self.setText(text)
        self.set_kind(kind)


class StatCard(QFrame):
    """A big number with a caption, e.g. a cyclist count."""

    def __init__(self, caption: str, value: str = "0", parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("StatCard")
        self.value = QLabel(value)
        self.value.setObjectName("StatValue")
        self.value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.caption = QLabel(caption)
        self.caption.setObjectName("StatCaption")
        self.caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.caption.setWordWrap(True)
        layout = QVBoxLayout(self)
        layout.addWidget(self.value)
        layout.addWidget(self.caption)

    def set(self, value: str, caption: str | None = None) -> None:
        self.value.setText(value)
        if caption is not None:
            self.caption.setText(caption)


class ImageView(QLabel):
    """Shows an image scaled to the available space, keeping its aspect ratio."""

    def __init__(self, placeholder: str = "", parent: QWidget | None = None):
        super().__init__(placeholder, parent)
        self._pixmap: QPixmap | None = None
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(320, 180)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet("QLabel { background: #20242a; color: #c8ccd2; border-radius: 6px; }")

    def set_image(self, image_bgr: np.ndarray) -> None:
        self._pixmap = QPixmap.fromImage(bgr_to_qimage(image_bgr))
        self._rescale()

    def resizeEvent(self, event) -> None:  # noqa: N802 (Qt API)
        super().resizeEvent(event)
        self._rescale()

    def _rescale(self) -> None:
        if self._pixmap is not None:
            self.setPixmap(self._pixmap.scaled(self.size(), Qt.AspectRatioMode.KeepAspectRatio,
                                               Qt.TransformationMode.SmoothTransformation))


def heading(text: str, sub: str = "") -> QWidget:
    box = QWidget()
    layout = QVBoxLayout(box)
    layout.setContentsMargins(0, 0, 0, 4)
    title = QLabel(text)
    title.setObjectName("Heading")
    layout.addWidget(title)
    if sub:
        subtitle = QLabel(sub)
        subtitle.setObjectName("SubHeading")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)
    return box


# ------------------------------------------------------------------------------ background work
class Task(QObject):
    """Runs ``fn(task)`` in a background thread.

    ``fn`` receives this task and may call :meth:`report` (progress),
    :meth:`send_preview` (an image) and :meth:`emit_item` (any object, e.g. a
    counted cyclist); it should poll :meth:`should_stop` to support the Stop button.
    """

    progress = pyqtSignal(int, int, object)
    preview = pyqtSignal(object)
    item = pyqtSignal(object)
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, fn: Callable[[Task], object], preview_interval: float = 0.1):
        super().__init__()
        self._fn = fn
        self._stop = threading.Event()
        self._last_preview = 0.0
        self._preview_interval = preview_interval
        self._thread = QThread()
        self.moveToThread(self._thread)
        self._thread.started.connect(self._run)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def should_stop(self) -> bool:
        return self._stop.is_set()

    def report(self, done: int, total: int, extra: object = None) -> None:
        self.progress.emit(done, total, extra)

    def send_preview(self, image: np.ndarray) -> None:
        now = time.monotonic()
        if now - self._last_preview >= self._preview_interval:
            self._last_preview = now
            self.preview.emit(image)

    def emit_item(self, obj: object) -> None:
        self.item.emit(obj)

    def is_running(self) -> bool:
        return self._thread.isRunning()

    def wait(self, ms: int = 30000) -> bool:
        return self._thread.wait(ms)

    def _run(self) -> None:
        try:
            result = self._fn(self)
        except Exception as exc:  # shown to the user, full trace in the log
            log.error("Task failed:\n%s", traceback.format_exc())
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        else:
            self.finished.emit(result)
        finally:
            self._thread.quit()


class _LogEmitter(QObject):
    message = pyqtSignal(str, int)


class QtLogHandler(logging.Handler):
    """Forwards log records to the GUI message panel (thread-safe through a Qt signal)."""

    def __init__(self):
        super().__init__()
        self.emitter = _LogEmitter()
        self.setFormatter(logging.Formatter("%(asctime)s  %(message)s", "%H:%M:%S"))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.emitter.message.emit(self.format(record), record.levelno)
        except RuntimeError:  # the GUI was closed
            pass
