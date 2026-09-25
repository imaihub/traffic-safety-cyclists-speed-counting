"""Tab 2 - draw the region of interest, the counting lines and the calibration on a camera image."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PyQt6.QtCore import QPointF, Qt, pyqtSignal
from PyQt6.QtGui import QBrush, QColor, QFont, QPainter, QPen, QPixmap, QPolygonF
from PyQt6.QtWidgets import (
    QButtonGroup,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QPushButton,
    QRadioButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..config import SceneConfig, utrecht_scene
from ..dataset import read_cvat_polygons
from ..geometry import calibration_error, compute_homography, world_distance
from .common import InfoBox, bgr_to_qimage, heading, show_error

MODES = {
    "roi": ("Region of interest", "#3ccf5a",
            "Click around the cycle track (at least 3 points). Only cyclists inside this area are used."),
    "line_a": ("Counting line A", "#ff8c00",
               "Click the two ends of the first counting line, straight across the track."),
    "line_b": ("Counting line B", "#1e90ff",
               "Click the two ends of the second counting line. A cyclist is counted when it passes A and B."),
    "calibration": ("Calibration points", "#e040e0",
                    "Click 4 or more road points whose real position you know (e.g. corners of road markings), "
                    "then type their position in metres in the table."),
}


class SceneCanvas(QGraphicsView):
    """Shows the camera image with the scene drawn on top; emits clicks in image pixels."""

    clicked = pyqtSignal(float, float)
    dragged = pyqtSignal(int, float, float)  # index of the moved point in the current mode

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setBackgroundBrush(QColor("#20242a"))
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self._pixmap_item = None
        self._points: list[tuple[float, float]] = []
        self._drag_index: int | None = None
        self.image_size = (0, 0)

    def set_image(self, image_bgr: np.ndarray) -> None:
        self._scene.clear()
        self._pixmap_item = self._scene.addPixmap(QPixmap.fromImage(bgr_to_qimage(image_bgr)))
        self.image_size = (image_bgr.shape[1], image_bgr.shape[0])
        self._scene.setSceneRect(0, 0, *self.image_size)
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def has_image(self) -> bool:
        return self._pixmap_item is not None

    def draw(self, scene: SceneConfig, mode: str) -> None:
        for item in list(self._scene.items()):
            if item is not self._pixmap_item:
                self._scene.removeItem(item)
        width = max(2.0, self.image_size[0] / 600)
        font = QFont()
        font.setPixelSize(int(max(14, self.image_size[0] / 60)))

        def pen(colour: str, w: float = width) -> QPen:
            p = QPen(QColor(colour), w)
            p.setCosmetic(False)
            return p

        roi_colour = MODES["roi"][1]
        if len(scene.roi) >= 2:
            fill = QColor(roi_colour)
            fill.setAlpha(40)
            self._scene.addPolygon(QPolygonF([QPointF(*p) for p in scene.roi]), pen(roi_colour), QBrush(fill))
        for key in ("line_a", "line_b"):
            pts = getattr(scene, key)
            colour = MODES[key][1]
            if len(pts) == 2:
                self._scene.addLine(*pts[0], *pts[1], pen(colour, width * 2))
                text = self._scene.addText("A" if key == "line_a" else "B", font)
                text.setDefaultTextColor(QColor(colour))
                text.setPos(pts[1][0] + 4, pts[1][1] - font.pixelSize())
        for i, (x, y) in enumerate(scene.calibration_image_points):
            colour = MODES["calibration"][1]
            text = self._scene.addText(str(i + 1), font)
            text.setDefaultTextColor(QColor(colour))
            text.setPos(x + 6, y - font.pixelSize() - 4)
        self._points = {"roi": scene.roi, "line_a": scene.line_a, "line_b": scene.line_b,
                        "calibration": scene.calibration_image_points}[mode]
        for key, pts in (("roi", scene.roi), ("line_a", scene.line_a), ("line_b", scene.line_b),
                         ("calibration", scene.calibration_image_points)):
            r = width * (3 if key == mode else 2)
            colour = QColor(MODES[key][1])
            for x, y in pts:
                self._scene.addEllipse(x - r, y - r, 2 * r, 2 * r, pen("#ffffff", width / 2), QBrush(colour))

    # -------------------------------------------------------------------- mouse
    def _nearest(self, pos: QPointF) -> int | None:
        tolerance = 10 / max(1e-6, self.transform().m11())  # 10 screen pixels
        best, best_d = None, tolerance
        for i, (x, y) in enumerate(self._points):
            d = np.hypot(x - pos.x(), y - pos.y())
            if d <= best_d:
                best, best_d = i, d
        return best

    def mousePressEvent(self, event) -> None:  # noqa: N802
        if not self.has_image() or event.button() != Qt.MouseButton.LeftButton:
            return super().mousePressEvent(event)
        pos = self.mapToScene(event.position().toPoint())
        self._drag_index = self._nearest(pos)
        if self._drag_index is None:
            x = min(max(pos.x(), 0), self.image_size[0] - 1)
            y = min(max(pos.y(), 0), self.image_size[1] - 1)
            self.clicked.emit(round(x, 1), round(y, 1))

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        if self._drag_index is not None:
            pos = self.mapToScene(event.position().toPoint())
            self.dragged.emit(self._drag_index, round(pos.x(), 1), round(pos.y(), 1))
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._drag_index = None
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802
        factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        self.scale(factor, factor)

    def resizeEvent(self, event) -> None:  # noqa: N802
        super().resizeEvent(event)
        if self.has_image():
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


class SceneTab(QWidget):
    scene_saved = pyqtSignal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.scene = SceneConfig(name="My camera")
        self.mode = "roi"
        self._updating_table = False

        self.canvas = SceneCanvas()
        self.canvas.clicked.connect(self._add_point)
        self.canvas.dragged.connect(self._move_point)

        load_image = QPushButton("Open camera image or video…")
        load_image.clicked.connect(self.choose_image)
        fit = QPushButton("Fit to window")
        fit.clicked.connect(lambda: self.canvas.fitInView(self.canvas.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio))
        top = QHBoxLayout()
        top.addWidget(load_image)
        top.addWidget(fit)
        top.addStretch()

        # Drawing mode
        mode_box = QGroupBox("What are you drawing?")
        mode_layout = QVBoxLayout(mode_box)
        self.mode_buttons = QButtonGroup(self)
        self._mode_radio: dict[str, QRadioButton] = {}
        for key, (label, colour, _) in MODES.items():
            button = QRadioButton(label)
            self._mode_radio[key] = button
            button.setStyleSheet(f"QRadioButton {{ color: {colour}; font-weight: bold; }}")
            button.setChecked(key == self.mode)
            button.toggled.connect(lambda checked, k=key: checked and self._set_mode(k))
            self.mode_buttons.addButton(button)
            mode_layout.addWidget(button)
        self.mode_help = InfoBox(MODES[self.mode][2])
        mode_layout.addWidget(self.mode_help)
        row = QHBoxLayout()
        undo = QPushButton("Undo last point")
        undo.clicked.connect(self._undo)
        clear = QPushButton("Clear these points")
        clear.clicked.connect(self._clear_mode)
        row.addWidget(undo)
        row.addWidget(clear)
        mode_layout.addLayout(row)

        # Names and distances
        details = QGroupBox("Names and distances")
        form = QFormLayout(details)
        self.name_edit = QLineEdit(self.scene.name)
        self.name_edit.textChanged.connect(lambda t: self._set("name", t))
        self.dir_ab = QLineEdit(self.scene.direction_a_to_b)
        self.dir_ab.textChanged.connect(lambda t: self._set("direction_a_to_b", t.strip()))
        self.dir_ba = QLineEdit(self.scene.direction_b_to_a)
        self.dir_ba.textChanged.connect(lambda t: self._set("direction_b_to_a", t.strip()))
        self.ref_distance = QDoubleSpinBox()
        self.ref_distance.setRange(0, 1000)
        self.ref_distance.setDecimals(2)
        self.ref_distance.setSuffix(" m")
        self.ref_distance.valueChanged.connect(lambda v: self._set("reference_distance_m", v))
        form.addRow("Scene name", self.name_edit)
        form.addRow("Direction A → B", self.dir_ab)
        form.addRow("Direction B → A", self.dir_ba)
        form.addRow("Real distance A–B", self.ref_distance)

        # Calibration table
        calib = QGroupBox("Calibration: road position of each point (metres)")
        calib_layout = QVBoxLayout(calib)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Pixel x", "Pixel y", "Road x (m)", "Road y (m)"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(True)
        self.table.itemChanged.connect(self._table_changed)
        calib_layout.addWidget(self.table)
        rect_row = QHBoxLayout()
        self.rect_w = QDoubleSpinBox(suffix=" m", maximum=1000, value=4.0, decimals=2)
        self.rect_l = QDoubleSpinBox(suffix=" m", maximum=1000, value=10.5, decimals=2)
        fill_rect = QPushButton("Fill as rectangle")
        fill_rect.setToolTip("For 4 points clicked in the order top-left, top-right, bottom-right, bottom-left "
                             "of a rectangle on the road with this width and length.")
        fill_rect.clicked.connect(self._fill_rectangle)
        rect_row.addWidget(self.rect_w)
        rect_row.addWidget(self.rect_l)
        rect_row.addWidget(fill_rect)
        calib_layout.addLayout(rect_row)

        self.status = InfoBox()
        files = QHBoxLayout()
        for text, slot in (("Open scene…", self.choose_scene), ("Save scene…", self.save_dialog),
                           ("Use paper scene", self.load_paper_scene), ("Import CVAT polygon…", self.import_cvat)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            files.addWidget(b)

        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.addWidget(mode_box)
        side_layout.addWidget(details)
        side_layout.addWidget(calib, 1)
        side_layout.addWidget(self.status)
        side_layout.addLayout(files)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addLayout(top)
        left_layout.addWidget(self.canvas, 1)

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(side)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)

        layout = QVBoxLayout(self)
        layout.addWidget(heading("Step 2 · Camera scene",
                                 "Tell the program where the cycle track is, where to count, and how pixels relate "
                                 "to metres. You do this once per camera position and save it as a scene file."))
        layout.addWidget(splitter, 1)
        self._refresh()

    # ------------------------------------------------------------------ files
    def choose_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Camera image or video", "",
                                              "Images and videos (*.jpg *.jpeg *.png *.bmp *.mp4 *.avi *.mov *.mkv);;"
                                              "All files (*)")
        if path:
            self.load_image(path)

    def load_image(self, path: str | Path) -> bool:
        path = Path(path)
        if path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv", ".m4v"}:
            cap = cv2.VideoCapture(str(path))
            ok, image = cap.read()
            cap.release()
            image = image if ok else None
        else:
            image = cv2.imread(str(path))
        if image is None:
            show_error(self, "Cannot open image", f"Could not read an image from\n{path}")
            return False
        self.canvas.set_image(image)
        self._refresh()
        return True

    def choose_scene(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Open scene", "", "Scene files (*.json)")
        if path:
            self.load_scene(path)

    def load_scene(self, path: str | Path) -> bool:
        try:
            self.set_scene(SceneConfig.load(path))
        except (OSError, ValueError) as exc:
            show_error(self, "Cannot open scene", str(exc))
            return False
        return True

    def load_paper_scene(self) -> None:
        self.set_scene(utrecht_scene())

    def set_scene(self, scene: SceneConfig) -> None:
        self.scene = scene
        for widget, value in ((self.name_edit, scene.name), (self.dir_ab, scene.direction_a_to_b),
                              (self.dir_ba, scene.direction_b_to_a)):
            widget.blockSignals(True)
            widget.setText(value)
            widget.blockSignals(False)
        self.ref_distance.blockSignals(True)
        self.ref_distance.setValue(scene.reference_distance_m)
        self.ref_distance.blockSignals(False)
        self._refresh()

    def save_dialog(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save scene", "scene.json", "Scene files (*.json)")
        if path:
            self.save_scene(path)

    def save_scene(self, path: str | Path) -> None:
        self.scene.save(path)
        self.status.show_message(f"Scene saved to {path}", "ok")
        self.scene_saved.emit(str(path))

    def import_cvat(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "CVAT XML export", "", "CVAT XML (*.xml)")
        if not path:
            return
        polygons = read_cvat_polygons(path)
        if not polygons:
            show_error(self, "No polygon", "The file contains no polygons.")
            return
        label, points = polygons[0]
        if self.mode == "calibration":
            self.scene.calibration_image_points = points
            self.scene.calibration_world_points = [(0.0, 0.0)] * len(points)
        else:
            self.scene.roi = points
        self.status.show_message(f"Imported polygon '{label}' with {len(points)} points.", "ok")
        self._refresh()

    # ----------------------------------------------------------------- editing
    def _set(self, field: str, value) -> None:
        setattr(self.scene, field, value)
        self._refresh(table=False)

    def _set_mode(self, mode: str) -> None:
        self.mode = mode
        if not self._mode_radio[mode].isChecked():
            self._mode_radio[mode].setChecked(True)
        self.mode_help.show_message(MODES[mode][2])
        self._refresh(table=False)

    def _points(self) -> list:
        return {"roi": self.scene.roi, "line_a": self.scene.line_a, "line_b": self.scene.line_b,
                "calibration": self.scene.calibration_image_points}[self.mode]

    def _add_point(self, x: float, y: float) -> None:
        if self.mode in ("line_a", "line_b"):
            pts = self._points()
            if len(pts) >= 2:
                pts.clear()
            pts.append((x, y))
        elif self.mode == "calibration":
            self.scene.calibration_image_points.append((x, y))
            self.scene.calibration_world_points.append((0.0, 0.0))
        else:
            self.scene.roi.append((x, y))
        self._refresh()

    def _move_point(self, index: int, x: float, y: float) -> None:
        pts = self._points()
        if 0 <= index < len(pts):
            pts[index] = (x, y)
            self._refresh()

    def _undo(self) -> None:
        pts = self._points()
        if pts:
            pts.pop()
            if self.mode == "calibration":
                self.scene.calibration_world_points.pop()
        self._refresh()

    def _clear_mode(self) -> None:
        self._points().clear()
        if self.mode == "calibration":
            self.scene.calibration_world_points.clear()
        self._refresh()

    def _fill_rectangle(self) -> None:
        if len(self.scene.calibration_image_points) != 4:
            show_error(self, "Four points needed", "Click exactly 4 calibration points first "
                       "(top-left, top-right, bottom-right, bottom-left).")
            return
        w, length = self.rect_w.value(), self.rect_l.value()
        self.scene.calibration_world_points = [(0.0, 0.0), (w, 0.0), (w, length), (0.0, length)]
        self._refresh()

    def _table_changed(self, item: QTableWidgetItem) -> None:
        if self._updating_table or item.column() < 2:
            return
        row = item.row()
        try:
            value = float(item.text().replace(",", "."))
        except ValueError:
            self._refresh()
            return
        x, y = self.scene.calibration_world_points[row]
        self.scene.calibration_world_points[row] = (value, y) if item.column() == 2 else (x, value)
        self._refresh(table=False)

    # ----------------------------------------------------------------- display
    def _refresh(self, table: bool = True) -> None:
        if self.canvas.has_image():
            self.canvas.draw(self.scene, self.mode)
        if table:
            self._updating_table = True
            pts, world = self.scene.calibration_image_points, self.scene.calibration_world_points
            self.table.setRowCount(len(pts))
            for r, ((px, py), (wx, wy)) in enumerate(zip(pts, world, strict=True)):
                for c, value in enumerate((px, py, wx, wy)):
                    item = QTableWidgetItem(f"{value:g}")
                    if c < 2:
                        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    self.table.setItem(r, c, item)
            self._updating_table = False
        self._update_status()

    def _update_status(self) -> None:
        problems = self.scene.problems()
        if not self.canvas.has_image():
            problems = ["Open a camera image (or the first frame of a video) to draw on."] + problems
        if problems:
            self.status.show_message("<b>Not finished yet:</b><br>• " + "<br>• ".join(problems), "warning")
            return
        lines = ["<b>✓ The scene is complete.</b>"]
        if self.scene.has_calibration:
            try:
                h = compute_homography(self.scene.calibration_image_points, self.scene.calibration_world_points)
                error = calibration_error(self.scene.calibration_image_points, self.scene.calibration_world_points, h)
                mid = lambda line: ((line[0][0] + line[1][0]) / 2, (line[0][1] + line[1][1]) / 2)  # noqa: E731
                measured = world_distance(mid(self.scene.line_a), mid(self.scene.line_b), h)
                lines.append(f"Distance between the middles of line A and B according to the calibration: "
                             f"<b>{measured:.2f} m</b>.")
                if error > 0.05:
                    lines.append(f"Calibration points fit with an average error of {error:.2f} m.")
            except ValueError as exc:
                self.status.show_message(f"<b>Calibration problem:</b> {exc}", "error")
                return
        lines.append("Save the scene to use it in step 4.")
        self.status.show_message("<br>".join(lines), "ok")
