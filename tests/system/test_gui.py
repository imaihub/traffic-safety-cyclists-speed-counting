"""System tests of the desktop application (offscreen, driven with pytest-qt)."""

import pytest
from PyQt6.QtCore import QPoint, Qt
from PyQt6.QtWidgets import QMessageBox

pytestmark = pytest.mark.system


@pytest.fixture
def window(qtbot, monkeypatch):
    from cyclist_tracker.gui import common
    from cyclist_tracker.gui.app import MainWindow, create_app

    monkeypatch.setattr(QMessageBox, "information", lambda *a, **k: None)
    errors = []
    monkeypatch.setattr(common, "show_error", lambda parent, title, msg: errors.append((title, msg)))
    for module in ("tab_analyse", "tab_dataset", "tab_scene", "tab_evaluate", "tab_train", "tab_start"):
        monkeypatch.setattr(f"cyclist_tracker.gui.{module}.show_error",
                            lambda parent, title, msg: errors.append((title, msg)))
    create_app([])
    w = MainWindow()
    w.errors = errors
    qtbot.addWidget(w)
    w.show()
    return w


def test_all_tabs_are_present(window):
    titles = [window.tabs.tabText(i) for i in range(window.tabs.count())]
    assert titles == ["Start", "1 · Dataset", "2 · Camera scene", "3 · Train", "4 · Analyse", "5 · Evaluate", "Help"]
    assert window.start.system_table.rowCount() >= 5


def test_start_page_buttons_open_the_steps(window, qtbot):
    window.start.go_to_step.emit(4)
    assert window.tabs.currentWidget() is window.analyse


def test_demo_button_creates_data_and_fills_all_tabs(window, qtbot, tmp_path):
    with qtbot.waitSignal(window.start.demo_created, timeout=60000):
        window.start.create_demo(tmp_path / "demo")
    assert window.dataset.report.ready_for_analysis
    assert window.analyse.detector.currentData() == "annotations"
    assert window.analyse.scene_file.path().endswith("scene.json")
    assert window.evaluate.speed.gt.path().endswith("speeds_test.csv")
    assert window.scene.canvas.has_image() and not window.scene.scene.problems()


def test_analysis_through_the_gui(window, qtbot, demo_dataset, tmp_path):
    window.load_demo(str(demo_dataset))
    tab = window.analyse
    tab.set_combo(tab.embedder, "histogram")
    tab.use_gpu.setChecked(False)
    tab.output_dir.set_path(tmp_path)
    tab.save_video.setChecked(False)
    with qtbot.waitSignal(tab.analysis_finished, timeout=120000) as blocker:
        qtbot.mouseClick(tab.start_button, Qt.MouseButton.LeftButton)
    result = blocker.args[0]
    assert sum(result.counts.values()) == 10
    assert tab.card_total.value.text() == "10"
    assert tab.crossings.rowCount() == 10
    assert "km/h" in tab.card_speed.value.text()
    assert tab.start_button.isEnabled() and not tab.stop_button.isEnabled()

    # results flow into the Evaluate tab
    for page in (window.evaluate.tracking, window.evaluate.speed, window.evaluate.counting):
        page.run()
        assert page.result is not None and page.table.rowCount() > 0
    assert window.errors == []


def test_analysis_explains_what_is_missing(window, qtbot):
    window.analyse.start()
    text = window.analyse.status.text()
    assert "Cannot start yet" in text and "video file or a folder" in text and "camera scene" in text


def test_dataset_check_reports_problems(window, tmp_path):
    window.dataset.set_folder(tmp_path)
    report = window.dataset.check()
    assert not report.ready_for_analysis
    assert "not ready for analysis" in window.dataset.summary.text()
    assert window.dataset.table.rowCount() == len(report.checks)


def test_drawing_a_scene_with_the_mouse(window, qtbot, demo_dataset, tmp_path):
    tab = window.scene
    window.tabs.setCurrentWidget(tab)
    assert tab.load_image(next((demo_dataset / "test").glob("*.jpg")))
    view = tab.canvas

    def click(x, y):
        pos = view.mapFromScene(float(x), float(y))
        qtbot.mouseClick(view.viewport(), Qt.MouseButton.LeftButton, pos=QPoint(pos.x(), pos.y()))

    for x, y in [(400, 100), (560, 100), (820, 530), (160, 530)]:
        click(x, y)
    assert len(tab.scene.roi) == 4
    tab._set_mode("line_a")
    click(250, 400)
    click(750, 400)
    tab._set_mode("line_b")
    click(380, 200)
    click(600, 200)
    tab._set_mode("calibration")
    for x, y in [(420, 150), (560, 150), (700, 450), (280, 450)]:
        click(x, y)
    tab.rect_w.setValue(4)
    tab.rect_l.setValue(20)
    tab._fill_rectangle()
    tab.dir_ab.setText("north")
    tab.dir_ba.setText("south")
    assert tab.scene.problems() == []
    assert "scene is complete" in tab.status.text()
    tab.save_scene(tmp_path / "scene.json")
    assert window.analyse.scene_file.path() == str(tmp_path / "scene.json")
    tab._undo()
    assert len(tab.scene.calibration_image_points) == 3


def test_paper_scene_button(window):
    window.scene.load_paper_scene()
    assert window.scene.scene.reference_distance_m == pytest.approx(13.5)
    assert window.scene.dir_ab.text() == "down"


def test_paper_settings_preset(window):
    from cyclist_tracker.config import AnalysisSettings

    window.analyse.apply_settings(AnalysisSettings.paper())
    assert window.analyse.settings() == AnalysisSettings.paper()


def test_sample_dataset_button_fills_all_tabs(window):
    from cyclist_tracker.gui.app import sample_path

    if sample_path() is None:
        pytest.skip("sample dataset not present")
    window.start.sample_requested.emit()
    assert window.dataset.report.ready_for_training
    assert window.analyse.tracking_point.currentData() == "center"  # from the sample's settings.json
    assert window.analyse.fps.value() == pytest.approx(4.995, abs=0.01)
    assert window.scene.scene.direction_a_to_b == "to roundabout"
    assert window.evaluate.counting.gt.path().endswith("counts_test.csv")
    assert window.errors == []
