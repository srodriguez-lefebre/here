from pathlib import Path

import pytest
from here.application import ApplicationSnapshot, ApplicationState
from here.ui.app import HereDesktop
from here.ui.contract import ApplicationUiAdapter
from here.ui.fake_controller import FakeApplicationController
from PySide6.QtCore import QSettings, Qt
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication, QMessageBox
from pytestqt.qtbot import QtBot


def _desktop(tmp_path: Path, qtbot: QtBot) -> tuple[FakeApplicationController, HereDesktop]:
    core = FakeApplicationController()
    adapter = ApplicationUiAdapter(core, tmp_path)
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    application = QApplication.instance()
    assert application is not None
    desktop = HereDesktop(application, adapter, settings=settings)
    qtbot.addWidget(desktop.main_window)
    qtbot.addWidget(desktop.overlay)
    return core, desktop


def test_active_work_close_hides_main_window_and_overlay_click_restores_it(
    tmp_path: Path,
    qtbot: QtBot,
) -> None:
    core, desktop = _desktop(tmp_path, qtbot)
    desktop.show()
    core.set_state(ApplicationState.RECORDING)
    qtbot.waitUntil(desktop.overlay.isVisible)

    desktop.main_window.close()

    assert not desktop.main_window.isVisible()
    assert desktop.overlay.isVisible()
    desktop.overlay.restoreRequested.emit()
    assert desktop.main_window.isVisible()


def test_idle_close_requests_process_exit(tmp_path: Path, qtbot: QtBot) -> None:
    _, desktop = _desktop(tmp_path, qtbot)
    desktop.show()
    close_requested = QSignalSpy(desktop.main_window.idleCloseRequested)

    desktop.main_window.close()

    assert close_requested.count() == 1


def test_close_uses_authoritative_controller_state_when_ui_snapshot_is_stale(
    tmp_path: Path,
    qtbot: QtBot,
) -> None:
    core, desktop = _desktop(tmp_path, qtbot)
    desktop.show()
    core.set_state(ApplicationState.PREPARING)
    desktop.main_window.set_snapshot(ApplicationSnapshot(state=ApplicationState.IDLE))
    close_requested = QSignalSpy(desktop.main_window.idleCloseRequested)

    desktop.main_window.close()

    assert not desktop.main_window.isVisible()
    assert close_requested.count() == 0


def test_main_window_exposes_only_compatible_actions(tmp_path: Path, qtbot: QtBot) -> None:
    _, desktop = _desktop(tmp_path, qtbot)
    window = desktop.main_window

    window.set_snapshot(ApplicationSnapshot(state=ApplicationState.IDLE))
    assert window.findChild(object, "startButton").isVisibleTo(window)
    assert not window.findChild(object, "pauseButton").isVisibleTo(window)

    window.set_snapshot(ApplicationSnapshot(state=ApplicationState.PAUSED))
    pause_button = window.findChild(object, "pauseButton")
    assert pause_button.text() == "Reanudar"
    assert "Micrófono + audio del sistema" in window.findChild(
        object, "sourceLabel"
    ).text()
    assert str(tmp_path) in window.findChild(object, "destinationLabel").text()

    window.set_snapshot(
        ApplicationSnapshot(state=ApplicationState.FAILED, recoverable=True)
    )
    assert window.findChild(object, "retryButton").isVisibleTo(window)
    assert window.findChild(object, "startButton").text() == "Nueva grabación"


def test_main_window_allows_recoverable_cancellation_while_stopping(
    tmp_path: Path,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core, desktop = _desktop(tmp_path, qtbot)
    core.set_state(ApplicationState.STOPPING)
    window = desktop.main_window
    window.set_snapshot(core.snapshot)
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Yes,
    )
    cancel_button = window.findChild(object, "cancelButton")

    assert cancel_button.isVisibleTo(window)
    qtbot.mouseClick(cancel_button, Qt.MouseButton.LeftButton)

    assert core.command_log == ["cancel"]
    assert core.snapshot.recoverable is True


def test_accent_preference_is_persisted_but_overlay_position_is_not(
    tmp_path: Path,
    qtbot: QtBot,
) -> None:
    _, desktop = _desktop(tmp_path, qtbot)
    settings = desktop.preferences._settings
    desktop.preferences.set_accent("#123456")
    desktop.overlay.move(5, 5)

    reloaded = type(desktop.preferences)(settings)

    assert reloaded.accent.name() == "#123456"
    assert settings.value("visual/position") is None
