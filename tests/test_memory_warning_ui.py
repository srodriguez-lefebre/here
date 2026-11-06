"""Render recoverable catalog warnings without turning successful capture into failure."""

from __future__ import annotations

import pytest
from here.application import ApplicationError, ApplicationSnapshot, ApplicationState
from here.ui.app import HereDesktop
from here.ui.contract import ApplicationUiAdapter
from here.ui.fake_controller import FakeApplicationController
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication, QLabel, QPushButton

pytestmark = pytest.mark.usefixtures("owned_desktops")

WARNING = "Audio guardado; biblioteca pendiente de recuperar."


def _desktop(tmp_path, qtbot):
    core = FakeApplicationController()
    adapter = ApplicationUiAdapter(core, tmp_path)
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    application = QApplication.instance()
    assert application is not None
    desktop = HereDesktop(application, adapter, settings=settings)
    desktop.show()
    qtbot.waitUntil(lambda: not desktop.jobs.busy)
    return desktop


def test_completed_capture_displays_library_warning_and_allows_new_recording(tmp_path, qtbot):
    desktop = _desktop(tmp_path, qtbot)
    window = desktop.main_window
    window.set_snapshot(
        ApplicationSnapshot(
            state=ApplicationState.COMPLETED,
            memory_warning=WARNING,
            last_error=None,
            worker_complete=True,
        )
    )
    detail = window.findChild(QLabel, "detailLabel")
    assert detail.isVisibleTo(window)
    assert detail.text() == WARNING
    assert "completada" in window.findChild(QLabel, "stateLabel").text().lower()
    start = window.findChild(QPushButton, "startButton")
    assert start.isVisibleTo(window) and start.isEnabled()
    assert window.snapshot.state is ApplicationState.COMPLETED


def test_application_error_takes_precedence_over_recoverable_library_warning(tmp_path, qtbot):
    desktop = _desktop(tmp_path, qtbot)
    window = desktop.main_window
    failure = ApplicationError(
        stage="processing",
        error_type="RuntimeError",
        message="Falló la transcripción sintética.",
        retryable=True,
    )
    window.set_snapshot(
        ApplicationSnapshot(
            state=ApplicationState.FAILED,
            last_error=failure,
            memory_warning=WARNING,
            recoverable=True,
        )
    )
    detail = window.findChild(QLabel, "detailLabel")
    assert detail.isVisibleTo(window)
    assert detail.text() == failure.message
    assert "error" in window.findChild(QLabel, "stateLabel").text().lower()
    window.set_snapshot(
        ApplicationSnapshot(
            state=ApplicationState.FAILED,
            memory_warning=WARNING,
            recoverable=True,
        )
    )
    assert detail.text() == WARNING


def test_success_and_new_start_snapshots_remove_previous_library_warning(tmp_path, qtbot):
    desktop = _desktop(tmp_path, qtbot)
    window = desktop.main_window
    detail = window.findChild(QLabel, "detailLabel")
    for next_state in (ApplicationState.COMPLETED, ApplicationState.PREPARING):
        window.set_snapshot(
            ApplicationSnapshot(
                state=ApplicationState.COMPLETED,
                memory_warning=WARNING,
            )
        )
        assert detail.isVisibleTo(window) and detail.text() == WARNING
        window.set_snapshot(
            ApplicationSnapshot(
                state=next_state,
                worker_complete=next_state is ApplicationState.COMPLETED,
            )
        )
        assert detail.text() == ""
        assert not detail.isVisibleTo(window)
        assert window.snapshot.state is next_state
