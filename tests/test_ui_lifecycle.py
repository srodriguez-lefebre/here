import threading
from pathlib import Path

import pytest
from here.application import ApplicationSnapshot, ApplicationState
from here.ui.app import HereDesktop
from here.ui.bridge import BackgroundJobs
from here.ui.contract import ApplicationUiAdapter
from here.ui.fake_controller import FakeApplicationController
from PySide6.QtCore import QObject, QSettings, Qt, Slot
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication, QLabel, QMessageBox, QTabWidget
from pytestqt.qtbot import QtBot

pytestmark = pytest.mark.usefixtures("owned_desktops")


def _desktop(tmp_path: Path, qtbot: QtBot) -> tuple[FakeApplicationController, HereDesktop]:
    core = FakeApplicationController()
    adapter = ApplicationUiAdapter(core, tmp_path)
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    application = QApplication.instance()
    assert application is not None
    desktop = HereDesktop(application, adapter, settings=settings)
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
    assert "Micrófono + audio del sistema" in window.findChild(object, "sourceLabel").text()
    assert str(tmp_path) in window.findChild(object, "destinationLabel").text()

    window.set_snapshot(ApplicationSnapshot(state=ApplicationState.FAILED, recoverable=True))
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


def test_library_tab_is_next_to_capture_without_removing_capture_actions(
    tmp_path: Path,
    qtbot: QtBot,
) -> None:
    _, desktop = _desktop(tmp_path, qtbot)
    desktop.show()
    tabs = desktop.main_window.findChildren(QTabWidget)
    assert len(tabs) == 1, "M3 Biblioteca and capture must be reachable as tabs"
    library = [index for index in range(tabs[0].count()) if tabs[0].tabText(index) == "Biblioteca"]
    assert len(library) == 1
    capture = desktop.main_window.findChild(object, "startButton")
    assert capture is not None
    capture_tab = [
        index for index in range(tabs[0].count()) if tabs[0].widget(index).isAncestorOf(capture)
    ]
    assert len(capture_tab) == 1
    assert abs(capture_tab[0] - library[0]) == 1
    tabs[0].setCurrentIndex(capture_tab[0])
    assert capture.isVisibleTo(desktop.main_window)


def test_background_jobs_preserves_structured_catalog_errors_on_qt_thread(qtbot: QtBot) -> None:
    from here.memory.database import LibraryError

    # Worker behavior double only: no alternate InvalidInput class/DTO is declared.
    failure = LibraryError("Opaque localized diagnostic")
    setattr(failure, "reason", "corpus_changed")
    jobs = BackgroundJobs()
    worker_threads, deliveries = [], []

    class Receiver(QObject):
        @Slot(int, str, object, object)
        def receive(self, request_id, kind, value, error):
            deliveries.append((request_id, kind, value, error, threading.get_ident()))

    receiver = Receiver()
    jobs.finished.connect(receiver.receive)

    def fail():
        worker_threads.append(threading.get_ident())
        raise failure

    try:
        request = jobs.submit("library-search", fail)
        qtbot.waitUntil(lambda: not jobs.busy)
        assert len(deliveries) == 1
        request_id, kind, value, error, delivery_thread = deliveries[0]
        assert (request_id, kind, value) == (request, "library-search", None)
        assert error is failure, "Stringifying errors loses machine-readable corpus_changed"
        assert getattr(error, "reason") == "corpus_changed"
        assert worker_threads[0] != threading.get_ident()
        assert delivery_thread == threading.get_ident()
    finally:
        jobs.close()
        qtbot.waitUntil(lambda: not jobs.busy)


def test_closed_background_jobs_drops_delayed_library_delivery(qtbot: QtBot) -> None:
    jobs = BackgroundJobs()
    entered, release = threading.Event(), threading.Event()
    delivered = []
    jobs.finished.connect(lambda *args: delivered.append(args))

    def delayed_page():
        entered.set()
        assert release.wait(5)
        return "synthetic page"

    try:
        jobs.submit("library-search", delayed_page)
        qtbot.waitUntil(entered.is_set)
        jobs.close()
        assert jobs.busy, "Closing must retain the real worker until actual exit"
    finally:
        release.set()
        qtbot.waitUntil(lambda: not jobs.busy)
        jobs.close()
    assert delivered == []


def test_real_worker_recovery_failure_still_displays_an_ordinary_error(
    tmp_path: Path,
    qtbot: QtBot,
) -> None:
    class FailingRecovery:
        def discover(self):
            raise RuntimeError("Synthetic recovery unavailable")

    application = QApplication.instance()
    assert application is not None
    adapter = ApplicationUiAdapter(
        FakeApplicationController(),
        tmp_path,
        recovery_service=FailingRecovery(),
    )
    desktop = HereDesktop(
        application,
        adapter,
        settings=QSettings(str(tmp_path / "recovery.ini"), QSettings.Format.IniFormat),
    )
    desktop.show()
    qtbot.waitUntil(lambda: not desktop.jobs.busy)
    message = desktop.main_window.findChild(QLabel, "recoveryLabel")
    assert message is not None
    assert message.text() == "Synthetic recovery unavailable"
