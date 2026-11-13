"""Composition root for the Qt application."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from here.application import ApplicationController
from here.memory.database import LibraryError
from here.memory.service import MemoryService
from here.ui.bridge import ApplicationEventBridge, BackgroundJobs, _DeleteAcknowledgement
from here.ui.contract import ApplicationState, ApplicationUiAdapter, EventKind, VisualController
from here.ui.main_window import MainWindow
from here.ui.overlay import LiveLogoOverlay
from here.ui.preferences import VisualPreferences
from PySide6.QtCore import QSettings, Slot
from PySide6.QtWidgets import QApplication


class HereDesktop:
    """Own the main window, overlay, and their lifecycle in one process."""

    def __init__(
        self,
        application: QApplication,
        controller: VisualController,
        *,
        settings: QSettings | None = None,
    ) -> None:
        self.application = application
        self.application.setQuitOnLastWindowClosed(False)
        self.controller = controller
        self.jobs = BackgroundJobs()
        self._exit_intent = False
        self._quit_sent = False
        self._requests = {}
        self._epoch = 0
        self._was_active = controller.snapshot.has_active_work
        self._refresh_pending = False
        self.preferences = VisualPreferences(settings)
        self.bridge = ApplicationEventBridge(controller)
        self.main_window = MainWindow(controller, self.preferences, self.jobs)
        self.overlay = LiveLogoOverlay(controller, self.preferences)

        self.bridge.snapshotChanged.connect(self.main_window.set_snapshot)
        self.bridge.snapshotChanged.connect(self.overlay.set_snapshot)
        self.bridge.audioLevelChanged.connect(self.main_window.set_audio_level)
        self.bridge.audioLevelChanged.connect(self.overlay.set_audio_level)
        self.overlay.restoreRequested.connect(self.restore_main_window)
        self.overlay.terminalDisplayFinished.connect(self._terminal_display_finished)
        self.main_window.idleCloseRequested.connect(self.request_exit)
        self.main_window.exitRequested.connect(self.request_exit)
        self.main_window.diagnosticsRequested.connect(self._diagnose)
        self.main_window.recoveryRequested.connect(self._recover)
        self.jobs.finished.connect(self._job_finished)
        self.jobs.idleChanged.connect(self._jobs_changed)
        self.jobs.busyChanged.connect(self._jobs_changed)
        self.bridge.snapshotChanged.connect(self._snapshot_changed)
        self.bridge.applicationEvent.connect(self._application_event)
        self.application.aboutToQuit.connect(self.bridge.close)
        self.application.aboutToQuit.connect(self.jobs.close)
        self.application.aboutToQuit.connect(self.main_window.library.shutdown)
        self._connect_delete_acknowledgement(controller)
        self._discover()

    def _connect_delete_acknowledgement(self, controller: VisualController) -> None:
        if not isinstance(controller, ApplicationUiAdapter):
            return
        try:
            memory = controller._memory_backend()
        except LibraryError:
            return
        if isinstance(memory, MemoryService):
            view = self.main_window.library
            acknowledgement = _DeleteAcknowledgement(
                self.jobs, memory.before_delete, view._pause_for_delete, view
            )
            view._closing.connect(acknowledgement.close)

    def show(self) -> None:
        self.main_window.show()

    @Slot()
    def restore_main_window(self) -> None:
        self.main_window.showNormal()
        self.main_window.raise_()
        self.main_window.activateWindow()

    @Slot()
    def _terminal_display_finished(self) -> None:
        if not self.main_window.isVisible():
            self.request_exit()

    def _submit(self, kind, operation):
        if self._exit_intent or self.jobs.busy:
            return
        request = self.jobs.submit(kind, operation)
        self._requests[kind] = (request, self._epoch)
        self._jobs_changed()

    def _discover(self):
        if self.jobs.busy:
            self._refresh_pending = True
            return
        self._refresh_pending = False
        self._submit("recovery", self.controller.discover_recovery)

    def _diagnose(self, source):
        if self.controller.snapshot.has_active_work:
            return
        self._submit("diagnostics", lambda: self.controller.diagnose(source))

    def _recover(self, candidate):
        if candidate is not None:
            self._submit("retry", lambda: self.controller.retry_candidate(candidate))

    def _job_finished(self, request, kind, value, error):
        expected = self._requests.get(kind)
        if self._exit_intent or expected != (request, self._epoch):
            return
        if kind == "diagnostics":
            if not self.controller.snapshot.has_active_work:
                self.main_window.set_diagnostics(value, error)
        elif kind == "recovery":
            self.main_window.set_recovery(value, error)
        elif error:
            self.main_window.set_recovery_error(error)

    def _jobs_changed(self, _busy=None):
        if self._refresh_pending and not self.jobs.busy and not self._exit_intent:
            self._discover()
        self.main_window.set_background_busy(self.jobs.busy, exiting=self._exit_intent)
        self._check_exit()

    def _snapshot_changed(self, snapshot):
        active = snapshot.has_active_work
        if active and not self._was_active:
            self._epoch += 1
        self._was_active = active
        self._check_exit()

    def _application_event(self, event):
        if self._exit_intent:
            return
        if event.kind is EventKind.MEMORY_INGESTED:
            self.main_window.library.invalidate_catalog()
        elif event.kind is EventKind.WORKER_COMPLETED:
            self._discover()
        elif event.kind is EventKind.STATE_CHANGED and event.state is ApplicationState.PREPARING:
            # Immutable events retain fast lifecycle transitions even if queued snapshot
            # delivery already sees a terminal state.
            self._epoch += 1

    @Slot()
    def request_exit(self):
        if self._exit_intent:
            return
        self._exit_intent = True
        self._epoch += 1
        self.main_window.library.shutdown()
        self.jobs.close()
        snapshot = self.controller.snapshot
        if snapshot.state in {
            ApplicationState.PREPARING,
            ApplicationState.RECORDING,
            ApplicationState.PAUSED,
        }:
            self.controller.stop_and_process()
        self._jobs_changed()

    def _check_exit(self):
        if not self._exit_intent or self._quit_sent:
            return
        snapshot = self.controller.snapshot
        if not snapshot.has_active_work and snapshot.worker_complete and not self.jobs.busy:
            self._quit_sent = True
            self.bridge.close()
            self.application.quit()


def create_desktop(
    application: QApplication,
    controller: ApplicationController,
    *,
    output_dir: Path,
    settings: QSettings | None = None,
) -> HereDesktop:
    """Compose a real application controller with the visual adapter."""

    return HereDesktop(
        application,
        ApplicationUiAdapter(controller, output_dir),
        settings=settings,
    )


def application_instance(arguments: list[str]) -> QApplication:
    """Return the process QApplication with a precise type for composition roots."""

    existing = QApplication.instance()
    if existing is not None:
        return cast(QApplication, existing)
    return QApplication(arguments)
