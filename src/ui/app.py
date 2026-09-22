"""Composition root for the Qt application."""

from __future__ import annotations

from pathlib import Path
from typing import cast

from PySide6.QtCore import QSettings, Slot
from PySide6.QtWidgets import QApplication

from ..application import ApplicationController
from .bridge import ApplicationEventBridge
from .contract import ApplicationUiAdapter, VisualController
from .main_window import MainWindow
from .overlay import LiveLogoOverlay
from .preferences import VisualPreferences


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
        self.preferences = VisualPreferences(settings)
        self.bridge = ApplicationEventBridge(controller)
        self.main_window = MainWindow(controller, self.preferences)
        self.overlay = LiveLogoOverlay(controller, self.preferences)

        self.bridge.snapshotChanged.connect(self.main_window.set_snapshot)
        self.bridge.snapshotChanged.connect(self.overlay.set_snapshot)
        self.bridge.audioLevelChanged.connect(self.main_window.set_audio_level)
        self.bridge.audioLevelChanged.connect(self.overlay.set_audio_level)
        self.overlay.restoreRequested.connect(self.restore_main_window)
        self.overlay.terminalDisplayFinished.connect(self._terminal_display_finished)
        self.main_window.idleCloseRequested.connect(self.application.quit)
        self.application.aboutToQuit.connect(self.bridge.close)

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
