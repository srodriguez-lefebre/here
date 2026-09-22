"""Thread-safe delivery of application events to Qt widgets."""

from __future__ import annotations

from PySide6.QtCore import QObject, Signal, Slot

from .contract import (
    ApplicationEvent,
    ApplicationSnapshot,
    EventKind,
    VisualController,
)


class ApplicationEventBridge(QObject):
    """Marshal core callbacks onto the Qt event loop."""

    snapshotChanged = Signal(object)
    audioLevelChanged = Signal(float)
    applicationEvent = Signal(object)
    _incomingEvent = Signal(object)

    def __init__(self, controller: VisualController) -> None:
        super().__init__()
        self._controller = controller
        self._incomingEvent.connect(self._deliver_event)
        self._unsubscribe = controller.subscribe(self._incomingEvent.emit)

    @property
    def snapshot(self) -> ApplicationSnapshot:
        return self._controller.snapshot

    @Slot(object)
    def _deliver_event(self, event: ApplicationEvent) -> None:
        self.applicationEvent.emit(event)
        if event.kind is EventKind.AUDIO_LEVEL and event.audio_level is not None:
            self.audioLevelChanged.emit(event.audio_level.peak)
        else:
            self.snapshotChanged.emit(self._controller.snapshot)

    def close(self) -> None:
        unsubscribe, self._unsubscribe = self._unsubscribe, lambda: None
        unsubscribe()
