"""Thread-safe delivery of application events to Qt widgets."""

from __future__ import annotations

import threading

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
        self._closed = False
        self._incomingEvent.connect(self._deliver_event)
        self._unsubscribe = controller.subscribe(self._incomingEvent.emit)

    @property
    def snapshot(self) -> ApplicationSnapshot:
        return self._controller.snapshot

    @Slot(object)
    def _deliver_event(self, event: ApplicationEvent) -> None:
        if self._closed:
            return
        self.applicationEvent.emit(event)
        if event.kind is EventKind.AUDIO_LEVEL and event.audio_level is not None:
            self.audioLevelChanged.emit(event.audio_level.peak)
        else:
            self.snapshotChanged.emit(self._controller.snapshot)

    def close(self) -> None:
        self._closed = True
        unsubscribe, self._unsubscribe = self._unsubscribe, lambda: None
        unsubscribe()


class BackgroundJobs(QObject):
    """Own Python jobs until joined; only queued Qt delivery touches widgets."""

    finished = Signal(int, str, object, object)
    idleChanged = Signal()
    _incoming = Signal(int, str, object, object)

    def __init__(self):
        super().__init__()
        self._serial = 0
        self._pending = {}
        self._closed = False
        self._incoming.connect(self._deliver)

    @property
    def busy(self):
        return bool(self._pending)

    def submit(self, kind, operation):
        if self._closed:
            return None
        self._serial += 1
        request_id = self._serial
        outcome = [None, None]

        def work():
            try:
                outcome[0] = operation()
            except Exception as exc:
                outcome[1] = str(exc)

        worker = threading.Thread(target=work, name=f"here-ui-{kind}", daemon=True)

        def observe():
            worker.join()
            self._incoming.emit(request_id, kind, *outcome)

        observer = threading.Thread(target=observe, name="here-ui-completion", daemon=True)
        self._pending[request_id] = (worker, observer)
        worker.start()
        observer.start()
        return request_id

    @Slot(int, str, object, object)
    def _deliver(self, request_id, kind, value, error):
        if request_id not in self._pending:
            return
        del self._pending[request_id]
        if not self._closed:
            self.finished.emit(request_id, kind, value, error)
        self.idleChanged.emit()

    def close(self):
        self._closed = True
