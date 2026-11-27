"""Thread-safe delivery of application events to Qt widgets."""

from __future__ import annotations

import threading
from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass

from here.ui.contract import (
    ApplicationEvent,
    ApplicationSnapshot,
    EventKind,
    VisualController,
)
from PySide6.QtCore import QObject, Qt, QTimer, Signal, Slot


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


@dataclass
class _JobOutcome:
    value: object = None
    error: Exception | None = None


class BackgroundJobs(QObject):
    """Retain jobs until actual thread exit; all Qt delivery stays on the Qt thread."""

    finished = Signal(int, str, object, object)
    settled = Signal(int)
    busyChanged = Signal(bool)
    idleChanged = Signal()

    def __init__(self):
        super().__init__()
        self._serial = 0
        self._pending = {}
        self._closed = False
        self._reported_busy = False
        self._poll = QTimer(self)
        self._poll.setInterval(10)
        self._poll.timeout.connect(self._collect_finished)

    @property
    def busy(self):
        return bool(self._pending)

    def submit(self, kind, operation):
        if self._closed:
            return None
        self._serial += 1
        request_id = self._serial
        outcome = _JobOutcome()

        def work():
            try:
                outcome.value = operation()
            except Exception as exc:
                outcome.error = exc

        worker = threading.Thread(target=work, name=f"here-ui-{kind}", daemon=True)

        self._pending[request_id] = (worker, kind, outcome)
        worker.start()
        self._poll.start()
        self._notify_busy()
        return request_id

    def _notify_busy(self):
        busy = self.busy
        if busy != self._reported_busy:
            self._reported_busy = busy
            self.busyChanged.emit(busy)

    @Slot()
    def _collect_finished(self):
        for request_id, (worker, kind, outcome) in tuple(self._pending.items()):
            # An emitted signal can arrive before its sending thread returns. Poll
            # actual termination instead: no worker touches a QObject or calls Qt.
            if worker.is_alive():
                continue
            del self._pending[request_id]
            if not self._closed:
                self.finished.emit(request_id, kind, outcome.value, outcome.error)
            # Internal scheduling ownership ends only after actual worker exit,
            # including when shutdown suppresses result/error delivery.
            self.settled.emit(request_id)
            self.idleChanged.emit()
            # Completion callbacks may submit replacement work. Publish the
            # resulting shared state after those callbacks, not a transient idle.
            self._notify_busy()
        if not self._pending:
            self._poll.stop()

    def close(self):
        self._closed = True

    def _request_for_worker(self, worker: threading.Thread) -> int | None:
        for request, (pending_worker, _, _) in self._pending.items():
            if pending_worker is worker:
                return request
        return None


class _DeleteAcknowledgement(QObject):
    """Acknowledge Qt invalidation before a tracked worker can accept deletion."""

    _requested = Signal(object, object)
    _TIMEOUT_SECONDS = 1.0

    def __init__(
        self,
        jobs: BackgroundJobs,
        callbacks: list[Callable[[str], None]],
        invalidate: Callable[[int], None],
        parent: QObject,
    ) -> None:
        super().__init__(parent)
        self._jobs = jobs
        self._callbacks = callbacks
        self._invalidate = invalidate
        self._qt_thread = threading.get_ident()
        self._lock = threading.Lock()
        self._closed = False
        self._tickets: set[Future[None]] = set()
        self._callback = self._acknowledge
        self._requested.connect(self._deliver, Qt.ConnectionType.QueuedConnection)
        # Keep teardown independent of Qt delivering to a child receiver.
        parent.destroyed.connect(lambda *_: self.close())
        callbacks.append(self._callback)

    def _acknowledge(self, _meeting_id: str) -> None:
        if threading.get_ident() == self._qt_thread:
            raise RuntimeError("Deletion acknowledgement requires a desktop worker")
        ticket: Future[None] = Future()
        with self._lock:
            if self._closed:
                raise RuntimeError("Deletion acknowledgement is closed")
            self._tickets.add(ticket)
        try:
            self._requested.emit(threading.current_thread(), ticket)
            ticket.result(timeout=self._TIMEOUT_SECONDS)
        finally:
            ticket.cancel()
            with self._lock:
                self._tickets.discard(ticket)

    @Slot(object, object)
    def _deliver(self, worker: threading.Thread, ticket: Future[None]) -> None:
        if not ticket.set_running_or_notify_cancel():
            return
        try:
            if self._closed:
                raise RuntimeError("Deletion acknowledgement is closed")
            request = self._jobs._request_for_worker(worker)
            if request is None:
                raise RuntimeError("Deletion acknowledgement requires a tracked desktop worker")
            self._invalidate(request)
        except Exception as exc:
            ticket.set_exception(exc)
        else:
            ticket.set_result(None)

    @Slot()
    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            if self._callback in self._callbacks:
                self._callbacks.remove(self._callback)
            tickets = tuple(self._tickets)
            self._tickets.clear()
        for ticket in tickets:
            ticket.cancel()
