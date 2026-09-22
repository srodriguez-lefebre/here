from __future__ import annotations

import math
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import numpy as np
from here.application.contracts import EventListener, Unsubscribe
from here.application.models import (
    ApplicationError,
    ApplicationEvent,
    ApplicationSnapshot,
    ApplicationState,
    AudioLevel,
    EventKind,
    StartRequest,
)
from here.application.processing import (
    ProcessingCancelled,
    SessionProcessingFailed,
    SessionProcessor,
    error_metadata,
)
from here.live_processing import LiveTranscriptionController
from here.output.metadata import SessionEventMetadata
from here.recording.control import ControllableRecording
from here.recording.service import start_recording

BlockSink = Callable[[str, np.ndarray, int, int], None]
CaptureFactory = Callable[[StartRequest, BlockSink], ControllableRecording]
LiveFactory = Callable[[int, bool], LiveTranscriptionController]


class InvalidApplicationCommand(RuntimeError):
    pass


def _default_capture_factory(request: StartRequest, block_sink: BlockSink) -> ControllableRecording:
    return start_recording(
        request.source_mode.value,
        block_sink=block_sink,
        microphone_device_id=request.microphone_device_id,
        system_device_id=request.system_device_id,
    )


def _default_live_factory(
    source_count: int,
    use_alt_transcription_model: bool,
) -> LiveTranscriptionController:
    return LiveTranscriptionController(
        expected_source_count=source_count,
        use_alt_transcription_model=use_alt_transcription_model,
    )


class HereApplicationController:
    """Thread-safe application use case used directly by CLI and Qt adapters."""

    _STARTABLE = {
        ApplicationState.IDLE,
        ApplicationState.COMPLETED,
        ApplicationState.FAILED,
        ApplicationState.CANCELLED,
    }

    def __init__(
        self,
        *,
        capture_factory: CaptureFactory = _default_capture_factory,
        live_factory: LiveFactory = _default_live_factory,
        processor: SessionProcessor | None = None,
        clock: Callable[[], datetime] = lambda: datetime.now().astimezone(),
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._capture_factory = capture_factory
        self._live_factory = live_factory
        self._processor = processor or SessionProcessor()
        self._clock = clock
        self._monotonic = monotonic
        self._lock = threading.RLock()
        self._listeners: list[EventListener] = []
        self._snapshot = ApplicationSnapshot()
        self._capture: ControllableRecording | None = None
        self._live: LiveTranscriptionController | None = None
        self._request: StartRequest | None = None
        self._worker: threading.Thread | None = None
        self._cancel_recording = False
        self._processing_cancel = threading.Event()
        self._pause_started_at: datetime | None = None
        self._events: list[SessionEventMetadata] = []
        self._captured_seconds: dict[str, float] = {}
        self._levels: dict[str, tuple[float, float]] = {}
        self._last_level_emitted = 0.0

    @property
    def snapshot(self) -> ApplicationSnapshot:
        with self._lock:
            return self._snapshot

    def subscribe(self, listener: EventListener) -> Unsubscribe:
        with self._lock:
            self._listeners.append(listener)

        def unsubscribe() -> None:
            with self._lock:
                if listener in self._listeners:
                    self._listeners.remove(listener)

        return unsubscribe

    def _emit(self, event: ApplicationEvent) -> None:
        with self._lock:
            listeners = tuple(self._listeners)
        for listener in listeners:
            try:
                listener(event)
            except Exception:
                continue

    def _recorded_duration(self) -> float:
        return max(self._captured_seconds.values(), default=0.0)

    def _append_session_event(
        self,
        kind: str,
        *,
        details: dict[str, str | int | float | bool | None] | None = None,
    ) -> None:
        self._events.append(
            SessionEventMetadata(
                kind=kind,
                occurred_at=self._clock(),
                recorded_duration_seconds=self._recorded_duration(),
                total_paused_seconds=self._snapshot.total_paused_seconds,
                details=details or {},
            )
        )

    def _transition(
        self,
        state: ApplicationState,
        *,
        details: dict[str, str | int | float | bool | None] | None = None,
    ) -> None:
        with self._lock:
            previous = self._snapshot.state
            self._snapshot = replace(self._snapshot, state=state)
            event = ApplicationEvent(
                kind=EventKind.STATE_CHANGED,
                state=state,
                previous_state=previous,
                details=details or {},
            )
        self._emit(event)

    def start(self, request: StartRequest) -> None:
        with self._lock:
            if self._snapshot.state not in self._STARTABLE:
                raise InvalidApplicationCommand(
                    f"Cannot start while application is {self._snapshot.state.value}"
                )
            previous = self._snapshot.state
            started_at = self._clock()
            self._request = request
            self._capture = None
            self._live = None
            self._cancel_recording = False
            self._processing_cancel.clear()
            self._pause_started_at = None
            self._events = []
            self._captured_seconds = {}
            self._levels = {}
            self._last_level_emitted = 0.0
            self._snapshot = ApplicationSnapshot(
                state=ApplicationState.PREPARING,
                started_at=started_at,
            )
            event = ApplicationEvent(
                kind=EventKind.STATE_CHANGED,
                state=ApplicationState.PREPARING,
                previous_state=previous,
            )
            self._append_session_event("preparing")
            self._worker = threading.Thread(
                target=self._run_recording,
                daemon=True,
                name="here-application-job",
            )
        self._emit(event)
        self._worker.start()

    def _block_sink(self, label: str, data: np.ndarray, sample_rate: int, channels: int) -> None:
        del channels
        with self._lock:
            live = self._live
            state = self._snapshot.state
            if state not in {
                ApplicationState.PREPARING,
                ApplicationState.RECORDING,
            }:
                return
        if live is not None:
            live.submit_block(label, data, sample_rate, data.shape[1] if data.ndim > 1 else 1)

        normalized = data.astype(np.float32, copy=False)
        if np.issubdtype(data.dtype, np.integer):
            scale = float(max(abs(np.iinfo(data.dtype).min), np.iinfo(data.dtype).max))
            normalized = normalized / scale
        peak = min(1.0, float(np.max(np.abs(normalized))) if normalized.size else 0.0)
        rms = min(
            1.0,
            math.sqrt(float(np.mean(np.square(normalized, dtype=np.float64))))
            if normalized.size
            else 0.0,
        )
        now = self._monotonic()
        with self._lock:
            self._captured_seconds[label] = self._captured_seconds.get(label, 0.0) + (
                data.shape[0] / sample_rate
            )
            self._levels[label] = (peak, rms)
            if state is not ApplicationState.RECORDING:
                return
            if now - self._last_level_emitted < (1 / 15):
                return
            self._last_level_emitted = now
            combined = AudioLevel(
                source="combined",
                peak=max((level[0] for level in self._levels.values()), default=0.0),
                rms=max((level[1] for level in self._levels.values()), default=0.0),
            )
            state = self._snapshot.state
        self._emit(
            ApplicationEvent(
                kind=EventKind.AUDIO_LEVEL,
                state=state,
                audio_level=combined,
            )
        )

    def _run_recording(self) -> None:
        request = self._request
        assert request is not None
        source_count = 2 if request.source_mode.value == "both" else 1
        try:
            live = self._live_factory(source_count, request.use_alt_transcription_model)
            with self._lock:
                self._live = live
            capture = self._capture_factory(request, self._block_sink)
            with self._lock:
                self._capture = capture
                should_cancel = self._cancel_recording
            if should_cancel:
                capture.cancel()
            else:
                self._transition(ApplicationState.RECORDING, details={"source_count": source_count})
                with self._lock:
                    self._append_session_event("recording", details={"source_count": source_count})

            session = capture.wait()
            with self._lock:
                destructive_cancel = self._cancel_recording
            if destructive_cancel:
                session.cleanup()
                live.abort()
                live.cleanup()
                self._transition(ApplicationState.CANCELLED, details={"recoverable": False})
                return

            self._transition(ApplicationState.PROCESSING)
            with self._lock:
                self._append_session_event("processing")
            artifacts = self._processor.process(
                session,
                request.output_dir,
                use_alt_transcription_model=request.use_alt_transcription_model,
                live_controller=live,
                cancel_event=self._processing_cancel,
                events=self._events,
                started_at=self._snapshot.started_at,
                total_paused_seconds=self._snapshot.total_paused_seconds,
            )
            self._persisted(artifacts.session_dir, recoverable=True)
            self._transition(ApplicationState.COMPLETED)
        except ProcessingCancelled as exc:
            self._persisted(exc.session_dir, recoverable=True)
            self._transition(ApplicationState.CANCELLED, details={"recoverable": True})
        except SessionProcessingFailed as exc:
            with self._lock:
                live = self._live
            if live is not None:
                live.abort()
                live.cleanup()
            self._persisted(exc.session_dir, recoverable=exc.recoverable)
            self._fail("processing", exc)
        except Exception as exc:
            with self._lock:
                destructive_cancel = self._cancel_recording
                live = self._live
            if destructive_cancel:
                if live is not None:
                    live.abort()
                    live.cleanup()
                self._transition(ApplicationState.CANCELLED, details={"recoverable": False})
            else:
                if live is not None:
                    live.abort()
                    live.cleanup()
                self._fail("application_job", exc)
        finally:
            with self._lock:
                self._capture = None
                self._live = None

    def _persisted(self, session_dir: Path, *, recoverable: bool) -> None:
        with self._lock:
            self._snapshot = replace(
                self._snapshot,
                session_id=session_dir.name,
                session_dir=session_dir,
                recoverable=recoverable,
            )
            state = self._snapshot.state
        self._emit(
            ApplicationEvent(
                kind=EventKind.SESSION_PERSISTED,
                state=state,
                session_dir=session_dir,
                details={"recoverable": recoverable},
            )
        )

    def _fail(self, stage: str, exc: BaseException) -> None:
        metadata = error_metadata(stage, exc)
        error = ApplicationError(
            stage=metadata.stage,
            error_type=metadata.type,
            message=metadata.message,
            retryable=metadata.retryable,
            occurred_at=metadata.occurred_at,
        )
        with self._lock:
            previous = self._snapshot.state
            self._snapshot = replace(
                self._snapshot,
                state=ApplicationState.FAILED,
                last_error=error,
            )
            self._append_session_event("error", details={"stage": stage, "type": error.error_type})
        self._emit(
            ApplicationEvent(
                kind=EventKind.ERROR_RECORDED,
                state=ApplicationState.FAILED,
                previous_state=previous,
                error=error,
            )
        )
        self._emit(
            ApplicationEvent(
                kind=EventKind.STATE_CHANGED,
                state=ApplicationState.FAILED,
                previous_state=previous,
            )
        )

    def pause(self) -> None:
        with self._lock:
            if self._snapshot.state is not ApplicationState.RECORDING or self._capture is None:
                raise InvalidApplicationCommand("Pause is only available while recording")
            self._capture.pause()
            self._pause_started_at = self._clock()
            self._append_session_event("paused")
        self._transition(ApplicationState.PAUSED)

    def _finish_pause(self) -> None:
        if self._pause_started_at is None:
            return
        paused_for = max(0.0, (self._clock() - self._pause_started_at).total_seconds())
        self._snapshot = replace(
            self._snapshot,
            total_paused_seconds=self._snapshot.total_paused_seconds + paused_for,
        )
        self._pause_started_at = None

    def resume(self) -> None:
        with self._lock:
            if self._snapshot.state is not ApplicationState.PAUSED or self._capture is None:
                raise InvalidApplicationCommand("Resume is only available while paused")
            self._capture.resume()
            self._finish_pause()
            self._append_session_event("resumed")
        self._transition(ApplicationState.RECORDING)

    def stop(self) -> None:
        with self._lock:
            if (
                self._snapshot.state
                not in {
                    ApplicationState.RECORDING,
                    ApplicationState.PAUSED,
                }
                or self._capture is None
            ):
                raise InvalidApplicationCommand("Stop is only available during capture")
            if self._snapshot.state is ApplicationState.PAUSED:
                self._finish_pause()
            self._append_session_event("stopping")
            self._capture.stop()
        self._transition(ApplicationState.STOPPING)

    def cancel(self) -> None:
        with self._lock:
            state = self._snapshot.state
            if state in {
                ApplicationState.PREPARING,
                ApplicationState.RECORDING,
                ApplicationState.PAUSED,
            }:
                if state is ApplicationState.PAUSED:
                    self._finish_pause()
                self._cancel_recording = True
                self._append_session_event("recording_cancelled", details={"recoverable": False})
                if self._capture is not None:
                    self._capture.cancel()
                return
            if state in {ApplicationState.STOPPING, ApplicationState.PROCESSING}:
                self._processing_cancel.set()
                self._append_session_event("processing_cancelled", details={"recoverable": True})
                live = self._live
                if live is not None:
                    live.abort()
                return
            raise InvalidApplicationCommand(f"Cannot cancel while application is {state.value}")

    def retry(self, session_dir: Path | None = None) -> None:
        with self._lock:
            if self._snapshot.has_active_work:
                raise InvalidApplicationCommand("Cannot retry while work is active")
            target = session_dir or self._snapshot.session_dir
            if target is None:
                raise InvalidApplicationCommand("No recoverable session is available")
            self._processing_cancel.clear()
            previous = self._snapshot.state
            self._snapshot = replace(self._snapshot, state=ApplicationState.PROCESSING)
            worker = threading.Thread(
                target=self._run_retry,
                args=(target,),
                daemon=True,
                name="here-application-retry",
            )
            self._worker = worker
        self._emit(
            ApplicationEvent(
                kind=EventKind.STATE_CHANGED,
                state=ApplicationState.PROCESSING,
                previous_state=previous,
                details={"retry": True},
            )
        )
        worker.start()

    def _run_retry(self, session_dir: Path) -> None:
        try:
            artifacts = self._processor.retry(session_dir, cancel_event=self._processing_cancel)
            self._persisted(artifacts.session_dir, recoverable=True)
            self._transition(ApplicationState.COMPLETED, details={"retry": True})
        except ProcessingCancelled as exc:
            self._persisted(exc.session_dir, recoverable=True)
            self._transition(ApplicationState.CANCELLED, details={"recoverable": True})
        except Exception as exc:
            self._fail("retry", exc)

    def wait_until_terminal(self, timeout: float | None = None) -> ApplicationSnapshot:
        with self._lock:
            worker = self._worker
        if worker is not None:
            worker.join(timeout)
            if worker.is_alive():
                raise TimeoutError("Timed out waiting for application job")
        return self.snapshot
