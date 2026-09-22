"""Deterministic application-contract implementation for UI tests and preview."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from ..application import (
    ApplicationController,
    ApplicationError,
    ApplicationEvent,
    ApplicationSnapshot,
    ApplicationState,
    AudioLevel,
    EventKind,
    EventListener,
    StartRequest,
    Unsubscribe,
)


class FakeApplicationController(ApplicationController):
    """In-memory core double that publishes the real application event types."""

    def __init__(self) -> None:
        self._snapshot = ApplicationSnapshot()
        self._listeners: list[EventListener] = []
        self.command_log: list[str] = []
        self.last_request: StartRequest | None = None

    @property
    def snapshot(self) -> ApplicationSnapshot:
        return self._snapshot

    def subscribe(self, listener: EventListener) -> Unsubscribe:
        self._listeners.append(listener)

        def unsubscribe() -> None:
            if listener in self._listeners:
                self._listeners.remove(listener)

        return unsubscribe

    def publish(self, event: ApplicationEvent) -> None:
        for listener in tuple(self._listeners):
            listener(event)

    def set_state(
        self,
        state: ApplicationState,
        *,
        recoverable: bool = False,
        error: ApplicationError | None = None,
        session_dir: Path | None = None,
    ) -> None:
        previous = self._snapshot.state
        self._snapshot = replace(
            self._snapshot,
            state=state,
            recoverable=recoverable,
            last_error=error,
            session_dir=session_dir,
        )
        self.publish(
            ApplicationEvent(
                kind=EventKind.STATE_CHANGED,
                state=state,
                previous_state=previous,
                error=error,
                session_dir=session_dir,
            )
        )

    def set_audio_level(self, level: float) -> None:
        self.publish(
            ApplicationEvent(
                kind=EventKind.AUDIO_LEVEL,
                state=self._snapshot.state,
                audio_level=AudioLevel(source="combined", peak=level, rms=level * 0.7),
            )
        )

    def start(self, request: StartRequest) -> None:
        self.command_log.append("start")
        self.last_request = request
        self._snapshot = ApplicationSnapshot(
            state=ApplicationState.RECORDING,
            started_at=datetime.now(timezone.utc),
        )
        self.publish(
            ApplicationEvent(
                kind=EventKind.STATE_CHANGED,
                state=ApplicationState.RECORDING,
                previous_state=ApplicationState.IDLE,
            )
        )

    def pause(self) -> None:
        self.command_log.append("pause")
        self.set_state(ApplicationState.PAUSED)

    def resume(self) -> None:
        self.command_log.append("resume")
        self.set_state(ApplicationState.RECORDING)

    def stop(self) -> None:
        self.command_log.append("stop")
        self.set_state(ApplicationState.PROCESSING, recoverable=True)

    def cancel(self) -> None:
        state = self._snapshot.state
        self.command_log.append("cancel")
        self.set_state(
            ApplicationState.CANCELLED,
            recoverable=state in {ApplicationState.STOPPING, ApplicationState.PROCESSING},
        )

    def retry(self, session_dir: Path | None = None) -> None:
        self.command_log.append("retry")
        self.set_state(
            ApplicationState.PROCESSING,
            recoverable=True,
            session_dir=session_dir,
        )

    def complete(self, session_dir: Path | None = None) -> None:
        self.set_state(
            ApplicationState.COMPLETED,
            recoverable=True,
            session_dir=session_dir,
        )

    def fail(self, detail: str = "El procesamiento falló") -> None:
        error = ApplicationError(
            stage="processing",
            error_type="RuntimeError",
            message=detail,
            retryable=True,
        )
        self.set_state(ApplicationState.FAILED, recoverable=True, error=error)
