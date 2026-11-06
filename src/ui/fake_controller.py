"""Deterministic application-contract implementation for UI tests and preview."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from here.application import (
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
from here.memory.models import (
    Meeting,
    MeetingDetail,
    MeetingPage,
    MeetingPatch,
    MeetingStatus,
    MeetingSummary,
    Participant,
    SearchHit,
    SearchPage,
    Segment,
    TranscriptRevision,
)
from here.memory.service import Conflict


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
        worker_complete: bool | None = None,
    ) -> None:
        previous = self._snapshot.state
        self._snapshot = replace(
            self._snapshot,
            state=state,
            recoverable=recoverable,
            last_error=error,
            session_dir=session_dir,
            worker_complete=worker_complete
            if worker_complete is not None
            else state
            in {
                ApplicationState.IDLE,
                ApplicationState.COMPLETED,
                ApplicationState.FAILED,
                ApplicationState.CANCELLED,
            },
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
            worker_complete=False,
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


class FakeLibraryBackend:
    """Explicit synthetic preview data, with no filesystem or persistence authority."""

    def __init__(self):
        meeting_id = "00000000-0000-0000-0000-000000000101"
        revision_id = "00000000-0000-0000-0000-000000000102"
        self.meeting = Meeting(
            id=meeting_id,
            title="Planificación · ejemplo sintético",
            started_at="2026-11-05T12:00:00+00:00",
            duration_ms=60000,
            status=MeetingStatus.COMPLETED,
            active_revision_id=revision_id,
        )
        self.revision = TranscriptRevision(
            id=revision_id,
            meeting_id=meeting_id,
            revision_number=1,
            raw_text="Revisamos el plan y los próximos pasos del ejemplo.",
            provider="synthetic",
            model="synthetic",
            origin="preview",
            content_hash="a" * 64,
        )
        self.segment = Segment(
            id="00000000-0000-0000-0000-000000000103",
            revision_id=revision_id,
            ordinal=0,
            text=self.revision.raw_text,
        )

    def list_meetings(self, query):
        matches = (not query.statuses or self.meeting.status in query.statuses) and (
            query.participant_id is None
        )
        return MeetingPage(items=(MeetingSummary(self.meeting),) if matches else ())

    def search(self, query):
        text = f"{self.meeting.title} {self.segment.text}".casefold()
        if not all(token.casefold() in text for token in query.text.split()):
            return SearchPage()
        return SearchPage(
            items=(
                SearchHit(
                    meeting_id=self.meeting.id,
                    revision_id=self.revision.id,
                    segment_id=self.segment.id,
                    snippet=self.segment.text,
                    title=self.meeting.title,
                    started_at=self.meeting.started_at,
                ),
            )
        )

    def get_meeting(self, meeting_id):
        if meeting_id != self.meeting.id:
            return None
        return MeetingDetail(self.meeting, revisions=(self.revision,), segments=(self.segment,))

    def list_participants(self, meeting_id=None) -> tuple[Participant, ...]:
        return ()

    def update_meeting(self, meeting_id, patch: MeetingPatch, expected_version):
        if meeting_id != self.meeting.id or expected_version != self.meeting.row_version:
            raise Conflict("El ejemplo cambió; volvé a abrirlo.")
        self.meeting = replace(
            self.meeting,
            title=self.meeting.title if patch.title is None else patch.title,
            notes=self.meeting.notes if patch.notes is None else patch.notes,
            row_version=self.meeting.row_version + 1,
        )
        return self.meeting
