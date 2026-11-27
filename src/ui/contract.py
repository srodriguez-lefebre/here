"""Single adapter boundary between Qt and :mod:`here.application`.

Widgets import application vocabulary only from this module. Keeping command
translation here prevents the visual layer from depending on capture,
transcription, persistence, or concrete controller implementations.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from here.application import (
    ApplicationController,
    ApplicationEvent,
    ApplicationSnapshot,
    ApplicationState,
    AudioLevel,
    EventKind,
    EventListener,
    HereApplicationController,
    SourceMode,
    StartRequest,
    Unsubscribe,
)
from here.application.diagnostics import AudioDiagnosticsService
from here.application.recovery import RecoveryService
from here.config.settings import require_provider_key, settings_operation
from here.memory.database import LibraryError, open_library
from here.memory.models import (
    MeetingDetail,
    MeetingPage,
    MeetingPatch,
    MeetingQuery,
    Participant,
    ReindexReport,
    SearchPage,
    SearchQuery,
)
from here.memory.search import SearchService
from here.memory.service import MemoryService


class _MemoryBackend(Protocol):
    def get_meeting(self, meeting_id: str) -> MeetingDetail: ...

    def list_participants(self, meeting_id: str | None = None) -> tuple[Participant, ...]: ...

    def update_meeting(
        self, meeting_id: str, patch: MeetingPatch, expected_version: int
    ) -> MeetingDetail: ...


class _QueryBackend(Protocol):
    def list_meetings(self, query: MeetingQuery) -> MeetingPage: ...

    def search(self, query: SearchQuery) -> SearchPage | MeetingPage: ...


@runtime_checkable
class _MemoryOwner(Protocol):
    @property
    def memory_service(self) -> _MemoryBackend: ...


@runtime_checkable
class _CatalogRoot(Protocol):
    @property
    def data_dir(self) -> Path: ...


@runtime_checkable
class _MaintenanceBackend(Protocol):
    def reindex(self) -> ReindexReport: ...


class VisualController(Protocol):
    """Command names used by the presentation layer."""

    @property
    def snapshot(self) -> ApplicationSnapshot: ...

    @property
    def output_dir(self) -> Path: ...

    @property
    def source_label(self) -> str: ...

    def subscribe(self, listener: EventListener) -> Unsubscribe: ...

    def start_recording(self) -> None: ...

    def stop_and_process(self) -> None: ...

    def pause_recording(self) -> None: ...

    def resume_recording(self) -> None: ...

    def cancel_recording(self) -> None: ...

    def cancel_processing(self) -> None: ...

    def retry_processing(self) -> None: ...

    def discover_recovery(self): ...

    def retry_candidate(self, candidate): ...

    def diagnose(self, source): ...

    def list_meetings(self, query: MeetingQuery) -> MeetingPage: ...

    def search(self, query: SearchQuery) -> SearchPage | MeetingPage: ...

    def get_meeting(self, meeting_id: str) -> MeetingDetail: ...

    def list_participants(self, meeting_id: str | None = None) -> tuple[Participant, ...]: ...

    def update_meeting(self, meeting_id: str, patch: MeetingPatch, expected_version: int): ...


class ApplicationUiAdapter:
    """Translate visual intent into the stable application-core contract."""

    def __init__(
        self,
        controller: ApplicationController,
        output_dir: Path,
        *,
        diagnostics_service=None,
        recovery_service=None,
        memory_service: _MemoryBackend | None = None,
        search_service: _QueryBackend | None = None,
    ) -> None:
        self._controller = controller
        self._output_dir = output_dir
        self._diagnostics = diagnostics_service or AudioDiagnosticsService()
        if recovery_service is None:
            recovery_memory = memory_service
            if recovery_memory is None and isinstance(controller, HereApplicationController):
                recovery_memory = controller.memory_service
            catalog = recovery_memory if isinstance(recovery_memory, MemoryService) else None
            recovery_service = RecoveryService(output_dir, memory_service=catalog)
        self._recovery = recovery_service
        self._memory = memory_service
        self._search = search_service

    def _memory_backend(self) -> _MemoryBackend:
        # The controller's accepted R48 getter preserves injected/custom data roots.
        # Missing handoff is explicit; opening another default-root service is forbidden.
        backend = self._memory
        if backend is not None:
            return backend
        if isinstance(self._controller, _MemoryOwner):
            return self._controller.memory_service
        raise LibraryError("La biblioteca aún no está disponible. Reintentá más tarde.")

    def _query_backend(self) -> _QueryBackend:
        if self._search is not None:
            return self._search
        memory = self._memory_backend()
        if not isinstance(memory, _CatalogRoot):
            raise LibraryError("El backend de consultas de biblioteca aún no está disponible.")
        reindexer = memory.reindex if isinstance(memory, _MaintenanceBackend) else None
        return SearchService(lambda: open_library(memory.data_dir), reindexer=reindexer)

    def list_meetings(self, query: MeetingQuery) -> MeetingPage:
        return self._query_backend().list_meetings(query)

    def search(self, query: SearchQuery) -> SearchPage | MeetingPage:
        return self._query_backend().search(query)

    def get_meeting(self, meeting_id: str) -> MeetingDetail:
        return self._memory_backend().get_meeting(meeting_id)

    def list_participants(self, meeting_id: str | None = None) -> tuple[Participant, ...]:
        return self._memory_backend().list_participants(meeting_id)

    def update_meeting(self, meeting_id: str, patch: MeetingPatch, expected_version: int):
        return self._memory_backend().update_meeting(meeting_id, patch, expected_version)

    @property
    def snapshot(self) -> ApplicationSnapshot:
        return self._controller.snapshot

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    @property
    def source_label(self) -> str:
        return "Micrófono + audio del sistema"

    def subscribe(self, listener: EventListener) -> Unsubscribe:
        return self._controller.subscribe(listener)

    def start_recording(self) -> None:
        self._controller.start(
            StartRequest(output_dir=self._output_dir, source_mode=SourceMode.BOTH)
        )

    def stop_and_process(self) -> None:
        self._controller.stop()

    def pause_recording(self) -> None:
        self._controller.pause()

    def resume_recording(self) -> None:
        self._controller.resume()

    def cancel_recording(self) -> None:
        self._controller.cancel()

    def cancel_processing(self) -> None:
        self._controller.cancel()

    def retry_processing(self) -> None:
        self._controller.retry(self.snapshot.session_dir)

    def discover_recovery(self):
        return self._recovery.discover()

    @settings_operation
    def retry_candidate(self, candidate):
        if self.snapshot.has_active_work or not self.snapshot.worker_complete:
            raise RuntimeError("Espere a que termine la sesión actual.")
        if not candidate.can_retry:
            raise RuntimeError("Esta sesión no contiene audio recuperable.")
        require_provider_key()
        self._controller.retry(self._recovery.materialize(candidate))

    def diagnose(self, source):
        if self.snapshot.has_active_work or not self.snapshot.worker_complete:
            raise RuntimeError("Espere a que termine la sesión actual.")
        if source == "devices":
            return self._diagnostics.devices()
        return self._diagnostics.test_signal(source)


__all__ = [
    "ApplicationEvent",
    "ApplicationSnapshot",
    "ApplicationState",
    "ApplicationUiAdapter",
    "AudioLevel",
    "EventKind",
    "VisualController",
]
