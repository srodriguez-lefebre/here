from __future__ import annotations

import re
import shutil
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from functools import wraps
from pathlib import Path
from typing import Concatenate, ParamSpec, cast
from uuid import UUID, uuid4

import soundfile as sf
from here.audio.mix import materialize_normalized_session
from here.config.paths import get_data_dir
from here.config.settings import get_settings, require_provider_key, settings_operation
from here.live_processing import LiveTranscriptionController
from here.memory.artifacts import read_pending_intent, write_durable_json
from here.memory.database import LibraryError
from here.memory.service import IngestResult, MemoryService, OwnershipBusy
from here.output.metadata import (
    CaptureSourceMetadata,
    ChunkMetadata,
    ChunkMetadataDocument,
    ErrorMetadata,
    ErrorMetadataDocument,
    SessionEventMetadata,
    SessionEventMetadataDocument,
    SessionMetadata,
    SourceMetadata,
    build_session_metadata,
    capture_metadata,
    source_metadata,
)
from here.output.ownership import FileLease, LeaseBusy
from here.output.paths import (
    UnsafeSessionPath,
    session_artifact_path,
    staged_artifact_path,
    write_artifact_text,
)
from here.output.session_writer import (
    AUDIO_FILE,
    CHUNKS_FILE,
    ERRORS_FILE,
    EVENTS_FILE,
    METADATA_FILE,
    SessionArtifactPaths,
    create_session_dir,
    read_session_metadata,
    validate_session_artifacts,
    write_session_artifacts,
)
from here.recording.models import CaptureFailed, RecordedAudioSource, RecordingSession
from here.transcriber import transcribe_recording_session
from here.transcription.client import TranscriptionResult
from loguru import logger


class ProcessingCancelled(RuntimeError):
    def __init__(self, session_dir: Path) -> None:
        super().__init__("Processing cancelled; recoverable audio was preserved")
        self.session_dir = session_dir


class SessionProcessingFailed(RuntimeError):
    def __init__(self, message: str, session_dir: Path, *, recoverable: bool) -> None:
        super().__init__(message)
        self.session_dir = session_dir
        self.recoverable = recoverable


class TranscriptionFailure(RuntimeError):
    def __init__(
        self,
        *,
        chunks: list[ChunkMetadata],
        errors: list[ErrorMetadata],
        live_pipeline_attempted: bool,
        fallback_used: bool,
    ) -> None:
        super().__init__("Transcription failed")
        self.chunks = chunks
        self.errors = errors
        self.live_pipeline_attempted = live_pipeline_attempted
        self.fallback_used = fallback_used


@dataclass(slots=True)
class TranscriptionOutcome:
    result: TranscriptionResult
    live_pipeline_attempted: bool
    live_pipeline_used: bool
    fallback_used: bool
    errors: list[ErrorMetadata]


def error_metadata(stage: str, exc: BaseException, *, retryable: bool = True) -> ErrorMetadata:
    cause = exc.__cause__ or exc.__context__
    return ErrorMetadata(
        stage=stage,
        type=type(exc).__name__,
        message=str(exc),
        cause_type=type(cause).__name__ if cause is not None else None,
        cause_message=str(cause) if cause is not None else None,
        retryable=retryable,
        occurred_at=datetime.now().astimezone(),
    )


def session_from_audio_file(audio_path: Path) -> RecordingSession:
    info = sf.info(audio_path)
    return RecordingSession(
        sources=[
            RecordedAudioSource(
                path=audio_path,
                sample_rate=info.samplerate,
                channels=info.channels,
                frames=info.frames,
                label=audio_path.stem,
                device_name=audio_path.name,
            )
        ]
    )


def session_audio_path(session_dir: Path, audio_file: str) -> Path:
    """Only persisted local filenames may select audio during recovery."""
    return session_artifact_path(session_dir, audio_file)


def _is_canonical_capture_id(capture_id: str) -> bool:
    """An optional session identity may locate a journal only as a canonical UUID."""
    try:
        identity = UUID(capture_id)
    except ValueError:
        return False
    return str(identity) == capture_id


def _check_capture_owner(root: Path, identity: str | None) -> None:
    if identity is None or not _is_canonical_capture_id(identity):
        return
    from here.recording.journal import CaptureJournal

    path = CaptureJournal.owner_path(root, identity)
    if path.exists():
        try:
            lease = FileLease(path, timeout=0)
        except LeaseBusy as exc:
            raise OwnershipBusy(str(exc)) from exc
        lease.close()


def _validated_recovery_source(
    audio_path: Path, expected: SourceMetadata | None = None
) -> RecordedAudioSource:
    info = sf.info(audio_path)
    if info.samplerate <= 0 or info.channels <= 0 or info.frames < 0:
        raise ValueError(f"Invalid recovery audio geometry: {audio_path.name}")
    if expected is not None and (info.samplerate, info.channels, info.frames) != (
        expected.sample_rate,
        expected.channels,
        expected.frames,
    ):
        raise ValueError(
            f"Recovery audio geometry disagrees with session metadata: {audio_path.name}"
        )
    return RecordedAudioSource(
        path=audio_path,
        sample_rate=info.samplerate,
        channels=info.channels,
        frames=info.frames,
        label=expected.label if expected is not None else audio_path.stem,
        device_name=expected.device_name if expected is not None else audio_path.name,
    )


def _cleanup_capture_after_commit(cleanup: Callable[[], None], session_dir: Path) -> None:
    """Remove capture files after completed metadata commits, retaining cleanup failures.

    Only file/journal cleanup belongs here; live/native worker closure stays strict.
    Journal validation may reject residual paths or unreadable media before removal.
    """
    try:
        cleanup()
    except (OSError, ValueError, RuntimeError) as exc:
        logger.warning(
            "Session metadata committed at {}; capture file cleanup failed; "
            "remaining capture files retained: {}",
            session_dir,
            exc,
        )


def _load_chunk_metadata(session_dir: Path) -> list[ChunkMetadata]:
    chunks_path = session_dir / CHUNKS_FILE
    if not chunks_path.exists():
        return []
    return ChunkMetadataDocument.model_validate_json(chunks_path.read_text(encoding="utf-8")).chunks


def _load_error_metadata(session_dir: Path) -> list[ErrorMetadata]:
    errors_path = session_dir / ERRORS_FILE
    if not errors_path.exists():
        return []
    return ErrorMetadataDocument.model_validate_json(errors_path.read_text(encoding="utf-8")).errors


def _cleanup_replaced_audio(
    session_dir: Path,
    previous: SessionMetadata | None,
    *,
    input_path: Path,
    current_audio: str,
) -> None:
    """Best-effort removal of one known derived file after final publication."""
    if previous is None or previous.recoverable_audio is None:
        return
    name = previous.recoverable_audio
    if re.fullmatch(r"audio(?:_[0-9a-f]{32})?\.wav", name) is None:
        logger.warning("Previous recoverable audio has an unknown name; retained: {}", name)
        return
    try:
        current = read_session_metadata(session_dir)
        if (
            current is None
            or current.status not in {"completed", "failed"}
            or current.session_id != previous.session_id
            or current.meeting_id != previous.meeting_id
            or current.recoverable_audio != current_audio
        ):
            raise ValueError("Final session metadata could not be confirmed")
        references = [*current.output_files, current_audio]
        references.extend(
            item.audio_file
            for item in (*previous.capture_sources, *current.capture_sources)
            if item.audio_file
        )
        protected = {
            input_path.absolute(),
            *((session_dir / item).absolute() for item in references),
        }
        candidate = session_artifact_path(session_dir, name)
        if candidate.absolute() in protected:
            return
        candidate.unlink(missing_ok=True)
    except (OSError, ValueError, RuntimeError) as exc:
        logger.warning(
            "Session metadata committed; replaced audio retained because cleanup failed: {}: {}",
            name,
            exc,
        )


P = ParamSpec("P")


def _catalog_producer(
    method: Callable[Concatenate[SessionProcessor, P], SessionArtifactPaths],
) -> Callable[Concatenate[SessionProcessor, P], SessionArtifactPaths]:
    """All application producers publish locally after compatibility artifacts commit."""

    @wraps(method)
    def publish(self: SessionProcessor, *args: P.args, **kwargs: P.kwargs) -> SessionArtifactPaths:
        expected = None
        meeting_id = None
        if method.__name__ in {"retry", "transcribe_audio_path"}:
            path = cast(
                Path,
                args[0]
                if args
                else kwargs["session_dir" if method.__name__ == "retry" else "audio_path"],
            )
            directory = path if method.__name__ == "retry" else path.parent
            metadata = read_session_metadata(directory)
            meeting_id = metadata.meeting_id if metadata else None
            # A rejected recovery preflight must not create or mutate a catalog.
            # WAL readers can inspect an existing active revision while another
            # operation holds BEGIN IMMEDIATE; they do not need a write reservation.
            catalog_root = self._memory.data_dir if self._memory else get_data_dir()
            if (catalog_root / "library.sqlite3").exists():
                try:
                    parent = self._memory_service().retry_parent(directory, meeting_id)
                    if parent:
                        meeting_id, expected = parent
                except (LibraryError, OSError):
                    pass
            self._retry_meeting_id = meeting_id
            self._retry_revision = expected
        try:
            artifacts = method(self, *args, **kwargs)
        except (SessionProcessingFailed, ProcessingCancelled) as exc:
            self.publish_memory(
                exc.session_dir,
                revision_of=self._retry_revision,
                status_only=bool(self._retry_revision and self._retry_meeting_id),
            )
            raise
        else:
            self.publish_memory(artifacts.session_dir, revision_of=self._retry_revision)
            return artifacts
        finally:
            self._release_processing()
            self._retry_meeting_id = None
            self._retry_revision = None

    return publish


class SessionProcessor:
    """Owns recoverable persistence, live fallback, retries and session reprocessing."""

    def __init__(
        self,
        *,
        transcribe: Callable[..., TranscriptionResult] = transcribe_recording_session,
        retry_delays: Sequence[float] = (0.5, 1.0),
        sleeper: Callable[[float], None] = time.sleep,
        normalize: Callable[..., RecordingSession] | None = None,
        clock: Callable[[], datetime] | None = None,
        memory_service: MemoryService | None = None,
    ) -> None:
        self._normalize = normalize
        self._clock = clock or (lambda: datetime.now().astimezone())
        self._transcribe = transcribe
        self._retry_delays = tuple(retry_delays)
        self._sleeper = sleeper
        self._memory = memory_service
        self._owned_processing_id: str | None = None
        self._retry_meeting_id: str | None = None
        self._retry_revision: str | None = None

    def _memory_service(self) -> MemoryService:
        if self._memory is None:
            self._memory = MemoryService(get_data_dir())
        return self._memory

    def _claim_processing(self, meeting_id: str | None) -> None:
        if meeting_id is None or not _is_canonical_capture_id(meeting_id):
            meeting_id = self._retry_meeting_id
        if not meeting_id or not _is_canonical_capture_id(meeting_id):
            raise LibraryError("Processing requires a resolved canonical identity")
        try:
            self._memory_service().begin_processing(meeting_id)
        except OwnershipBusy:
            raise
        except (LibraryError, OSError):
            # Compatibility artifacts remain the recovery authority when the
            # independent catalog cannot write its processing state.
            pass
        self._owned_processing_id = meeting_id

    def _resolve_retry(self, directory: Path, source_id: str | None) -> None:
        if self._retry_revision is None:
            self._retry_meeting_id, self._retry_revision = self._memory_service().prepare_retry(
                directory, source_id
            )

    def _release_processing(self) -> None:
        identity, self._owned_processing_id = self._owned_processing_id, None
        if identity and self._memory is not None:
            self._memory.finish_capture(identity)

    def publish_memory(
        self, session_dir: Path, *, revision_of: str | None = None, status_only: bool = False
    ) -> IngestResult:
        try:
            metadata = read_session_metadata(session_dir)
            if metadata is None:
                raise ValueError("Session metadata is missing")
            marker = session_artifact_path(session_dir, "memory_pending.json")
            if marker.exists():
                expected, intent = read_pending_intent(session_dir, metadata.meeting_id)
                if revision_of is None:
                    revision_of, status_only = expected, intent == "status"
            if revision_of is not None and (
                not isinstance(revision_of, str) or str(UUID(revision_of)) != revision_of
            ):
                raise ValueError("Invalid pending revision UUID")
            write_durable_json(
                marker,
                {
                    "schema_version": 1,
                    "meeting_id": metadata.meeting_id,
                    "revision_of": revision_of,
                    "intent": "status" if status_only else "ingest",
                },
            )
            if status_only:
                if revision_of is None:
                    raise ValueError("A state-only retry requires an expected active revision")
                result = self._memory_service().finish_attempt(
                    metadata.meeting_id, metadata.status, revision_of=revision_of
                )
            else:
                result = (
                    self._memory_service().ingest_retry(session_dir, revision_of=revision_of)
                    if revision_of is not None
                    else self._memory_service().ingest_session(session_dir)
                )
            if result.status != "rejected":
                session_artifact_path(session_dir, marker.name).unlink(missing_ok=True)
        except Exception as exc:
            result = IngestResult("rejected", warnings=(str(exc),))
        if result.status == "rejected":
            logger.warning(
                "Audio guardado; biblioteca pendiente de recuperación local: {}", session_dir
            )
        return result

    @_catalog_producer
    def transcribe_audio_path(
        self,
        audio_path: Path,
        target_dir: Path,
        *,
        use_alt_transcription_model: bool = False,
    ) -> SessionArtifactPaths:
        """Transcribe an explicit file without taking ownership of its external origin."""
        from here.application.recovery import validate_recovery_audio

        require_provider_key()
        if not audio_path.exists():
            raise RuntimeError(f"Audio file does not exist: {audio_path}")

        settings = get_settings()
        transcription_model = (
            settings.ALT_TRANSCRIPTION_MODEL
            if use_alt_transcription_model
            else settings.TRANSCRIPTION_MODEL
        )
        completed_at = self._clock()
        existing_metadata = read_session_metadata(audio_path.parent)
        if existing_metadata is not None:
            _check_capture_owner(audio_path.parent.parent, existing_metadata.meeting_id)
            MemoryService.check_owner(get_data_dir(), existing_metadata.meeting_id)
            session_artifact_path(audio_path.parent, audio_path.name)
            validate_recovery_audio(audio_path.parent, existing_metadata)
        source_session = session_from_audio_file(audio_path)
        source_session.meeting_id = (
            existing_metadata.meeting_id if existing_metadata else str(uuid4())
        )
        previous_chunks = _load_chunk_metadata(audio_path.parent) if existing_metadata else []
        previous_errors = _load_error_metadata(audio_path.parent) if existing_metadata else []

        if existing_metadata is not None:
            self._resolve_retry(audio_path.parent, existing_metadata.meeting_id)
        self._claim_processing(source_session.meeting_id)

        if existing_metadata is not None:
            session_dir = audio_path.parent
            session_id = existing_metadata.session_id
        else:
            session_id, session_dir = create_session_dir(target_dir, completed_at)

        recoverable_audio = AUDIO_FILE
        if existing_metadata is not None:
            if audio_path.name == existing_metadata.recoverable_audio:
                recoverable_audio = audio_path.name
            else:
                # A managed raw input must not replace audio referenced by the old
                # completed manifest before pending publication has committed.
                recoverable_audio = f"audio_{uuid4().hex}.wav"
        recoverable_session: RecordingSession | None = None
        newly_owned_audio = False
        try:
            if audio_path.parent == session_dir and audio_path.name == recoverable_audio:
                recoverable_session = source_session
            else:
                recoverable_session = (self._normalize or materialize_normalized_session)(
                    source_session,
                    session_dir,
                    output_name=recoverable_audio,
                )
                newly_owned_audio = True
        except Exception as exc:
            if existing_metadata is not None:
                raise RuntimeError("Recoverable audio preparation failed") from exc
            errors = [
                *previous_errors,
                error_metadata("recoverable_audio", exc),
            ]
            artifacts = write_session_artifacts(
                session=source_session,
                capture_sources=existing_metadata.capture_sources
                if existing_metadata
                else capture_metadata(source_session),
                target_dir=target_dir,
                completed_at=existing_metadata.completed_at if existing_metadata else completed_at,
                transcription_model=transcription_model,
                cleanup_model=settings.CLEANUP_MODEL,
                cleanup_enabled=settings.CLEANUP_ENABLED,
                alt_model_used=use_alt_transcription_model,
                live_pipeline_attempted=existing_metadata.live_pipeline_attempted
                if existing_metadata
                else False,
                live_pipeline_used=existing_metadata.live_pipeline_used
                if existing_metadata
                else False,
                fallback_used=existing_metadata.fallback_used if existing_metadata else False,
                chunks=previous_chunks,
                errors=errors,
                status="failed",
                failure_stage="recoverable_audio",
                session_dir=session_dir,
                session_id=session_id,
            )
            raise SessionProcessingFailed(
                "Recoverable audio preparation failed", artifacts.session_dir, recoverable=False
            ) from exc

        self.publish_pending(
            recoverable_session,
            session_dir,
            session_id=session_id,
            completed_at=completed_at,
            recoverable_audio=recoverable_audio,
            use_alt_transcription_model=use_alt_transcription_model,
            capture_sources=existing_metadata.capture_sources
            if existing_metadata
            else capture_metadata(source_session),
            newly_owned_audio=newly_owned_audio,
        )
        try:
            result = self._transcribe(
                recoverable_session,
                use_alt_transcription_model=use_alt_transcription_model,
            )
        except Exception as exc:
            failed_chunks = previous_chunks + list(getattr(exc, "chunks", []))
            errors = [
                *previous_errors,
                error_metadata("offline_transcription", exc),
            ]
            artifacts = write_session_artifacts(
                session=recoverable_session,
                capture_sources=existing_metadata.capture_sources
                if existing_metadata
                else capture_metadata(source_session),
                target_dir=target_dir,
                completed_at=existing_metadata.completed_at if existing_metadata else completed_at,
                transcription_model=transcription_model,
                cleanup_model=settings.CLEANUP_MODEL,
                cleanup_enabled=settings.CLEANUP_ENABLED,
                alt_model_used=use_alt_transcription_model,
                live_pipeline_attempted=existing_metadata.live_pipeline_attempted
                if existing_metadata
                else False,
                live_pipeline_used=existing_metadata.live_pipeline_used
                if existing_metadata
                else False,
                fallback_used=existing_metadata.fallback_used if existing_metadata else False,
                chunks=failed_chunks,
                errors=errors,
                status="failed",
                failure_stage="offline_transcription",
                recoverable_audio=recoverable_audio,
                session_dir=session_dir,
                session_id=session_id,
            )
            _cleanup_replaced_audio(
                session_dir,
                existing_metadata,
                input_path=audio_path,
                current_audio=recoverable_audio,
            )
            raise SessionProcessingFailed(
                "Transcription failed", artifacts.session_dir, recoverable=True
            ) from exc

        artifacts = write_session_artifacts(
            session=recoverable_session,
            capture_sources=existing_metadata.capture_sources
            if existing_metadata
            else capture_metadata(source_session),
            target_dir=target_dir,
            transcript_text=result.final_text,
            raw_transcript_text=getattr(result, "raw_text", None),
            segments=getattr(result, "segments", None),
            completed_at=existing_metadata.completed_at if existing_metadata else completed_at,
            transcription_model=transcription_model,
            cleanup_model=settings.CLEANUP_MODEL,
            cleanup_enabled=settings.CLEANUP_ENABLED,
            alt_model_used=use_alt_transcription_model,
            live_pipeline_attempted=existing_metadata.live_pipeline_attempted
            if existing_metadata
            else False,
            live_pipeline_used=existing_metadata.live_pipeline_used if existing_metadata else False,
            fallback_used=existing_metadata.fallback_used if existing_metadata else False,
            chunks=previous_chunks + list(getattr(result, "chunks", [])),
            status="completed",
            recoverable_audio=recoverable_audio,
            session_dir=session_dir,
            session_id=session_id,
        )
        _cleanup_replaced_audio(
            session_dir, existing_metadata, input_path=audio_path, current_audio=recoverable_audio
        )
        return artifacts

    def publish_pending(
        self,
        session: RecordingSession,
        session_dir: Path,
        *,
        session_id: str,
        completed_at: datetime,
        recoverable_audio: str,
        use_alt_transcription_model: bool = False,
        capture_sources: list[CaptureSourceMetadata] | None = None,
        newly_owned_audio: bool = False,
    ) -> SessionMetadata:
        """Commit only the pending manifest, preserving all prior artifact bytes.

        Callers preparing different audio for an existing session must use a new
        owned filename so the old manifest remains coherent until this commit.
        """
        previous = read_session_metadata(session_dir)
        references = list(previous.output_files) if previous else [METADATA_FILE]
        references.append(recoverable_audio)
        provenance = (
            capture_sources
            if capture_sources is not None
            else (previous.capture_sources if previous else capture_metadata(session))
        )
        references.extend(item.audio_file for item in provenance if item.audio_file)
        validate_session_artifacts(session_dir, references)
        for source in session.sources:
            if getattr(source, "path", None) is not None:
                path = session_artifact_path(session_dir, source.path.name)
                if path.absolute() != source.path.absolute():
                    raise UnsafeSessionPath("Pending audio must belong to its session")
                _validated_recovery_source(path, source_metadata(source))
        for item in provenance:
            if item.audio_file:
                _validated_recovery_source(
                    session_artifact_path(session_dir, item.audio_file), item
                )
        settings = get_settings()
        metadata = build_session_metadata(
            session=session,
            session_id=previous.session_id if previous else session_id,
            completed_at=previous.completed_at if previous else completed_at,
            transcription_model=settings.ALT_TRANSCRIPTION_MODEL
            if use_alt_transcription_model
            else settings.TRANSCRIPTION_MODEL,
            cleanup_model=settings.CLEANUP_MODEL,
            cleanup_enabled=settings.CLEANUP_ENABLED,
            alt_model_used=use_alt_transcription_model,
            live_pipeline_attempted=previous.live_pipeline_attempted if previous else False,
            live_pipeline_used=previous.live_pipeline_used if previous else False,
            fallback_used=previous.fallback_used if previous else False,
            status="pending",
            recoverable_audio=recoverable_audio,
            started_at=previous.started_at if previous else None,
            total_paused_seconds=previous.total_paused_seconds if previous else 0.0,
            output_files=list(dict.fromkeys(references)),
            capture_sources=provenance,
        )
        if previous is not None:
            metadata.meeting_id = previous.meeting_id
        try:
            write_artifact_text(session_dir / METADATA_FILE, metadata.model_dump_json(indent=2))
        except Exception as publication_error:
            if newly_owned_audio:
                try:
                    # A failure can be reported after replace succeeded. Re-read
                    # authority before removing only this operation's new audio.
                    current = read_session_metadata(session_dir)
                    referenced = set(current.output_files) if current else set()
                    if current:
                        referenced.add(current.recoverable_audio)
                        referenced.update(item.audio_file for item in current.capture_sources)
                    if recoverable_audio not in referenced:
                        owned = session_artifact_path(session_dir, recoverable_audio)
                        _validated_recovery_source(owned)
                        owned.unlink()
                        if previous is None:
                            try:
                                session_dir.rmdir()  # succeeds only for an empty owned directory
                            except OSError:
                                pass
                except Exception as cleanup_error:
                    publication_error.add_note(
                        "Prepared audio retained because cleanup failed: "
                        f"{type(cleanup_error).__name__}"
                    )
            raise
        return metadata

    def _materialize(
        self, session: RecordingSession, directory: Path, *, output_name: str
    ) -> RecordingSession:
        normalize = self._normalize or materialize_normalized_session
        for source in session.sources:
            if getattr(source, "path", None) is not None:
                path = session_artifact_path(source.path.parent, source.path.name)
                _validated_recovery_source(path, source_metadata(source))
        result = normalize(session, directory, output_name=output_name)
        for source in result.sources:
            if getattr(source, "path", None) is not None:
                path = session_artifact_path(directory, source.path.name)
                if path.absolute() != source.path.absolute():
                    raise UnsafeSessionPath("Normalized source escaped the session")
                _validated_recovery_source(path, source_metadata(source))
        return result

    def _destination(
        self, session: RecordingSession, target_dir: Path, completed_at: datetime
    ) -> tuple[str, Path]:
        journal = getattr(session, "journal", None)
        if journal is None:
            for source in session.sources:
                if getattr(source, "path", None) is not None:
                    path = session_artifact_path(source.path.parent, source.path.name)
                    _validated_recovery_source(path, source_metadata(source))
            return create_session_dir(target_dir, completed_at)
        current = journal.load(journal.root, journal.document.capture_id)
        if current.root != target_dir.absolute():
            raise UnsafeSessionPath("Capture root disagrees with processing destination")
        session.sources = current.recording_session().sources  # authoritative validated disk state
        directory = current.destination
        existing = read_session_metadata(directory)
        if existing is not None and existing.meeting_id != current.document.capture_id:
            raise UnsafeSessionPath("Destination belongs to another session")
        directory.mkdir(parents=True, exist_ok=True)
        return current.document.started_at.strftime("%Y%m%d_%H%M%S"), directory

    def _capture_events(
        self, session: RecordingSession, events: list[SessionEventMetadata] | None
    ) -> list[SessionEventMetadata] | None:
        journal = getattr(session, "journal", None)
        if journal is None:
            return events
        current = journal.load(journal.root, journal.document.capture_id)
        return [
            *(events or []),
            *(SessionEventMetadata.model_validate(event) for event in current.document.events),
        ]

    def _preserve_sources(
        self, session: RecordingSession, session_dir: Path
    ) -> list[CaptureSourceMetadata]:
        validate_session_artifacts(
            session_dir, [f"source_{index:02d}.wav" for index in range(1, len(session.sources) + 1)]
        )
        for source in session.sources:
            if getattr(source, "path", None) is not None:
                path = session_artifact_path(source.path.parent, source.path.name)
                _validated_recovery_source(path, source_metadata(source))
        provenance = []
        for index, source in enumerate(session.sources, start=1):
            metadata = CaptureSourceMetadata(**source_metadata(source).model_dump())
            if getattr(source, "path", None) is not None and source.path.exists():
                name = f"source_{index:02d}.wav"
                with staged_artifact_path(session_dir / name) as staged:
                    shutil.copy2(source.path, staged)
                metadata.audio_file = name
            provenance.append(metadata)
        return provenance

    @_catalog_producer
    def persist_capture_failure(
        self,
        failure: CaptureFailed,
        target_dir: Path,
        *,
        use_alt_transcription_model: bool = False,
        events: list[SessionEventMetadata] | None = None,
        started_at: datetime | None = None,
        total_paused_seconds: float = 0.0,
        original_errors: list[ErrorMetadata] | None = None,
    ) -> SessionArtifactPaths:
        settings = get_settings()
        completed_at = self._clock()
        session = failure.session
        session_id, session_dir = self._destination(session, target_dir, completed_at)
        capture_events = self._capture_events(session, None) or []
        events = [*(events or []), *capture_events]
        provenance = self._preserve_sources(session, session_dir)
        errors = list(original_errors or [])
        if not errors:
            # Only the current validated journal owns original capture failures;
            # caller lifecycle events and IPC wrappers are not additional causes.
            errors = [
                ErrorMetadata(
                    stage="capture",
                    type=error_type,
                    message=message,
                    retryable=True,
                    occurred_at=event.occurred_at,
                )
                for event in capture_events
                if event.kind == "capture_error"
                and isinstance(error_type := event.details.get("error_type"), str)
                and error_type
                and isinstance(message := event.details.get("error_message"), str)
            ]
        if not errors:
            errors.append(error_metadata("capture", failure))
        recoverable_audio = None
        try:
            material = RecordingSession(
                [source for source in session.sources if source.frames > 0],
                meeting_id=session.meeting_id,
            )
            normalized = self._materialize(material, session_dir, output_name=AUDIO_FILE)
            recoverable_audio = AUDIO_FILE
        except UnsafeSessionPath:
            raise
        except Exception as exc:
            errors.append(error_metadata("recoverable_audio", exc))
            normalized = session
        artifacts = write_session_artifacts(
            session=normalized,
            target_dir=target_dir,
            completed_at=completed_at,
            transcription_model=settings.ALT_TRANSCRIPTION_MODEL
            if use_alt_transcription_model
            else settings.TRANSCRIPTION_MODEL,
            cleanup_model=settings.CLEANUP_MODEL,
            cleanup_enabled=settings.CLEANUP_ENABLED,
            alt_model_used=use_alt_transcription_model,
            live_pipeline_attempted=False,
            live_pipeline_used=False,
            fallback_used=False,
            errors=errors,
            status="failed",
            failure_stage="capture",
            recoverable_audio=recoverable_audio,
            session_dir=session_dir,
            session_id=session_id,
            events=events,
            started_at=started_at,
            total_paused_seconds=total_paused_seconds,
            capture_sources=provenance,
        )
        session.cleanup()
        return artifacts

    def mark_capture_failure_cancelled(
        self, artifacts: SessionArtifactPaths, *, events: list[SessionEventMetadata]
    ) -> None:
        """Finalize a recovery cancellation without losing the capture-error evidence."""
        extra_files = list(artifacts.metadata.output_files)
        extra_files.extend(
            source.audio_file for source in artifacts.metadata.capture_sources if source.audio_file
        )
        if artifacts.metadata.recoverable_audio:
            extra_files.append(artifacts.metadata.recoverable_audio)
        validate_session_artifacts(artifacts.session_dir, extra_files)
        metadata = artifacts.metadata.model_copy(
            update={"status": "cancelled", "failure_stage": "processing_cancelled"}
        )
        write_artifact_text(
            artifacts.session_dir / EVENTS_FILE,
            SessionEventMetadataDocument(events=events).model_dump_json(indent=2),
            encoding="utf-8",
        )
        write_artifact_text(
            artifacts.session_dir / METADATA_FILE, metadata.model_dump_json(indent=2)
        )
        artifacts.metadata = metadata

    def _offline_with_retries(
        self,
        session: RecordingSession,
        *,
        use_alt_transcription_model: bool,
        cancel_event: threading.Event,
    ) -> TranscriptionResult:
        last_error: Exception | None = None
        attempts = len(self._retry_delays) + 1
        for attempt in range(attempts):
            if cancel_event.is_set():
                raise InterruptedError("Processing cancelled")
            try:
                return self._transcribe(
                    session,
                    use_alt_transcription_model=use_alt_transcription_model,
                )
            except Exception as exc:
                last_error = exc
                if attempt < len(self._retry_delays):
                    self._sleeper(self._retry_delays[attempt])
        assert last_error is not None
        raise last_error

    def _transcribe_outcome(
        self,
        session: RecordingSession,
        *,
        use_alt_transcription_model: bool,
        live_controller: LiveTranscriptionController | None,
        cancel_event: threading.Event,
    ) -> TranscriptionOutcome:
        live_chunks: list[ChunkMetadata] = []
        errors: list[ErrorMetadata] = []
        if live_controller is not None and not cancel_event.is_set():
            try:
                return TranscriptionOutcome(
                    result=live_controller.complete(),
                    live_pipeline_attempted=True,
                    live_pipeline_used=True,
                    fallback_used=False,
                    errors=[],
                )
            except Exception as exc:
                errors.append(error_metadata("live_transcription", exc))
                live_chunks = live_controller.chunk_metadata()

        if cancel_event.is_set():
            raise InterruptedError("Processing cancelled")

        try:
            result = self._offline_with_retries(
                session,
                use_alt_transcription_model=use_alt_transcription_model,
                cancel_event=cancel_event,
            )
        except InterruptedError:
            raise
        except Exception as exc:
            errors.append(error_metadata("offline_transcription", exc))
            raise TranscriptionFailure(
                chunks=live_chunks + list(getattr(exc, "chunks", [])),
                errors=errors,
                live_pipeline_attempted=live_controller is not None,
                fallback_used=live_controller is not None,
            ) from exc

        if live_chunks:
            result.chunks = live_chunks + list(getattr(result, "chunks", []))
        return TranscriptionOutcome(
            result=result,
            live_pipeline_attempted=live_controller is not None,
            live_pipeline_used=False,
            fallback_used=live_controller is not None,
            errors=errors,
        )

    @settings_operation
    @_catalog_producer
    def process(
        self,
        session: RecordingSession,
        target_dir: Path,
        *,
        use_alt_transcription_model: bool = False,
        live_controller: LiveTranscriptionController | None = None,
        cancel_event: threading.Event | None = None,
        events: list[SessionEventMetadata] | None = None,
        started_at: datetime | None = None,
        total_paused_seconds: float = 0.0,
    ) -> SessionArtifactPaths:
        if self._transcribe is transcribe_recording_session:
            require_provider_key()
        cancellation = cancel_event or threading.Event()
        completed_at = self._clock()
        settings = get_settings()
        transcription_model = (
            settings.ALT_TRANSCRIPTION_MODEL
            if use_alt_transcription_model
            else settings.TRANSCRIPTION_MODEL
        )
        session_id, session_dir = self._destination(session, target_dir, completed_at)
        events = self._capture_events(session, events)
        recoverable: RecordingSession | None = None

        try:
            recoverable = self._materialize(
                session,
                session_dir,
                output_name=AUDIO_FILE,
            )
        except UnsafeSessionPath:
            raise
        except Exception as exc:
            provenance = self._preserve_sources(session, session_dir)
            write_session_artifacts(
                session=session,
                target_dir=target_dir,
                completed_at=completed_at,
                transcription_model=transcription_model,
                cleanup_model=settings.CLEANUP_MODEL,
                cleanup_enabled=settings.CLEANUP_ENABLED,
                alt_model_used=use_alt_transcription_model,
                live_pipeline_attempted=False,
                live_pipeline_used=False,
                fallback_used=False,
                errors=[error_metadata("recoverable_audio", exc)],
                status="failed",
                failure_stage="recoverable_audio",
                session_dir=session_dir,
                session_id=session_id,
                events=events,
                started_at=started_at,
                total_paused_seconds=total_paused_seconds,
                capture_sources=provenance,
            )
            if live_controller is not None:
                live_controller.abort()
            raise SessionProcessingFailed(
                "Recoverable audio preparation failed",
                session_dir,
                recoverable=any(source.audio_file and source.frames > 0 for source in provenance),
            ) from exc

        write_session_artifacts(
            session=recoverable,
            target_dir=target_dir,
            completed_at=completed_at,
            transcription_model=transcription_model,
            cleanup_model=settings.CLEANUP_MODEL,
            cleanup_enabled=settings.CLEANUP_ENABLED,
            alt_model_used=use_alt_transcription_model,
            live_pipeline_attempted=live_controller is not None,
            live_pipeline_used=False,
            fallback_used=False,
            status="pending",
            recoverable_audio=AUDIO_FILE,
            session_dir=session_dir,
            session_id=session_id,
            capture_sources=capture_metadata(session),
            events=events,
            started_at=started_at,
            total_paused_seconds=total_paused_seconds,
        )
        if cancellation.is_set():
            return self._persist_cancelled(
                original=session,
                recoverable=recoverable,
                target_dir=target_dir,
                session_dir=session_dir,
                session_id=session_id,
                completed_at=completed_at,
                transcription_model=transcription_model,
                use_alt_transcription_model=use_alt_transcription_model,
                live_controller=live_controller,
                events=events,
                started_at=started_at,
                total_paused_seconds=total_paused_seconds,
            )

        try:
            outcome = self._transcribe_outcome(
                recoverable,
                use_alt_transcription_model=use_alt_transcription_model,
                live_controller=live_controller,
                cancel_event=cancellation,
            )
        except InterruptedError:
            return self._persist_cancelled(
                original=session,
                recoverable=recoverable,
                target_dir=target_dir,
                session_dir=session_dir,
                session_id=session_id,
                completed_at=completed_at,
                transcription_model=transcription_model,
                use_alt_transcription_model=use_alt_transcription_model,
                live_controller=live_controller,
                events=events,
                started_at=started_at,
                total_paused_seconds=total_paused_seconds,
            )
        except TranscriptionFailure as exc:
            artifacts = write_session_artifacts(
                session=recoverable,
                capture_sources=capture_metadata(session),
                target_dir=target_dir,
                completed_at=completed_at,
                transcription_model=transcription_model,
                cleanup_model=settings.CLEANUP_MODEL,
                cleanup_enabled=settings.CLEANUP_ENABLED,
                alt_model_used=use_alt_transcription_model,
                live_pipeline_attempted=exc.live_pipeline_attempted,
                live_pipeline_used=False,
                fallback_used=exc.fallback_used,
                chunks=exc.chunks,
                errors=exc.errors,
                status="failed",
                failure_stage="offline_transcription",
                recoverable_audio=AUDIO_FILE,
                session_dir=session_dir,
                session_id=session_id,
                events=events,
                started_at=started_at,
                total_paused_seconds=total_paused_seconds,
            )
            session.cleanup()
            if live_controller is not None:
                live_controller.cleanup()
            raise SessionProcessingFailed(
                "Transcription failed",
                artifacts.session_dir,
                recoverable=True,
            ) from exc

        if cancellation.is_set():
            return self._persist_cancelled(
                original=session,
                recoverable=recoverable,
                target_dir=target_dir,
                session_dir=session_dir,
                session_id=session_id,
                completed_at=completed_at,
                transcription_model=transcription_model,
                use_alt_transcription_model=use_alt_transcription_model,
                live_controller=live_controller,
                events=events,
                started_at=started_at,
                total_paused_seconds=total_paused_seconds,
            )

        artifacts = write_session_artifacts(
            session=recoverable,
            capture_sources=capture_metadata(session),
            target_dir=target_dir,
            transcript_text=outcome.result.final_text,
            raw_transcript_text=getattr(outcome.result, "raw_text", None),
            segments=getattr(outcome.result, "segments", None),
            completed_at=completed_at,
            transcription_model=transcription_model,
            cleanup_model=settings.CLEANUP_MODEL,
            cleanup_enabled=settings.CLEANUP_ENABLED,
            alt_model_used=use_alt_transcription_model,
            live_pipeline_attempted=outcome.live_pipeline_attempted,
            live_pipeline_used=outcome.live_pipeline_used,
            fallback_used=outcome.fallback_used,
            chunks=list(getattr(outcome.result, "chunks", [])),
            errors=outcome.errors,
            recoverable_audio=AUDIO_FILE,
            session_dir=session_dir,
            session_id=session_id,
            events=events,
            started_at=started_at,
            total_paused_seconds=total_paused_seconds,
        )
        _cleanup_capture_after_commit(session.cleanup, artifacts.session_dir)
        if live_controller is not None:
            live_controller.cleanup()
        return artifacts

    def _persist_cancelled(
        self,
        *,
        original: RecordingSession,
        recoverable: RecordingSession,
        target_dir: Path,
        session_dir: Path,
        session_id: str,
        completed_at: datetime,
        transcription_model: str,
        use_alt_transcription_model: bool,
        live_controller: LiveTranscriptionController | None,
        events: list[SessionEventMetadata] | None,
        started_at: datetime | None,
        total_paused_seconds: float,
    ) -> SessionArtifactPaths:
        settings = get_settings()
        if live_controller is not None:
            live_controller.abort()
            live_controller.cleanup()
        artifacts = write_session_artifacts(
            session=recoverable,
            capture_sources=capture_metadata(original),
            target_dir=target_dir,
            completed_at=completed_at,
            transcription_model=transcription_model,
            cleanup_model=settings.CLEANUP_MODEL,
            cleanup_enabled=settings.CLEANUP_ENABLED,
            alt_model_used=use_alt_transcription_model,
            live_pipeline_attempted=live_controller is not None,
            live_pipeline_used=False,
            fallback_used=False,
            status="cancelled",
            failure_stage="processing_cancelled",
            recoverable_audio=AUDIO_FILE,
            session_dir=session_dir,
            session_id=session_id,
            events=events,
            started_at=started_at,
            total_paused_seconds=total_paused_seconds,
        )
        original.cleanup()
        raise ProcessingCancelled(artifacts.session_dir)

    @settings_operation
    @_catalog_producer
    def retry(
        self,
        session_dir: Path,
        *,
        cancel_event: threading.Event | None = None,
    ) -> SessionArtifactPaths:
        if self._transcribe is transcribe_recording_session:
            require_provider_key()
        validate_session_artifacts(session_dir)
        metadata = read_session_metadata(session_dir)
        _check_capture_owner(session_dir.parent, metadata.meeting_id if metadata else None)
        MemoryService.check_owner(get_data_dir(), metadata.meeting_id if metadata else None)
        audio_file = (
            metadata.recoverable_audio if metadata and metadata.recoverable_audio else AUDIO_FILE
        )
        audio_path = session_audio_path(session_dir, audio_file)
        journal = None
        if (
            metadata is not None
            and metadata.meeting_id is not None
            and _is_canonical_capture_id(metadata.meeting_id)
        ):
            from here.recording.journal import CaptureJournal

            try:
                candidate_journal = CaptureJournal.load(session_dir.parent, metadata.meeting_id)
            except FileNotFoundError:
                pass
            else:
                # Copied metadata may share an ID with another capture's reservation.
                if candidate_journal.destination == session_dir.absolute():
                    candidate_journal.recording_session()
                    journal = candidate_journal
        raw_sources = []
        if metadata is not None:
            extra_files = list(metadata.output_files)
            extra_files.extend(
                source.audio_file for source in metadata.capture_sources if source.audio_file
            )
            if metadata.recoverable_audio:
                extra_files.append(metadata.recoverable_audio)
            validate_session_artifacts(session_dir, extra_files)
            for item in metadata.capture_sources:
                if item.audio_file:
                    actual = _validated_recovery_source(
                        session_audio_path(session_dir, item.audio_file), item
                    )
                    raw_sources.append(actual)
                    item.duration_seconds = actual.duration_seconds

        chunks_path = session_dir / CHUNKS_FILE
        errors_path = session_dir / ERRORS_FILE
        events_path = session_dir / EVENTS_FILE
        chunks = (
            ChunkMetadataDocument.model_validate_json(
                chunks_path.read_text(encoding="utf-8")
            ).chunks
            if chunks_path.exists()
            else []
        )
        errors = (
            ErrorMetadataDocument.model_validate_json(
                errors_path.read_text(encoding="utf-8")
            ).errors
            if errors_path.exists()
            else []
        )
        events = (
            SessionEventMetadataDocument.model_validate_json(
                events_path.read_text(encoding="utf-8")
            ).events
            if events_path.exists()
            else []
        )
        if audio_path.exists():
            # Only this shape identifies a single persisted source as audio.wav;
            # legacy multi-source manifests may instead describe the original inputs.
            expected = (
                metadata.sources[0]
                if metadata is not None
                and metadata.recoverable_audio == audio_file
                and len(metadata.sources) == 1
                else None
            )
            recovered_source = _validated_recovery_source(audio_path, expected)
        else:
            raw_sources = [source for source in raw_sources if source.frames > 0]
            if not raw_sources:
                raise RuntimeError(f"Recoverable audio does not exist: {audio_path}")
            self._resolve_retry(session_dir, metadata.meeting_id if metadata else None)
            self._claim_processing(metadata.meeting_id if metadata else None)
            self._materialize(RecordingSession(raw_sources), session_dir, output_name=audio_file)
            recovered_source = _validated_recovery_source(audio_path)
        source = RecordingSession(
            [recovered_source], meeting_id=metadata.meeting_id if metadata else None
        )
        cancellation = cancel_event or threading.Event()
        settings = get_settings()
        if self._owned_processing_id is None:
            self._resolve_retry(session_dir, source.meeting_id)
            self._claim_processing(source.meeting_id)
        self.publish_pending(
            source,
            session_dir,
            session_id=metadata.session_id if metadata else session_dir.name,
            completed_at=metadata.completed_at if metadata else self._clock(),
            recoverable_audio=audio_file,
            use_alt_transcription_model=metadata.alt_model_used if metadata else False,
            capture_sources=metadata.capture_sources if metadata else None,
        )
        try:
            result = self._offline_with_retries(
                source,
                use_alt_transcription_model=metadata.alt_model_used if metadata else False,
                cancel_event=cancellation,
            )
            if cancellation.is_set():
                raise InterruptedError("Processing cancelled")
        except InterruptedError as exc:
            events.append(
                SessionEventMetadata(
                    kind="processing_cancelled",
                    occurred_at=datetime.now().astimezone(),
                    recorded_duration_seconds=source.duration_seconds,
                    total_paused_seconds=metadata.total_paused_seconds if metadata else 0.0,
                    details={"recoverable": True},
                )
            )
            write_session_artifacts(
                session=source,
                capture_sources=metadata.capture_sources if metadata else None,
                target_dir=session_dir.parent,
                completed_at=metadata.completed_at if metadata else datetime.now().astimezone(),
                transcription_model=settings.ALT_TRANSCRIPTION_MODEL
                if metadata and metadata.alt_model_used
                else settings.TRANSCRIPTION_MODEL,
                cleanup_model=settings.CLEANUP_MODEL,
                cleanup_enabled=settings.CLEANUP_ENABLED,
                alt_model_used=metadata.alt_model_used if metadata else False,
                live_pipeline_attempted=metadata.live_pipeline_attempted if metadata else False,
                live_pipeline_used=False,
                fallback_used=metadata.fallback_used if metadata else False,
                chunks=chunks,
                errors=errors,
                status="cancelled",
                failure_stage="processing_cancelled",
                recoverable_audio=audio_file,
                session_dir=session_dir,
                session_id=metadata.session_id if metadata else session_dir.name,
                events=events,
                started_at=metadata.started_at if metadata else None,
                total_paused_seconds=metadata.total_paused_seconds if metadata else 0.0,
            )
            raise ProcessingCancelled(session_dir) from exc
        except Exception as exc:
            failure = error_metadata("offline_transcription", exc)
            write_session_artifacts(
                session=source,
                capture_sources=metadata.capture_sources if metadata else None,
                target_dir=session_dir.parent,
                completed_at=metadata.completed_at if metadata else datetime.now().astimezone(),
                transcription_model=settings.ALT_TRANSCRIPTION_MODEL
                if metadata and metadata.alt_model_used
                else settings.TRANSCRIPTION_MODEL,
                cleanup_model=settings.CLEANUP_MODEL,
                cleanup_enabled=settings.CLEANUP_ENABLED,
                alt_model_used=metadata.alt_model_used if metadata else False,
                live_pipeline_attempted=metadata.live_pipeline_attempted if metadata else False,
                live_pipeline_used=False,
                fallback_used=metadata.fallback_used if metadata else False,
                chunks=chunks + list(getattr(exc, "chunks", [])),
                errors=[*errors, failure],
                status="failed",
                failure_stage="offline_transcription",
                recoverable_audio=audio_file,
                session_dir=session_dir,
                session_id=metadata.session_id if metadata else session_dir.name,
                events=events,
                started_at=metadata.started_at if metadata else None,
                total_paused_seconds=metadata.total_paused_seconds if metadata else 0.0,
            )
            raise SessionProcessingFailed(
                "Transcription failed",
                session_dir,
                recoverable=True,
            ) from exc

        artifacts = write_session_artifacts(
            session=source,
            target_dir=session_dir.parent,
            transcript_text=result.final_text,
            raw_transcript_text=getattr(result, "raw_text", None),
            segments=getattr(result, "segments", None),
            capture_sources=metadata.capture_sources if metadata else None,
            completed_at=metadata.completed_at if metadata else datetime.now().astimezone(),
            transcription_model=settings.ALT_TRANSCRIPTION_MODEL
            if metadata and metadata.alt_model_used
            else settings.TRANSCRIPTION_MODEL,
            cleanup_model=settings.CLEANUP_MODEL,
            cleanup_enabled=settings.CLEANUP_ENABLED,
            alt_model_used=metadata.alt_model_used if metadata else False,
            live_pipeline_attempted=metadata.live_pipeline_attempted if metadata else False,
            live_pipeline_used=False,
            fallback_used=metadata.fallback_used if metadata else False,
            chunks=chunks + list(getattr(result, "chunks", [])),
            status="completed",
            recoverable_audio=audio_file,
            session_dir=session_dir,
            session_id=metadata.session_id if metadata else session_dir.name,
            events=events,
            started_at=metadata.started_at if metadata else None,
            total_paused_seconds=metadata.total_paused_seconds if metadata else 0.0,
        )

        if journal is not None:
            _cleanup_capture_after_commit(journal.discard, artifacts.session_dir)
        return artifacts
