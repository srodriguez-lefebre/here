from __future__ import annotations

import shutil
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import soundfile as sf
from here.audio.mix import materialize_normalized_session
from here.config.settings import get_settings, require_provider_key, settings_operation
from here.live_processing import LiveTranscriptionController
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
    ) -> None:
        self._normalize = normalize
        self._clock = clock or (lambda: datetime.now().astimezone())
        self._transcribe = transcribe
        self._retry_delays = tuple(retry_delays)
        self._sleeper = sleeper

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
        events = self._capture_events(session, events)
        provenance = self._preserve_sources(session, session_dir)
        errors = original_errors or [error_metadata("capture", failure)]
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
        session.cleanup()
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
        audio_file = (
            metadata.recoverable_audio if metadata and metadata.recoverable_audio else AUDIO_FILE
        )
        audio_path = session_audio_path(session_dir, audio_file)
        journal = None
        if metadata is not None and metadata.meeting_id is not None:
            from here.recording.journal import CaptureJournal

            try:
                journal = CaptureJournal.load(session_dir.parent, metadata.meeting_id)
            except FileNotFoundError:
                pass
            if journal is not None:
                if journal.destination != session_dir.absolute():
                    raise UnsafeSessionPath("Recovery journal points to another session")
                journal.recording_session()
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
            self._materialize(RecordingSession(raw_sources), session_dir, output_name=audio_file)
            recovered_source = _validated_recovery_source(audio_path)
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
        source = RecordingSession(
            [recovered_source], meeting_id=metadata.meeting_id if metadata else None
        )
        cancellation = cancel_event or threading.Event()
        settings = get_settings()
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
            journal.discard()
        return artifacts
