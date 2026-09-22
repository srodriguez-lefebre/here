from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import soundfile as sf
from here.audio.mix import materialize_normalized_session
from here.config.settings import get_settings
from here.live_processing import LiveTranscriptionController
from here.output.metadata import (
    ChunkMetadata,
    ChunkMetadataDocument,
    ErrorMetadata,
    ErrorMetadataDocument,
    SessionEventMetadata,
    SessionMetadata,
)
from here.output.session_writer import (
    AUDIO_FILE,
    CHUNKS_FILE,
    ERRORS_FILE,
    METADATA_FILE,
    SessionArtifactPaths,
    create_session_dir,
    write_session_artifacts,
)
from here.recording.models import RecordedAudioSource, RecordingSession
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


def _read_model(path: Path, model: type[SessionMetadata]) -> SessionMetadata | None:
    if not path.exists():
        return None
    return model.model_validate_json(path.read_text(encoding="utf-8"))


class SessionProcessor:
    """Owns recoverable persistence, live fallback, retries and session reprocessing."""

    def __init__(
        self,
        *,
        transcribe: Callable[..., TranscriptionResult] = transcribe_recording_session,
        retry_delays: Sequence[float] = (0.5, 1.0),
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self._transcribe = transcribe
        self._retry_delays = tuple(retry_delays)
        self._sleeper = sleeper

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
        cancellation = cancel_event or threading.Event()
        completed_at = datetime.now().astimezone()
        settings = get_settings()
        transcription_model = (
            settings.ALT_TRANSCRIPTION_MODEL
            if use_alt_transcription_model
            else settings.TRANSCRIPTION_MODEL
        )
        session_id, session_dir = create_session_dir(target_dir, completed_at)
        recoverable: RecordingSession | None = None

        try:
            recoverable = materialize_normalized_session(
                session,
                session_dir,
                output_name=AUDIO_FILE,
            )
        except Exception as exc:
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
            )
            if live_controller is not None:
                live_controller.abort()
            raise SessionProcessingFailed(
                "Recoverable audio preparation failed",
                session_dir,
                recoverable=False,
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

        artifacts = write_session_artifacts(
            session=recoverable,
            target_dir=target_dir,
            transcript_text=outcome.result.final_text,
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

    def retry(
        self,
        session_dir: Path,
        *,
        cancel_event: threading.Event | None = None,
    ) -> SessionArtifactPaths:
        audio_path = session_dir / AUDIO_FILE
        if not audio_path.exists():
            raise RuntimeError(f"Recoverable audio does not exist: {audio_path}")
        metadata_path = session_dir / METADATA_FILE
        metadata = _read_model(metadata_path, SessionMetadata)
        chunks_path = session_dir / CHUNKS_FILE
        errors_path = session_dir / ERRORS_FILE
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
        source = session_from_audio_file(audio_path)
        cancellation = cancel_event or threading.Event()
        try:
            result = self._offline_with_retries(
                source,
                use_alt_transcription_model=metadata.alt_model_used if metadata else False,
                cancel_event=cancellation,
            )
        except InterruptedError as exc:
            raise ProcessingCancelled(session_dir) from exc
        except Exception as exc:
            settings = get_settings()
            failure = error_metadata("offline_transcription", exc)
            write_session_artifacts(
                session=source,
                target_dir=session_dir.parent,
                completed_at=metadata.completed_at if metadata else datetime.now().astimezone(),
                transcription_model=metadata.transcription_model
                if metadata
                else settings.TRANSCRIPTION_MODEL,
                cleanup_model=metadata.cleanup_model if metadata else settings.CLEANUP_MODEL,
                cleanup_enabled=metadata.cleanup_enabled if metadata else settings.CLEANUP_ENABLED,
                alt_model_used=metadata.alt_model_used if metadata else False,
                live_pipeline_attempted=metadata.live_pipeline_attempted if metadata else False,
                live_pipeline_used=False,
                fallback_used=metadata.fallback_used if metadata else False,
                chunks=chunks + list(getattr(exc, "chunks", [])),
                errors=[*errors, failure],
                status="failed",
                failure_stage="offline_transcription",
                recoverable_audio=AUDIO_FILE,
                session_dir=session_dir,
                session_id=metadata.session_id if metadata else session_dir.name,
                started_at=metadata.started_at if metadata else None,
                total_paused_seconds=metadata.total_paused_seconds if metadata else 0.0,
            )
            raise SessionProcessingFailed(
                "Transcription failed",
                session_dir,
                recoverable=True,
            ) from exc

        settings = get_settings()
        return write_session_artifacts(
            session=source,
            target_dir=session_dir.parent,
            transcript_text=result.final_text,
            completed_at=metadata.completed_at if metadata else datetime.now().astimezone(),
            transcription_model=metadata.transcription_model
            if metadata
            else settings.TRANSCRIPTION_MODEL,
            cleanup_model=metadata.cleanup_model if metadata else settings.CLEANUP_MODEL,
            cleanup_enabled=metadata.cleanup_enabled if metadata else settings.CLEANUP_ENABLED,
            alt_model_used=metadata.alt_model_used if metadata else False,
            live_pipeline_attempted=metadata.live_pipeline_attempted if metadata else False,
            live_pipeline_used=False,
            fallback_used=metadata.fallback_used if metadata else False,
            chunks=chunks + list(getattr(result, "chunks", [])),
            status="completed",
            recoverable_audio=AUDIO_FILE,
            session_dir=session_dir,
            session_id=metadata.session_id if metadata else session_dir.name,
            started_at=metadata.started_at if metadata else None,
            total_paused_seconds=metadata.total_paused_seconds if metadata else 0.0,
        )
