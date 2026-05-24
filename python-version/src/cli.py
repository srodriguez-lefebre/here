from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated

import soundfile as sf
import typer
from loguru import logger

from here.audio.mix import materialize_normalized_session
from here.config.settings import get_settings
from here.live_processing import LiveTranscriptionController
from here.output.metadata import (
    ChunkMetadata,
    ChunkMetadataDocument,
    ErrorMetadata,
    ErrorMetadataDocument,
    SessionMetadata,
)
from here.output.session_writer import (
    AUDIO_FILE,
    CHUNKS_FILE,
    ERRORS_FILE,
    METADATA_FILE,
    TRANSCRIPT_ENCODING,
    create_session_dir,
    write_session_artifacts,
)
from here.recording.diagnostics import (
    AudioDeviceInfo,
    SignalTestResult,
    get_windows_audio_devices,
    test_windows_audio_signal,
)
from here.recorder import (
    RecordedAudioSource,
    RecordingSession,
    record_both_until_enter,
    record_mic_until_enter,
    record_os_until_enter,
)
from here.transcription.client import TranscriptionResult
from here.transcriber import transcribe_recording_session

app = typer.Typer()
record_app = typer.Typer(invoke_without_command=True)
mic_app = typer.Typer(invoke_without_command=True)
os_app = typer.Typer(invoke_without_command=True)
test_app = typer.Typer()
app.add_typer(record_app, name="record")
app.add_typer(test_app, name="test")
record_app.add_typer(mic_app, name="mic")
record_app.add_typer(os_app, name="os")
OutputDirOption = Annotated[
    Path | None,
    typer.Option(
        "--output-dir",
        "-o",
        help="Directory to save the transcription. Defaults to TRANSCRIPTIONS_DIR from settings.",
    ),
]
AudioFileArgument = Annotated[
    Path,
    typer.Argument(help="Audio file to transcribe."),
]


@dataclass(slots=True)
class _TranscriptionOutcome:
    result: TranscriptionResult
    live_pipeline_attempted: bool
    live_pipeline_used: bool
    fallback_used: bool
    errors: list[ErrorMetadata]


class _TranscriptionFailure(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        chunks: list[ChunkMetadata],
        errors: list[ErrorMetadata],
        live_pipeline_attempted: bool,
        live_pipeline_used: bool,
        fallback_used: bool,
        failure_stage: str,
    ) -> None:
        super().__init__(message)
        self.chunks = chunks
        self.errors = errors
        self.live_pipeline_attempted = live_pipeline_attempted
        self.live_pipeline_used = live_pipeline_used
        self.fallback_used = fallback_used
        self.failure_stage = failure_stage


def _resolve_target_dir(output_dir: Path | None) -> Path:
    return output_dir or get_settings().TRANSCRIPTIONS_DIR


def _format_device_info(device: AudioDeviceInfo) -> str:
    return (
        f"{device.source}: {device.name} "
        f"(index={device.index}, {device.channels} channel(s), {device.sample_rate} Hz)"
    )


def _format_signal_result(result: SignalTestResult) -> str:
    signal = "signal detected" if result.has_signal else "no signal detected"
    return (
        f"{_format_device_info(result.device)}\n"
        f"duration={result.duration_seconds:.1f}s peak={result.peak:.4f} "
        f"rms={result.rms:.4f} status={signal}"
    )


def _run_audio_diagnostic(action: Callable[[], str]) -> None:
    try:
        typer.echo(action())
    except RuntimeError as exc:
        logger.error(str(exc))
        raise typer.Exit(code=1) from exc


def _load_session_metadata(session_dir: Path) -> SessionMetadata | None:
    metadata_path = session_dir / METADATA_FILE
    if not metadata_path.exists():
        return None
    return SessionMetadata.model_validate_json(metadata_path.read_text(encoding="utf-8"))


def _load_chunk_metadata(session_dir: Path) -> list[ChunkMetadata]:
    chunks_path = session_dir / CHUNKS_FILE
    if not chunks_path.exists():
        return []
    return ChunkMetadataDocument.model_validate_json(
        chunks_path.read_text(encoding="utf-8")
    ).chunks


def _load_error_metadata(session_dir: Path) -> list[ErrorMetadata]:
    errors_path = session_dir / ERRORS_FILE
    if not errors_path.exists():
        return []
    return ErrorMetadataDocument.model_validate_json(
        errors_path.read_text(encoding="utf-8")
    ).errors


def _session_from_audio_file(audio_path: Path) -> RecordingSession:
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


def _save_transcription(
    session: RecordingSession,
    target_dir: Path,
    *,
    use_alt_transcription_model: bool = False,
    live_controller: LiveTranscriptionController | None = None,
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    recording_completed_at = datetime.now().astimezone()
    settings = get_settings()
    transcription_model = (
        settings.ALT_TRANSCRIPTION_MODEL
        if use_alt_transcription_model
        else settings.TRANSCRIPTION_MODEL
    )
    session_id, session_dir = create_session_dir(target_dir, recording_completed_at)
    recoverable_session: RecordingSession | None = None
    raw_audio_is_recoverable = False

    try:
        recoverable_session = materialize_normalized_session(
            session,
            session_dir,
            output_name=AUDIO_FILE,
        )
        raw_audio_is_recoverable = True

        outcome = _transcribe_session_outcome(
            recoverable_session,
            use_alt_transcription_model=use_alt_transcription_model,
            live_controller=live_controller,
        )
    except _TranscriptionFailure as exc:
        write_session_artifacts(
            session=recoverable_session or session,
            target_dir=target_dir,
            completed_at=recording_completed_at,
            transcription_model=transcription_model,
            cleanup_model=settings.CLEANUP_MODEL,
            cleanup_enabled=settings.CLEANUP_ENABLED,
            alt_model_used=use_alt_transcription_model,
            live_pipeline_attempted=exc.live_pipeline_attempted,
            live_pipeline_used=exc.live_pipeline_used,
            fallback_used=exc.fallback_used,
            chunks=exc.chunks,
            errors=exc.errors,
            status="failed",
            failure_stage=exc.failure_stage,
            recoverable_audio=AUDIO_FILE if raw_audio_is_recoverable else None,
            session_dir=session_dir,
            session_id=session_id,
        )
        if raw_audio_is_recoverable:
            session.cleanup()
            if live_controller is not None:
                live_controller.cleanup()
        else:
            if live_controller is not None:
                live_controller.abort()
            logger.warning("Temporary audio files were preserved after recoverable audio failure.")
        logger.error("Saved failed recoverable session to {path}", path=session_dir)
        raise
    except Exception as exc:
        if live_controller is not None:
            live_controller.abort()
        logger.warning("Temporary audio files were preserved after transcription failure.")
        raise

    artifacts = write_session_artifacts(
        session=recoverable_session,
        target_dir=target_dir,
        transcript_text=outcome.result.final_text,
        completed_at=recording_completed_at,
        transcription_model=transcription_model,
        cleanup_model=settings.CLEANUP_MODEL,
        cleanup_enabled=settings.CLEANUP_ENABLED,
        alt_model_used=use_alt_transcription_model,
        live_pipeline_attempted=outcome.live_pipeline_attempted,
        live_pipeline_used=outcome.live_pipeline_used,
        fallback_used=outcome.fallback_used,
        chunks=list(getattr(outcome.result, "chunks", [])),
        errors=outcome.errors,
        status="completed",
        recoverable_audio=AUDIO_FILE,
        session_dir=session_dir,
        session_id=session_id,
    )
    session.cleanup()
    if live_controller is not None:
        live_controller.cleanup()
    logger.info("Temporary audio files deleted.")
    logger.success("Saved to {path}", path=artifacts.session_dir)


def _transcribe_audio_path(
    audio_path: Path,
    target_dir: Path,
    *,
    use_alt_transcription_model: bool = False,
) -> None:
    if not audio_path.exists():
        raise RuntimeError(f"Audio file does not exist: {audio_path}")

    settings = get_settings()
    transcription_model = (
        settings.ALT_TRANSCRIPTION_MODEL
        if use_alt_transcription_model
        else settings.TRANSCRIPTION_MODEL
    )
    completed_at = datetime.now().astimezone()
    source_session = _session_from_audio_file(audio_path)
    existing_metadata = _load_session_metadata(audio_path.parent)
    previous_chunks = _load_chunk_metadata(audio_path.parent) if existing_metadata else []
    previous_errors = _load_error_metadata(audio_path.parent) if existing_metadata else []

    if existing_metadata is not None:
        session_dir = audio_path.parent
        session_id = existing_metadata.session_id
    else:
        session_id, session_dir = create_session_dir(target_dir, completed_at)

    if audio_path.parent == session_dir and audio_path.name == AUDIO_FILE:
        recoverable_session = source_session
    else:
        recoverable_session = materialize_normalized_session(
            source_session,
            session_dir,
            output_name=AUDIO_FILE,
        )

    try:
        result = transcribe_recording_session(
            recoverable_session,
            use_alt_transcription_model=use_alt_transcription_model,
        )
    except Exception as exc:
        failed_chunks = previous_chunks + list(getattr(exc, "chunks", []))
        errors = [
            *previous_errors,
            _error_metadata("offline_transcription", exc),
        ]
        write_session_artifacts(
            session=recoverable_session,
            target_dir=target_dir,
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
            chunks=failed_chunks,
            errors=errors,
            status="failed",
            failure_stage="offline_transcription",
            recoverable_audio=AUDIO_FILE,
            session_dir=session_dir,
            session_id=session_id,
        )
        raise RuntimeError("Transcription failed") from exc

    artifacts = write_session_artifacts(
        session=recoverable_session,
        target_dir=target_dir,
        transcript_text=result.final_text,
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
        recoverable_audio=AUDIO_FILE,
        session_dir=session_dir,
        session_id=session_id,
    )
    logger.success("Saved to {path}", path=artifacts.session_dir)


def _run_recording(
    capture_fn: Callable[..., RecordingSession],
    target_dir: Path,
    *,
    use_alt_transcription_model: bool = False,
    expected_source_count: int,
) -> None:
    logger.info(
        "Live chunk processing enabled for this recording ({sources} source(s)).",
        sources=expected_source_count,
    )
    live_controller = LiveTranscriptionController(
        expected_source_count=expected_source_count,
        use_alt_transcription_model=use_alt_transcription_model,
    )
    try:
        session = capture_fn(block_sink=live_controller.submit_block)
        _save_transcription(
            session,
            target_dir,
            use_alt_transcription_model=use_alt_transcription_model,
            live_controller=live_controller,
        )
    except RuntimeError as exc:
        live_controller.abort()
        live_controller.cleanup()
        logger.error(str(exc))
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        live_controller.abort()
        live_controller.cleanup()
        logger.exception("Unexpected error while recording or transcribing audio")
        raise typer.Exit(code=1) from exc


def _transcribe_session(
    session: RecordingSession,
    *,
    use_alt_transcription_model: bool = False,
    live_controller: LiveTranscriptionController | None = None,
) -> TranscriptionResult:
    return _transcribe_session_outcome(
        session,
        use_alt_transcription_model=use_alt_transcription_model,
        live_controller=live_controller,
    ).result


def _transcribe_session_outcome(
    session: RecordingSession,
    *,
    use_alt_transcription_model: bool = False,
    live_controller: LiveTranscriptionController | None = None,
) -> _TranscriptionOutcome:
    failed_live_chunks: list[ChunkMetadata] = []
    errors: list[ErrorMetadata] = []
    if live_controller is not None:
        try:
            logger.info("Completing live transcription from background chunks.")
            return _TranscriptionOutcome(
                result=live_controller.complete(),
                live_pipeline_attempted=True,
                live_pipeline_used=True,
                fallback_used=False,
                errors=[],
            )
        except Exception as exc:
            logger.warning(
                "Live transcription failed: {exc}. Falling back to offline post-processing.",
                exc=exc,
            )
            errors.append(_error_metadata("live_transcription", exc))
            failed_live_chunks = _chunk_metadata_from_controller(live_controller)

    try:
        result = transcribe_recording_session(
            session,
            use_alt_transcription_model=use_alt_transcription_model,
        )
    except Exception as exc:
        offline_chunks = getattr(exc, "chunks", [])
        errors.append(_error_metadata("offline_transcription", exc))
        raise _TranscriptionFailure(
            "Transcription failed",
            chunks=failed_live_chunks + list(offline_chunks),
            errors=errors,
            live_pipeline_attempted=live_controller is not None,
            live_pipeline_used=False,
            fallback_used=live_controller is not None,
            failure_stage="offline_transcription",
        ) from exc

    if failed_live_chunks:
        result.chunks = failed_live_chunks + list(getattr(result, "chunks", []))

    return _TranscriptionOutcome(
        result=result,
        live_pipeline_attempted=live_controller is not None,
        live_pipeline_used=False,
        fallback_used=live_controller is not None,
        errors=errors,
    )


def _chunk_metadata_from_controller(
    live_controller: LiveTranscriptionController,
) -> list[ChunkMetadata]:
    chunk_metadata = getattr(live_controller, "chunk_metadata", None)
    if not callable(chunk_metadata):
        return []
    return list(chunk_metadata())


def _error_metadata(stage: str, exc: Exception) -> ErrorMetadata:
    return ErrorMetadata(
        stage=stage,
        type=type(exc).__name__,
        message=str(exc),
        retryable=True,
        occurred_at=datetime.now().astimezone(),
    )


@app.callback()
def main() -> None:
    """here - record audio and transcribe it."""


@app.command("devices")
def devices() -> None:
    """Show the Windows audio devices used by here."""

    def _show_devices() -> str:
        return "\n".join(_format_device_info(device) for device in get_windows_audio_devices())

    _run_audio_diagnostic(_show_devices)


@app.command("trans")
def trans(
    audio_file: AudioFileArgument,
    output_dir: OutputDirOption = None,
) -> None:
    """Transcribe an existing audio file."""
    try:
        _transcribe_audio_path(audio_file, _resolve_target_dir(output_dir))
    except RuntimeError as exc:
        logger.error(str(exc))
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        logger.exception("Unexpected error while transcribing audio")
        raise typer.Exit(code=1) from exc


@test_app.command("mic")
def test_mic(
    duration: Annotated[
        float,
        typer.Option("--duration", "-d", help="Seconds to measure microphone signal."),
    ] = 3.0,
) -> None:
    """Measure signal from the default Windows microphone."""

    _run_audio_diagnostic(
        lambda: _format_signal_result(
            test_windows_audio_signal("microphone", duration_seconds=duration)
        )
    )


@test_app.command("os")
def test_os(
    duration: Annotated[
        float,
        typer.Option("--duration", "-d", help="Seconds to measure system audio signal."),
    ] = 3.0,
) -> None:
    """Measure signal from the default Windows WASAPI loopback device."""

    _run_audio_diagnostic(
        lambda: _format_signal_result(
            test_windows_audio_signal("system audio", duration_seconds=duration)
        )
    )


@record_app.callback(invoke_without_command=True)
def record_main(
    ctx: typer.Context,
    output_dir: OutputDirOption = None,
) -> None:
    """Record audio from different sources and transcribe it."""
    if ctx.invoked_subcommand is not None:
        return

    _run_recording(
        record_both_until_enter,
        _resolve_target_dir(output_dir),
        expected_source_count=2,
    )


@record_app.command("alt")
def record_alt(
    output_dir: OutputDirOption = None,
) -> None:
    """Record microphone + system audio with the alternate transcription model."""
    _run_recording(
        record_both_until_enter,
        _resolve_target_dir(output_dir),
        use_alt_transcription_model=True,
        expected_source_count=2,
    )


@mic_app.callback(invoke_without_command=True)
def mic_main(
    ctx: typer.Context,
    output_dir: OutputDirOption = None,
) -> None:
    """Record audio from the microphone only."""
    if ctx.invoked_subcommand is not None:
        return

    _run_recording(
        record_mic_until_enter,
        _resolve_target_dir(output_dir),
        expected_source_count=1,
    )


@mic_app.command("alt")
def mic_alt(
    output_dir: OutputDirOption = None,
) -> None:
    """Record microphone audio with the alternate transcription model."""
    _run_recording(
        record_mic_until_enter,
        _resolve_target_dir(output_dir),
        use_alt_transcription_model=True,
        expected_source_count=1,
    )


@os_app.callback(invoke_without_command=True)
def os_main(
    ctx: typer.Context,
    output_dir: OutputDirOption = None,
) -> None:
    """Record system audio only."""
    if ctx.invoked_subcommand is not None:
        return

    _run_recording(
        record_os_until_enter,
        _resolve_target_dir(output_dir),
        expected_source_count=1,
    )


@os_app.command("alt")
def os_alt(
    output_dir: OutputDirOption = None,
) -> None:
    """Record system audio with the alternate transcription model."""
    _run_recording(
        record_os_until_enter,
        _resolve_target_dir(output_dir),
        use_alt_transcription_model=True,
        expected_source_count=1,
    )
