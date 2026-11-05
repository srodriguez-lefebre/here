import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated
from uuid import UUID

import typer
from here.application import (
    ApplicationState,
    SourceMode,
    StartRequest,
    create_default_controller,
)
from here.application.processing import SessionProcessor
from here.audio.mix import materialize_normalized_session
from here.config.paths import get_data_dir
from here.config.settings import get_settings, settings_operation
from here.live_processing import LiveTranscriptionController
from here.memory.database import LibraryError
from here.memory.service import MemoryService
from here.output.metadata import (
    ChunkMetadata,
    ErrorMetadata,
)
from here.output.session_writer import (
    TRANSCRIPT_ENCODING as TRANSCRIPT_ENCODING,
)
from here.recorder import (
    RecordingSession,
    record_both_until_enter,
    record_mic_until_enter,
    record_os_until_enter,
)
from here.recording.diagnostics import (
    AudioDeviceInfo,
    SignalTestResult,
    get_windows_audio_devices,
    test_windows_audio_signal,
)
from here.transcriber import transcribe_recording_session
from here.transcription.client import TranscriptionResult
from loguru import logger

app = typer.Typer()
record_app = typer.Typer(invoke_without_command=True)
mic_app = typer.Typer(invoke_without_command=True)
os_app = typer.Typer(invoke_without_command=True)
test_app = typer.Typer()
app.add_typer(record_app, name="record")
app.add_typer(test_app, name="test")
memory_app = typer.Typer(help="Maintain the local meeting catalog without a provider.")
app.add_typer(memory_app, name="memory")
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


def _memory_output(value: object, json_output: bool) -> None:
    def encode(item):
        return asdict(item) if is_dataclass(item) and not isinstance(item, type) else str(item)

    serialized = json.dumps(value, default=encode, ensure_ascii=False)
    if json_output:
        typer.echo(serialized)
    else:
        logger.info(serialized)


def _memory_error(error: Exception, json_output: bool, code: int) -> None:
    _memory_output({"error": str(error)}, json_output)
    raise typer.Exit(code)


@memory_app.command("ingest")
def memory_ingest(path: Path, json_output: bool = typer.Option(False, "--json")) -> None:
    if not path.is_dir():
        _memory_error(ValueError("Session directory does not exist"), json_output, 2)
    try:
        result = MemoryService(get_data_dir()).ingest_session(path)
    except (LibraryError, OSError) as exc:
        _memory_error(exc, json_output, 4)
        return
    _memory_output(asdict(result), json_output)
    if result.status == "rejected":
        raise typer.Exit({"invalid_input": 2, "io": 4}.get(result.error_category or "conflict", 3))


@memory_app.command("reconcile")
def memory_reconcile(json_output: bool = typer.Option(False, "--json")) -> None:
    try:
        report = MemoryService(get_data_dir()).reconcile()
    except (LibraryError, OSError) as exc:
        _memory_error(exc, json_output, 4)
        return
    _memory_output(asdict(report), json_output)


@memory_app.command("show")
def memory_show(meeting_id: str, json_output: bool = typer.Option(False, "--json")) -> None:
    try:
        if str(UUID(meeting_id)) != meeting_id:
            raise ValueError("Use a canonical meeting UUID")
    except ValueError as exc:
        _memory_error(exc, json_output, 2)
    try:
        result = MemoryService(get_data_dir()).show(meeting_id)
    except (LibraryError, OSError) as exc:
        _memory_error(exc, json_output, 4)
        return
    if result is None:
        _memory_error(ValueError("Meeting does not exist"), json_output, 3)
    _memory_output(result, json_output)


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


@settings_operation
def _save_transcription(
    session: RecordingSession,
    target_dir: Path,
    *,
    use_alt_transcription_model: bool = False,
    live_controller: LiveTranscriptionController | None = None,
) -> None:
    processor = SessionProcessor(
        transcribe=transcribe_recording_session,
        normalize=materialize_normalized_session,
        clock=lambda: datetime.now().astimezone(),
        retry_delays=(),
    )
    artifacts = processor.process(
        session,
        target_dir,
        use_alt_transcription_model=use_alt_transcription_model,
        live_controller=live_controller,
    )
    logger.success("Saved to {path}", path=artifacts.session_dir)


@settings_operation
def _transcribe_audio_path(
    audio_path: Path,
    target_dir: Path,
    *,
    use_alt_transcription_model: bool = False,
) -> None:
    processor = SessionProcessor(
        transcribe=transcribe_recording_session,
        normalize=materialize_normalized_session,
        clock=lambda: datetime.now().astimezone(),
        retry_delays=(),
    )
    artifacts = processor.transcribe_audio_path(
        audio_path, target_dir, use_alt_transcription_model=use_alt_transcription_model
    )
    logger.success("Saved to {path}", path=artifacts.session_dir)


def _run_recording(
    capture_fn: Callable[..., RecordingSession],
    target_dir: Path,
    *,
    use_alt_transcription_model: bool = False,
    expected_source_count: int,
) -> None:
    del expected_source_count
    source_modes = {
        record_both_until_enter: SourceMode.BOTH,
        record_mic_until_enter: SourceMode.MICROPHONE,
        record_os_until_enter: SourceMode.SYSTEM_AUDIO,
    }
    source_mode = source_modes.get(capture_fn)
    if source_mode is None:
        raise ValueError("Unknown recording entry point")
    controller = create_default_controller()

    def finish_recording() -> None:
        try:
            if controller.snapshot.has_active_work:
                controller.cancel()
        finally:
            # A terminal state can precede live teardown and the completion observer.
            if not controller.snapshot.worker_complete:
                controller.wait_until_terminal()

    try:
        controller.start(
            StartRequest(
                output_dir=target_dir,
                source_mode=source_mode,
                use_alt_transcription_model=use_alt_transcription_model,
            )
        )
        while controller.snapshot.state is ApplicationState.PREPARING:
            time.sleep(0.01)
        if controller.snapshot.state is not ApplicationState.RECORDING:
            raise RuntimeError(
                controller.snapshot.last_error.message
                if controller.snapshot.last_error is not None
                else "Recording could not be started"
            )
        logger.info("Recording... Press Enter to stop.")
        input()
        controller.stop()
        snapshot = controller.wait_until_terminal()
        if snapshot.state is not ApplicationState.COMPLETED:
            raise RuntimeError(
                snapshot.last_error.message
                if snapshot.last_error is not None
                else f"Recording ended with state {snapshot.state.value}"
            )
        logger.success(
            "Saved to {path}",
            path=snapshot.session_dir,
        )
    except KeyboardInterrupt:
        try:
            finish_recording()
        except Exception:
            logger.exception("Failed to clean up the active recording")
        raise typer.Exit(code=130) from None
    except RuntimeError as exc:
        try:
            finish_recording()
        except Exception:
            logger.exception("Failed to clean up the active recording")
        logger.error(str(exc))
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        try:
            finish_recording()
        except Exception:
            logger.exception("Failed to clean up the active recording")
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
    cause = exc.__cause__ or exc.__context__
    return ErrorMetadata(
        stage=stage,
        type=type(exc).__name__,
        message=str(exc),
        cause_type=type(cause).__name__ if cause is not None else None,
        cause_message=str(cause) if cause is not None else None,
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
