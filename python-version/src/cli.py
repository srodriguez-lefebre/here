from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Annotated

import typer
from loguru import logger

from here.config.settings import get_settings
from here.live_processing import LiveTranscriptionController
from here.output.session_writer import TRANSCRIPT_ENCODING, write_session_artifacts
from here.recorder import RecordingSession, record_both_until_enter, record_mic_until_enter, record_os_until_enter
from here.transcription.client import TranscriptionResult
from here.transcriber import transcribe_recording_session

app = typer.Typer()
record_app = typer.Typer(invoke_without_command=True)
mic_app = typer.Typer(invoke_without_command=True)
os_app = typer.Typer(invoke_without_command=True)
app.add_typer(record_app, name="record")
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


@dataclass(slots=True)
class _TranscriptionOutcome:
    result: TranscriptionResult
    live_pipeline_attempted: bool
    live_pipeline_used: bool
    fallback_used: bool


def _resolve_target_dir(output_dir: Path | None) -> Path:
    return output_dir or get_settings().TRANSCRIPTIONS_DIR


def _save_transcription(
    session: RecordingSession,
    target_dir: Path,
    *,
    use_alt_transcription_model: bool = False,
    live_controller: LiveTranscriptionController | None = None,
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    should_cleanup = False

    try:
        outcome = _transcribe_session_outcome(
            session,
            use_alt_transcription_model=use_alt_transcription_model,
            live_controller=live_controller,
        )
        should_cleanup = True
    finally:
        if should_cleanup:
            session.cleanup()
            if live_controller is not None:
                live_controller.cleanup()
            logger.info("Temporary audio files deleted.")
        else:
            if live_controller is not None:
                live_controller.abort()
            logger.warning("Temporary audio files were preserved after transcription failure.")

    settings = get_settings()
    transcription_model = (
        settings.ALT_TRANSCRIPTION_MODEL
        if use_alt_transcription_model
        else settings.TRANSCRIPTION_MODEL
    )
    artifacts = write_session_artifacts(
        session=session,
        target_dir=target_dir,
        transcript_text=outcome.result.final_text,
        completed_at=datetime.now().astimezone(),
        transcription_model=transcription_model,
        cleanup_model=settings.CLEANUP_MODEL,
        cleanup_enabled=settings.CLEANUP_ENABLED,
        alt_model_used=use_alt_transcription_model,
        live_pipeline_attempted=outcome.live_pipeline_attempted,
        live_pipeline_used=outcome.live_pipeline_used,
        fallback_used=outcome.fallback_used,
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
    if live_controller is not None:
        try:
            logger.info("Completing live transcription from background chunks.")
            return _TranscriptionOutcome(
                result=live_controller.complete(),
                live_pipeline_attempted=True,
                live_pipeline_used=True,
                fallback_used=False,
            )
        except Exception as exc:
            logger.warning(
                "Live transcription failed: {exc}. Falling back to offline post-processing.",
                exc=exc,
            )

    return _TranscriptionOutcome(
        result=transcribe_recording_session(
            session,
            use_alt_transcription_model=use_alt_transcription_model,
        ),
        live_pipeline_attempted=live_controller is not None,
        live_pipeline_used=False,
        fallback_used=live_controller is not None,
    )


@app.callback()
def main() -> None:
    """here - record audio and transcribe it."""


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
