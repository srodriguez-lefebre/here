from collections.abc import Callable
from datetime import datetime
from pathlib import Path

import typer
from loguru import logger

from here.config.settings import get_settings
from here.recorder import RecordingSession, record_both_until_enter, record_mic_until_enter, record_os_until_enter
from here.transcriber import transcribe_recording_session

app = typer.Typer()
record_app = typer.Typer(invoke_without_command=True)
mic_app = typer.Typer(invoke_without_command=True)
os_app = typer.Typer(invoke_without_command=True)
app.add_typer(record_app, name="record")
record_app.add_typer(mic_app, name="mic")
record_app.add_typer(os_app, name="os")
TRANSCRIPT_ENCODING = "utf-8-sig"


def _resolve_target_dir(output_dir: Path | None) -> Path:
    return output_dir or get_settings().TRANSCRIPTIONS_DIR


def _save_transcription(
    session: RecordingSession,
    target_dir: Path,
    *,
    use_alt_transcription_model: bool = False,
) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = transcribe_recording_session(
            session,
            use_alt_transcription_model=use_alt_transcription_model,
        )
    finally:
        session.cleanup()
        logger.info("Temporary audio files deleted.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = target_dir / f"{timestamp}.txt"
    output_file.write_text(result.final_text, encoding=TRANSCRIPT_ENCODING)
    logger.success("Saved to {path}", path=output_file)


def _run_recording(
    capture_fn: Callable[[], RecordingSession],
    target_dir: Path,
    *,
    use_alt_transcription_model: bool = False,
) -> None:
    try:
        session = capture_fn()
        _save_transcription(
            session,
            target_dir,
            use_alt_transcription_model=use_alt_transcription_model,
        )
    except RuntimeError as exc:
        logger.error(str(exc))
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        logger.exception("Unexpected error while recording or transcribing audio")
        raise typer.Exit(code=1) from exc


@app.callback()
def main() -> None:
    """here - record audio and transcribe it."""


@record_app.callback(invoke_without_command=True)
def record_main(
    ctx: typer.Context,
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory to save the transcription. Defaults to TRANSCRIPTIONS_DIR from settings.",
    ),
) -> None:
    """Record audio from different sources and transcribe it."""
    if ctx.invoked_subcommand is not None:
        return

    _run_recording(record_both_until_enter, _resolve_target_dir(output_dir))


@record_app.command("alt")
def record_alt(
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory to save the transcription. Defaults to TRANSCRIPTIONS_DIR from settings.",
    ),
) -> None:
    """Record microphone + system audio with the alternate transcription model."""
    _run_recording(
        record_both_until_enter,
        _resolve_target_dir(output_dir),
        use_alt_transcription_model=True,
    )


@mic_app.callback(invoke_without_command=True)
def mic_main(
    ctx: typer.Context,
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory to save the transcription. Defaults to TRANSCRIPTIONS_DIR from settings.",
    ),
) -> None:
    """Record audio from the microphone only."""
    if ctx.invoked_subcommand is not None:
        return

    _run_recording(record_mic_until_enter, _resolve_target_dir(output_dir))


@mic_app.command("alt")
def mic_alt(
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory to save the transcription. Defaults to TRANSCRIPTIONS_DIR from settings.",
    ),
) -> None:
    """Record microphone audio with the alternate transcription model."""
    _run_recording(
        record_mic_until_enter,
        _resolve_target_dir(output_dir),
        use_alt_transcription_model=True,
    )


@os_app.callback(invoke_without_command=True)
def os_main(
    ctx: typer.Context,
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory to save the transcription. Defaults to TRANSCRIPTIONS_DIR from settings.",
    ),
) -> None:
    """Record system audio only."""
    if ctx.invoked_subcommand is not None:
        return

    _run_recording(record_os_until_enter, _resolve_target_dir(output_dir))


@os_app.command("alt")
def os_alt(
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory to save the transcription. Defaults to TRANSCRIPTIONS_DIR from settings.",
    ),
) -> None:
    """Record system audio with the alternate transcription model."""
    _run_recording(
        record_os_until_enter,
        _resolve_target_dir(output_dir),
        use_alt_transcription_model=True,
    )
