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
app.add_typer(record_app, name="record")
TRANSCRIPT_ENCODING = "utf-8-sig"


def _save_transcription(session: RecordingSession, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)

    try:
        result = transcribe_recording_session(session)
    finally:
        session.cleanup()
        logger.info("Temporary audio files deleted.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = target_dir / f"{timestamp}.txt"
    output_file.write_text(result.final_text, encoding=TRANSCRIPT_ENCODING)
    logger.success("Saved to {path}", path=output_file)


def _run_recording(capture_fn: Callable[[], RecordingSession], target_dir: Path) -> None:
    try:
        session = capture_fn()
        _save_transcription(session, target_dir)
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

    target_dir = output_dir or get_settings().TRANSCRIPTIONS_DIR
    _run_recording(record_both_until_enter, target_dir)


@record_app.command()
def mic(
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory to save the transcription. Defaults to TRANSCRIPTIONS_DIR from settings.",
    ),
) -> None:
    """Record audio from the microphone only."""
    target_dir = output_dir or get_settings().TRANSCRIPTIONS_DIR
    _run_recording(record_mic_until_enter, target_dir)


@record_app.command()
def os(
    output_dir: Path | None = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory to save the transcription. Defaults to TRANSCRIPTIONS_DIR from settings.",
    ),
) -> None:
    """Record system audio only."""
    target_dir = output_dir or get_settings().TRANSCRIPTIONS_DIR
    _run_recording(record_os_until_enter, target_dir)
