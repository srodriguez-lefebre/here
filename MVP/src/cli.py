from datetime import datetime
from pathlib import Path

import typer
from loguru import logger

from here.config.settings import get_settings
from here.recorder import record_mic_until_enter, record_os_until_enter
from here.transcriber import transcribe

app = typer.Typer()
record_app = typer.Typer()
app.add_typer(record_app, name="record")


@app.callback()
def main() -> None:
    """here — record audio and transcribe it."""


@record_app.callback()
def record_main() -> None:
    """Record audio from different sources and transcribe with Whisper."""


@record_app.command()
def mic(
    output_dir: Path = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory to save the transcription. Defaults to TRANSCRIPTIONS_DIR from settings.",
    ),
) -> None:
    """Record audio from the microphone."""
    target_dir = output_dir or get_settings().TRANSCRIPTIONS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    audio_path = record_mic_until_enter()

    try:
        text = transcribe(audio_path)
    finally:
        audio_path.unlink(missing_ok=True)
        logger.info("Temporary audio file deleted.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = target_dir / f"{timestamp}.txt"
    output_file.write_text(text, encoding="utf-8")

    logger.success("Saved to {path}", path=output_file)


@record_app.command()
def os(
    output_dir: Path = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory to save the transcription. Defaults to TRANSCRIPTIONS_DIR from settings.",
    ),
) -> None:
    """Record system audio and transcribe it with Whisper."""
    target_dir = output_dir or get_settings().TRANSCRIPTIONS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    audio_path = record_os_until_enter()

    try:
        text = transcribe(audio_path)
    finally:
        audio_path.unlink(missing_ok=True)
        logger.info("Temporary audio file deleted.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = target_dir / f"{timestamp}.txt"
    output_file.write_text(text, encoding="utf-8")

    logger.success("Saved to {path}", path=output_file)
