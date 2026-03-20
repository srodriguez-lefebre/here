from datetime import datetime
from pathlib import Path

import typer
from loguru import logger

from .config.settings import get_settings
from .recorder import record_until_enter
from .transcriber import transcribe

app = typer.Typer()


@app.callback()
def main() -> None:
    """here — record audio and transcribe its."""


@app.command()
def record(
    output_dir: Path = typer.Option(
        None,
        "--output-dir",
        "-o",
        help="Directory to save the transcription. Defaults to TRANSCRIPTIONS_DIR from settings.",
    ),
) -> None:
    """Record audio from the microphone and transcribe it with Whisper."""
    target_dir = output_dir or get_settings().TRANSCRIPTIONS_DIR
    target_dir.mkdir(parents=True, exist_ok=True)

    audio_path = record_until_enter()

    try:
        text = transcribe(audio_path)
    finally:
        audio_path.unlink(missing_ok=True)
        logger.info("Temporary audio file deleted.")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = target_dir / f"{timestamp}.txt"
    output_file.write_text(text, encoding="utf-8")

    logger.success("Saved to {path}", path=output_file)
