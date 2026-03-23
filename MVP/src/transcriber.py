from pathlib import Path

from loguru import logger
from openai import OpenAI

from here.config.settings import get_settings


def transcribe(audio_path: Path) -> str:
    """
    Transcribes a WAV file using OpenAI Whisper.

    Args:
        audio_path: Path to the WAV audio file.

    Returns:
        Transcribed text.
    """
    client = OpenAI(api_key=get_settings().OPENAI_API_KEY)

    logger.info("Transcribing {path}...", path=audio_path.name)

    try:
        with open(audio_path, "rb") as f:
            result = client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
            )
    except Exception as exc:
        logger.error("Transcription failed: {exc}", exc=exc)
        raise RuntimeError("Transcription failed") from exc

    logger.success("Transcription complete.")
    return result.text
