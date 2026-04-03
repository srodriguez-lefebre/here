from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from openai import OpenAI

from here.config.settings import get_settings


@dataclass(slots=True)
class TranscriptionResult:
    raw_text: str
    final_text: str


def _value_from_payload(payload: object, key: str) -> object | None:
    if isinstance(payload, Mapping):
        return payload.get(key)

    return getattr(payload, key, None)


def extract_transcript_text(payload: object) -> str:
    """Extracts transcript text from different OpenAI response payload shapes."""
    text = _value_from_payload(payload, "text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    segments = _value_from_payload(payload, "segments")
    if isinstance(segments, list):
        lines: list[str] = []
        for segment in segments:
            segment_text = _value_from_payload(segment, "text")
            if not isinstance(segment_text, str) or not segment_text.strip():
                continue

            speaker = _value_from_payload(segment, "speaker")
            if isinstance(speaker, str) and speaker.strip():
                lines.append(f"{speaker.strip()}: {segment_text.strip()}")
            else:
                lines.append(segment_text.strip())

        if lines:
            return "\n".join(lines)

    raise ValueError("No transcript text found in transcription response")


def _extract_cleanup_text(payload: object) -> str:
    choices = _value_from_payload(payload, "choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("No cleanup completion choices returned")

    first_choice = choices[0]
    message = _value_from_payload(first_choice, "message")
    content = _value_from_payload(message, "content")
    if isinstance(content, str) and content.strip():
        return content.strip()

    raise ValueError("No cleanup text found in completion response")


def _transcribe_audio_file(client: OpenAI, audio_path: Path, model: str) -> str:
    logger.info("Uploading audio with model {model}", model=model)

    with audio_path.open("rb") as audio_file:
        response = client.audio.transcriptions.create(
            model=model,
            file=audio_file,
            response_format="json",
        )

    logger.info("Transcription response received.")
    return extract_transcript_text(response)


def _cleanup_transcript(client: OpenAI, raw_text: str, model: str) -> str:
    logger.info("Cleaning transcript with model {model}", model=model)

    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Reestructura esta transcripcion a texto claro y legible en espanol. "
                    "Corrige puntuacion y saltos de linea, sin inventar contenido nuevo."
                ),
            },
            {"role": "user", "content": raw_text},
        ],
    )

    logger.info("Cleanup response received.")
    return _extract_cleanup_text(completion)


def transcribe(
    audio_path: Path,
    *,
    transcription_model: str | None = None,
    cleanup_model: str | None = None,
    skip_cleanup: bool = False,
) -> TranscriptionResult:
    """
    Transcribes a WAV file and optionally cleans up the text.

    Args:
        audio_path: Path to the WAV audio file.
        transcription_model: Optional override for the transcription model.
        cleanup_model: Optional override for the cleanup model.
        skip_cleanup: When True, returns the raw transcript without cleanup.

    Returns:
        Raw and final transcription text.
    """
    settings = get_settings()
    client = OpenAI(api_key=settings.OPENAI_API_KEY.get_secret_value())
    resolved_transcription_model = transcription_model or settings.TRANSCRIPTION_MODEL
    resolved_cleanup_model = cleanup_model or settings.CLEANUP_MODEL
    should_cleanup = settings.CLEANUP_ENABLED and not skip_cleanup

    logger.info("Transcribing {path}...", path=audio_path.name)

    try:
        raw_text = _transcribe_audio_file(client, audio_path, resolved_transcription_model)
        final_text = (
            _cleanup_transcript(client, raw_text, resolved_cleanup_model) if should_cleanup else raw_text
        )
    except Exception as exc:
        logger.error("Transcription failed: {exc}", exc=exc)
        raise RuntimeError("Transcription failed") from exc

    logger.success("Transcription complete.")
    return TranscriptionResult(raw_text=raw_text, final_text=final_text)
