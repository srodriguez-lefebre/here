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
    segments = _value_from_payload(payload, "segments")
    if isinstance(segments, list):
        lines: list[str] = []
        has_speaker_labels = False
        for segment in segments:
            segment_text = _value_from_payload(segment, "text")
            if not isinstance(segment_text, str) or not segment_text.strip():
                continue

            speaker = _value_from_payload(segment, "speaker")
            if isinstance(speaker, str) and speaker.strip():
                has_speaker_labels = True
                lines.append(f"{speaker.strip()}: {segment_text.strip()}")
            else:
                lines.append(segment_text.strip())

        if lines and has_speaker_labels:
            return "\n".join(lines)

    text = _value_from_payload(payload, "text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    if isinstance(segments, list):
        lines = []
        for segment in segments:
            segment_text = _value_from_payload(segment, "text")
            if isinstance(segment_text, str) and segment_text.strip():
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


def build_client() -> OpenAI:
    settings = get_settings()
    return OpenAI(api_key=settings.OPENAI_API_KEY.get_secret_value())


def resolve_transcription_models(
    *,
    transcription_model: str | None,
    cleanup_model: str | None,
    skip_cleanup: bool,
) -> tuple[str, str, bool]:
    settings = get_settings()
    resolved_transcription_model = transcription_model or settings.TRANSCRIPTION_MODEL
    resolved_cleanup_model = cleanup_model or settings.CLEANUP_MODEL
    should_cleanup = settings.CLEANUP_ENABLED and not skip_cleanup
    return resolved_transcription_model, resolved_cleanup_model, should_cleanup


def _uses_diarized_output(model: str) -> bool:
    return model.casefold().endswith("-diarize")


def model_supports_prompt(model: str) -> bool:
    return not _uses_diarized_output(model)


def transcribe_audio_file(
    client: OpenAI,
    audio_path: Path,
    model: str,
    *,
    prompt: str | None = None,
) -> str:
    logger.info("Uploading audio with model {model}", model=model)

    use_diarized_output = _uses_diarized_output(model)
    request: dict[str, object] = {
        "model": model,
        "response_format": "diarized_json" if use_diarized_output else "json",
    }
    if use_diarized_output:
        request["chunking_strategy"] = "auto"
    if prompt and not use_diarized_output:
        request["prompt"] = prompt

    with audio_path.open("rb") as audio_file:
        request["file"] = audio_file
        response = client.audio.transcriptions.create(**request)

    logger.info("Transcription response received.")
    return extract_transcript_text(response)


def cleanup_transcript(client: OpenAI, raw_text: str, model: str) -> str:
    logger.info("Cleaning transcript with model {model}", model=model)

    completion = client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {
                "role": "system",
                "content": (
                    "Rewrite this transcript into clear, readable text while preserving its "
                    "original language. Correct punctuation and line breaks without "
                    "inventing new content."
                ),
            },
            {"role": "user", "content": raw_text},
        ],
    )

    logger.info("Cleanup response received.")
    return _extract_cleanup_text(completion)


def finalize_transcription(
    *,
    client: OpenAI,
    raw_text: str,
    cleanup_model: str,
    should_cleanup: bool,
) -> TranscriptionResult:
    final_text = cleanup_transcript(client, raw_text, cleanup_model) if should_cleanup else raw_text
    logger.success("Transcription complete.")
    return TranscriptionResult(raw_text=raw_text, final_text=final_text)
