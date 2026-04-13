from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from loguru import logger
from openai import OpenAI

from here.config.settings import get_settings
from here.transcription.segments import TranscriptSegment, parse_transcript_segments


@dataclass(slots=True)
class TranscriptionResult:
    raw_text: str
    final_text: str


@dataclass(slots=True)
class AudioTranscription:
    text: str
    segments: list[TranscriptSegment]


def coerce_audio_transcription(value: AudioTranscription | str) -> AudioTranscription:
    if isinstance(value, AudioTranscription):
        return value
    if isinstance(value, str):
        return AudioTranscription(text=value, segments=[])
    raise TypeError("Unsupported transcription payload")


def _value_from_payload(payload: object, key: str) -> object | None:
    if isinstance(payload, Mapping):
        return payload.get(key)
    return getattr(payload, key, None)


def format_transcript_segments(segments: list[TranscriptSegment]) -> str:
    lines: list[str] = []
    for segment in segments:
        segment_text = segment.text.strip()
        if not segment_text:
            continue
        if segment.speaker:
            lines.append(f"{segment.speaker}: {segment_text}")
        else:
            lines.append(segment_text)

    if not lines:
        return ""
    return "\n".join(lines)


def _extract_transcript_text_from_parts(
    payload: object,
    parsed_segments: list[TranscriptSegment],
) -> str:
    if parsed_segments:
        rendered_segments = format_transcript_segments(parsed_segments)
        if rendered_segments and any(segment.speaker for segment in parsed_segments):
            return rendered_segments

    text = _value_from_payload(payload, "text")
    if isinstance(text, str) and text.strip():
        return text.strip()

    if parsed_segments:
        rendered_segments = format_transcript_segments(parsed_segments)
        if rendered_segments:
            return rendered_segments

    raise ValueError("No transcript text found in transcription response")


def extract_transcript_text(payload: object) -> str:
    parsed_segments = parse_transcript_segments(payload)
    return _extract_transcript_text_from_parts(payload, parsed_segments)


def extract_audio_transcription(payload: object) -> AudioTranscription:
    parsed_segments = parse_transcript_segments(payload)
    return AudioTranscription(
        text=_extract_transcript_text_from_parts(payload, parsed_segments),
        segments=parsed_segments,
    )


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
    use_alt_transcription_model: bool = False,
) -> tuple[str, str, bool]:
    settings = get_settings()
    default_transcription_model = (
        settings.ALT_TRANSCRIPTION_MODEL if use_alt_transcription_model else settings.TRANSCRIPTION_MODEL
    )
    resolved_transcription_model = transcription_model or default_transcription_model
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
) -> AudioTranscription:
    logger.info("Uploading audio with model {model}", model=model)

    use_diarized_output = _uses_diarized_output(model)
    request: dict[str, object] = {
        "model": model,
        "response_format": "diarized_json" if use_diarized_output else "verbose_json",
    }
    if use_diarized_output:
        request["chunking_strategy"] = "auto"
    else:
        request["timestamp_granularities"] = ["segment"]
    if prompt and not use_diarized_output:
        request["prompt"] = prompt

    with audio_path.open("rb") as audio_file:
        request["file"] = audio_file
        response = client.audio.transcriptions.create(**request)

    logger.info("Transcription response received.")
    return extract_audio_transcription(response)


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
