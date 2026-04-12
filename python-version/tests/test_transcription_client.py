from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import here.transcription.client as client_module
from here.transcription.client import (
    AudioTranscription,
    cleanup_transcript,
    extract_transcript_text,
    finalize_transcription,
    model_supports_prompt,
    resolve_transcription_models,
    transcribe_audio_file,
)


def test_extract_transcript_text_prefers_speaker_labeled_segments() -> None:
    payload = {
        "segments": [
            {"speaker": "Speaker 1", "text": "Hola"},
            {"speaker": "Speaker 2", "text": "Chau"},
        ]
    }

    assert extract_transcript_text(payload) == "Speaker 1: Hola\nSpeaker 2: Chau"


def test_extract_transcript_text_falls_back_to_text_field() -> None:
    payload = SimpleNamespace(text="Solo texto")

    assert extract_transcript_text(payload) == "Solo texto"


def test_extract_transcript_text_raises_when_payload_has_no_text() -> None:
    with pytest.raises(ValueError, match="No transcript text found"):
        extract_transcript_text({})


def test_resolve_transcription_models_respects_settings_and_skip_cleanup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        client_module,
        "get_settings",
        lambda: SimpleNamespace(
            TRANSCRIPTION_MODEL="default-transcribe",
            ALT_TRANSCRIPTION_MODEL="alt-transcribe",
            CLEANUP_MODEL="default-cleanup",
            CLEANUP_ENABLED=True,
        ),
    )

    models = resolve_transcription_models(
        transcription_model=None,
        cleanup_model=None,
        skip_cleanup=True,
    )

    assert models == ("default-transcribe", "default-cleanup", False)


def test_resolve_transcription_models_can_use_alt_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        client_module,
        "get_settings",
        lambda: SimpleNamespace(
            TRANSCRIPTION_MODEL="default-transcribe",
            ALT_TRANSCRIPTION_MODEL="alt-transcribe",
            CLEANUP_MODEL="default-cleanup",
            CLEANUP_ENABLED=False,
        ),
    )

    models = resolve_transcription_models(
        transcription_model=None,
        cleanup_model=None,
        skip_cleanup=False,
        use_alt_transcription_model=True,
    )

    assert models == ("alt-transcribe", "default-cleanup", False)


def test_model_supports_prompt_is_false_for_diarized_models() -> None:
    assert model_supports_prompt("gpt-4o-transcribe")
    assert not model_supports_prompt("gpt-4o-transcribe-diarize")


def test_transcribe_audio_file_passes_prompt_for_non_diarized_models(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")
    captured: dict[str, object] = {}

    class _Transcriptions:
        def create(self, **kwargs: object) -> dict[str, str]:
            captured.update(kwargs)
            return {"text": "done"}

    client = SimpleNamespace(audio=SimpleNamespace(transcriptions=_Transcriptions()))

    result = transcribe_audio_file(client, audio_path, "gpt-4o-transcribe", prompt="tail context")

    assert result == AudioTranscription(text="done", segments=[])
    assert captured["response_format"] == "verbose_json"
    assert captured["timestamp_granularities"] == ["segment"]
    assert captured["prompt"] == "tail context"
    assert "chunking_strategy" not in captured
    assert Path(captured["file"].name) == audio_path


def test_transcribe_audio_file_uses_diarized_request_shape(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")
    captured: dict[str, object] = {}

    class _Transcriptions:
        def create(self, **kwargs: object) -> dict[str, str]:
            captured.update(kwargs)
            return {"text": "done"}

    client = SimpleNamespace(audio=SimpleNamespace(transcriptions=_Transcriptions()))

    transcribe_audio_file(client, audio_path, "gpt-4o-transcribe-diarize", prompt="ignored prompt")

    assert captured["response_format"] == "diarized_json"
    assert captured["chunking_strategy"] == "auto"
    assert "prompt" not in captured


def test_cleanup_transcript_extracts_completion_text() -> None:
    captured: dict[str, object] = {}

    class _Completions:
        def create(self, **kwargs: object) -> dict[str, object]:
            captured.update(kwargs)
            return {"choices": [{"message": {"content": "clean transcript"}}]}

    client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions()))

    result = cleanup_transcript(client, "raw transcript", "cleanup-model")

    assert result == "clean transcript"
    assert captured["model"] == "cleanup-model"
    assert captured["temperature"] == 0
    assert captured["messages"][1]["content"] == "raw transcript"


def test_finalize_transcription_uses_cleanup_only_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client_module, "cleanup_transcript", lambda *args, **kwargs: "cleaned")

    result = finalize_transcription(
        client=object(),
        raw_text="raw",
        cleanup_model="cleanup-model",
        should_cleanup=True,
    )

    assert result.raw_text == "raw"
    assert result.final_text == "cleaned"
