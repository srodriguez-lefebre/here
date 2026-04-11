from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import here.transcription.service as service_module
from here.audio.models import ChunkWindow
from here.recording.models import RecordingSession
from here.transcription.client import TranscriptionResult


def test_transcribe_returns_finalized_result(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")
    client = object()

    monkeypatch.setattr(service_module, "build_client", lambda: client)
    monkeypatch.setattr(
        service_module,
        "resolve_transcription_models",
        lambda **kwargs: ("transcribe-model", "cleanup-model", True),
    )
    monkeypatch.setattr(service_module, "transcribe_audio_file", lambda *args, **kwargs: "raw transcript")
    monkeypatch.setattr(
        service_module,
        "finalize_transcription",
        lambda **kwargs: TranscriptionResult(raw_text=kwargs["raw_text"], final_text="clean transcript"),
    )

    result = service_module.transcribe(audio_path, prompt="carryover")

    assert result.raw_text == "raw transcript"
    assert result.final_text == "clean transcript"


def test_transcribe_wraps_transcription_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")

    monkeypatch.setattr(service_module, "build_client", lambda: object())
    monkeypatch.setattr(
        service_module,
        "resolve_transcription_models",
        lambda **kwargs: ("transcribe-model", "cleanup-model", False),
    )
    monkeypatch.setattr(
        service_module,
        "transcribe_audio_file",
        lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("boom")),
    )

    with pytest.raises(RuntimeError, match="Transcription failed"):
        service_module.transcribe(audio_path)


def test_transcribe_recording_session_processes_chunks_in_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    session = RecordingSession(sources=[])
    normalized_session = RecordingSession(sources=[])
    windows = [
        ChunkWindow(index=1, start_frame=0, end_frame=10),
        ChunkWindow(index=2, start_frame=8, end_frame=20),
    ]
    prompts: list[str | None] = []
    rendered_paths: list[Path] = []
    chunk_texts = iter(["chunk one", "chunk two"])
    finalized: dict[str, str] = {}

    monkeypatch.setattr(service_module, "build_client", lambda: object())
    monkeypatch.setattr(
        service_module,
        "resolve_transcription_models",
        lambda **kwargs: ("transcribe-model", "cleanup-model", False),
    )
    monkeypatch.setattr(service_module, "materialize_normalized_session", lambda *args: normalized_session)
    monkeypatch.setattr(service_module, "plan_chunk_windows", lambda *args: windows)
    monkeypatch.setattr(service_module, "model_supports_prompt", lambda model: True)
    monkeypatch.setattr(
        service_module,
        "build_chunk_prompt",
        lambda text, max_words: None if not text else f"prompt::{text}",
    )

    def _render_chunk(*args: object, **kwargs: object) -> Path:
        window = args[1]
        working_dir = args[2]
        path = working_dir / f"chunk_{window.index}.wav"
        path.write_bytes(b"chunk")
        rendered_paths.append(path)
        return path

    def _transcribe_audio_file(*args: object, **kwargs: object) -> str:
        prompts.append(kwargs.get("prompt"))
        return next(chunk_texts)

    def _merge_transcript_parts(parts: list[str], config: object) -> str:
        del config
        return " | ".join(part for part in parts if part)

    def _finalize_transcription(**kwargs: object) -> TranscriptionResult:
        finalized["raw_text"] = str(kwargs["raw_text"])
        return TranscriptionResult(raw_text=str(kwargs["raw_text"]), final_text="final transcript")

    monkeypatch.setattr(service_module, "render_chunk_window", _render_chunk)
    monkeypatch.setattr(service_module, "transcribe_audio_file", _transcribe_audio_file)
    monkeypatch.setattr(service_module, "merge_transcript_parts", _merge_transcript_parts)
    monkeypatch.setattr(service_module, "finalize_transcription", _finalize_transcription)

    result = service_module.transcribe_recording_session(session)

    assert prompts == [None, "prompt::chunk one"]
    assert finalized["raw_text"] == "chunk one | chunk two"
    assert result.final_text == "final transcript"
    assert all(not path.exists() for path in rendered_paths)


def test_transcribe_recording_session_raises_when_no_chunk_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = RecordingSession(sources=[])

    monkeypatch.setattr(service_module, "build_client", lambda: object())
    monkeypatch.setattr(
        service_module,
        "resolve_transcription_models",
        lambda **kwargs: ("transcribe-model", "cleanup-model", False),
    )
    monkeypatch.setattr(service_module, "materialize_normalized_session", lambda *args: RecordingSession(sources=[]))
    monkeypatch.setattr(service_module, "plan_chunk_windows", lambda *args: [])

    with pytest.raises(RuntimeError, match="Transcription failed"):
        service_module.transcribe_recording_session(session)
