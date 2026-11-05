from __future__ import annotations

from pathlib import Path

import here.transcription.service as service_module
import numpy as np
import pytest
import soundfile as sf
from here.audio.models import ChunkingConfig, ChunkWindow
from here.recording.models import RecordedAudioSource, RecordingSession
from here.transcription.client import AudioTranscription, TranscriptionResult
from here.transcription.segments import TranscriptSegment


def test_transcribe_returns_finalized_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")
    client = object()

    monkeypatch.setattr(service_module, "build_client", lambda: client)
    monkeypatch.setattr(
        service_module,
        "resolve_transcription_models",
        lambda **kwargs: ("transcribe-model", "cleanup-model", True),
    )
    monkeypatch.setattr(
        service_module, "transcribe_audio_file", lambda *args, **kwargs: "raw transcript"
    )
    monkeypatch.setattr(
        service_module,
        "finalize_transcription",
        lambda **kwargs: TranscriptionResult(
            raw_text=kwargs["raw_text"], final_text="clean transcript"
        ),
    )

    result = service_module.transcribe(audio_path, prompt="carryover")

    assert result.raw_text == "raw transcript"
    assert result.final_text == "clean transcript"


def test_transcribe_wraps_transcription_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
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


def test_transcribe_passes_alt_flag_to_model_resolution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"audio")
    captured: dict[str, object] = {}

    monkeypatch.setattr(service_module, "build_client", lambda: object())

    def _resolve_transcription_models(**kwargs: object) -> tuple[str, str, bool]:
        captured.update(kwargs)
        return ("alt-model", "cleanup-model", False)

    monkeypatch.setattr(
        service_module, "resolve_transcription_models", _resolve_transcription_models
    )
    monkeypatch.setattr(
        service_module, "transcribe_audio_file", lambda *args, **kwargs: "raw transcript"
    )
    monkeypatch.setattr(
        service_module,
        "finalize_transcription",
        lambda **kwargs: TranscriptionResult(
            raw_text=kwargs["raw_text"], final_text="raw transcript"
        ),
    )

    service_module.transcribe(audio_path, use_alt_transcription_model=True)

    assert captured["use_alt_transcription_model"] is True


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
    finalized: dict[str, object] = {}

    monkeypatch.setattr(service_module, "build_client", lambda: object())
    monkeypatch.setattr(
        service_module,
        "resolve_transcription_models",
        lambda **kwargs: ("transcribe-model", "cleanup-model", False),
    )
    monkeypatch.setattr(
        service_module, "materialize_normalized_session", lambda *args: normalized_session
    )
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
        finalized["chunks"] = kwargs["chunks"]
        return TranscriptionResult(
            raw_text=str(kwargs["raw_text"]),
            final_text="final transcript",
            chunks=kwargs["chunks"],
        )

    monkeypatch.setattr(service_module, "render_chunk_window", _render_chunk)
    monkeypatch.setattr(service_module, "transcribe_audio_file", _transcribe_audio_file)
    monkeypatch.setattr(service_module, "merge_transcript_parts", _merge_transcript_parts)
    monkeypatch.setattr(service_module, "finalize_transcription", _finalize_transcription)

    result = service_module.transcribe_recording_session(session)

    assert prompts == [None, "prompt::chunk one"]
    assert finalized["raw_text"] == "chunk one | chunk two"
    assert result.final_text == "final transcript"
    assert len(result.chunks) == 2
    assert result.chunks[0].mode == "offline"
    assert result.chunks[0].index == 1
    assert result.chunks[0].start_seconds == 0.0
    assert result.chunks[0].end_seconds == 10 / 16000
    assert result.chunks[0].status == "completed"
    assert all(not path.exists() for path in rendered_paths)


def test_transcribe_recording_session_merges_timestamped_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = RecordingSession(sources=[])
    normalized_session = RecordingSession(sources=[])
    windows = [
        ChunkWindow(index=1, start_frame=0, end_frame=10),
        ChunkWindow(index=2, start_frame=8, end_frame=20),
    ]
    finalized: dict[str, str] = {}
    chunk_results = iter(
        [
            AudioTranscription(
                text="Hola mundo",
                segments=[TranscriptSegment(text="Hola mundo", start=0.0, end=1.5)],
            ),
            AudioTranscription(
                text="Hola mundo\nSiguiente idea",
                segments=[
                    TranscriptSegment(text="Hola mundo", start=0.0, end=0.8),
                    TranscriptSegment(text="Siguiente idea", start=0.8, end=1.5),
                ],
            ),
        ]
    )

    monkeypatch.setattr(service_module, "build_client", lambda: object())
    monkeypatch.setattr(
        service_module,
        "resolve_transcription_models",
        lambda **kwargs: ("transcribe-model", "cleanup-model", False),
    )
    monkeypatch.setattr(
        service_module, "materialize_normalized_session", lambda *args: normalized_session
    )
    monkeypatch.setattr(service_module, "plan_chunk_windows", lambda *args: windows)
    monkeypatch.setattr(service_module, "model_supports_prompt", lambda model: True)
    monkeypatch.setattr(
        service_module,
        "build_chunk_prompt",
        lambda text, max_words: None if not text else f"prompt::{text}",
    )
    monkeypatch.setattr(
        service_module,
        "render_chunk_window",
        lambda session, window, working_dir, config: working_dir / f"chunk_{window.index}.wav",
    )
    monkeypatch.setattr(
        service_module, "transcribe_audio_file", lambda *args, **kwargs: next(chunk_results)
    )

    def _finalize_transcription(**kwargs: object) -> TranscriptionResult:
        finalized["raw_text"] = str(kwargs["raw_text"])
        return TranscriptionResult(
            raw_text=str(kwargs["raw_text"]), final_text=str(kwargs["raw_text"])
        )

    monkeypatch.setattr(service_module, "finalize_transcription", _finalize_transcription)

    result = service_module.transcribe_recording_session(session)

    assert finalized["raw_text"] == "Hola mundo\nSiguiente idea"
    assert result.final_text == "Hola mundo\nSiguiente idea"


def test_transcribe_recording_session_raises_when_no_chunk_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    session = RecordingSession(sources=[])
    captured: dict[str, object] = {}

    monkeypatch.setattr(service_module, "build_client", lambda: object())

    def _resolve_transcription_models(**kwargs: object) -> tuple[str, str, bool]:
        captured.update(kwargs)
        return ("transcribe-model", "cleanup-model", False)

    monkeypatch.setattr(
        service_module, "resolve_transcription_models", _resolve_transcription_models
    )
    monkeypatch.setattr(
        service_module, "materialize_normalized_session", lambda *args: RecordingSession(sources=[])
    )
    monkeypatch.setattr(service_module, "plan_chunk_windows", lambda *args: [])

    with pytest.raises(RuntimeError, match="Transcription failed"):
        service_module.transcribe_recording_session(session, use_alt_transcription_model=True)

    assert captured["use_alt_transcription_model"] is True


@pytest.mark.parametrize("entrypoint", ["file", "recording"])
def test_text_only_provider_response_survives_as_untimed_segment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, entrypoint: str
) -> None:
    audio = tmp_path / "synthetic.wav"
    sf.write(audio, np.zeros(1600, dtype=np.float32), 16000)
    monkeypatch.setattr(service_module, "build_client", lambda: object())
    monkeypatch.setattr(
        service_module,
        "resolve_transcription_models",
        lambda **kwargs: ("synthetic-transcribe", "synthetic-cleanup", False),
    )
    monkeypatch.setattr(
        service_module, "transcribe_audio_file", lambda *args, **kwargs: "untimed words"
    )

    if entrypoint == "file":
        result = service_module.transcribe(audio)
    else:
        session = RecordingSession([RecordedAudioSource(audio, 16000, 1, 1600, "microphone")])
        result = service_module.transcribe_recording_session(session)

    assert result.raw_text == "untimed words"
    assert [segment.text for segment in result.segments] == ["untimed words"]
    assert (result.segments[0].start, result.segments[0].end) == (None, None)
    assert result.segments[0].speaker is None


def test_offline_mixed_timed_and_text_only_chunks_keep_every_ordinal(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    audio = tmp_path / "three-seconds.wav"
    sf.write(audio, np.zeros(3 * 16000, dtype=np.float32), 16000)
    session = RecordingSession([RecordedAudioSource(audio, 16000, 1, 3 * 16000, "microphone")])
    responses = iter(
        [
            AudioTranscription("timed first", [TranscriptSegment("timed first", 0.0, 0.5, "A")]),
            "untimed middle",
            AudioTranscription(
                "contradictory last", [TranscriptSegment("contradictory last", 0.5, 0.0, "A")]
            ),
        ]
    )
    monkeypatch.setattr(service_module, "build_client", lambda: object())
    monkeypatch.setattr(
        service_module,
        "resolve_transcription_models",
        lambda **kwargs: ("synthetic-transcribe", "synthetic-cleanup", False),
    )
    monkeypatch.setattr(
        service_module, "transcribe_audio_file", lambda *args, **kwargs: next(responses)
    )

    result = service_module.transcribe_recording_session(
        session,
        chunking_config=ChunkingConfig(max_chunk_bytes=32000, overlap_seconds=0),
    )

    assert [segment.text for segment in result.segments] == [
        "timed first",
        "untimed middle",
        "contradictory last",
    ]
    assert [segment.chunk_index for segment in result.segments] == [1, 2, 3]
    assert [(segment.start, segment.end) for segment in result.segments] == [
        (0.0, 0.5),
        (None, None),
        (None, None),
    ]
    assert [segment.speaker_scope for segment in result.segments] == [
        "chunk:1",
        "chunk:2",
        "chunk:3",
    ]
