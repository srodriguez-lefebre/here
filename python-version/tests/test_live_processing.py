from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import here.live_processing as live_module
from here.audio.models import ChunkingConfig
from here.transcription.client import AudioTranscription, TranscriptionResult
from here.transcription.segments import TranscriptSegment


def test_live_transcription_controller_processes_chunks_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[Path] = []
    chunk_texts = iter(["chunk one", "chunk two"])

    monkeypatch.setattr(live_module, "build_client", lambda: object())
    monkeypatch.setattr(
        live_module,
        "resolve_transcription_models",
        lambda **kwargs: ("model", "cleanup-model", False),
    )
    monkeypatch.setattr(
        live_module,
        "finalize_transcription",
        lambda **kwargs: TranscriptionResult(raw_text=str(kwargs["raw_text"]), final_text=str(kwargs["raw_text"])),
    )

    def _transcribe_audio_file(client: object, audio_path: Path, model: str, *, prompt: str | None = None) -> str:
        del client, model, prompt
        calls.append(audio_path)
        return next(chunk_texts)

    monkeypatch.setattr(live_module, "transcribe_audio_file", _transcribe_audio_file)

    controller = live_module.LiveTranscriptionController(
        expected_source_count=1,
        chunking_config=ChunkingConfig(
            target_sample_rate=4,
            live_chunk_seconds=2,
            overlap_seconds=1,
            silence_search_seconds=0.0,
            silence_window_seconds=0.1,
            silence_candidate_step_seconds=0.01,
        ),
    )

    controller.submit_block("mic", np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float32), 4, 1)
    controller.submit_block("mic", np.array([0.4, 0.5, 0.6, 0.7], dtype=np.float32), 4, 1)
    controller.submit_block("mic", np.array([0.8, 0.9, 1.0, 1.1], dtype=np.float32), 4, 1)

    result = controller.complete()
    controller.cleanup()

    assert len(calls) == 2
    assert result.raw_text == "chunk one\nchunk two"


def test_live_transcription_controller_can_merge_two_sources(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transcribed: list[Path] = []

    monkeypatch.setattr(live_module, "build_client", lambda: object())
    monkeypatch.setattr(
        live_module,
        "resolve_transcription_models",
        lambda **kwargs: ("model", "cleanup-model", False),
    )
    monkeypatch.setattr(
        live_module,
        "finalize_transcription",
        lambda **kwargs: TranscriptionResult(raw_text=str(kwargs["raw_text"]), final_text=str(kwargs["raw_text"])),
    )
    monkeypatch.setattr(
        live_module,
        "transcribe_audio_file",
        lambda client, audio_path, model, *, prompt=None: transcribed.append(audio_path) or "merged chunk",
    )

    controller = live_module.LiveTranscriptionController(
        expected_source_count=2,
        chunking_config=ChunkingConfig(
            target_sample_rate=4,
            live_chunk_seconds=2,
            overlap_seconds=1,
            silence_search_seconds=0.0,
            silence_window_seconds=0.1,
            silence_candidate_step_seconds=0.01,
        ),
    )

    first = np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float32)
    second = np.array([0.4, 0.5, 0.6, 0.7], dtype=np.float32)
    for block in (first, second):
        controller.submit_block("microphone", block, 4, 1)
        controller.submit_block("system audio", block, 4, 1)

    result = controller.complete()
    controller.cleanup()

    assert len(transcribed) == 1
    assert result.final_text == "merged chunk"


def test_live_transcription_controller_raises_on_background_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(live_module, "build_client", lambda: object())
    monkeypatch.setattr(
        live_module,
        "resolve_transcription_models",
        lambda **kwargs: ("model", "cleanup-model", False),
    )
    monkeypatch.setattr(
        live_module,
        "transcribe_audio_file",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    controller = live_module.LiveTranscriptionController(
        expected_source_count=1,
        chunking_config=ChunkingConfig(
            target_sample_rate=4,
            live_chunk_seconds=2,
            overlap_seconds=1,
            silence_search_seconds=0.0,
            silence_window_seconds=0.1,
            silence_candidate_step_seconds=0.01,
        ),
    )
    controller.submit_block("mic", np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float32), 4, 1)
    controller.submit_block("mic", np.array([0.4, 0.5, 0.6, 0.7], dtype=np.float32), 4, 1)

    with pytest.raises(RuntimeError, match="Live transcription failed"):
        controller.complete()

    controller.cleanup()


def test_live_transcription_controller_merges_timestamped_segments_across_overlap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
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
                    TranscriptSegment(text="Siguiente idea", start=0.8, end=1.6),
                ],
            ),
        ]
    )

    monkeypatch.setattr(live_module, "build_client", lambda: object())
    monkeypatch.setattr(
        live_module,
        "resolve_transcription_models",
        lambda **kwargs: ("model", "cleanup-model", False),
    )
    monkeypatch.setattr(
        live_module,
        "finalize_transcription",
        lambda **kwargs: TranscriptionResult(raw_text=str(kwargs["raw_text"]), final_text=str(kwargs["raw_text"])),
    )
    monkeypatch.setattr(
        live_module,
        "transcribe_audio_file",
        lambda *args, **kwargs: next(chunk_results),
    )

    controller = live_module.LiveTranscriptionController(
        expected_source_count=1,
        chunking_config=ChunkingConfig(
            target_sample_rate=4,
            live_chunk_seconds=2,
            overlap_seconds=1,
            silence_search_seconds=0.0,
            silence_window_seconds=0.1,
            silence_candidate_step_seconds=0.01,
        ),
    )

    controller.submit_block("mic", np.array([0.0, 0.1, 0.2, 0.3], dtype=np.float32), 4, 1)
    controller.submit_block("mic", np.array([0.4, 0.5, 0.6, 0.7], dtype=np.float32), 4, 1)
    controller.submit_block("mic", np.array([0.8, 0.9, 1.0, 1.1], dtype=np.float32), 4, 1)

    result = controller.complete()
    controller.cleanup()

    assert result.raw_text == "Hola mundo\nSiguiente idea"
