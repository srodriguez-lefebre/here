from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from here.audio.chunking import (
    build_chunk_prompt,
    dedupe_overlap,
    merge_transcript_parts,
    plan_chunk_windows,
    render_chunk_window,
)
from here.audio.models import ChunkWindow, ChunkingConfig
from here.recording.models import RecordedAudioSource, RecordingSession


def _write_source(
    tmp_path: Path,
    name: str,
    data: np.ndarray,
    *,
    sample_rate: int,
) -> RecordedAudioSource:
    path = tmp_path / name
    sf.write(path, data, sample_rate, subtype="PCM_16")
    channels = data.shape[1] if data.ndim == 2 else 1
    return RecordedAudioSource(
        path=path,
        sample_rate=sample_rate,
        channels=channels,
        frames=int(data.shape[0]),
        label=name,
    )


def test_plan_chunk_windows_applies_overlap_and_indices(tmp_path: Path) -> None:
    source = _write_source(
        tmp_path,
        "source.wav",
        np.zeros(7, dtype=np.float32),
        sample_rate=2,
    )
    session = RecordingSession(sources=[source])
    config = ChunkingConfig(target_sample_rate=2, max_chunk_bytes=8, overlap_seconds=1)

    windows = plan_chunk_windows(session, config)

    assert [(window.index, window.start_frame, window.end_frame) for window in windows] == [
        (1, 0, 4),
        (2, 2, 6),
        (3, 4, 7),
    ]


def test_build_chunk_prompt_returns_only_the_tail_words() -> None:
    text = "uno dos tres cuatro cinco"

    assert build_chunk_prompt(text, max_words=3) == "tres cuatro cinco"
    assert build_chunk_prompt("  ", max_words=3) is None


def test_dedupe_overlap_trims_matching_prefix_after_normalization() -> None:
    config = ChunkingConfig(dedupe_min_words=4, dedupe_max_words=10)
    previous = "Hola, esto es una prueba de overlap muy util"
    next_text = "hola esto es una prueba de overlap muy util y sigue"

    merged_tail = dedupe_overlap(previous, next_text, config)

    assert merged_tail == "y sigue"


def test_merge_transcript_parts_skips_empty_parts_and_dedupes() -> None:
    config = ChunkingConfig(dedupe_min_words=3, dedupe_max_words=10)

    merged = merge_transcript_parts(
        [
            "",
            "alpha beta gamma delta epsilon",
            "gamma delta epsilon zeta eta",
            "   ",
        ],
        config,
    )

    assert merged == "alpha beta gamma delta epsilon\nzeta eta"


def test_render_chunk_window_writes_expected_frame_slice(tmp_path: Path) -> None:
    source = _write_source(
        tmp_path,
        "slice.wav",
        np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
        sample_rate=4,
    )
    session = RecordingSession(sources=[source])
    window = ChunkWindow(index=1, start_frame=1, end_frame=3)
    config = ChunkingConfig(target_sample_rate=4, process_block_seconds=1)

    chunk_path = render_chunk_window(session, window, tmp_path / "chunks", config)

    rendered_audio, sample_rate = sf.read(chunk_path, dtype="float32")
    assert sample_rate == 4
    assert rendered_audio.shape[0] == 2
    assert rendered_audio.tolist() == [pytest.approx(0.2, abs=0.02), pytest.approx(0.3, abs=0.02)]
