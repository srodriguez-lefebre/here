from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from here.audio.mix import (
    downmix_to_mono,
    fit_audio_to_target_frames,
    get_total_target_frames,
    materialize_normalized_session,
    mix_audio_blocks,
    resample_mono_audio,
)
from here.audio.models import ChunkingConfig
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


def test_downmix_to_mono_averages_channels() -> None:
    stereo = np.array([[0.0, 0.4], [0.2, 0.6]], dtype=np.float32)

    assert downmix_to_mono(stereo).tolist() == pytest.approx([0.2, 0.4])


def test_resample_mono_audio_repeats_single_frame_to_target_length() -> None:
    source = np.array([0.25], dtype=np.float32)

    resampled = resample_mono_audio(source, source_rate=1, target_rate=4)

    assert resampled.tolist() == pytest.approx([0.25, 0.25, 0.25, 0.25])


def test_fit_audio_to_target_frames_trims_or_pads() -> None:
    assert fit_audio_to_target_frames(np.array([1.0, 2.0, 3.0], dtype=np.float32), 2).tolist() == [
        1.0,
        2.0,
    ]
    assert fit_audio_to_target_frames(np.array([1.0], dtype=np.float32), 3).tolist() == [1.0, 0.0, 0.0]


def test_mix_audio_blocks_normalizes_peak_when_sources_clip() -> None:
    mixed = mix_audio_blocks(
        [
            np.array([0.8, 0.8], dtype=np.float32),
            np.array([0.8, 0.8], dtype=np.float32),
        ],
        target_frames=2,
    )

    assert float(np.max(np.abs(mixed))) == pytest.approx(1.0)


def test_get_total_target_frames_uses_longest_resampled_source(tmp_path: Path) -> None:
    first = _write_source(tmp_path, "first.wav", np.zeros(4, dtype=np.float32), sample_rate=4)
    second = _write_source(tmp_path, "second.wav", np.zeros(3, dtype=np.float32), sample_rate=2)
    session = RecordingSession(sources=[first, second])

    total_frames = get_total_target_frames(session, ChunkingConfig(target_sample_rate=4))

    assert total_frames == 6


def test_materialize_normalized_session_creates_single_mono_source(tmp_path: Path) -> None:
    source = _write_source(
        tmp_path,
        "input.wav",
        np.array([0.0, 0.5, -0.5, 0.25], dtype=np.float32),
        sample_rate=4,
    )
    session = RecordingSession(sources=[source])
    config = ChunkingConfig(target_sample_rate=4, process_block_seconds=1)

    normalized_session = materialize_normalized_session(session, tmp_path / "working", config)

    assert len(normalized_session.sources) == 1
    normalized_source = normalized_session.sources[0]
    assert normalized_source.label == "normalized-mix"
    assert normalized_source.sample_rate == 4
    assert normalized_source.channels == 1
    assert normalized_source.frames == 4
    assert normalized_source.path.exists()

    normalized_audio, sample_rate = sf.read(normalized_source.path, dtype="float32")
    assert sample_rate == 4
    assert normalized_audio.shape[0] == 4
    assert 0.35 <= float(np.max(np.abs(normalized_audio))) <= 0.45

    normalized_session.cleanup()
