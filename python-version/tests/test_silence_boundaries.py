from __future__ import annotations

import numpy as np

from here.audio.silence_boundaries import (
    choose_silence_cut_index,
    score_boundary_window,
    score_boundary_windows,
    to_mono_float32,
)


def test_score_boundary_window_prefers_silence_over_speech() -> None:
    sample_rate = 100
    audio = np.concatenate(
        [
            np.full(40, 0.4, dtype=np.float32),
            np.zeros(20, dtype=np.float32),
            np.full(40, 0.4, dtype=np.float32),
        ]
    )

    silence_score = score_boundary_window(audio, 50, sample_rate=sample_rate, analysis_window_seconds=0.1)
    speech_score = score_boundary_window(audio, 20, sample_rate=sample_rate, analysis_window_seconds=0.1)

    assert silence_score.silence_ratio > speech_score.silence_ratio
    assert silence_score.energy_score < speech_score.energy_score


def test_score_boundary_windows_scores_each_candidate() -> None:
    sample_rate = 10
    audio = np.concatenate([np.ones(10, dtype=np.float32), np.zeros(10, dtype=np.float32)])

    scores = score_boundary_windows(audio, [5, 10, 15], sample_rate=sample_rate, analysis_window_seconds=0.2)

    assert [score.index for score in scores] == [5, 10, 15]
    assert scores[1].silence_ratio >= scores[0].silence_ratio


def test_choose_silence_cut_index_snaps_to_silence_near_target() -> None:
    sample_rate = 100
    audio = np.concatenate(
        [
            np.full(40, 0.35, dtype=np.float32),
            np.zeros(20, dtype=np.float32),
            np.full(40, 0.35, dtype=np.float32),
        ]
    )

    cut_index = choose_silence_cut_index(
        audio,
        39,
        sample_rate=sample_rate,
        search_radius_seconds=1.0,
        analysis_window_seconds=0.1,
        candidate_step_seconds=0.01,
        silence_threshold=0.02,
    )

    assert cut_index == 50


def test_choose_silence_cut_index_falls_back_to_target_without_silence() -> None:
    sample_rate = 100
    audio = np.full(100, 0.2, dtype=np.float32)

    cut_index = choose_silence_cut_index(
        audio,
        55,
        sample_rate=sample_rate,
        search_radius_seconds=1.0,
        analysis_window_seconds=0.1,
        candidate_step_seconds=0.01,
        silence_threshold=0.02,
    )

    assert cut_index == 55


def test_choose_silence_cut_index_respects_min_index_guard() -> None:
    sample_rate = 100
    audio = np.concatenate(
        [
            np.full(40, 0.35, dtype=np.float32),
            np.zeros(20, dtype=np.float32),
            np.full(40, 0.35, dtype=np.float32),
        ]
    )

    cut_index = choose_silence_cut_index(
        audio,
        39,
        sample_rate=sample_rate,
        min_index=65,
        search_radius_seconds=1.0,
        analysis_window_seconds=0.1,
        candidate_step_seconds=0.01,
        silence_threshold=0.02,
    )

    assert cut_index == 65


def test_choose_silence_cut_index_handles_short_audio() -> None:
    sample_rate = 100
    audio = np.full(12, 0.2, dtype=np.float32)

    cut_index = choose_silence_cut_index(
        audio,
        50,
        sample_rate=sample_rate,
        search_radius_seconds=1.0,
        analysis_window_seconds=0.1,
        candidate_step_seconds=0.01,
    )

    assert cut_index == len(audio)


def test_to_mono_float32_collapses_multichannel_audio() -> None:
    stereo = np.array([[0.0, 1.0], [0.5, -0.5]], dtype=np.float32)

    mono = to_mono_float32(stereo)

    assert mono.dtype == np.float32
    assert mono.tolist() == [0.5, 0.0]
