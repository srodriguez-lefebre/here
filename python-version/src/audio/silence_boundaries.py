from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_ANALYSIS_WINDOW_SECONDS = 0.5
DEFAULT_CANDIDATE_STEP_SECONDS = 0.05
DEFAULT_SEARCH_RADIUS_SECONDS = 10.0
DEFAULT_SILENCE_BONUS_WEIGHT = 0.25
DEFAULT_DISTANCE_WEIGHT = 0.05
DEFAULT_SILENCE_THRESHOLD = 0.015


@dataclass(slots=True, frozen=True)
class BoundaryScore:
    index: int
    left_rms: float
    right_rms: float
    combined_rms: float
    silence_ratio: float
    energy_score: float


def to_mono_float32(audio: np.ndarray) -> np.ndarray:
    """Return audio as a contiguous mono float32 buffer."""
    samples = np.asarray(audio)

    if samples.ndim == 0:
        return np.asarray([samples.item()], dtype=np.float32)
    if samples.ndim == 1:
        return np.asarray(samples, dtype=np.float32)
    if samples.ndim == 2:
        return np.asarray(samples.mean(axis=-1), dtype=np.float32)

    raise ValueError("audio must be a mono or stereo-like one- or two-dimensional array")


def window_rms(samples: np.ndarray) -> float:
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float32), dtype=np.float32)))


def window_silence_ratio(samples: np.ndarray, silence_threshold: float) -> float:
    if samples.size == 0:
        return 1.0
    return float(np.mean(np.abs(samples) <= silence_threshold, dtype=np.float32))


def score_boundary_window(
    audio: np.ndarray,
    index: int,
    *,
    sample_rate: int,
    analysis_window_seconds: float = DEFAULT_ANALYSIS_WINDOW_SECONDS,
    silence_threshold: float = DEFAULT_SILENCE_THRESHOLD,
) -> BoundaryScore:
    """Score a candidate cut point using local energy and silence density."""
    mono_audio = to_mono_float32(audio)
    sample_rate = max(1, int(sample_rate))
    window_samples = max(1, int(round(analysis_window_seconds * sample_rate)))
    candidate_index = int(np.clip(index, 0, mono_audio.shape[0]))

    left = mono_audio[max(0, candidate_index - window_samples) : candidate_index]
    right = mono_audio[candidate_index : min(mono_audio.shape[0], candidate_index + window_samples)]
    combined = mono_audio[
        max(0, candidate_index - window_samples) : min(
            mono_audio.shape[0],
            candidate_index + window_samples,
        )
    ]

    left_rms = window_rms(left)
    right_rms = window_rms(right)
    combined_rms = window_rms(combined)
    silence_ratio = window_silence_ratio(combined, silence_threshold)
    energy_score = max(left_rms, right_rms, combined_rms) - (silence_ratio * DEFAULT_SILENCE_BONUS_WEIGHT)

    return BoundaryScore(
        index=candidate_index,
        left_rms=left_rms,
        right_rms=right_rms,
        combined_rms=combined_rms,
        silence_ratio=silence_ratio,
        energy_score=energy_score,
    )


def score_boundary_windows(
    audio: np.ndarray,
    indices: list[int] | tuple[int, ...],
    *,
    sample_rate: int,
    analysis_window_seconds: float = DEFAULT_ANALYSIS_WINDOW_SECONDS,
    silence_threshold: float = DEFAULT_SILENCE_THRESHOLD,
) -> list[BoundaryScore]:
    """Score multiple candidate cut points."""
    return [
        score_boundary_window(
            audio,
            index,
            sample_rate=sample_rate,
            analysis_window_seconds=analysis_window_seconds,
            silence_threshold=silence_threshold,
        )
        for index in indices
    ]


def choose_silence_cut_index(
    audio: np.ndarray,
    target_index: int,
    *,
    sample_rate: int,
    min_index: int = 0,
    max_index: int | None = None,
    search_radius_seconds: float = DEFAULT_SEARCH_RADIUS_SECONDS,
    analysis_window_seconds: float = DEFAULT_ANALYSIS_WINDOW_SECONDS,
    candidate_step_seconds: float = DEFAULT_CANDIDATE_STEP_SECONDS,
    silence_threshold: float = DEFAULT_SILENCE_THRESHOLD,
    distance_weight: float = DEFAULT_DISTANCE_WEIGHT,
) -> int:
    """Pick the best cut point near ``target_index``.

    The chooser prefers low-energy, silence-like windows and only moves away
    from the target when a quieter boundary is found inside the search radius.
    ``min_index`` can be used to keep enough fresh audio before a cut, which is
    how a live pipeline can account for overlap-based chunk spacing.
    """
    mono_audio = to_mono_float32(audio)
    sample_rate = max(1, int(sample_rate))
    total_frames = mono_audio.shape[0]
    if total_frames == 0:
        return 0

    inclusive_max_index = total_frames if max_index is None else int(np.clip(max_index, 0, total_frames))
    inclusive_min_index = int(np.clip(min_index, 0, inclusive_max_index))
    clamped_target = int(np.clip(target_index, inclusive_min_index, inclusive_max_index))

    search_radius_samples = max(1, int(round(search_radius_seconds * sample_rate)))
    candidate_step_samples = max(1, int(round(candidate_step_seconds * sample_rate)))

    search_start = max(inclusive_min_index, clamped_target - search_radius_samples)
    search_end = min(inclusive_max_index, clamped_target + search_radius_samples)
    if search_start >= search_end:
        return clamped_target

    candidate_indices = list(range(search_start, search_end + 1, candidate_step_samples))
    if candidate_indices[-1] != search_end:
        candidate_indices.append(search_end)
    if clamped_target not in candidate_indices:
        candidate_indices.append(clamped_target)
        candidate_indices.sort()

    scored_candidates = score_boundary_windows(
        mono_audio,
        candidate_indices,
        sample_rate=sample_rate,
        analysis_window_seconds=analysis_window_seconds,
        silence_threshold=silence_threshold,
    )

    best_candidate = min(
        scored_candidates,
        key=lambda candidate: (
            candidate.energy_score
            + (abs(candidate.index - clamped_target) / max(search_radius_samples, 1)) * distance_weight,
            abs(candidate.index - clamped_target),
            candidate.index,
        ),
    )

    fine_start = max(search_start, best_candidate.index - candidate_step_samples)
    fine_end = min(search_end, best_candidate.index + candidate_step_samples)
    if fine_start >= fine_end:
        return best_candidate.index

    fine_candidates = list(range(fine_start, fine_end + 1))
    fine_best = min(
        score_boundary_windows(
            mono_audio,
            fine_candidates,
            sample_rate=sample_rate,
            analysis_window_seconds=analysis_window_seconds,
            silence_threshold=silence_threshold,
        ),
        key=lambda candidate: (
            candidate.energy_score
            + (abs(candidate.index - clamped_target) / max(search_radius_samples, 1)) * distance_weight,
            abs(candidate.index - clamped_target),
            candidate.index,
        ),
    )
    return fine_best.index
