from pathlib import Path

import numpy as np
import soundfile as sf
from loguru import logger

from here.audio.models import ChunkingConfig
from here.recording.models import RecordedAudioSource, RecordingSession

ACTIVE_BLOCK_PEAK_THRESHOLD = 1e-4
TARGET_SOURCE_PEAK = 0.4
MIN_SOURCE_GAIN = 0.25
MAX_SOURCE_GAIN = 4.0


def downmix_to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32)
    return audio.mean(axis=1, dtype=np.float32)


def resample_mono_audio(audio: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate or audio.size == 0:
        return audio.astype(np.float32)

    target_length = max(1, int(round(audio.shape[0] * target_rate / source_rate)))
    if audio.shape[0] == 1:
        return np.full(target_length, audio[0], dtype=np.float32)

    source_positions = np.linspace(0.0, 1.0, num=audio.shape[0], endpoint=False)
    target_positions = np.linspace(0.0, 1.0, num=target_length, endpoint=False)
    resampled = np.interp(target_positions, source_positions, audio)
    return resampled.astype(np.float32)


def fit_audio_to_target_frames(audio: np.ndarray, target_frames: int) -> np.ndarray:
    if audio.shape[0] == target_frames:
        return audio
    if audio.shape[0] > target_frames:
        return audio[:target_frames]

    padding = np.zeros(target_frames - audio.shape[0], dtype=np.float32)
    return np.concatenate([audio, padding])


def read_source_block(
    source_file: sf.SoundFile,
    source: RecordedAudioSource,
    *,
    target_start_frame: int,
    target_end_frame: int,
    target_sample_rate: int,
) -> np.ndarray:
    target_frames = target_end_frame - target_start_frame
    if target_frames <= 0:
        return np.zeros(0, dtype=np.float32)

    source_start = int(round(target_start_frame * source.sample_rate / target_sample_rate))
    source_end = int(round(target_end_frame * source.sample_rate / target_sample_rate))
    requested_frames = max(0, source_end - source_start)
    if requested_frames <= 0:
        return np.zeros(target_frames, dtype=np.float32)

    source_file.seek(source_start)
    audio = source_file.read(requested_frames, dtype="float32", always_2d=True)
    if audio.shape[0] == 0:
        return np.zeros(target_frames, dtype=np.float32)

    mono_audio = downmix_to_mono(audio)
    resampled_audio = resample_mono_audio(mono_audio, source.sample_rate, target_sample_rate)
    return fit_audio_to_target_frames(resampled_audio, target_frames)


def _estimate_source_peak(source: RecordedAudioSource, config: ChunkingConfig) -> float:
    peak = 0.0
    block_frames = max(1, config.process_block_seconds * source.sample_rate)
    with sf.SoundFile(source.path) as source_file:
        while True:
            audio = source_file.read(block_frames, dtype="float32", always_2d=True)
            if audio.shape[0] == 0:
                break
            mono_audio = downmix_to_mono(audio)
            block_peak = float(np.max(np.abs(mono_audio))) if mono_audio.size else 0.0
            peak = max(peak, block_peak)
    return peak


def _compute_source_gains(
    session: RecordingSession,
    config: ChunkingConfig,
) -> list[float]:
    gains: list[float] = []
    for source in session.sources:
        peak = _estimate_source_peak(source, config)
        if peak <= 0.0:
            logger.warning("Source {label} has no measurable peak. Using unity gain.", label=source.label)
            gains.append(1.0)
            continue

        gain = TARGET_SOURCE_PEAK / peak
        gain = min(MAX_SOURCE_GAIN, max(MIN_SOURCE_GAIN, gain))
        logger.info(
            "Source {label}: peak={peak:.4f}, gain={gain:.3f}",
            label=source.label,
            peak=peak,
            gain=gain,
        )
        gains.append(gain)

    return gains


def mix_audio_blocks(
    blocks: list[np.ndarray],
    target_frames: int,
    gains: list[float] | None = None,
) -> np.ndarray:
    if not blocks:
        return np.zeros(target_frames, dtype=np.float32)

    active_blocks: list[np.ndarray] = []
    for index, block in enumerate(blocks):
        fitted_block = fit_audio_to_target_frames(block.astype(np.float32), target_frames)
        if gains is not None:
            fitted_block = fitted_block * gains[index]
        peak = float(np.max(np.abs(fitted_block))) if fitted_block.size else 0.0
        if peak > ACTIVE_BLOCK_PEAK_THRESHOLD:
            active_blocks.append(fitted_block)

    if not active_blocks:
        return np.zeros(target_frames, dtype=np.float32)

    mixed = np.zeros(target_frames, dtype=np.float32)
    for block in active_blocks:
        mixed += block

    peak = float(np.max(np.abs(mixed))) if mixed.size else 0.0
    if peak > 1.0:
        mixed = mixed / peak

    return mixed.astype(np.float32)


def get_total_target_frames(session: RecordingSession, config: ChunkingConfig) -> int:
    if not session.sources:
        return 0
    return max(
        int(round(source.frames * config.target_sample_rate / source.sample_rate))
        for source in session.sources
    )


def materialize_normalized_session(
    session: RecordingSession,
    working_dir: Path,
    config: ChunkingConfig | None = None,
) -> RecordingSession:
    resolved_config = config or ChunkingConfig()
    total_frames = get_total_target_frames(session, resolved_config)
    if total_frames <= 0:
        raise RuntimeError("No audio available to normalize.")

    working_dir.mkdir(parents=True, exist_ok=True)
    normalized_path = working_dir / "normalized_source.wav"
    block_frames = max(1, resolved_config.process_block_seconds * resolved_config.target_sample_rate)
    gains = _compute_source_gains(session, resolved_config)
    source_files = [sf.SoundFile(source.path) for source in session.sources]

    try:
        with sf.SoundFile(
            normalized_path,
            mode="w",
            samplerate=resolved_config.target_sample_rate,
            channels=1,
            subtype="PCM_16",
        ) as writer:
            written_frames = 0
            for block_start in range(0, total_frames, block_frames):
                block_end = min(total_frames, block_start + block_frames)
                target_frames = block_end - block_start
                blocks = [
                    read_source_block(
                        source_file,
                        source,
                        target_start_frame=block_start,
                        target_end_frame=block_end,
                        target_sample_rate=resolved_config.target_sample_rate,
                    )
                    for source_file, source in zip(source_files, session.sources, strict=True)
                ]
                mixed_block = mix_audio_blocks(blocks, target_frames, gains)
                writer.write(mixed_block)
                written_frames += target_frames
    finally:
        for source_file in source_files:
            source_file.close()

    return RecordingSession(
        sources=[
            RecordedAudioSource(
                path=normalized_path,
                sample_rate=resolved_config.target_sample_rate,
                channels=1,
                frames=written_frames,
                label="normalized-mix",
            )
        ]
    )
