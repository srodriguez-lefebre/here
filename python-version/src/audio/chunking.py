from pathlib import Path

import soundfile as sf

from here.audio.mix import get_total_target_frames, mix_audio_blocks, read_source_block
from here.audio.models import ChunkWindow, ChunkingConfig
from here.audio.text_merge import (
    TextMergeConfig,
    dedupe_overlap as dedupe_text_overlap,
    merge_transcript_parts as merge_text_parts,
)
from here.recording.models import RecordingSession

PCM_16_BYTES_PER_FRAME = 2

def _chunk_frames_from_config(config: ChunkingConfig) -> int:
    frames = config.max_chunk_bytes // PCM_16_BYTES_PER_FRAME
    return max(1, frames)


def _overlap_frames_from_config(config: ChunkingConfig) -> int:
    return max(0, int(config.overlap_seconds * config.target_sample_rate))


def plan_chunk_windows(session: RecordingSession, config: ChunkingConfig | None = None) -> list[ChunkWindow]:
    resolved_config = config or ChunkingConfig()
    total_frames = get_total_target_frames(session, resolved_config)
    if total_frames <= 0:
        return []

    chunk_frames = _chunk_frames_from_config(resolved_config)
    overlap_frames = min(_overlap_frames_from_config(resolved_config), chunk_frames - 1)
    step_frames = max(1, chunk_frames - overlap_frames)

    windows: list[ChunkWindow] = []
    index = 1
    start_frame = 0
    while start_frame < total_frames:
        end_frame = min(total_frames, start_frame + chunk_frames)
        windows.append(ChunkWindow(index=index, start_frame=start_frame, end_frame=end_frame))
        if end_frame >= total_frames:
            break
        start_frame += step_frames
        index += 1

    return windows


def render_chunk_window(
    session: RecordingSession,
    window: ChunkWindow,
    working_dir: Path,
    config: ChunkingConfig | None = None,
) -> Path:
    resolved_config = config or ChunkingConfig()
    working_dir.mkdir(parents=True, exist_ok=True)

    chunk_path = working_dir / f"chunk_{window.index:04d}.wav"
    block_frames = max(1, resolved_config.process_block_seconds * resolved_config.target_sample_rate)

    source_files = [sf.SoundFile(source.path) for source in session.sources]
    try:
        with sf.SoundFile(
            chunk_path,
            mode="w",
            samplerate=resolved_config.target_sample_rate,
            channels=1,
            subtype="PCM_16",
        ) as writer:
            for block_start in range(window.start_frame, window.end_frame, block_frames):
                block_end = min(window.end_frame, block_start + block_frames)
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
                mixed_block = mix_audio_blocks(blocks, target_frames)
                writer.write(mixed_block)
    finally:
        for source_file in source_files:
            source_file.close()

    return chunk_path


def build_chunk_prompt(text: str, max_words: int) -> str | None:
    words = text.split()
    if not words:
        return None
    if len(words) <= max_words:
        return text.strip()
    return " ".join(words[-max_words:]).strip()


def _text_merge_config(config: ChunkingConfig | None) -> TextMergeConfig:
    resolved_config = config or ChunkingConfig()
    return TextMergeConfig(
        min_overlap_words=resolved_config.dedupe_min_words,
        max_overlap_words=resolved_config.dedupe_max_words,
    )


def dedupe_overlap(previous_text: str, next_text: str, config: ChunkingConfig | None = None) -> str:
    return dedupe_text_overlap(previous_text, next_text, _text_merge_config(config))


def merge_transcript_parts(parts: list[str], config: ChunkingConfig | None = None) -> str:
    return merge_text_parts(parts, _text_merge_config(config))
