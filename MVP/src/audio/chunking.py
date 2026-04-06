import re
from pathlib import Path

import soundfile as sf

from here.audio.mix import get_total_target_frames, mix_audio_blocks, read_source_block
from here.audio.models import ChunkWindow, ChunkingConfig
from here.recording.models import RecordingSession

PCM_16_BYTES_PER_FRAME = 2


def _normalize_word(word: str) -> str:
    return re.sub(r"[^\w]+", "", word, flags=re.UNICODE).casefold()


def _non_empty_normalized_words(words: list[str]) -> list[str]:
    return [normalized for normalized in (_normalize_word(word) for word in words) if normalized]


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


def dedupe_overlap(previous_text: str, next_text: str, config: ChunkingConfig | None = None) -> str:
    resolved_config = config or ChunkingConfig()
    previous_words = previous_text.split()
    next_words = next_text.split()
    if not previous_words or not next_words:
        return next_text.strip()

    previous_normalized = _non_empty_normalized_words(previous_words)
    next_normalized = _non_empty_normalized_words(next_words)
    if not previous_normalized or not next_normalized:
        return next_text.strip()

    max_overlap = min(
        resolved_config.dedupe_max_words,
        len(previous_normalized),
        len(next_normalized),
        len(next_words),
    )

    for overlap_size in range(max_overlap, resolved_config.dedupe_min_words - 1, -1):
        if previous_normalized[-overlap_size:] == next_normalized[:overlap_size]:
            trimmed_words = next_words[overlap_size:]
            return " ".join(trimmed_words).strip()

    return next_text.strip()


def merge_transcript_parts(parts: list[str], config: ChunkingConfig | None = None) -> str:
    resolved_config = config or ChunkingConfig()
    merged_text = ""

    for part in parts:
        stripped_part = part.strip()
        if not stripped_part:
            continue
        if not merged_text:
            merged_text = stripped_part
            continue

        trimmed_part = dedupe_overlap(merged_text, stripped_part, resolved_config)
        if not trimmed_part:
            continue
        merged_text = f"{merged_text.rstrip()}\n{trimmed_part}"

    return merged_text.strip()
