from dataclasses import dataclass


@dataclass(slots=True)
class ChunkingConfig:
    target_sample_rate: int = 16_000
    max_chunk_bytes: int = 20 * 1024 * 1024
    overlap_seconds: int = 5
    process_block_seconds: int = 30
    prompt_tail_words: int = 120
    dedupe_min_words: int = 8
    dedupe_max_words: int = 80


@dataclass(slots=True)
class ChunkWindow:
    index: int
    start_frame: int
    end_frame: int

    @property
    def frame_count(self) -> int:
        return self.end_frame - self.start_frame
