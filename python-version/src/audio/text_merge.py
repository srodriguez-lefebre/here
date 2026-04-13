from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import SequenceMatcher

_NON_WORD_RE = re.compile(r"[^\w]+", re.UNICODE)


@dataclass(frozen=True, slots=True)
class TextMergeConfig:
    min_overlap_words: int = 4
    max_overlap_words: int = 80
    min_similarity: float = 0.82
    min_line_similarity: float = 0.9
    max_line_overlap_lines: int = 8


def _normalize_word(word: str) -> str:
    return _NON_WORD_RE.sub("", word).casefold()


def _normalized_words(words: list[str]) -> list[str]:
    return [normalized for normalized in (_normalize_word(word) for word in words) if normalized]


def _normalized_word_positions(words: list[str]) -> list[tuple[int, str]]:
    positions: list[tuple[int, str]] = []
    for index, word in enumerate(words):
        normalized = _normalize_word(word)
        if normalized:
            positions.append((index, normalized))
    return positions


def _canonical_line(line: str) -> str:
    return " ".join(_normalized_words(line.split()))


def _non_empty_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


def _word_count(lines: list[str]) -> int:
    return sum(len(line.split()) for line in lines)


def _line_overlap_size(
    previous_lines: list[str],
    next_lines: list[str],
    config: TextMergeConfig,
) -> int:
    max_lines = min(config.max_line_overlap_lines, len(previous_lines), len(next_lines))
    if max_lines <= 0:
        return 0

    for line_count in range(max_lines, 0, -1):
        previous_window = previous_lines[-line_count:]
        next_window = next_lines[:line_count]
        if _word_count(previous_window) < config.min_overlap_words:
            continue

        previous_canonical = [_canonical_line(line) for line in previous_window]
        next_canonical = [_canonical_line(line) for line in next_window]
        if not all(previous_canonical) or not all(next_canonical):
            continue

        if SequenceMatcher(None, previous_canonical, next_canonical).ratio() >= config.min_line_similarity:
            return line_count

    return 0


def _token_overlap_size(previous_text: str, next_text: str, config: TextMergeConfig) -> int:
    previous_words = previous_text.split()
    next_words = next_text.split()
    if not previous_words or not next_words:
        return 0

    previous_positions = _normalized_word_positions(previous_words)
    next_positions = _normalized_word_positions(next_words)
    if not previous_positions or not next_positions:
        return 0

    max_overlap = min(
        config.max_overlap_words,
        len(previous_positions),
        len(next_positions),
    )
    if max_overlap < config.min_overlap_words:
        return 0

    for overlap_size in range(max_overlap, config.min_overlap_words - 1, -1):
        previous_window = [normalized for _, normalized in previous_positions[-overlap_size:]]
        next_window = [normalized for _, normalized in next_positions[:overlap_size]]
        if SequenceMatcher(None, previous_window, next_window).ratio() >= config.min_similarity:
            return next_positions[overlap_size - 1][0] + 1

    return 0


def trim_overlapping_prefix(
    previous_text: str,
    next_text: str,
    config: TextMergeConfig | None = None,
) -> str:
    resolved_config = config or TextMergeConfig()
    previous_text = previous_text.strip()
    next_text = next_text.strip()

    if not previous_text or not next_text:
        return next_text

    previous_lines = _non_empty_lines(previous_text)
    next_lines = _non_empty_lines(next_text)
    if previous_lines and next_lines:
        line_overlap = _line_overlap_size(previous_lines, next_lines, resolved_config)
        if line_overlap:
            trimmed_lines = next_lines[line_overlap:]
            return "\n".join(trimmed_lines).strip()

    overlap_words = _token_overlap_size(previous_text, next_text, resolved_config)
    if not overlap_words:
        return next_text

    next_words = next_text.split()
    trimmed_words = next_words[overlap_words:]
    return " ".join(trimmed_words).strip()


def merge_transcript_pair(
    previous_text: str,
    next_text: str,
    config: TextMergeConfig | None = None,
) -> str:
    previous_text = previous_text.strip()
    trimmed_next = trim_overlapping_prefix(previous_text, next_text, config)

    if not previous_text:
        return trimmed_next
    if not trimmed_next:
        return previous_text
    return f"{previous_text.rstrip()}\n{trimmed_next}"


def merge_transcript_parts(
    parts: list[str],
    config: TextMergeConfig | None = None,
) -> str:
    merged_text = ""

    for part in parts:
        stripped_part = part.strip()
        if not stripped_part:
            continue
        if not merged_text:
            merged_text = stripped_part
            continue
        merged_text = merge_transcript_pair(merged_text, stripped_part, config)

    return merged_text.strip()


def dedupe_overlap(
    previous_text: str,
    next_text: str,
    config: TextMergeConfig | None = None,
) -> str:
    return trim_overlapping_prefix(previous_text, next_text, config)
