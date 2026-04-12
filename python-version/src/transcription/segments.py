from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field


_SEGMENT_CONTAINER_KEYS = ("segments", "items", "results")
_TEXT_KEYS = ("text", "transcript", "utterance")
_SPEAKER_KEYS = ("speaker", "speaker_label", "speaker_id")
_START_KEYS = ("start", "start_time", "start_seconds", "start_ts")
_END_KEYS = ("end", "end_time", "end_seconds", "end_ts")
_TIMESTAMP_KEYS = ("timestamp", "timestamps", "time", "times")


@dataclass(slots=True, frozen=True)
class TranscriptSegment:
    text: str
    start: float | None = None
    end: float | None = None
    speaker: str | None = None

    @property
    def has_timestamps(self) -> bool:
        return self.start is not None and self.end is not None


@dataclass(slots=True)
class SegmentTimeline:
    segments: list[TranscriptSegment] = field(default_factory=list)

    def extend(self, segments: Sequence[TranscriptSegment]) -> None:
        self.segments = merge_segments(self.segments, segments)

    def extend_payload(self, payload: object, *, offset_seconds: float = 0.0) -> None:
        self.extend(parse_transcript_segments(payload, offset_seconds=offset_seconds))

    def to_list(self) -> list[TranscriptSegment]:
        return list(self.segments)


def _value_from_payload(payload: object, key: str) -> object | None:
    if isinstance(payload, Mapping):
        return payload.get(key)
    return getattr(payload, key, None)


def _iter_candidate_segments(payload: object) -> list[object]:
    if isinstance(payload, list):
        return list(payload)
    if isinstance(payload, tuple):
        return list(payload)

    for key in _SEGMENT_CONTAINER_KEYS:
        segments = _value_from_payload(payload, key)
        if isinstance(segments, list):
            return list(segments)
        if isinstance(segments, tuple):
            return list(segments)

    return []


def _coerce_float(value: object | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return None
        try:
            return float(stripped)
        except ValueError:
            return None
    return None


def _coerce_text(value: object | None) -> str | None:
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return None


def _coerce_speaker(value: object | None) -> str | None:
    speaker = _coerce_text(value)
    return speaker or None


def _extract_timing_bounds(payload: object) -> tuple[float | None, float | None]:
    start = None
    end = None

    for key in _START_KEYS:
        start = _coerce_float(_value_from_payload(payload, key))
        if start is not None:
            break

    for key in _END_KEYS:
        end = _coerce_float(_value_from_payload(payload, key))
        if end is not None:
            break

    if start is not None or end is not None:
        return start, end

    for key in _TIMESTAMP_KEYS:
        timestamps = _value_from_payload(payload, key)
        if isinstance(timestamps, Mapping):
            start = _coerce_float(
                timestamps.get("start")
                or timestamps.get("begin")
                or timestamps.get("from")
                or timestamps.get("start_time")
            )
            end = _coerce_float(
                timestamps.get("end")
                or timestamps.get("finish")
                or timestamps.get("to")
                or timestamps.get("end_time")
            )
            if start is not None or end is not None:
                return start, end
        if isinstance(timestamps, (list, tuple)) and len(timestamps) >= 2:
            start = _coerce_float(timestamps[0])
            end = _coerce_float(timestamps[1])
            if start is not None or end is not None:
                return start, end
        scalar_timestamp = _coerce_float(timestamps)
        if scalar_timestamp is not None:
            return scalar_timestamp, scalar_timestamp

    return None, None


def _extract_segment_text(payload: object) -> str | None:
    for key in _TEXT_KEYS:
        text = _coerce_text(_value_from_payload(payload, key))
        if text is not None:
            return text
    return None


def _extract_segment_speaker(payload: object) -> str | None:
    for key in _SPEAKER_KEYS:
        speaker = _coerce_speaker(_value_from_payload(payload, key))
        if speaker is not None:
            return speaker
    return None


def parse_transcript_segments(payload: object, *, offset_seconds: float = 0.0) -> list[TranscriptSegment]:
    segments: list[TranscriptSegment] = []
    for candidate in _iter_candidate_segments(payload):
        text = _extract_segment_text(candidate)
        if text is None:
            continue

        start, end = _extract_timing_bounds(candidate)
        if start is not None:
            start += offset_seconds
        if end is not None:
            end += offset_seconds

        segments.append(
            TranscriptSegment(
                text=text,
                start=start,
                end=end,
                speaker=_extract_segment_speaker(candidate),
            )
        )

    return segments


def shift_segments(segments: Sequence[TranscriptSegment], offset_seconds: float) -> list[TranscriptSegment]:
    shifted: list[TranscriptSegment] = []
    for segment in segments:
        start = segment.start + offset_seconds if segment.start is not None else None
        end = segment.end + offset_seconds if segment.end is not None else None
        shifted.append(
            TranscriptSegment(
                text=segment.text,
                start=start,
                end=end,
                speaker=segment.speaker,
            )
        )
    return shifted


def _normalized_words(text: str) -> list[str]:
    words = []
    for word in text.split():
        normalized = re.sub(r"[^\w]+", "", word, flags=re.UNICODE).casefold()
        if normalized:
            words.append(normalized)
    return words


def _contains_contiguous_words(haystack: list[str], needle: list[str]) -> bool:
    if not needle or len(needle) > len(haystack):
        return False
    window_size = len(needle)
    for index in range(len(haystack) - window_size + 1):
        if haystack[index : index + window_size] == needle:
            return True
    return False


def _text_overlap_is_duplicate(previous_text: str, next_text: str) -> bool:
    previous_words = _normalized_words(previous_text)
    next_words = _normalized_words(next_text)
    if not previous_words or not next_words:
        return False
    if previous_words == next_words:
        return True
    return _contains_contiguous_words(previous_words, next_words) or _contains_contiguous_words(
        next_words,
        previous_words,
    )


def _segments_overlap(previous: TranscriptSegment, current: TranscriptSegment) -> bool:
    if previous.start is None or previous.end is None:
        return False
    if current.start is None or current.end is None:
        return False
    return min(previous.end, current.end) > max(previous.start, current.start)


def _merge_segment_pair(previous: TranscriptSegment, current: TranscriptSegment) -> TranscriptSegment:
    start = previous.start
    if current.start is not None:
        start = current.start if start is None else min(start, current.start)

    end = previous.end
    if current.end is not None:
        end = current.end if end is None else max(end, current.end)

    if len(current.text.strip()) > len(previous.text.strip()):
        merged_text = current.text.strip()
    else:
        merged_text = previous.text.strip()

    speaker = previous.speaker or current.speaker

    return TranscriptSegment(text=merged_text, start=start, end=end, speaker=speaker)


def _can_merge_segments(previous: TranscriptSegment, current: TranscriptSegment) -> bool:
    if previous.speaker and current.speaker and previous.speaker != current.speaker:
        return False
    if not previous.text.strip() or not current.text.strip():
        return False
    if previous.has_timestamps and current.has_timestamps:
        if not _segments_overlap(previous, current):
            return False
    elif previous.text.strip().casefold() != current.text.strip().casefold():
        return False

    return _text_overlap_is_duplicate(previous.text, current.text)


def merge_segments(
    existing: Sequence[TranscriptSegment],
    incoming: Sequence[TranscriptSegment],
) -> list[TranscriptSegment]:
    merged = list(existing)
    for segment in incoming:
        if not merged:
            merged.append(segment)
            continue

        previous = merged[-1]
        if _can_merge_segments(previous, segment):
            merged[-1] = _merge_segment_pair(previous, segment)
            continue

        merged.append(segment)

    return merged


def merge_segment_payloads(
    existing: Sequence[TranscriptSegment],
    payload: object,
    *,
    offset_seconds: float = 0.0,
) -> list[TranscriptSegment]:
    return merge_segments(existing, parse_transcript_segments(payload, offset_seconds=offset_seconds))
