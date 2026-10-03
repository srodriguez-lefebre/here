from __future__ import annotations

from types import SimpleNamespace

import pytest
from here.transcription.segments import (
    SegmentTimeline,
    TranscriptSegment,
    merge_segment_payloads,
    merge_segments,
    parse_transcript_segments,
    scoped_segments,
    shift_segments,
)


def test_parse_transcript_segments_extracts_timestamps_and_speakers_from_dict_payload() -> None:
    payload = {
        "segments": [
            {"text": "Hola", "start": 1.25, "end": 2.5, "speaker": "Speaker 1"},
            {"text": "Chau", "start_time": 2.5, "end_time": 3.75, "speaker_label": "Speaker 2"},
        ]
    }

    segments = parse_transcript_segments(payload)

    assert segments == [
        TranscriptSegment(text="Hola", start=1.25, end=2.5, speaker="Speaker 1"),
        TranscriptSegment(text="Chau", start=2.5, end=3.75, speaker="Speaker 2"),
    ]


def test_parse_transcript_segments_supports_object_payloads_and_timestamp_tuples() -> None:
    payload = SimpleNamespace(
        segments=[
            SimpleNamespace(text="Test", timestamps=(4, 6), speaker_id="Speaker A"),
            SimpleNamespace(text="Otro", timestamp=7.5, speaker="Speaker B"),
        ]
    )

    segments = parse_transcript_segments(payload, offset_seconds=10)

    assert segments == [
        TranscriptSegment(text="Test", start=14.0, end=16.0, speaker="Speaker A"),
        TranscriptSegment(text="Otro", start=17.5, end=17.5, speaker="Speaker B"),
    ]


def test_shift_segments_offsets_relative_timestamps() -> None:
    segments = [TranscriptSegment(text="Hola", start=1.0, end=2.0, speaker="Speaker 1")]

    shifted = shift_segments(segments, 8.5)

    assert shifted == [TranscriptSegment(text="Hola", start=9.5, end=10.5, speaker="Speaker 1")]
    assert segments == [TranscriptSegment(text="Hola", start=1.0, end=2.0, speaker="Speaker 1")]


def test_merge_segments_dedupes_overlapping_duplicate_segments() -> None:
    existing = [TranscriptSegment(text="Hola mundo", start=0.0, end=5.0, speaker="Speaker 1")]
    incoming = [TranscriptSegment(text="Hola mundo", start=4.0, end=9.0, speaker="Speaker 1")]

    merged = merge_segments(existing, incoming)

    assert merged == [
        TranscriptSegment(text="Hola mundo", start=0.0, end=9.0, speaker="Speaker 1"),
    ]


def test_merge_segments_keeps_distinct_text_even_if_timestamps_overlap() -> None:
    existing = [TranscriptSegment(text="Hola mundo", start=0.0, end=5.0, speaker="Speaker 1")]
    incoming = [TranscriptSegment(text="Adios mundo", start=4.0, end=7.0, speaker="Speaker 1")]

    merged = merge_segments(existing, incoming)

    assert merged == [
        TranscriptSegment(text="Hola mundo", start=0.0, end=5.0, speaker="Speaker 1"),
        TranscriptSegment(text="Adios mundo", start=4.0, end=7.0, speaker="Speaker 1"),
    ]


def test_merge_segment_payloads_applies_offset_before_merge() -> None:
    existing = [TranscriptSegment(text="Hola mundo", start=0.0, end=5.0, speaker="Speaker 1")]
    payload = {
        "segments": [
            {"text": "Hola mundo", "start": 4.0, "end": 9.0, "speaker": "Speaker 1"},
        ]
    }

    merged = merge_segment_payloads(existing, payload, offset_seconds=6.0)

    assert merged == [
        TranscriptSegment(text="Hola mundo", start=0.0, end=5.0, speaker="Speaker 1"),
        TranscriptSegment(text="Hola mundo", start=10.0, end=15.0, speaker="Speaker 1"),
    ]


def test_segment_timeline_extends_payload_incrementally() -> None:
    timeline = SegmentTimeline(
        [TranscriptSegment(text="Hola mundo", start=0.0, end=5.0, speaker="Speaker 1")]
    )

    timeline.extend_payload(
        {
            "segments": [
                {"text": "Hola mundo", "start": 4.5, "end": 9.0, "speaker": "Speaker 1"},
                {"text": "Siguiente", "start": 9.5, "end": 11.0, "speaker": "Speaker 2"},
            ]
        }
    )

    assert timeline.to_list() == [
        TranscriptSegment(text="Hola mundo", start=0.0, end=9.0, speaker="Speaker 1"),
        TranscriptSegment(text="Siguiente", start=9.5, end=11.0, speaker="Speaker 2"),
    ]


_REVERSED_TIMINGS = [
    {"start": 2, "end": 1},
    {"start_time": "2", "end_time": "1"},
    {"start_seconds": 2, "end_seconds": 1},
    {"start_ts": 2, "end_ts": 1},
    *[
        {key: bounds}
        for key in ("timestamp", "timestamps", "time", "times")
        for bounds in (
            {"start": 2, "end": 1},
            {"begin": 2, "finish": 1},
            {"from": "2", "to": "1"},
            {"start_time": 2, "end_time": 1},
            [2, 1],
            (2, 1),
        )
    ],
]


@pytest.mark.parametrize("timing", _REVERSED_TIMINGS)
@pytest.mark.parametrize("as_object", [False, True], ids=["mapping", "object"])
def test_reversed_provider_times_lose_both_bounds_but_keep_evidence(timing, as_object):
    candidate = {"text": "original evidence", "speaker": "A", **timing}
    if as_object:
        candidate = SimpleNamespace(**candidate)
    result = parse_transcript_segments(
        [candidate, {"text": "next evidence", "start": 0, "end": 1}], offset_seconds=10
    )
    assert result[0].text == "original evidence"
    assert result[0].speaker == "A"
    assert result[0].start is None
    assert result[0].end is None
    assert not result[0].has_timestamps
    assert result[1].text == "next evidence"
    assert (result[1].start, result[1].end) == (10.0, 11.0)


def test_reversed_provider_times_are_validated_before_offset_rounding():
    result = parse_transcript_segments(
        [{"text": "original", "start": 2, "end": 1}], offset_seconds=1e20
    )
    assert (result[0].start, result[0].end) == (None, None)


@pytest.mark.parametrize(
    "timing, expected",
    [
        ({"start": 0, "end": 0}, (10.0, 10.0)),
        ({"start": 2, "end": 2}, (12.0, 12.0)),
        ({"start": 0, "end": 3}, (10.0, 13.0)),
        ({"start": None, "end": 1}, (None, 11.0)),
        ({"start": 1, "end": None}, (11.0, None)),
        ({"timestamp": 0}, (10.0, 10.0)),
        ({"times": "2"}, (12.0, 12.0)),
        ({}, (None, None)),
    ],
)
def test_valid_and_one_sided_provider_times_preserve_bounds_and_apply_offset_once(timing, expected):
    result = parse_transcript_segments([{"text": "evidence", **timing}], offset_seconds=10)
    assert (result[0].start, result[0].end) == expected


def test_direct_typed_reversed_segment_stays_untimed_when_scoped_and_shifted():
    original = TranscriptSegment("evidence", 2.0, 1.0, "A")
    scoped = scoped_segments([original], chunk_index=3, offset_seconds=10)[0]
    assert (scoped.start, scoped.end) == (None, None)
    assert scoped.text == "evidence"
    assert scoped.speaker == "A"
    assert scoped.chunk_index == 3
    assert scoped.speaker_scope == "chunk:3"
    valid = scoped_segments(
        [TranscriptSegment("valid", 0.0, 1.0)], chunk_index=3, offset_seconds=10
    )
    assert (valid[0].start, valid[0].end) == (10.0, 11.0)
