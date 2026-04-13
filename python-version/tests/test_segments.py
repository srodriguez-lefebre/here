from __future__ import annotations

from types import SimpleNamespace

from here.transcription.segments import (
    SegmentTimeline,
    TranscriptSegment,
    merge_segment_payloads,
    merge_segments,
    parse_transcript_segments,
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
