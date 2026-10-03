import threading

import here.recording.windows as windows
import numpy as np
import pytest
import soundfile as sf
from here.recording.models import RecordedAudioSource, RecordingSession
from here.transcription.segments import TranscriptSegment, parse_transcript_segments


@pytest.mark.parametrize("after_done", [False, True])
def test_cancel_after_result_check_deletes_successful_recording(monkeypatch, tmp_path, after_done):
    path = tmp_path / "synthetic.wav"
    sf.write(path, np.full(32, 0.25), 8000)
    session = RecordingSession([RecordedAudioSource(path, 8000, 1, 32, "mic", "Synthetic")])
    publishing = threading.Event()
    release = threading.Event()

    def controlled(mode, **kwargs):
        kwargs["ready_event"].set()
        assert kwargs["stop_event"].wait(3)
        return session

    class GatedCompletion:
        def __init__(self):
            self.done = threading.Event()

        def set(self):
            publishing.set()
            assert release.wait(3)
            self.done.set()

        def wait(self, timeout):
            return self.done.wait(timeout)

    monkeypatch.setattr(windows, "_record_windows_controlled", controlled)
    handle = windows.start_windows_recording("microphone")
    completion = GatedCompletion()
    handle._done_event = completion
    try:
        handle.stop()
        assert publishing.wait(2), "worker must pass its successful-result cancellation check"
        assert sf.info(path).frames == 32
        assert handle._result is session
        if after_done:
            release.set()
            assert completion.wait(2)
        handle.cancel()
    finally:
        release.set()
    with pytest.raises(RuntimeError, match="cancelled"):
        handle.wait(2)
    assert not path.exists(), "cancelled wait must delete successful material too"
    assert handle._result is None
    with pytest.raises(RuntimeError, match="cancelled"):
        handle.wait(2)


@pytest.mark.parametrize("invalid", [None, "invalid", True, float("nan"), float("inf"), -1.0])
def test_nested_timing_skips_invalid_alias_and_retains_later_valid_value(invalid):
    segments = parse_transcript_segments(
        {
            "segments": [
                {
                    "text": "original",
                    "timestamp": {"start": invalid, "begin": 1.0, "end": invalid, "finish": 2.0},
                }
            ]
        }
    )
    assert segments[0].start == 1.0
    assert segments[0].end == 2.0


def test_nested_timing_tries_all_aliases_until_a_valid_value():
    segments = parse_transcript_segments(
        {
            "segments": [
                {
                    "text": "original",
                    "timestamp": {
                        "start": None,
                        "begin": "invalid",
                        "from": -1,
                        "start_time": 1.0,
                        "end": None,
                        "finish": "invalid",
                        "to": False,
                        "end_time": 2.0,
                    },
                }
            ]
        }
    )
    assert segments[0].start == 1.0
    assert segments[0].end == 2.0


def test_nested_timing_valid_zero_takes_precedence_over_later_aliases():
    segments = parse_transcript_segments(
        {
            "segments": [
                {
                    "text": "original",
                    "timestamp": {"start": 0.0, "begin": 1.0, "end": 0.0, "finish": 2.0},
                }
            ]
        }
    )
    assert segments[0].start == 0.0
    assert segments[0].end == 0.0


@pytest.mark.parametrize("oversized", [10**400, -(10**400)], ids=["positive", "negative"])
@pytest.mark.parametrize("nested", [False, True], ids=["primary", "nested"])
def test_oversized_timing_preserves_text_speaker_and_other_timing_evidence(oversized, nested):
    timings = [
        {"start": oversized, "end": oversized},
        {"start": oversized, "end": 2},
        {"start": 1, "end": oversized},
        {"start": 1, "end": 2},
    ]
    candidates = [
        {
            "text": text,
            "speaker": "A",
            **({"timestamp": timing} if nested else timing),
        }
        for text, timing in zip(["both invalid", "start invalid", "end invalid", "valid"], timings)
    ]

    assert parse_transcript_segments({"segments": candidates}) == [
        TranscriptSegment("both invalid", speaker="A"),
        TranscriptSegment("start invalid", end=2.0, speaker="A"),
        TranscriptSegment("end invalid", start=1.0, speaker="A"),
        TranscriptSegment("valid", start=1.0, end=2.0, speaker="A"),
    ]


@pytest.mark.parametrize("oversized", [10**400, -(10**400)], ids=["positive", "negative"])
@pytest.mark.parametrize("nested", [False, True], ids=["primary", "nested"])
def test_oversized_timing_aliases_allow_later_valid_values(oversized, nested):
    timing = {"start": oversized, "end": oversized}
    timing.update({"begin": 1, "finish": 2} if nested else {"start_time": 1, "end_time": 2})
    candidate = {"text": "original", **({"timestamp": timing} if nested else timing)}

    assert parse_transcript_segments({"segments": [candidate]}) == [
        TranscriptSegment("original", start=1.0, end=2.0)
    ]


@pytest.mark.parametrize("oversized", [10**400, -(10**400)], ids=["positive", "negative"])
@pytest.mark.parametrize("nested", [False, True], ids=["primary", "nested"])
def test_valid_zero_timing_precedes_later_oversized_aliases(oversized, nested):
    timing = {"start": 0, "end": 0}
    timing.update(
        {"begin": oversized, "finish": oversized}
        if nested
        else {"start_time": oversized, "end_time": oversized}
    )
    candidate = {"text": "original", **({"timestamp": timing} if nested else timing)}

    assert parse_transcript_segments({"segments": [candidate]}) == [
        TranscriptSegment("original", start=0.0, end=0.0)
    ]
