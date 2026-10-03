import json
from types import SimpleNamespace

import here.cli as cli
import here.live_processing as live
import here.transcription.client as client
import here.transcription.service as service
import numpy as np
import pytest
import soundfile as sf
from here.application.processing import SessionProcessor
from here.audio.models import ChunkingConfig
from here.recording.models import RecordedAudioSource, RecordingSession
from here.transcription.client import AudioTranscription
from here.transcription.segments import TranscriptSegment, parse_transcript_segments


@pytest.mark.parametrize("mode", ["live", "offline", "single"])
def test_provider_segment_evidence_survives_cleanup_overlap_and_missing_values(
    monkeypatch, tmp_path, mode
):
    path = tmp_path / "synthetic.wav"
    sf.write(path, np.full(12, 0.25), 4)
    session = RecordingSession([RecordedAudioSource(path, 4, 1, 12, "mic", "Synthetic Mic")])
    repeated = TranscriptSegment("original repeated words", 1.0, 2.0, "A")
    untimed = TranscriptSegment("original without time or speaker")
    payloads = iter(
        [
            AudioTranscription("first", [repeated]),
            AudioTranscription(
                "second", [TranscriptSegment(repeated.text, 0.0, 1.0, "A"), untimed]
            ),
        ]
    )
    module = live if mode == "live" else service
    monkeypatch.setattr(module, "build_client", lambda: object())
    monkeypatch.setattr(
        module, "resolve_transcription_models", lambda **kwargs: ("model", "cleanup", True)
    )
    monkeypatch.setattr(module, "transcribe_audio_file", lambda *args, **kwargs: next(payloads))
    monkeypatch.setattr(client, "cleanup_transcript", lambda *args: "cleaned display")
    config = ChunkingConfig(
        target_sample_rate=4,
        max_chunk_bytes=16,
        live_chunk_seconds=2,
        overlap_seconds=1,
        silence_search_seconds=0,
    )
    if mode == "live":
        pipeline = live.LiveTranscriptionController(expected_source_count=1, chunking_config=config)
        try:
            pipeline.submit_block("mic", np.full(12, 0.25, dtype=np.float32), 4, 1)
            result = pipeline.complete()
        finally:
            pipeline.abort()
            pipeline.cleanup()
    elif mode == "offline":
        result = service.transcribe_recording_session(session, chunking_config=config)
    else:
        result = service.transcribe(path)
    assert result.final_text == "cleaned display"
    assert hasattr(result, "segments"), "final result must retain original structured evidence"
    assert result.segments[0].text == repeated.text
    assert result.segments[0].start == 1.0
    if mode != "single":
        assert len(result.segments) == 3, "overlap evidence must not be deleted by display merging"
        assert result.segments[1].start == 1.0
        assert result.segments[1].end == 2.0
        assert result.segments[0].speaker_scope != result.segments[1].speaker_scope
        assert result.segments[1].speaker == "A"
        assert result.segments[2].start is None
        assert result.segments[2].end is None
        assert result.segments[2].speaker is None


@pytest.mark.parametrize("entrypoint", ["processor", "cli"])
def test_session_artifact_keeps_segments_and_original_devices_after_mixing(
    monkeypatch, tmp_path, entrypoint
):
    sources = []
    for index, rate in enumerate((8000, 16000)):
        path = tmp_path / f"source-{index}.wav"
        sf.write(path, np.full((rate, 2), 0.25), rate)
        sources.append(
            RecordedAudioSource(path, rate, 2, rate, f"source-{index}", f"Device {index}")
        )
    session = RecordingSession(sources)
    evidence = [TranscriptSegment("original", None, 0.5, None)]
    result = SimpleNamespace(
        raw_text="original", final_text="cleaned", chunks=[], segments=evidence
    )
    if entrypoint == "processor":
        artifacts = SessionProcessor(
            transcribe=lambda *args, **kwargs: result, retry_delays=()
        ).process(session, tmp_path / "sessions")
        session_dir = artifacts.session_dir
    else:
        monkeypatch.setattr(cli, "transcribe_recording_session", lambda *args, **kwargs: result)
        cli._save_transcription(session, tmp_path / "sessions")
        session_dir = next((tmp_path / "sessions").iterdir())
    assert (session_dir / "segments.json").exists(), "structured evidence artifact is required"
    doc = json.loads((session_dir / "segments.json").read_text())
    assert doc["schema_version"] == 1
    assert doc["semantics"] == "provider_evidence"
    assert doc["segments"][0]["text"] == "original"
    assert doc["segments"][0]["start"] is None
    assert doc["segments"][0]["speaker"] is None
    metadata = json.loads((session_dir / "session.json").read_text())
    assert "segments.json" in metadata["output_files"]
    assert [
        (source["device_name"], source["sample_rate"], source["channels"])
        for source in metadata["capture_sources"]
    ] == [("Device 0", 8000, 2), ("Device 1", 16000, 2)]
    if entrypoint == "processor":
        SessionProcessor(transcribe=lambda *args, **kwargs: result, retry_delays=()).retry(
            session_dir
        )
        retried = json.loads((session_dir / "session.json").read_text())
        assert retried["capture_sources"] == metadata["capture_sources"]


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -1.0])
def test_invalid_provider_times_remain_missing(invalid):
    segment = parse_transcript_segments(
        {"segments": [{"text": "original", "start": invalid, "end": invalid}]}
    )[0]
    assert segment.start is None
    assert segment.end is None


def test_nested_provider_timestamps_preserve_zero():
    segment = parse_transcript_segments(
        {"segments": [{"text": "original", "timestamp": {"start": 0.0, "end": 1.0}}]}
    )[0]
    assert segment.start == 0.0
    assert segment.end == 1.0


def test_partial_live_success_fallback_uses_only_complete_offline_evidence(monkeypatch, tmp_path):
    from here.transcription.client import TranscriptionResult

    path = tmp_path / "synthetic.wav"
    sf.write(path, np.full(12, 0.25), 4)
    session = RecordingSession([RecordedAudioSource(path, 4, 1, 12, "mic", "Synthetic")])
    monkeypatch.setattr(live, "build_client", lambda: object())
    monkeypatch.setattr(
        live, "resolve_transcription_models", lambda **kwargs: ("model", "cleanup", False)
    )
    calls = []

    def provider(*args, **kwargs):
        calls.append(True)
        if len(calls) == 2:
            raise OSError("synthetic provider failure")
        return AudioTranscription("partial live", [TranscriptSegment("partial live", 0, 1, "A")])

    monkeypatch.setattr(live, "transcribe_audio_file", provider)
    config = ChunkingConfig(
        target_sample_rate=4, live_chunk_seconds=2, overlap_seconds=1, silence_search_seconds=0
    )
    pipeline = live.LiveTranscriptionController(expected_source_count=1, chunking_config=config)
    pipeline.submit_block("mic", np.full(12, 0.25, dtype=np.float32), 4, 1)
    observed = []

    def offline(material, **kwargs):
        observed.append(sf.info(material.sources[0].path).frames)
        return TranscriptionResult(
            "all saved audio",
            "all saved audio",
            segments=[TranscriptSegment("all saved audio", None, None, None)],
        )

    try:
        artifacts = SessionProcessor(transcribe=offline, retry_delays=()).process(
            session, tmp_path / "sessions", live_controller=pipeline
        )
        assert observed == [48000]
        assert artifacts.metadata.fallback_used
        assert [chunk.status for chunk in artifacts.chunks.chunks] == ["completed", "failed"]
        evidence = json.loads(artifacts.segments_path.read_text())["segments"]
        assert [segment["text"] for segment in evidence] == ["all saved audio"]
    finally:
        pipeline.abort()
        pipeline.cleanup()
