from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from here.application import processing
from here.application.processing import (
    ProcessingCancelled,
    SessionProcessingFailed,
    SessionProcessor,
)
from here.audio.mix import materialize_normalized_session
from here.output.metadata import CaptureSourceMetadata
from here.output.session_writer import write_session_artifacts
from here.recording.models import RecordedAudioSource, RecordingSession
from here.transcription.client import TranscriptionResult


def _recovery_session(tmp_path: Path, *, normalized: bool = False):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    raw_path = session_dir / "source_01.wav"
    sf.write(raw_path, np.full(80, 0.25), 8000)
    raw = RecordingSession([RecordedAudioSource(raw_path, 8000, 1, 80, "mic", "original")])
    session = (
        materialize_normalized_session(raw, session_dir, output_name="audio.wav")
        if normalized
        else raw
    )
    artifacts = write_session_artifacts(
        session=session,
        target_dir=tmp_path,
        session_dir=session_dir,
        completed_at=datetime(2026, 10, 3),
        transcription_model="synthetic",
        cleanup_model="synthetic",
        cleanup_enabled=False,
        alt_model_used=False,
        live_pipeline_attempted=False,
        live_pipeline_used=False,
        fallback_used=False,
        status="failed",
        recoverable_audio="audio.wav" if normalized else None,
        capture_sources=[
            CaptureSourceMetadata(
                label="mic",
                device_name="original",
                sample_rate=8000,
                channels=1,
                frames=80,
                duration_seconds=0.01,
                audio_file="source_01.wav",
            )
        ],
    )
    return artifacts


@pytest.mark.parametrize("normalized", [False, True], ids=["raw-only", "normalized-exists"])
@pytest.mark.parametrize(
    "field,value",
    [
        ("frames", 10**12),
        ("frames", 0),
        ("frames", -1),
        ("sample_rate", 16000),
        ("sample_rate", 0),
        ("channels", 2),
        ("channels", 0),
    ],
)
def test_retry_rejects_manifest_geometry_before_normalization_or_provider(
    tmp_path, monkeypatch, normalized, field, value
):
    artifacts = _recovery_session(tmp_path, normalized=normalized)
    metadata = json.loads(artifacts.metadata_path.read_text())
    metadata["capture_sources"][0][field] = value
    artifacts.metadata_path.write_text(json.dumps(metadata))
    before = {path.name: path.read_bytes() for path in artifacts.session_dir.iterdir()}
    normalization_calls = []
    provider_calls = []

    def forbid_normalization(*args, **kwargs):
        normalization_calls.append(True)
        raise AssertionError("Unvalidated geometry reached normalization")

    monkeypatch.setattr(processing, "materialize_normalized_session", forbid_normalization)
    processor = SessionProcessor(
        transcribe=lambda *args, **kwargs: (
            provider_calls.append(True) or TranscriptionResult("synthetic", "synthetic")
        ),
        retry_delays=(),
    )
    with pytest.raises(ValueError, match="geometry"):
        processor.retry(artifacts.session_dir)
    assert normalization_calls == []
    assert provider_calls == []
    assert {path.name: path.read_bytes() for path in artifacts.session_dir.iterdir()} == before


def test_retry_validates_later_raw_sources_before_creating_audio(tmp_path, monkeypatch):
    artifacts = _recovery_session(tmp_path)
    metadata = json.loads(artifacts.metadata_path.read_text())
    second_path = artifacts.session_dir / "source_02.wav"
    sf.write(second_path, np.full(40, 0.25), 8000)
    metadata["capture_sources"].append(
        {**metadata["capture_sources"][0], "audio_file": "source_02.wav", "frames": 80}
    )
    artifacts.metadata_path.write_text(json.dumps(metadata))
    before = {path.name: path.read_bytes() for path in artifacts.session_dir.iterdir()}
    calls = []
    monkeypatch.setattr(
        processing,
        "materialize_normalized_session",
        lambda *args, **kwargs: (
            calls.append("normalize") or (_ for _ in ()).throw(AssertionError("unsafe"))
        ),
    )
    processor = SessionProcessor(transcribe=lambda *args, **kwargs: calls.append("provider"))
    with pytest.raises(ValueError, match="geometry"):
        processor.retry(artifacts.session_dir)
    assert calls == []
    assert {path.name: path.read_bytes() for path in artifacts.session_dir.iterdir()} == before


@pytest.mark.parametrize("field,value", [("frames", 320), ("sample_rate", 8000), ("channels", 2)])
def test_retry_checks_manifest_geometry_of_identified_normalized_audio(tmp_path, field, value):
    artifacts = _recovery_session(tmp_path, normalized=True)
    metadata = json.loads(artifacts.metadata_path.read_text())
    metadata["sources"][0][field] = value
    artifacts.metadata_path.write_text(json.dumps(metadata))
    before = {path.name: path.read_bytes() for path in artifacts.session_dir.iterdir()}
    calls = []
    with pytest.raises(ValueError, match="geometry"):
        SessionProcessor(
            transcribe=lambda *args, **kwargs: (
                calls.append(True) or TranscriptionResult("synthetic", "synthetic")
            )
        ).retry(artifacts.session_dir)
    assert calls == []
    assert {path.name: path.read_bytes() for path in artifacts.session_dir.iterdir()} == before


@pytest.mark.parametrize("normalized", [False, True])
def test_matching_recovery_geometry_preserves_audio_and_derives_duration(tmp_path, normalized):
    artifacts = _recovery_session(tmp_path, normalized=normalized)
    raw_before = (artifacts.session_dir / "source_01.wav").read_bytes()
    metadata = json.loads(artifacts.metadata_path.read_text())
    metadata["capture_sources"][0]["duration_seconds"] = 999
    artifacts.metadata_path.write_text(json.dumps(metadata))
    seen = []

    def transcribe(session, **kwargs):
        source = session.sources[0]
        seen.append((source.sample_rate, source.channels, source.frames, session.duration_seconds))
        assert sf.info(source.path).frames == 160
        return TranscriptionResult("synthetic", "synthetic")

    result = SessionProcessor(transcribe=transcribe, retry_delays=()).retry(artifacts.session_dir)
    assert seen == [(16000, 1, 160, 0.01)]
    assert result.metadata.duration_seconds == 0.01
    assert result.metadata.capture_sources[0].duration_seconds == 0.01
    assert result.metadata.capture_sources[0].device_name == "original"
    assert (artifacts.session_dir / "source_01.wav").read_bytes() == raw_before


@pytest.mark.parametrize("field,value", [("samplerate", 0), ("channels", 0), ("frames", -1)])
def test_retry_rejects_invalid_header_geometry(tmp_path, monkeypatch, field, value):
    artifacts = _recovery_session(tmp_path)
    raw_path = artifacts.session_dir / "source_01.wav"
    info = sf.info

    def invalid_info(path, *args, **kwargs):
        actual = info(path, *args, **kwargs)
        if Path(path) == raw_path:
            values = {
                "samplerate": actual.samplerate,
                "channels": actual.channels,
                "frames": actual.frames,
            }
            values[field] = value
            return SimpleNamespace(**values)
        return actual

    monkeypatch.setattr(sf, "info", invalid_info)
    monkeypatch.setattr(
        processing,
        "materialize_normalized_session",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("invalid header reached normalization")
        ),
    )
    with pytest.raises(ValueError, match="geometry"):
        SessionProcessor().retry(artifacts.session_dir)
    assert not (artifacts.session_dir / "audio.wav").exists()


@pytest.mark.parametrize("outcome", ["failed", "cancelled"])
def test_failed_or_cancelled_retry_persists_duration_from_validated_geometry(tmp_path, outcome):
    artifacts = _recovery_session(tmp_path)
    metadata = json.loads(artifacts.metadata_path.read_text())
    metadata["capture_sources"][0]["duration_seconds"] = 999
    artifacts.metadata_path.write_text(json.dumps(metadata))
    cancellation = threading.Event()
    if outcome == "cancelled":
        cancellation.set()

    def fail_transcription(*args, **kwargs):
        raise RuntimeError("synthetic failure")

    with pytest.raises(ProcessingCancelled if outcome == "cancelled" else SessionProcessingFailed):
        SessionProcessor(transcribe=fail_transcription, retry_delays=()).retry(
            artifacts.session_dir, cancel_event=cancellation
        )
    persisted = json.loads(artifacts.metadata_path.read_text())
    assert persisted["status"] == outcome
    assert persisted["duration_seconds"] == 0.01
    assert persisted["capture_sources"][0]["duration_seconds"] == 0.01
    assert persisted["capture_sources"][0]["device_name"] == "original"
    assert persisted["capture_sources"][0]["audio_file"] == "source_01.wav"


@pytest.mark.parametrize("with_nonempty_source", [False, True])
def test_honest_empty_wav_retains_existing_recovery_behavior(tmp_path, with_nonempty_source):
    artifacts = _recovery_session(tmp_path)
    metadata = json.loads(artifacts.metadata_path.read_text())
    empty_path = artifacts.session_dir / "empty.wav"
    sf.write(empty_path, np.empty(0), 8000)
    empty_source = {
        **metadata["capture_sources"][0],
        "audio_file": "empty.wav",
        "frames": 0,
        "duration_seconds": 0.0,
    }
    metadata["capture_sources"] = (
        [empty_source, *metadata["capture_sources"]] if with_nonempty_source else [empty_source]
    )
    artifacts.metadata_path.write_text(json.dumps(metadata))
    calls = []
    processor = SessionProcessor(
        transcribe=lambda *args, **kwargs: (
            calls.append(True) or TranscriptionResult("valid", "valid")
        )
    )
    if with_nonempty_source:
        result = processor.retry(artifacts.session_dir)
        assert result.metadata.duration_seconds == 0.01
        assert result.metadata.capture_sources[0].frames == 0
        assert result.metadata.capture_sources[0].duration_seconds == 0.0
        assert calls == [True]
    else:
        before = artifacts.metadata_path.read_bytes()
        with pytest.raises(RuntimeError, match="Recoverable audio does not exist"):
            processor.retry(artifacts.session_dir)
        assert calls == []
        assert artifacts.metadata_path.read_bytes() == before
        assert not (artifacts.session_dir / "audio.wav").exists()


def test_legacy_multisource_manifest_does_not_define_normalized_mix_geometry(tmp_path):
    artifacts = _recovery_session(tmp_path, normalized=True)
    metadata = json.loads(artifacts.metadata_path.read_text())
    original_source = {**metadata["sources"][0], "sample_rate": 8000, "frames": 80, "channels": 2}
    metadata["sources"] = [original_source, {**original_source, "label": "second"}]
    artifacts.metadata_path.write_text(json.dumps(metadata))
    seen = []

    def transcribe(session, **kwargs):
        seen.append(
            (session.sources[0].sample_rate, session.sources[0].channels, session.duration_seconds)
        )
        return TranscriptionResult("legacy", "legacy")

    result = SessionProcessor(transcribe=transcribe).retry(artifacts.session_dir)
    assert seen == [(16000, 1, 0.01)]
    assert result.metadata.duration_seconds == 0.01
