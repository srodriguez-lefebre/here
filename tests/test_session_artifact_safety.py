from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from here.application.processing import SessionProcessor
from here.audio.mix import materialize_normalized_session
from here.output.metadata import CaptureSourceMetadata, ErrorMetadata, SessionEventMetadata
from here.output.session_writer import write_session_artifacts
from here.recording.models import CaptureFailed, RecordedAudioSource, RecordingSession
from here.transcription.client import TranscriptionResult
from here.transcription.segments import TranscriptSegment

FIXED_FILES = [
    "session.json",
    "segments.json",
    "transcript.txt",
    "transcript.md",
    "chunks.json",
    "errors.json",
    "events.json",
    "audio.wav",
]


def _audio(path: Path) -> RecordingSession:
    sf.write(path, np.full(80, 0.25), 8000)
    return RecordingSession([RecordedAudioSource(path, 8000, 1, 80, "mic")])


def _save(session_dir: Path, session: RecordingSession):
    return write_session_artifacts(
        session=session,
        session_dir=session_dir,
        target_dir=session_dir.parent,
        completed_at=datetime(2026, 10, 3),
        transcription_model="synthetic",
        cleanup_model="synthetic",
        cleanup_enabled=False,
        alt_model_used=False,
        live_pipeline_attempted=False,
        live_pipeline_used=False,
        fallback_used=False,
        transcript_text="new transcript",
        segments=[TranscriptSegment("new evidence")],
        recoverable_audio="audio.wav",
        errors=[
            ErrorMetadata(
                stage="capture",
                type="OSError",
                message="lost",
                retryable=True,
                occurred_at=datetime(2026, 10, 3),
            )
        ],
        events=[
            SessionEventMetadata(
                kind="stopped", occurred_at=datetime(2026, 10, 3), recorded_duration_seconds=0.01
            )
        ],
    )


def _session(tmp_path: Path):
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    session = _audio(session_dir / "audio.wav")
    return _save(session_dir, session), session


def _link(entry: Path, outside: Path, kind: str) -> None:
    if entry.exists():
        entry.unlink()
    if kind == "hardlink":
        os.link(outside, entry)
    elif kind == "symlink":
        try:
            entry.symlink_to(outside, target_is_directory=outside.is_dir())
        except OSError as error:
            pytest.skip(f"Host cannot create symlinks: {error}")
    elif kind == "junction":
        if sys.platform != "win32":
            pytest.skip("Junctions are Windows-specific")
        subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(entry), str(outside)], check=True, capture_output=True
        )
    else:
        entry.mkdir()
        (entry / "canary").write_bytes(b"keep directory contents")


@pytest.mark.parametrize("name", FIXED_FILES)
@pytest.mark.parametrize("kind", ["hardlink", "directory"])
@pytest.mark.parametrize("operation", ["writer", "retry"])
def test_session_artifacts_rejected_before_reads_writes_or_provider(
    tmp_path, monkeypatch, name, kind, operation
):
    artifacts, session = _session(tmp_path)
    entry = artifacts.session_dir / name
    outside = tmp_path / "outside-canary"
    outside.write_bytes(entry.read_bytes())
    original = outside.read_bytes()
    metadata_before = artifacts.metadata_path.read_bytes()
    _link(entry, outside, kind)
    provider_calls = []
    read_text = Path.read_text
    audio_info = sf.info

    def forbid_entry_read(path, *args, **kwargs):
        assert path != entry, "tampered session entry was read"
        return read_text(path, *args, **kwargs)

    def forbid_audio_read(path, *args, **kwargs):
        assert Path(path) != entry, "tampered session audio was read"
        return audio_info(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", forbid_entry_read)
    monkeypatch.setattr(sf, "info", forbid_audio_read)
    processor = SessionProcessor(
        transcribe=lambda *args, **kwargs: (
            provider_calls.append(True) or TranscriptionResult("provider", "provider")
        ),
        retry_delays=(),
    )
    with pytest.raises(ValueError, match="session"):
        if operation == "writer":
            _save(artifacts.session_dir, session)
        else:
            processor.retry(artifacts.session_dir)
    assert provider_calls == []
    assert outside.read_bytes() == original
    if name != "session.json":
        assert artifacts.metadata_path.read_bytes() == metadata_before
    if kind == "directory":
        assert (entry / "canary").read_bytes() == b"keep directory contents"


@pytest.mark.parametrize("name", ["transcript.txt", "session.json", "audio.wav"])
def test_retry_rejects_actual_symlink_entries(tmp_path, name):
    artifacts, _ = _session(tmp_path)
    entry = artifacts.session_dir / name
    outside = tmp_path / "outside-canary"
    outside.write_bytes(entry.read_bytes())
    original = outside.read_bytes()
    _link(entry, outside, "symlink")
    calls = []
    with pytest.raises(ValueError, match="session"):
        SessionProcessor(
            transcribe=lambda *args, **kwargs: (
                calls.append(True) or TranscriptionResult("new", "new")
            )
        ).retry(artifacts.session_dir)
    assert calls == []
    assert outside.read_bytes() == original


@pytest.mark.parametrize("nested", [False, True], ids=["session-directory", "ancestor"])
@pytest.mark.parametrize("kind", ["symlink", "junction"])
def test_linked_session_directory_is_rejected(tmp_path, nested, kind):
    outside = tmp_path / "outside"
    outside.mkdir()
    original_dir = outside / "nested" if nested else outside
    original_dir.mkdir(exist_ok=True)
    session = _audio(original_dir / "audio.wav")
    artifacts = _save(original_dir, session)
    before = {path.name: path.read_bytes() for path in original_dir.iterdir()}
    link = tmp_path / "linked"
    _link(link, outside, kind)
    linked_session = link / "nested" if nested else link
    calls = []
    with pytest.raises(ValueError, match="session"):
        SessionProcessor(
            transcribe=lambda *args, **kwargs: (
                calls.append(True) or TranscriptionResult("new", "new")
            )
        ).retry(linked_session)
    with pytest.raises(ValueError, match="session"):
        _save(linked_session, session)
    assert calls == []
    assert {path.name: path.read_bytes() for path in artifacts.session_dir.iterdir()} == before


@pytest.mark.parametrize("name", ["session.json", "events.json"])
def test_capture_failure_cancellation_rejects_linked_outputs(tmp_path, name):
    artifacts, _ = _session(tmp_path)
    outside = tmp_path / "outside-canary"
    outside.write_bytes((artifacts.session_dir / name).read_bytes())
    original = outside.read_bytes()
    metadata_before = artifacts.metadata_path.read_bytes()
    _link(artifacts.session_dir / name, outside, "hardlink")
    with pytest.raises(ValueError, match="session"):
        SessionProcessor().mark_capture_failure_cancelled(artifacts, events=[])
    assert outside.read_bytes() == original
    assert artifacts.metadata_path.read_bytes() == metadata_before


@pytest.mark.parametrize("operation", ["normalize", "preserve-sources"])
@pytest.mark.parametrize("kind", ["hardlink", "directory"])
def test_audio_output_destinations_reject_unsafe_entries(tmp_path, operation, kind):
    session = _audio(tmp_path / "capture.wav")
    destination = tmp_path / "session"
    destination.mkdir()
    name = "audio.wav" if operation == "normalize" else "source_01.wav"
    outside = tmp_path / "outside-canary"
    outside.write_bytes(b"outside original")
    _link(destination / name, outside, kind)
    with pytest.raises(ValueError, match="session"):
        if operation == "normalize":
            materialize_normalized_session(session, destination, output_name=name)
        else:
            SessionProcessor()._preserve_sources(session, destination)
    assert outside.read_bytes() == b"outside original"


@pytest.mark.parametrize("field", ["output_files", "recoverable_audio", "capture_sources"])
def test_retry_preflights_all_manifest_references_even_with_existing_normalized_audio(
    tmp_path, field
):
    artifacts, _ = _session(tmp_path)
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"do not read")
    linked = artifacts.session_dir / "source_01.wav"
    os.link(outside, linked)
    metadata = json.loads(artifacts.metadata_path.read_text())
    if field == "output_files":
        metadata[field].append("source_01.wav")
    elif field == "recoverable_audio":
        metadata[field] = "source_01.wav"
    else:
        metadata[field] = [
            CaptureSourceMetadata(
                label="mic",
                sample_rate=8000,
                channels=1,
                frames=80,
                duration_seconds=0.01,
                audio_file="source_01.wav",
            ).model_dump()
        ]
    artifacts.metadata_path.write_text(json.dumps(metadata))
    calls = []
    with pytest.raises(ValueError, match="session"):
        SessionProcessor(
            transcribe=lambda *args, **kwargs: (
                calls.append(True) or TranscriptionResult("new", "new")
            )
        ).retry(artifacts.session_dir)
    assert calls == []
    assert outside.read_bytes() == b"do not read"


def test_legacy_audio_only_session_still_retries(tmp_path):
    session_dir = tmp_path / "legacy"
    session_dir.mkdir()
    _audio(session_dir / "audio.wav")
    result = SessionProcessor(
        transcribe=lambda *args, **kwargs: TranscriptionResult("legacy", "legacy"), retry_delays=()
    ).retry(session_dir)
    assert result.metadata.status == "completed"
    assert result.transcript_path.read_text(encoding="utf-8-sig") == "legacy"


@pytest.mark.parametrize(
    "name", ["transcript.txt", "transcript.md", "chunks.json", "errors.json", "events.json"]
)
def test_partial_text_write_does_not_truncate_previous_artifact(tmp_path, monkeypatch, name):
    artifacts, session = _session(tmp_path)
    final_path = artifacts.session_dir / name
    previous = final_path.read_bytes()
    write_text = Path.write_text

    def fail_partial_write(path, text, *args, **kwargs):
        if path.name == name or path.name.startswith(f".{name}."):
            write_text(path, "partial", *args, **kwargs)
            raise OSError("partial artifact write")
        return write_text(path, text, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_partial_write)
    with pytest.raises(OSError, match="partial artifact write"):
        _save(artifacts.session_dir, session)
    assert final_path.read_bytes() == previous
    assert not list(artifacts.session_dir.glob(".*.stage"))


@pytest.mark.parametrize("operation", ["normalize", "preserve-sources"])
def test_partial_audio_output_does_not_truncate_previous_artifact(tmp_path, monkeypatch, operation):
    source = _audio(tmp_path / "source.wav")
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    name = "audio.wav" if operation == "normalize" else "source_01.wav"
    final_path = session_dir / name
    final_path.write_bytes(b"prior owned audio")
    if operation == "normalize":
        write = sf.SoundFile.write

        def fail_write(writer, data):
            write(writer, data)
            raise OSError("partial audio write")

        monkeypatch.setattr(sf.SoundFile, "write", fail_write)
    else:

        def fail_copy(source, target):
            Path(target).write_bytes(b"partial")
            raise OSError("partial audio write")

        monkeypatch.setattr(shutil, "copy2", fail_copy)
    with pytest.raises(OSError, match="partial audio write"):
        if operation == "normalize":
            materialize_normalized_session(source, session_dir, output_name=name)
        else:
            SessionProcessor()._preserve_sources(source, session_dir)
    assert final_path.read_bytes() == b"prior owned audio"
    assert not list(session_dir.glob(".*.stage"))


@pytest.mark.parametrize("operation", ["process", "capture-failure"])
def test_audio_destination_rejection_does_not_enter_recovery_cleanup(
    tmp_path, monkeypatch, operation
):
    import here.application.processing as processing

    session = _audio(tmp_path / "capture.wav")
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    outside = tmp_path / "outside.wav"
    outside.write_bytes(b"outside audio")
    os.link(outside, session_dir / "audio.wav")
    monkeypatch.setattr(processing, "create_session_dir", lambda *args: ("session", session_dir))
    calls = []
    processor = SessionProcessor(
        transcribe=lambda *args, **kwargs: calls.append(True) or TranscriptionResult("new", "new")
    )
    with pytest.raises(ValueError, match="session"):
        if operation == "process":
            processor.process(session, tmp_path)
        else:
            processor.persist_capture_failure(CaptureFailed(session, OSError("capture")), tmp_path)
    assert calls == []
    assert session.sources[0].path.exists()
    assert (session_dir / "audio.wav").samefile(outside)
    assert outside.read_bytes() == b"outside audio"
