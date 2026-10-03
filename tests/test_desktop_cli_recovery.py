import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from here.application.processing import SessionProcessor, session_from_audio_file
from here.transcription.client import TranscriptionResult


def managed_file_session(root):
    from datetime import datetime

    from here.output.metadata import ChunkMetadata, ErrorMetadata, capture_metadata
    from here.output.session_writer import write_session_artifacts

    directory = root / "human-session"
    directory.mkdir(parents=True)
    sf.write(directory / "audio.wav", np.full(160, 1000, dtype=np.int16), 16000)
    sf.write(directory / "source_01.wav", np.full(120, 2345, dtype=np.int16), 8000)
    session = session_from_audio_file(directory / "audio.wav")
    session.meeting_id = "5375f538-dc5b-4be9-8a0e-322552ef5967"
    provenance = capture_metadata(session_from_audio_file(directory / "source_01.wav"))
    provenance[0].audio_file = "source_01.wav"
    now = datetime.now().astimezone()
    write_session_artifacts(
        session=session,
        target_dir=root,
        session_dir=directory,
        session_id="human-session",
        completed_at=now,
        transcription_model="old-model",
        cleanup_model="old-cleanup",
        cleanup_enabled=False,
        alt_model_used=False,
        live_pipeline_attempted=True,
        live_pipeline_used=True,
        fallback_used=False,
        transcript_text="previous completed text",
        recoverable_audio="audio.wav",
        chunks=[
            ChunkMetadata(
                index=1,
                mode="offline",
                start_seconds=0,
                end_seconds=0.01,
                duration_seconds=0.01,
                source_count=1,
                transcription_started_at=now,
                transcription_finished_at=now,
                status="completed",
            )
        ],
        capture_sources=provenance,
        errors=[
            ErrorMetadata(
                stage="previous",
                type="OSError",
                message="previous evidence",
                retryable=True,
                occurred_at=now,
            )
        ],
    )
    return directory


@pytest.mark.parametrize("kind", ["external", "managed_audio", "managed_raw"])
def test_cli_provider_interruption_leaves_discoverable_pending(tmp_path, monkeypatch, kind):
    import here.cli as cli
    from here.application.recovery import RecoveryService

    root = tmp_path / "sessions"
    previous = None
    evidence = {}
    if kind == "external":
        audio = tmp_path / "external.wav"
        sf.write(audio, np.full(80, 1234, dtype=np.int16), 8000)
    else:
        directory = managed_file_session(root)
        previous = json.loads((directory / "session.json").read_text())
        evidence = {
            name: (directory / name).read_bytes() for name in ("chunks.json", "errors.json")
        }
        audio = directory / ("source_01.wav" if kind == "managed_raw" else "audio.wav")
    original = audio.read_bytes()

    def provider(*args, **kwargs):
        raise SystemExit("process interrupted at provider boundary")

    monkeypatch.setattr(cli, "transcribe_recording_session", provider)
    with pytest.raises(SystemExit):
        cli._transcribe_audio_path(audio, root)
    assert audio.read_bytes() == original
    (candidate,) = RecoveryService(root).discover()
    assert candidate.status == "pending" and candidate.can_retry
    metadata = json.loads((candidate.session_dir / "session.json").read_text())
    if previous:
        for name in ("meeting_id", "session_id", "capture_sources"):
            assert metadata[name] == previous[name]
        assert "previous evidence" in (candidate.session_dir / "errors.json").read_text()
        assert {name: (directory / name).read_bytes() for name in evidence} == evidence
    if kind == "managed_raw":
        assert metadata["recoverable_audio"] != "audio.wav"
        assert sf.info(candidate.session_dir / metadata["recoverable_audio"]).frames == 240
    resumed = RecoveryService(root).materialize(candidate)
    if kind == "managed_raw":
        # Retry must also rebuild the selected filename from preserved raw audio
        # if that newly prepared normalized file is missing.
        (resumed / metadata["recoverable_audio"]).unlink()
    result = SessionProcessor(transcribe=lambda *a, **kw: TranscriptionResult("new", "new")).retry(
        resumed
    )
    assert result.metadata.status == "completed"
    assert result.metadata.meeting_id == (previous["meeting_id"] if previous else None)


@pytest.mark.parametrize("kind", ["external", "managed_audio", "managed_raw"])
def test_cli_pending_publication_failure_does_not_call_provider_or_change_prior_record(
    tmp_path, monkeypatch, kind
):
    import here.cli as cli
    from here.application.recovery import RecoveryService

    root = tmp_path / "sessions"
    before = {}
    if kind == "external":
        audio = tmp_path / "external.wav"
        sf.write(audio, np.full(80, 1234, dtype=np.int16), 8000)
    else:
        directory = managed_file_session(root)
        before = {path.name: path.read_bytes() for path in directory.iterdir()}
        audio = directory / ("source_01.wav" if kind == "managed_raw" else "audio.wav")
    calls = []
    monkeypatch.setattr(
        cli,
        "transcribe_recording_session",
        lambda *a, **kw: (calls.append(1), TranscriptionResult("unexpected", "unexpected"))[1],
    )
    original_replace = Path.replace

    def replace(path, target):
        if (
            Path(target).name == "session.json"
            and json.loads(path.read_text())["status"] == "pending"
        ):
            raise OSError("pending publication blocked")
        return original_replace(path, target)

    monkeypatch.setattr(Path, "replace", replace)
    with pytest.raises(OSError, match="pending publication blocked"):
        cli._transcribe_audio_path(audio, root)
    assert calls == []
    if before:
        assert {name: (directory / name).read_bytes() for name in before} == before
        assert {path.name for path in directory.iterdir()} == set(before)
        assert RecoveryService(root).discover() == []
        assert sf.info(directory / "audio.wav").frames == 160
    else:
        assert not list(root.rglob("*.wav"))
        assert audio.exists()


@pytest.mark.parametrize("managed", [False, True])
def test_failure_reported_after_pending_replace_keeps_referenced_audio(
    tmp_path, monkeypatch, managed
):
    import here.cli as cli
    from here.application.recovery import RecoveryService

    root = tmp_path / "sessions"
    if managed:
        directory = managed_file_session(root)
        audio = directory / "source_01.wav"
        prior_audio = (directory / "audio.wav").read_bytes()
    else:
        audio = tmp_path / "external.wav"
        sf.write(audio, np.full(80, 1234, dtype=np.int16), 8000)
    original_replace = Path.replace

    def replace(path, target):
        pending = (
            Path(target).name == "session.json"
            and json.loads(path.read_text())["status"] == "pending"
        )
        result = original_replace(path, target)
        if pending:
            raise OSError("failure after manifest commit")
        return result

    monkeypatch.setattr(Path, "replace", replace)
    monkeypatch.setattr(
        cli, "transcribe_recording_session", lambda *a, **kw: pytest.fail("provider")
    )
    with pytest.raises(OSError, match="after manifest commit"):
        cli._transcribe_audio_path(audio, root)
    (candidate,) = RecoveryService(root).discover()
    assert candidate.status == "pending" and candidate.can_retry
    assert RecoveryService(root).materialize(candidate) == candidate.session_dir
    if managed:
        assert (directory / "audio.wav").read_bytes() == prior_audio


def test_legacy_save_real_wav_normalization_failure_has_local_retry_source(tmp_path, monkeypatch):
    import here.cli as cli

    audio = tmp_path / "original.wav"
    sf.write(audio, np.full(80, 1234, dtype=np.int16), 8000, subtype="PCM_16")
    session = session_from_audio_file(audio)

    def fail(*args, **kwargs):
        raise OSError("synthetic normalization failure")

    monkeypatch.setattr(cli, "materialize_normalized_session", fail)
    with pytest.raises(RuntimeError):
        cli._save_transcription(session, tmp_path / "sessions")
    saved = next((tmp_path / "sessions").iterdir())
    metadata = json.loads((saved / "session.json").read_text())
    local = metadata["capture_sources"][0]["audio_file"]
    assert local is not None
    audio.unlink(missing_ok=True)
    assert sf.info(saved / local).frames == 80
    error = json.loads((saved / "errors.json").read_text())["errors"][0]
    assert error["type"] == "OSError" and error["message"] == "synthetic normalization failure"
    result = SessionProcessor(transcribe=lambda *a, **kw: TranscriptionResult("ok", "ok")).retry(
        saved
    )
    assert result.metadata.status == "completed"


@pytest.mark.parametrize("reference", ["../outside.wav", "C:/outside.wav", "nested/file.wav"])
def test_cli_preflights_all_advertised_references_before_provider(tmp_path, monkeypatch, reference):
    import here.cli as cli

    audio = tmp_path / "original.wav"
    sf.write(audio, np.zeros(80), 8000)
    saved = SessionProcessor(transcribe=lambda *a, **kw: TranscriptionResult("ok", "ok")).process(
        session_from_audio_file(audio), tmp_path / "sessions"
    )
    path = saved.metadata_path
    metadata = json.loads(path.read_text())
    metadata["output_files"].append(reference)
    path.write_text(json.dumps(metadata))
    called = []

    def provider(*args, **kwargs):
        called.append(1)
        return TranscriptionResult("wrong", "wrong")

    monkeypatch.setattr(cli, "transcribe_recording_session", provider)
    with pytest.raises(ValueError):
        cli._transcribe_audio_path(saved.audio_path, tmp_path / "sessions")
    assert called == []


@pytest.mark.parametrize("name", ["session.json", "chunks.json", "errors.json", "audio.wav"])
def test_cli_redirected_neighbor_never_reads_canary(tmp_path, monkeypatch, name):
    import here.cli as cli

    audio = tmp_path / "original.wav"
    sf.write(audio, np.zeros(80), 8000)
    saved = SessionProcessor(transcribe=lambda *a, **kw: TranscriptionResult("ok", "ok")).process(
        session_from_audio_file(audio), tmp_path / "sessions"
    )
    canary = tmp_path / "external-canary"
    canary.write_text("private canary")
    entry = saved.session_dir / name
    entry.unlink(missing_ok=True)
    try:
        entry.symlink_to(canary)
    except OSError:
        entry.hardlink_to(canary)
    original_read = Path.read_text

    def read(path, *args, **kwargs):
        assert path.resolve() != canary.resolve(), "read external canary"
        return original_read(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", read)
    monkeypatch.setattr(
        cli, "transcribe_recording_session", lambda *a, **kw: pytest.fail("provider")
    )
    with pytest.raises(ValueError):
        cli._transcribe_audio_path(saved.audio_path, tmp_path / "sessions")
    assert canary.read_bytes() == b"private canary"
