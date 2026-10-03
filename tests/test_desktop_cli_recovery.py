import json
import sys
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


@pytest.mark.parametrize(
    "outcomes",
    [
        ("completed", "completed"),
        ("completed", "failed"),
        ("failed", "completed"),
        ("failed", "failed"),
    ],
)
def test_repeated_managed_raw_transcription_removes_only_replaced_audio(
    tmp_path, monkeypatch, outcomes
):
    import here.cli as cli

    root = tmp_path / "sessions"
    directory = managed_file_session(root)
    raw = directory / "source_01.wav"
    original_raw = raw.read_bytes()
    canaries = {"unknown.wav": b"unknown", "audio_" + "f" * 32 + ".wav": b"unowned"}
    for name, content in canaries.items():
        (directory / name).write_bytes(content)
    metadata_path = directory / "session.json"
    initial = json.loads(metadata_path.read_text())
    for outcome in outcomes:
        previous = json.loads(metadata_path.read_text())
        old = directory / previous["recoverable_audio"]
        old_bytes = old.read_bytes()
        provider_error = OSError("synthetic original provider failure")

        def provider(session, **kwargs):
            pending = json.loads(metadata_path.read_text())
            assert pending["status"] == "pending"
            assert set(previous["output_files"]) <= set(pending["output_files"])
            assert old.read_bytes() == old_bytes
            assert raw.read_bytes() == original_raw
            assert session.sources[0].path.name == pending["recoverable_audio"]
            assert sf.info(session.sources[0].path).frames == 240
            if outcome == "failed":
                raise provider_error
            return TranscriptionResult("new text", "new text")

        monkeypatch.setattr(cli, "transcribe_recording_session", provider)
        if outcome == "failed":
            with pytest.raises(RuntimeError, match="Transcription failed") as caught:
                cli._transcribe_audio_path(raw, root)
            assert caught.value.__cause__ is provider_error
        else:
            cli._transcribe_audio_path(raw, root)
        current = json.loads(metadata_path.read_text())
        assert current["status"] == outcome
        for key in ("session_id", "meeting_id", "capture_sources"):
            assert current[key] == initial[key]
        assert current["recoverable_audio"] != old.name
        assert not old.exists(), "Replaced full-length normalized WAV became an orphan"
        assert sf.info(directory / current["recoverable_audio"]).frames == 240
        assert raw.read_bytes() == original_raw
        assert {name: (directory / name).read_bytes() for name in canaries} == canaries
        assert {path.name for path in directory.glob("*.wav")} == {
            "source_01.wav",
            current["recoverable_audio"],
            *canaries,
        }


@pytest.mark.parametrize("outcome", ["completed", "failed"])
@pytest.mark.parametrize("failure", ["unlink", "read", "path", "missing_metadata", "invalid_json"])
def test_postcommit_audio_cleanup_fault_preserves_committed_outcome(
    tmp_path, monkeypatch, outcome, failure
):
    import here.cli as cli
    from here.output.paths import UnsafeSessionPath

    root = tmp_path / "sessions"
    directory = managed_file_session(root)
    old = directory / "audio.wav"
    old_bytes = old.read_bytes()
    committed = False
    attempted = []
    warnings = []
    provider_error = OSError("original provider failure")
    original_write = cli.write_session_artifacts
    original_read = cli.read_session_metadata
    original_unlink = Path.unlink
    original_path = cli.session_artifact_path

    def write(**kwargs):
        nonlocal committed
        result = original_write(**kwargs)
        committed = True
        if failure == "invalid_json":
            (directory / "session.json").write_text("invalid metadata")
        return result

    def read(path):
        if committed and failure == "invalid_json":
            attempted.append(failure)
        if committed and failure in {"read", "missing_metadata"}:
            attempted.append(failure)
            if failure == "missing_metadata":
                return None
            raise OSError("postcommit metadata read failed")
        return original_read(path)

    def unlink(path, *args, **kwargs):
        if path == old and committed and failure == "unlink":
            attempted.append(failure)
            raise OSError("postcommit unlink failed")
        return original_unlink(path, *args, **kwargs)

    def artifact_path(parent, name):
        if parent / name == old and committed and failure == "path":
            attempted.append(failure)
            raise UnsafeSessionPath("postcommit path rejected")
        return original_path(parent, name)

    def provider(*args, **kwargs):
        if outcome == "failed":
            raise provider_error
        return TranscriptionResult("new", "new")

    monkeypatch.setattr(cli, "write_session_artifacts", write)
    monkeypatch.setattr(cli, "read_session_metadata", read)
    monkeypatch.setattr(cli, "session_artifact_path", artifact_path)
    monkeypatch.setattr(Path, "unlink", unlink)
    monkeypatch.setattr(cli, "transcribe_recording_session", provider)
    token = cli.logger.add(lambda message: warnings.append(str(message)), level="WARNING")
    try:
        if outcome == "failed":
            with pytest.raises(RuntimeError, match="Transcription failed") as caught:
                cli._transcribe_audio_path(directory / "source_01.wav", root)
            assert caught.value.__cause__ is provider_error
        else:
            cli._transcribe_audio_path(directory / "source_01.wav", root)
    finally:
        cli.logger.remove(token)
    assert attempted == [failure], "Committed cleanup was not attempted"
    assert warnings
    assert old.read_bytes() == old_bytes
    if failure != "invalid_json":
        current = json.loads((directory / "session.json").read_text())
        assert current["status"] == outcome
        assert current["recoverable_audio"] != "audio.wav"
        assert sf.info(directory / current["recoverable_audio"]).frames == 240


@pytest.mark.parametrize("protection", ["output", "capture", "prior_capture", "current", "unknown"])
def test_audio_cleanup_preserves_protected_and_unknown_previous_file(
    tmp_path, monkeypatch, protection
):
    import here.cli as cli

    root = tmp_path / "sessions"
    directory = managed_file_session(root)
    metadata_path = directory / "session.json"
    old = directory / "audio.wav"
    metadata = json.loads(metadata_path.read_text())
    if protection == "unknown":
        old = old.rename(directory / "audio_selected.wav")
        metadata["recoverable_audio"] = old.name
        metadata["output_files"] = [
            old.name if name == "audio.wav" else name for name in metadata["output_files"]
        ]
    elif protection == "prior_capture":
        original_capture = metadata["sources"][0]
        metadata["capture_sources"].append({**original_capture, "audio_file": old.name})
    metadata_path.write_text(json.dumps(metadata))
    old_bytes = old.read_bytes()
    raw_bytes = (directory / "source_01.wav").read_bytes()
    original_write = cli.write_session_artifacts

    def write(**kwargs):
        # Remove old capture provenance at final publication: even then the
        # original capture file must remain protected by the prior manifest.
        if protection == "prior_capture":
            kwargs["capture_sources"] = kwargs["capture_sources"][:1]
        result = original_write(**kwargs)
        current = json.loads(metadata_path.read_text())
        if protection == "output":
            # Windows paths ignore case; retain the same entry under either spelling.
            current["output_files"].append(
                old.name.upper() if sys.platform == "win32" else old.name
            )
        elif protection == "capture":
            current["capture_sources"].append({**metadata["sources"][0], "audio_file": old.name})
        metadata_path.write_text(json.dumps(current))
        return result

    monkeypatch.setattr(cli, "write_session_artifacts", write)
    monkeypatch.setattr(
        cli, "transcribe_recording_session", lambda *a, **kw: TranscriptionResult("ok", "ok")
    )
    selected = old if protection == "current" else directory / "source_01.wav"
    cli._transcribe_audio_path(selected, root)
    assert old.read_bytes() == old_bytes
    assert (directory / "source_01.wav").read_bytes() == raw_bytes
    current = json.loads(metadata_path.read_text())
    assert current["status"] == "completed"
    assert (directory / current["recoverable_audio"]).exists()


@pytest.mark.parametrize("redirect", ["hardlink", "directory"])
def test_postcommit_cleanup_rechecks_previous_path_safety(tmp_path, monkeypatch, redirect):
    import here.cli as cli

    root = tmp_path / "sessions"
    directory = managed_file_session(root)
    old = directory / "audio.wav"
    external = tmp_path / "canary.wav"
    external.write_bytes(b"external canary")
    original_write = cli.write_session_artifacts
    warnings = []

    def write(**kwargs):
        result = original_write(**kwargs)
        old.unlink()
        if redirect == "hardlink":
            old.hardlink_to(external)
        else:
            old.mkdir()
        return result

    monkeypatch.setattr(cli, "write_session_artifacts", write)
    monkeypatch.setattr(
        cli, "transcribe_recording_session", lambda *a, **kw: TranscriptionResult("ok", "ok")
    )
    token = cli.logger.add(lambda message: warnings.append(str(message)), level="WARNING")
    try:
        cli._transcribe_audio_path(directory / "source_01.wav", root)
    finally:
        cli.logger.remove(token)
    assert warnings
    assert old.exists()
    assert external.read_bytes() == b"external canary"
    assert json.loads((directory / "session.json").read_text())["status"] == "completed"


@pytest.mark.parametrize("failure", ["normalize", "final_completed", "final_failed"])
def test_managed_audio_is_retained_before_final_publication(tmp_path, monkeypatch, failure):
    import here.cli as cli

    root = tmp_path / "sessions"
    directory = managed_file_session(root)
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    original_replace = Path.replace
    normalization_error = OSError("normalization blocked")
    provider_calls = []

    def normalize(*args, **kwargs):
        raise normalization_error

    def replace(path, target):
        if Path(target).name == "session.json":
            metadata = json.loads(path.read_text())
            if metadata["status"] != "pending":
                raise OSError("final publication blocked")
        return original_replace(path, target)

    def provider(*args, **kwargs):
        provider_calls.append(True)
        if failure == "final_failed":
            raise RuntimeError("synthetic provider error")
        return TranscriptionResult("new", "new")

    monkeypatch.setattr(cli, "transcribe_recording_session", provider)
    if failure == "normalize":
        monkeypatch.setattr(cli, "materialize_normalized_session", normalize)
    else:
        monkeypatch.setattr(Path, "replace", replace)
    with pytest.raises((OSError, RuntimeError)) as caught:
        cli._transcribe_audio_path(directory / "source_01.wav", root)
    assert (directory / "audio.wav").read_bytes() == before["audio.wav"]
    assert (directory / "source_01.wav").read_bytes() == before["source_01.wav"]
    if failure == "normalize":
        assert str(caught.value) == "Recoverable audio preparation failed"
        assert caught.value.__cause__ is normalization_error
        assert not provider_calls
        assert {path.name: path.read_bytes() for path in directory.iterdir()} == before
    else:
        current = json.loads((directory / "session.json").read_text())
        assert current["status"] == "pending"
        assert "audio.wav" in current["output_files"]
        assert sf.info(directory / current["recoverable_audio"]).frames == 240


@pytest.mark.parametrize("audio_name", ["audio.wav", "audio_selected.wav"])
@pytest.mark.parametrize("status", ["pending", "failed"])
def test_fresh_discovery_materialize_retry_rebuilds_absent_normalized_audio(
    tmp_path, audio_name, status
):
    from here.application.recovery import RecoveryService

    root = tmp_path / "sessions"
    directory = managed_file_session(root)
    path = directory / "session.json"
    metadata = json.loads(path.read_text())
    metadata["status"] = status
    metadata["recoverable_audio"] = audio_name
    metadata["output_files"] = [
        audio_name if name == "audio.wav" else name for name in metadata["output_files"]
    ]
    path.write_text(json.dumps(metadata))
    (directory / "audio.wav").unlink()
    before = {entry.name: entry.read_bytes() for entry in directory.iterdir()}

    # Restart with the derived WAV already absent, before either public recovery step.
    service = RecoveryService(root)
    (candidate,) = service.discover()
    assert candidate.can_retry
    assert candidate.capture_id == metadata["meeting_id"]
    assert service.materialize(candidate) == directory
    assert {entry.name: entry.read_bytes() for entry in directory.iterdir()} == before
    calls = []

    def synthetic_transcribe(session, **kwargs):
        pending = json.loads(path.read_text())
        assert pending["status"] == "pending"
        assert pending["recoverable_audio"] == audio_name
        assert pending["meeting_id"] == metadata["meeting_id"]
        calls.append(session.sources[0].frames)
        return TranscriptionResult("synthetic", "synthetic")

    result = SessionProcessor(transcribe=synthetic_transcribe).retry(directory)
    assert calls == [240]
    assert sf.info(directory / audio_name).frames == 240
    assert result.metadata.status == "completed"
    assert result.metadata.meeting_id == metadata["meeting_id"]
    assert result.metadata.session_id == metadata["session_id"]
    assert (directory / "source_01.wav").read_bytes() == before["source_01.wav"]


@pytest.mark.parametrize(
    "damage",
    ["corrupt", "geometry", "directory", "hardlink", "traversal", "absolute", "raw_missing"],
)
def test_raw_backing_never_hides_existing_normalized_damage(tmp_path, monkeypatch, damage):
    from here.application.recovery import RecoveryService

    root = tmp_path / "sessions"
    directory = managed_file_session(root)
    metadata_path = directory / "session.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["status"] = "pending"
    metadata_path.write_text(json.dumps(metadata))
    service = RecoveryService(root)
    (candidate,) = service.discover()
    normalized = directory / "audio.wav"
    external = tmp_path / "external.wav"
    sf.write(external, np.ones(160), 16000)
    external_bytes = external.read_bytes()
    if damage == "corrupt":
        normalized.write_bytes(b"not audio")
    elif damage == "geometry":
        sf.write(normalized, np.ones(1), 8000)
    elif damage == "directory":
        normalized.unlink()
        normalized.mkdir()
    elif damage == "hardlink":
        normalized.unlink()
        normalized.hardlink_to(external)
    elif damage in {"traversal", "absolute"}:
        metadata["recoverable_audio"] = (
            "../../external.wav" if damage == "traversal" else str(external)
        )
        metadata_path.write_text(json.dumps(metadata))
    else:
        normalized.unlink()
        (directory / "source_01.wav").unlink()
    before = {entry.name: entry.read_bytes() for entry in directory.iterdir() if entry.is_file()}
    original_info = sf.info

    def guarded_info(path, *args, **kwargs):
        candidate_path = Path(path)
        if candidate_path.exists():
            assert not candidate_path.samefile(external), "Read redirected audio"
        return original_info(path, *args, **kwargs)

    monkeypatch.setattr(sf, "info", guarded_info)
    monkeypatch.setattr(
        "here.transcription.client.build_client", lambda: pytest.fail("provider allocation")
    )
    assert RecoveryService(root).discover() == []
    with pytest.raises((OSError, ValueError, RuntimeError)):
        service.materialize(candidate)
    calls = []

    def forbidden(*args, **kwargs):
        calls.append("normalize/provider")
        pytest.fail("Unsafe recovery allocated normalization/provider work")

    with pytest.raises((OSError, ValueError, RuntimeError)):
        SessionProcessor(normalize=forbidden, transcribe=forbidden).retry(directory)
    assert calls == []
    assert {
        entry.name: entry.read_bytes() for entry in directory.iterdir() if entry.is_file()
    } == before
    assert external.read_bytes() == external_bytes


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
