import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf


def journal_type():
    from here.recording import journal

    return journal.CaptureJournal


@pytest.mark.parametrize("channels", [1, 3])
def test_open_production_writer_survives_process_kill(tmp_path, channels):
    journal_type()
    script = """
import sys, time
from pathlib import Path
import numpy as np
from here.recording.journal import CaptureJournal
j = CaptureJournal.create(Path(sys.argv[1]))
w = j.open_writer(label="synthetic", sample_rate=8000, channels=int(sys.argv[2]))
w.write(np.tile(np.array([1000, -2000, 3000], dtype=np.int16)[:int(sys.argv[2])], (80, 1)))
w.checkpoint()
Path(sys.argv[1], "ready").write_text(j.directory.name)
while True: time.sleep(1)
"""
    process = subprocess.Popen([sys.executable, "-c", script, str(tmp_path), str(channels)])
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / "ready").exists() and time.monotonic() < deadline:
            assert process.poll() is None
            time.sleep(0.02)
        assert (tmp_path / "ready").exists()
        process.kill()
        process.wait(5)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(5)
    from here.application.recovery import RecoveryService

    service = RecoveryService(tmp_path)
    candidates = service.discover()
    assert len(candidates) == 1 and candidates[0].can_retry
    capture = journal_type().load(tmp_path, candidates[0].capture_id)
    source = capture.recording_session().sources[0]
    data, rate = sf.read(source.path, dtype="int16", always_2d=True)
    assert rate == 8000 and data.shape == (80, channels)
    np.testing.assert_array_equal(data, np.tile([1000, -2000, 3000][:channels], (80, 1)))
    fresh_reader = subprocess.check_output(
        [
            sys.executable,
            "-c",
            "import soundfile as s,sys,json; "
            "d,r=s.read(sys.argv[1],dtype='int16',always_2d=True); "
            "print(json.dumps([r,d.tolist()]))",
            str(source.path),
        ],
        text=True,
    )
    assert json.loads(fresh_reader) == [8000, [[1000, -2000, 3000][:channels]] * 80]
    first = service.materialize(candidates[0])
    second = service.materialize(candidates[0])
    assert first == second == candidates[0].session_dir
    meta = json.loads((first / "session.json").read_text())
    assert meta["meeting_id"] == capture.document.capture_id
    assert meta["status"] == "failed"
    assert service.discover()[0].session_dir == first


def make_capture(root):
    capture = journal_type().create(root)
    writer = capture.open_writer(label="mic", sample_rate=8000, channels=1)
    writer.write(np.full((80, 1), 1234, dtype=np.int16))
    writer.close()
    return capture


def write_recovery_manifest(directory, capture_id, status):
    """An imported manifest and authored PCM, independent of the session writer."""
    directory.mkdir()
    sf.write(directory / "audio.wav", np.full(80, 1234, dtype=np.int16), 8000, subtype="PCM_16")
    (directory / "session.json").write_text(
        json.dumps(
            {
                "session_id": "copied-human-id",
                "meeting_id": capture_id,
                "started_at": "2026-10-03T12:00:00+00:00",
                "completed_at": "2026-10-03T12:00:01+00:00",
                "duration_seconds": 0.01,
                "status": status,
                "recoverable_audio": "audio.wav",
                "sources": [
                    {
                        "label": "mic",
                        "sample_rate": 8000,
                        "channels": 1,
                        "frames": 80,
                        "duration_seconds": 0.01,
                    }
                ],
                "transcription_model": "test-model",
                "cleanup_model": "test-cleanup",
                "cleanup_enabled": False,
                "alt_model_used": False,
                "live_pipeline_attempted": False,
                "live_pipeline_used": False,
                "fallback_used": False,
                "output_files": ["session.json", "audio.wav"],
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize("status", ["failed", "completed"])
def test_foreign_manifest_with_same_uuid_does_not_hide_interrupted_capture(tmp_path, status):
    from here.application.recovery import RecoveryService

    capture = make_capture(tmp_path)
    foreign = tmp_path / "foreign-copy"
    write_recovery_manifest(foreign, capture.document.capture_id, status)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    candidates = RecoveryService(tmp_path).discover()

    expected = {capture.destination: "interrupted"}
    if status == "failed":
        expected[foreign] = "failed"
    assert {item.session_dir: item.status for item in candidates} == expected
    assert all(item.capture_id == capture.document.capture_id for item in candidates)
    assert all(item.can_retry and item.recorded_duration_seconds == 0.01 for item in candidates)
    assert not capture.destination.exists()
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_failed_manifest_copies_with_same_uuid_are_both_discoverable(tmp_path):
    from here.application.recovery import RecoveryService

    capture = make_capture(tmp_path)
    write_recovery_manifest(capture.destination, capture.document.capture_id, "failed")
    copied = tmp_path / "copied-session"
    shutil.copytree(capture.destination, copied)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    service = RecoveryService(tmp_path)

    candidates = service.discover()

    assert {item.session_dir for item in candidates} == {capture.destination, copied}
    assert len(candidates) == 2
    assert all(item.status == "failed" and item.can_retry for item in candidates)
    assert all(item.capture_id == capture.document.capture_id for item in candidates)
    assert all(item.display_id == "copied-human-id" for item in candidates)
    for candidate in candidates:
        assert service.materialize(candidate) == candidate.session_dir
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("original_status", ["interrupted", "failed"])
@pytest.mark.parametrize("identity_spelling", ["canonical", "hex", "braced"])
def test_retry_copied_session_preserves_independent_capture(
    tmp_path, original_status, identity_spelling
):
    from here.application.processing import SessionProcessor
    from here.application.recovery import RecoveryService
    from here.transcription.client import TranscriptionResult

    root = tmp_path / "sessions"
    capture = make_capture(root)
    seed = tmp_path / "authored-session"
    write_recovery_manifest(seed, capture.document.capture_id, "failed")
    copied = root / "copied-session"
    shutil.copytree(seed, copied)
    meeting_id = capture.document.capture_id
    if identity_spelling == "hex":
        meeting_id = meeting_id.replace("-", "")
    elif identity_spelling == "braced":
        meeting_id = "{" + meeting_id + "}"
    metadata = json.loads((copied / "session.json").read_text())
    metadata["meeting_id"] = meeting_id
    (copied / "session.json").write_text(json.dumps(metadata), encoding="utf-8")
    sf.write(copied / "audio.wav", np.full(80, 4321, dtype=np.int16), 8000, subtype="PCM_16")
    if original_status == "failed":
        shutil.copytree(seed, capture.destination)
    original_bytes = {
        path: path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and not path.is_relative_to(copied)
    }
    copied_audio = (copied / "audio.wav").read_bytes()
    service = RecoveryService(root)
    candidate = next(item for item in service.discover() if item.session_dir == copied)
    assert candidate.can_retry
    assert candidate.capture_id == meeting_id
    selected = service.materialize(candidate)
    assert selected == copied
    seen = []

    def transcribe(session, **kwargs):
        seen.append(session.sources[0].path)
        assert session.sources[0].path == copied / "audio.wav"
        np.testing.assert_array_equal(
            sf.read(session.sources[0].path, dtype="int16")[0], [4321] * 80
        )
        assert session.meeting_id == meeting_id
        assert json.loads((copied / "session.json").read_text())["status"] == "pending"
        return TranscriptionResult("copied words", "Recovered copy")

    result = SessionProcessor(transcribe=transcribe, retry_delays=()).retry(selected)

    assert seen == [copied / "audio.wav"]
    assert result.session_dir == copied
    assert result.metadata.status == "completed"
    assert result.metadata.meeting_id == meeting_id
    assert result.metadata.session_id == "copied-human-id"
    assert result.transcript_path.read_text(encoding="utf-8-sig") == "Recovered copy"
    assert (copied / "audio.wav").read_bytes() == copied_audio
    assert {
        path: path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and not path.is_relative_to(copied)
    } == original_bytes
    (remaining,) = service.discover()
    assert remaining.session_dir == capture.destination
    assert remaining.capture_id == capture.document.capture_id
    assert remaining.status == original_status
    assert remaining.can_retry
    original_source = capture.recording_session().sources[0]
    np.testing.assert_array_equal(sf.read(original_source.path, dtype="int16")[0], [1234] * 80)


@pytest.mark.parametrize(
    "meeting_id",
    [
        pytest.param(None, id="absent"),
        pytest.param("", id="empty"),
        pytest.param("imported-legacy-id", id="malformed"),
        pytest.param("../outside", id="posix-path"),
        pytest.param(r"..\outside", id="windows-path"),
        pytest.param("12345678-1234-4ABC-8DEF-123456789ABC", id="uppercase"),
        pytest.param("1234567812344abc8def123456789abc", id="hex"),
        pytest.param("{12345678-1234-4abc-8def-123456789abc}", id="braced"),
    ],
)
def test_retry_optional_identity_preserves_selected_audio_and_metadata(
    tmp_path, monkeypatch, meeting_id
):
    from here.application.processing import SessionProcessor
    from here.application.recovery import RecoveryService
    from here.output.session_writer import read_session_metadata
    from here.transcription.client import TranscriptionResult

    directory = tmp_path / "imported-session"
    write_recovery_manifest(directory, meeting_id, "failed")
    sf.write(directory / "raw.wav", np.full(80, 4321, dtype=np.int16), 8000, subtype="PCM_16")
    metadata = json.loads((directory / "session.json").read_text())
    raw_metadata = {
        **metadata["sources"][0],
        "audio_file": "raw.wav",
        "device_name": "original microphone",
    }
    metadata["capture_sources"] = [raw_metadata]
    metadata["output_files"].append("raw.wav")
    (directory / "session.json").write_text(json.dumps(metadata), encoding="utf-8")
    before = {path.name: path.read_bytes() for path in directory.iterdir()}
    service = RecoveryService(tmp_path)

    (candidate,) = service.discover()
    assert candidate.can_retry and candidate.capture_id == meeting_id
    assert candidate.display_id == "copied-human-id"
    selected = service.materialize(candidate)
    assert selected == directory
    assert {path.name: path.read_bytes() for path in directory.iterdir()} == before
    journal_loads = []
    load = journal_type().load

    def observe_load(root, capture_id):
        journal_loads.append(capture_id)
        return load(root, capture_id)

    monkeypatch.setattr(journal_type(), "load", staticmethod(observe_load))
    seen = []

    def transcribe(session, **kwargs):
        (source,) = session.sources
        seen.append(source.path)
        assert source.path == directory / "audio.wav"
        np.testing.assert_array_equal(sf.read(source.path, dtype="int16")[0], [1234] * 80)
        assert session.meeting_id == meeting_id
        pending = read_session_metadata(directory)
        assert pending.status == "pending"
        assert pending.meeting_id == meeting_id
        assert pending.session_id == "copied-human-id"
        return TranscriptionResult("selected words", "Recovered import")

    result = SessionProcessor(transcribe=transcribe, retry_delays=()).retry(selected)

    assert seen == [directory / "audio.wav"]
    assert journal_loads == []
    assert result.metadata == read_session_metadata(directory)
    assert result.metadata.status == "completed"
    assert result.metadata.meeting_id == meeting_id
    assert result.metadata.session_id == "copied-human-id"
    assert [item.model_dump() for item in result.metadata.capture_sources] == [raw_metadata]
    assert result.transcript_path.read_text(encoding="utf-8-sig") == "Recovered import"
    assert (directory / "audio.wav").read_bytes() == before["audio.wav"]
    assert (directory / "raw.wav").read_bytes() == before["raw.wav"]
    assert service.discover() == []


@pytest.mark.parametrize(
    "meeting_id", [None, "imported-legacy-id", "{12345678-1234-4abc-8def-123456789abc}"]
)
@pytest.mark.parametrize("fault", ["path", "geometry", "media"])
def test_retry_optional_identity_keeps_selected_audio_preflight(tmp_path, meeting_id, fault):
    from here.application.processing import SessionProcessor

    directory = tmp_path / "imported-session"
    write_recovery_manifest(directory, meeting_id, "failed")
    metadata_path = directory / "session.json"
    metadata = json.loads(metadata_path.read_text())
    if fault == "path":
        metadata["recoverable_audio"] = "../outside.wav"
    elif fault == "geometry":
        metadata["sources"][0]["frames"] = 160
    else:
        (directory / "audio.wav").write_bytes(b"invalid selected WAV")
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    with pytest.raises(
        RuntimeError if fault == "media" else ValueError,
        match="geometry" if fault == "geometry" else None,
    ):
        SessionProcessor(transcribe=lambda *a, **kw: pytest.fail("provider called")).retry(
            directory
        )

    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("fault", [None, "geometry", "media", "path"])
def test_retry_own_reservation_keeps_journal_validation_and_cleanup(tmp_path, fault):
    from here.application.processing import SessionProcessor
    from here.transcription.client import TranscriptionResult

    capture = make_capture(tmp_path)
    write_recovery_manifest(capture.destination, capture.document.capture_id, "failed")
    journal_path = capture.directory / "journal.json"
    if fault == "media":
        (capture.directory / "source_01.wav").write_bytes(b"invalid original WAV")
    elif fault is not None:
        document = json.loads(journal_path.read_text())
        if fault == "geometry":
            document["sources"][0]["channels"] = 2
        else:
            document["sources"][0]["audio_file"] = "../outside.wav"
        journal_path.write_text(json.dumps(document))
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    calls = []

    def transcribe(session, **kwargs):
        calls.append(session.sources[0].path)
        return TranscriptionResult("own words", "Recovered original")

    processor = SessionProcessor(transcribe=transcribe, retry_delays=())
    if fault is not None:
        with pytest.raises(RuntimeError if fault == "media" else ValueError):
            processor.retry(capture.destination)
        assert calls == []
        assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before
    else:
        result = processor.retry(capture.destination)
        assert calls == [capture.destination / "audio.wav"]
        assert result.metadata.status == "completed"
        assert result.metadata.meeting_id == capture.document.capture_id
        assert not capture.directory.exists()
        assert result.audio_path.read_bytes() == before[capture.destination / "audio.wav"]


@pytest.mark.parametrize("fault", ["identity", "destination", "source_path", "schema", "json"])
def test_retry_copy_rejects_invalid_journal_before_association(tmp_path, fault):
    from here.application.processing import SessionProcessor
    from here.output.paths import UnsafeSessionPath

    capture = make_capture(tmp_path)
    copied = tmp_path / "copied-session"
    write_recovery_manifest(copied, capture.document.capture_id, "failed")
    journal_path = capture.directory / "journal.json"
    document = json.loads(journal_path.read_text())
    if fault == "identity":
        document["capture_id"] = "12345678-1234-1234-1234-123456789abc"
    elif fault == "destination":
        document["destination"] = "../outside"
    elif fault == "schema":
        document["schema_version"] = 2
    elif fault == "source_path":
        document["sources"][0]["audio_file"] = "../outside.wav"
    journal_path.write_text("{" if fault == "json" else json.dumps(document))
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    with pytest.raises(ValueError if fault in {"schema", "json"} else UnsafeSessionPath):
        SessionProcessor(transcribe=lambda *a, **kw: pytest.fail("provider called")).retry(copied)
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


@pytest.mark.parametrize("status", ["failed", "completed"])
def test_manifest_supersedes_journal_only_at_same_identity_and_reserved_directory(tmp_path, status):
    from here.application.recovery import RecoveryService

    capture = make_capture(tmp_path)
    write_recovery_manifest(capture.destination, capture.document.capture_id, status)
    before = {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}

    candidates = RecoveryService(tmp_path).discover()

    if status == "completed":
        assert candidates == []
    else:
        assert len(candidates) == 1
        assert candidates[0].session_dir == capture.destination
        assert candidates[0].status == "failed"
        assert candidates[0].display_id == "copied-human-id"
    assert {path: path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before


def test_pending_exists_before_provider_and_journal_retained_on_commit_error(tmp_path, monkeypatch):
    from here.application.processing import SessionProcessor

    capture = make_capture(tmp_path)
    mapped = capture.destination
    assert not mapped.exists()

    def transcribe(session, **kwargs):
        assert json.loads((mapped / "session.json").read_text())["status"] == "pending"
        assert capture.directory.exists()
        raise SystemExit("killed provider")

    with pytest.raises(SystemExit):
        SessionProcessor(transcribe=transcribe).process(capture.recording_session(), tmp_path)
    assert capture.directory.exists()
    from here.application.recovery import RecoveryService

    assert RecoveryService(tmp_path).discover()[0].session_dir == mapped


@pytest.mark.parametrize(
    "field,value",
    [("destination", "../outside"), ("destination", "C:/outside"), ("schema_version", 999)],
)
def test_malformed_journal_does_not_hide_good_candidate(tmp_path, field, value):
    from here.application.recovery import RecoveryService

    good = make_capture(tmp_path)
    bad = make_capture(tmp_path)
    path = bad.directory / "journal.json"
    document = json.loads(path.read_text())
    document[field] = value
    path.write_text(json.dumps(document))
    candidates = RecoveryService(tmp_path).discover()
    assert [item.capture_id for item in candidates] == [good.document.capture_id]
    assert path.exists()


def test_header_ahead_of_journal_recovers_actual_frames(tmp_path, monkeypatch):
    capture = journal_type().create(tmp_path)
    writer = capture.open_writer(label="mic", sample_rate=8000, channels=1)
    writer.write(np.full(80, 3210, dtype=np.int16))
    publish = capture._publish
    monkeypatch.setattr(
        capture, "_publish", lambda: (_ for _ in ()).throw(OSError("journal replace"))
    )
    with pytest.raises(OSError):
        writer.checkpoint()
    assert json.loads((capture.directory / "journal.json").read_text())["sources"][0]["frames"] == 0
    recovered = capture.load(tmp_path, capture.document.capture_id).recording_session()
    assert recovered.sources[0].frames == 80
    np.testing.assert_array_equal(sf.read(recovered.sources[0].path, dtype="int16")[0], [3210] * 80)
    monkeypatch.setattr(capture, "_publish", publish)
    writer.close()


def test_local_recovery_preserves_original_capture_cause(tmp_path):
    from here.application.recovery import RecoveryService

    capture = make_capture(tmp_path)
    capture.finish(OSError("synthetic device disconnected"))
    service = RecoveryService(tmp_path)
    (candidate,) = service.discover()
    directory = service.materialize(candidate)
    errors = json.loads((directory / "errors.json").read_text())["errors"]
    assert any(
        error["type"] == "OSError" and error["message"] == "synthetic device disconnected"
        for error in errors
    )
    assert service.discover()[0].error_summary == "synthetic device disconnected"


def test_explicit_retry_publishes_pending_before_offline_work(tmp_path):
    from here.application.processing import SessionProcessor
    from here.application.recovery import RecoveryService
    from here.transcription.client import TranscriptionResult

    make_capture(tmp_path)
    service = RecoveryService(tmp_path)
    (candidate,) = service.discover()
    directory = service.materialize(candidate)

    def transcribe(*args, **kwargs):
        metadata = json.loads((directory / "session.json").read_text())
        assert metadata["status"] == "pending"
        assert metadata["recoverable_audio"] == "audio.wav"
        return TranscriptionResult("ok", "ok")

    assert (
        SessionProcessor(transcribe=transcribe, retry_delays=()).retry(directory).metadata.status
        == "completed"
    )


@pytest.mark.parametrize(
    "boundary", ["reservation", "normalize", "pending", "final_before", "final_after", "remove"]
)
def test_real_kill_at_publication_boundaries(tmp_path, boundary):
    script = """
import sys,time
from pathlib import Path
import numpy as np
import here.application.processing as processing
import here.output.session_writer as output
from here.recording.journal import CaptureJournal
from here.transcription.client import TranscriptionResult
root=Path(sys.argv[1]); phase=sys.argv[2]
j=CaptureJournal.create(root)
def block():
    (root/'ready').write_text(j.document.capture_id)
    while True: time.sleep(1)
if phase=='reservation': block()
w=j.open_writer(label='synthetic',sample_rate=8000,channels=1)
w.write(np.full(80,1234,dtype=np.int16)); w.close()
if phase=='normalize': processing.materialize_normalized_session=lambda *a,**kw:block()
if phase=='remove': CaptureJournal.discard=lambda *a:block()
original=output._publish_metadata_and_segments
def publish(path,meta,segments,obsolete_paths=()):
    import json
    state=json.loads(meta)['status']
    if phase=='final_before' and state=='completed': block()
    original(path,meta,segments,obsolete_paths)
    if (phase=='pending' and state=='pending') or (phase=='final_after' and state=='completed'):
        block()
output._publish_metadata_and_segments=publish
processor=processing.SessionProcessor(transcribe=lambda *a,**kw:TranscriptionResult('ok','ok'))
processor.process(j.recording_session(),root)
"""
    process = subprocess.Popen([sys.executable, "-c", script, str(tmp_path), boundary])
    try:
        deadline = time.monotonic() + 10
        while not (tmp_path / "ready").exists() and time.monotonic() < deadline:
            assert process.poll() is None
            time.sleep(0.02)
        assert (tmp_path / "ready").exists()
        process.kill()
        process.wait(5)
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(5)
    from here.application.recovery import RecoveryService

    capture_id = (tmp_path / "ready").read_text()
    journal = journal_type().load(tmp_path, capture_id)
    service = RecoveryService(tmp_path)
    if boundary in {"final_after", "remove"}:
        assert service.discover() == []
        assert (
            json.loads((journal.destination / "session.json").read_text())["meeting_id"]
            == capture_id
        )
    else:
        (candidate,) = service.discover()
        assert candidate.capture_id == capture_id
        assert candidate.can_retry == (boundary != "reservation")
        if candidate.can_retry:
            assert (
                service.materialize(candidate)
                == service.materialize(candidate)
                == journal.destination
            )


@pytest.mark.parametrize(
    "boundary",
    [
        "mkdir",
        "normalize",
        "pending_before",
        "pending_after",
        "final_before",
        "final_after",
        "remove",
    ],
)
def test_crash_boundaries_keep_mapping_and_completed_metadata_wins(tmp_path, monkeypatch, boundary):
    import here.output.session_writer as output
    from here.application.processing import SessionProcessor
    from here.application.recovery import RecoveryService
    from here.transcription.client import TranscriptionResult

    capture = make_capture(tmp_path)
    processor = SessionProcessor(transcribe=lambda *a, **kw: TranscriptionResult("ok", "ok"))

    def crash():
        raise SystemExit("synthetic process death")

    with monkeypatch.context() as patch:
        if boundary == "mkdir":
            original_mkdir = Path.mkdir

            def mkdir(path, *args, **kwargs):
                if path == capture.destination:
                    crash()
                return original_mkdir(path, *args, **kwargs)

            patch.setattr(Path, "mkdir", mkdir)
        elif boundary == "normalize":
            patch.setattr(processor, "_materialize", lambda *a, **kw: crash())
        elif boundary == "remove":
            patch.setattr(type(capture), "discard", lambda *a: crash())
        else:
            publish = output._publish_metadata_and_segments

            def publish_at(path, metadata, segments, obsolete_paths=()):
                state = json.loads(metadata)["status"]
                target = "pending" if boundary.startswith("pending") else "completed"
                if state == target and boundary.endswith("before"):
                    crash()
                publish(path, metadata, segments, obsolete_paths)
                if state == target and boundary.endswith("after"):
                    crash()

            patch.setattr(output, "_publish_metadata_and_segments", publish_at)
        with pytest.raises(SystemExit):
            processor.process(capture.recording_session(), tmp_path)
    assert capture.directory.exists()
    service = RecoveryService(tmp_path)
    if boundary in {"final_after", "remove"}:
        assert service.discover() == []
        assert (
            json.loads((capture.destination / "session.json").read_text())["status"] == "completed"
        )
    else:
        (candidate,) = service.discover()
        path = service.materialize(candidate)
        assert service.materialize(candidate) == path == capture.destination
        finished = processor.retry(path)
        assert finished.metadata.meeting_id == capture.document.capture_id
        assert finished.metadata.status == "completed"
        assert service.discover() == []
        assert not capture.directory.exists()


def test_revalidate_junction_replaced_after_discovery(tmp_path):
    from here.application.recovery import RecoveryService
    from here.output.paths import UnsafeSessionPath

    root = tmp_path / "sessions"
    make_capture(root)
    service = RecoveryService(root)
    (candidate,) = service.discover()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "canary").write_text("untouched")
    if sys.platform == "win32":
        result = subprocess.run(
            ["cmd", "/c", "mklink", "/J", str(candidate.session_dir), str(outside)],
            capture_output=True,
        )
        assert result.returncode == 0, result.stderr
    else:
        candidate.session_dir.symlink_to(outside, target_is_directory=True)
    with pytest.raises(UnsafeSessionPath):
        service.materialize(candidate)
    assert list(outside.iterdir()) == [outside / "canary"]
    assert (outside / "canary").read_text() == "untouched"


@pytest.mark.parametrize("tamper", ["traversal", "absolute", "geometry", "redirect"])
def test_revalidate_journal_sources_before_materialization(tmp_path, tamper):
    from here.application.recovery import RecoveryService

    capture = make_capture(tmp_path)
    service = RecoveryService(tmp_path)
    (candidate,) = service.discover()
    manifest = capture.directory / "journal.json"
    document = json.loads(manifest.read_text())
    canary = tmp_path / "outside.wav"
    sf.write(canary, np.full(80, 4567, dtype=np.int16), 8000)
    before = canary.read_bytes()
    if tamper == "traversal":
        document["sources"][0]["audio_file"] = "../../outside.wav"
    elif tamper == "absolute":
        document["sources"][0]["audio_file"] = str(canary)
    elif tamper == "geometry":
        document["sources"][0]["channels"] = 3
    else:
        source = capture.directory / document["sources"][0]["audio_file"]
        source.unlink()
        try:
            source.symlink_to(canary)
        except OSError:
            source.hardlink_to(canary)
    manifest.write_text(json.dumps(document))
    with pytest.raises(ValueError):
        service.materialize(candidate)
    assert canary.read_bytes() == before
    assert not candidate.session_dir.exists()


def test_cancel_deletes_only_capture_and_revalidates_sources(tmp_path):
    from here.output.paths import UnsafeSessionPath

    capture = make_capture(tmp_path)
    outside = tmp_path / "canary.wav"
    outside.write_bytes(b"canary")
    path = capture.directory / "journal.json"
    data = json.loads(path.read_text())
    data["sources"][0]["audio_file"] = "../canary.wav"
    path.write_text(json.dumps(data))
    with pytest.raises(UnsafeSessionPath):
        capture.discard()
    assert outside.read_bytes() == b"canary"
    assert path.exists()


@pytest.mark.parametrize("intent", ["stop", "cancel", "pause"])
def test_control_interrupts_catchup_between_blocks(tmp_path, monkeypatch, intent):
    import threading

    from here.recording import windows

    capture = journal_type().create(tmp_path)
    writer = capture.open_writer(label="mic", sample_rate=8000, channels=1)
    stop = threading.Event()
    pause = threading.Event()
    delivered = []
    reads = []
    errors = []

    class Stream:
        def get_read_available(self):
            if pause.is_set():
                stop.set()
            reads.append(1)
            return 0

    def sink(label, block, rate, channels):
        delivered.extend(block[:, 0].tolist())
        if len(delivered) == 80:
            (pause if intent == "pause" else stop).set()

    monkeypatch.setattr(windows.time, "perf_counter", lambda: 10)
    count = [0]
    windows._capture_windows_stream_to_file(
        Stream(),
        chunk=80,
        sample_rate=8000,
        channels=1,
        writer=writer,
        stop_event=stop,
        pause_event=pause,
        errors=errors,
        label="mic",
        written_frames=count,
        start_time=0,
        block_sink=sink,
    )
    writer.close()
    data, _ = sf.read(writer.path, dtype="int16")
    assert len(data) == count[0] == len(delivered) == 80
    assert len(reads) <= 1
    assert not errors
    assert any(event["kind"] == "scheduling_gap" for event in capture.document.events)
