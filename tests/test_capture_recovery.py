import json
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
def publish(path,meta,segments):
    import json
    state=json.loads(meta)['status']
    if phase=='final_before' and state=='completed': block()
    original(path,meta,segments)
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

            def publish_at(path, metadata, segments):
                state = json.loads(metadata)["status"]
                target = "pending" if boundary.startswith("pending") else "completed"
                if state == target and boundary.endswith("before"):
                    crash()
                publish(path, metadata, segments)
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
