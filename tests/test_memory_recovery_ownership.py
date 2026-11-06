"""Real process ownership and catalog-only recovery after successful publication."""

from __future__ import annotations

import importlib
import json
import os
import sqlite3
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from memory_fixtures import MEETING_ID, OTHER_MEETING_ID, make_session, write_json, write_marked_wav
from typer.testing import CliRunner


def _memory(root):
    return importlib.import_module("here.memory.service").MemoryService(root)


def _configure(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("HERE_DATA_DIR", str(root))
    monkeypatch.setenv("HERE_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-private-canary-not-for-memory")
    return root


def _row(root, sql, parameters=()):
    with sqlite3.connect(root / "library.sqlite3") as connection:
        return connection.execute(sql, parameters).fetchone()


_CHILD_HEADER = r"""
import json, os, pathlib, sqlite3, sys, types
source_root, data = map(pathlib.Path, sys.argv[1:3])
here = types.ModuleType("here")
here.__path__ = [str(source_root / "src")]
sys.modules["here"] = here
from here.memory.service import MemoryService
"""

_OWNER_SCRIPT = (
    _CHILD_HEADER
    + r"""
ready, status = pathlib.Path(sys.argv[3]), sys.argv[4]
identity = "11111111-1111-4111-8111-111111111111"
memory = MemoryService(data)
memory.begin_capture(identity)
if status == "processing": memory.set_status(identity, status)
stage = data / ".staging" / "owned-publication"
stage.mkdir(parents=True, exist_ok=True)
(stage / "audio.wav").write_bytes(b"synthetic staging material still owned by child")
with sqlite3.connect(data / "library.sqlite3") as connection:
    connection.execute(
        "INSERT INTO pending_operations "
        "(id,kind,phase,meeting_id,staging_path,destination_path,payload_json) "
        "VALUES ('owned-operation','publish_revision','prepared',?,?,?,'{}')",
        (identity, ".staging/owned-publication", "meetings/" + identity + "/revisions/owned"),
    )
ready.write_text("ownership established", encoding="utf-8")
sys.stdin.buffer.read(1)
"""
)

_RECONCILE_SCRIPT = (
    _CHILD_HEADER
    + r"""
MemoryService(data).reconcile()
"""
)


def _child_arguments(script, root, *extra):
    return [
        sys.executable,
        "-c",
        script,
        str(Path(__file__).resolve().parents[1]),
        str(root),
        *map(str, extra),
    ]


def _child_env(root, tmp_path):
    environment = dict(os.environ)
    environment.pop("OPENAI_API_KEY", None)
    environment["HERE_ENV_FILE"] = str(tmp_path / "absent.env")
    environment["HERE_DATA_DIR"] = str(root)
    return environment


@pytest.mark.parametrize("status", ["recording", "processing"])
def test_separate_reconciler_preserves_real_live_owner_then_detects_process_death(
    tmp_path,
    monkeypatch,
    status,
):
    root = _configure(tmp_path, monkeypatch)
    memory = _memory(root)
    ready = tmp_path / "owner-ready"
    owner = subprocess.Popen(
        _child_arguments(_OWNER_SCRIPT, root, ready, status),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_child_env(root, tmp_path),
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and owner.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.is_file(), "Owner process did not establish ownership and prepared materials"
        assert owner.poll() is None
        before = (root / ".staging" / "owned-publication" / "audio.wav").read_bytes()
        other = subprocess.run(
            _child_arguments(_RECONCILE_SCRIPT, root),
            env=_child_env(root, tmp_path),
            capture_output=True,
            text=True,
            timeout=10,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        assert other.returncode == 0, other.stdout + other.stderr
        assert _row(root, "SELECT status FROM meetings WHERE id=?", (MEETING_ID,)) == (status,)
        assert _row(root, "SELECT phase FROM pending_operations WHERE id='owned-operation'") == (
            "prepared",
        )
        assert (root / ".staging" / "owned-publication" / "audio.wav").read_bytes() == before
        owner.kill()
        owner.wait(timeout=5)
        monkeypatch.setattr(
            "socket.socket.connect", lambda *args: pytest.fail("Recovery used network")
        )
        memory.reconcile()
        assert _row(root, "SELECT status FROM meetings WHERE id=?", (MEETING_ID,)) == (
            "interrupted",
        )
        memory.reconcile()
        assert _row(root, "SELECT COUNT(*) FROM meetings")[0] == 1
    finally:
        if owner.poll() is None:
            owner.kill()
        owner.communicate(timeout=5)


class _Capture:
    meeting_id = MEETING_ID

    def __init__(self, session, *, failure=None):
        self.session = session
        self.failure = failure
        self.done = threading.Event()

    def stop(self):
        self.done.set()

    def cancel(self):
        self.done.set()

    def wait(self):
        assert self.done.wait(5)
        if self.failure:
            raise self.failure
        return self.session


class _Live:
    def abort(self):
        pass

    def cleanup(self):
        pass


def _wait_recording(controller):
    deadline = time.monotonic() + 5
    while controller.snapshot.state.value == "preparing" and time.monotonic() < deadline:
        time.sleep(0.01)
    assert controller.snapshot.state.value == "recording"


def _provider_result(text):
    client = importlib.import_module("here.transcription.client")
    segments = importlib.import_module("here.transcription.segments")
    return client.TranscriptionResult(
        raw_text=text,
        final_text=text,
        segments=[segments.TranscriptSegment(text, start=0.0, end=1.0)],
    )


def _forbid_provider(*args, **kwargs):
    pytest.fail("Catalog recovery must never invoke a provider")


@pytest.mark.parametrize("producer", ["recording", "file"])
def test_busy_catalog_leaves_owned_marker_and_startup_replays_only_local_publication(
    tmp_path,
    monkeypatch,
    producer,
):
    root = _configure(tmp_path, monkeypatch)
    memory = _memory(root)
    application = importlib.import_module("here.application")
    models = importlib.import_module("here.recording.models")
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    unrelated = make_session(tmp_path / "unselected-external", meeting_id=OTHER_MEETING_ID)
    unrelated_before = unrelated.metadata.read_bytes()
    with sqlite3.connect(root / "library.sqlite3", isolation_level=None) as blocker:
        blocker.execute("BEGIN IMMEDIATE")
        if producer == "recording":
            source = make_session(sessions / "recorded", meeting_id=MEETING_ID)
            capture = _Capture(models.RecordingSession([], meeting_id=MEETING_ID))

            class PublishedProcessor:
                def process(self, *args, **kwargs):
                    return SimpleNamespace(session_dir=source.directory)

            controller = application.HereApplicationController(
                capture_factory=lambda *args: capture,
                live_factory=lambda *args: _Live(),
                processor=PublishedProcessor(),
                memory_service=memory,
            )
            controller.start(application.StartRequest(output_dir=sessions))
            try:
                _wait_recording(controller)
                controller.stop()
                assert controller.wait_until_terminal(8).state.value == "completed"
            finally:
                capture.done.set()
                controller.wait_until_terminal(8)
            directory = source.directory
        else:
            cli = importlib.import_module("here.cli")
            audio = tmp_path / "external.wav"
            write_marked_wav(audio)
            monkeypatch.setattr(
                cli,
                "transcribe_recording_session",
                lambda *args, **kwargs: _provider_result("Synthetic file evidence"),
            )
            result = CliRunner().invoke(
                cli.app, ["trans", str(audio), "--output-dir", str(sessions)]
            )
            assert result.exit_code == 0, result.output + repr(result.exception)
            directories = [path for path in sessions.iterdir() if not path.name.startswith(".")]
            assert len(directories) == 1
            directory = directories[0]
        marker = directory / "memory_pending.json"
        assert marker.is_file(), "Successful file publication needs a durable local retry marker"
        document = json.loads(marker.read_text(encoding="utf-8"))
        assert document["revision_of"] is None
        assert len(marker.read_bytes()) < 2048
        assert "Synthetic file evidence" not in marker.read_text(encoding="utf-8")
        assert "Synthetic copper canary" not in marker.read_text(encoding="utf-8")
        assert "synthetic-private-canary-not-for-memory" not in marker.read_text(encoding="utf-8")
        preserved = (
            (directory / "audio.wav").read_bytes(),
            (directory / "session.json").read_bytes(),
        )
        blocker.rollback()
    processing = importlib.import_module("here.application.processing")
    cli = importlib.import_module("here.cli")
    monkeypatch.setattr(processing, "transcribe_recording_session", _forbid_provider)
    monkeypatch.setattr(cli, "transcribe_recording_session", _forbid_provider)
    monkeypatch.setattr("socket.socket.connect", _forbid_provider)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    recovery = importlib.import_module("here.application.recovery").RecoveryService(sessions)
    recovery.discover()
    recovery.discover()
    assert not marker.exists()
    assert _row(root, "SELECT COUNT(*) FROM meetings") == (1,)
    assert _row(root, "SELECT status FROM meetings") == ("completed",)
    assert _row(root, "SELECT COUNT(*) FROM audio_assets") == (1,)
    assert _row(root, "SELECT COUNT(*) FROM transcript_revisions") == (1,)
    assert (
        (directory / "audio.wav").read_bytes(),
        (directory / "session.json").read_bytes(),
    ) == preserved
    assert unrelated.metadata.read_bytes() == unrelated_before


def test_capture_failure_artifacts_are_cataloged_as_failed_and_emit_memory_result(
    tmp_path, monkeypatch
):
    root = _configure(tmp_path, monkeypatch)
    memory = _memory(root)
    application = importlib.import_module("here.application")
    models = importlib.import_module("here.recording.models")
    source = make_session(tmp_path / "sessions" / "failed", meeting_id=MEETING_ID)
    metadata = json.loads(source.metadata.read_text(encoding="utf-8"))
    metadata["status"] = "failed"
    write_json(source.metadata, metadata)
    failure = models.CaptureFailed(
        models.RecordingSession([], meeting_id=MEETING_ID),
        RuntimeError("Synthetic capture failure"),
    )
    capture = _Capture(failure.session, failure=failure)

    class FailureProcessor:
        def persist_capture_failure(self, *args, **kwargs):
            return SimpleNamespace(session_dir=source.directory, errors=None)

    controller = application.HereApplicationController(
        capture_factory=lambda *args: capture,
        live_factory=lambda *args: _Live(),
        processor=FailureProcessor(),
        memory_service=memory,
    )
    events = []
    controller.subscribe(events.append)
    controller.start(application.StartRequest(output_dir=source.directory.parent))
    try:
        _wait_recording(controller)
        controller.stop()
        snapshot = controller.wait_until_terminal(8)
        assert snapshot.state.value == "failed"
        assert snapshot.recoverable
        assert source.audio.is_file() and source.metadata.is_file()
        assert _row(root, "SELECT id,status FROM meetings") == (MEETING_ID, "failed")
        assert _row(root, "SELECT COUNT(*) FROM audio_assets") == (1,)
        result = [event for event in events if event.kind.value == "memory_ingested"]
        assert len(result) == 1
        assert result[0].details["meeting_id"] == MEETING_ID
        assert result[0].details["status"] in {"inserted", "revised"}
    finally:
        capture.done.set()
        controller.wait_until_terminal(8)


@pytest.mark.parametrize("advance_revision", [False, True], ids=["recover", "stale-intent"])
def test_controller_retry_catalog_fault_keeps_prior_revision_and_trusted_replay_intent(
    tmp_path,
    monkeypatch,
    advance_revision,
):
    root = _configure(tmp_path, monkeypatch)
    memory = _memory(root)
    source = make_session(tmp_path / "sessions" / "recorded", meeting_id=MEETING_ID)
    assert memory.ingest_session(source.directory).status == "inserted"
    previous = _row(root, "SELECT id,raw_text FROM transcript_revisions")
    processing = importlib.import_module("here.application.processing")
    application = importlib.import_module("here.application")
    processor = processing.SessionProcessor(
        transcribe=lambda *args, **kwargs: _provider_result("Pending retry evidence"),
        retry_delays=(),
    )
    controller = application.HereApplicationController(processor=processor, memory_service=memory)
    with sqlite3.connect(root / "library.sqlite3", isolation_level=None) as blocker:
        blocker.execute("BEGIN IMMEDIATE")
        controller.retry(source.directory)
        assert controller.wait_until_terminal(8).state.value == "completed"
        assert _row(root, "SELECT active_revision_id FROM meetings") == (previous[0],)
        assert _row(root, "SELECT id,raw_text FROM transcript_revisions") == previous
        marker = source.directory / "memory_pending.json"
        assert marker.is_file()
        document = json.loads(marker.read_text(encoding="utf-8"))
        assert document["revision_of"] == previous[0]
        assert "Pending retry evidence" not in marker.read_text(encoding="utf-8")
        assert "synthetic-private-canary-not-for-memory" not in marker.read_text(encoding="utf-8")
        blocker.rollback()
    expected_text = "Pending retry evidence"
    if advance_revision:
        fresh = make_session(
            tmp_path / "fresh-explicit-retry",
            meeting_id=MEETING_ID,
            transcript_text="Newer trusted evidence",
            segments=[{"text": "Newer trusted evidence", "start": 0.0, "end": 1.0}],
        )
        assert memory.ingest_session(fresh.directory, revision_of=previous[0]).status == "revised"
        expected_text = "Newer trusted evidence"
    monkeypatch.setattr(processing, "transcribe_recording_session", _forbid_provider)
    monkeypatch.setattr("socket.socket.connect", _forbid_provider)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    recovery = importlib.import_module("here.application.recovery").RecoveryService(
        source.directory.parent
    )
    recovery.discover()
    recovery.discover()
    if not advance_revision:
        assert not marker.exists()
    assert _row(root, "SELECT COUNT(*) FROM meetings") == (1,)
    assert _row(root, "SELECT COUNT(*) FROM transcript_revisions") == (2,)
    assert _row(root, "SELECT raw_text FROM transcript_revisions WHERE id=?", (previous[0],)) == (
        previous[1],
    )
    assert _row(
        root,
        "SELECT r.raw_text FROM transcript_revisions r JOIN meetings m "
        "ON r.id=m.active_revision_id",
    ) == (expected_text,)
