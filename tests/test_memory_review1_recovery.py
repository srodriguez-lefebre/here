"""Round-one review regressions for untrusted recovery envelopes."""

from __future__ import annotations

import importlib
import os
import sqlite3
import subprocess
import sys
import time
from contextlib import closing
from pathlib import Path

import numpy as np
import pytest
from memory_fixtures import MEETING_ID, OTHER_MEETING_ID, make_session, write_json


@pytest.mark.parametrize(
    "envelope",
    [
        [],
        None,
        {"schema_version": 1, "meeting_id": MEETING_ID, "revision_of": 42},
    ],
    ids=["array", "null", "non-string-revision"],
)
def test_malformed_memory_marker_cannot_abort_valid_local_sibling_replay(
    tmp_path,
    monkeypatch,
    envelope,
):
    root = tmp_path / "data"
    monkeypatch.setenv("HERE_DATA_DIR", str(root))
    monkeypatch.setenv("HERE_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    service = importlib.import_module("here.memory.service").MemoryService(root)
    sessions = tmp_path / "sessions"
    bad = make_session(sessions / "a-malformed", meeting_id=MEETING_ID)
    good = make_session(sessions / "z-valid", meeting_id=OTHER_MEETING_ID)
    bad_marker = bad.directory / "memory_pending.json"
    good_marker = good.directory / "memory_pending.json"
    write_json(bad_marker, envelope)
    write_json(
        good_marker, {"schema_version": 1, "meeting_id": OTHER_MEETING_ID, "revision_of": None}
    )
    original_bad = bad_marker.read_bytes()

    def forbidden(*args, **kwargs):
        pytest.fail("Local marker replay allocated provider/network work")

    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr("here.transcription.client.build_client", forbidden)
    monkeypatch.setattr("here.application.processing.transcribe_recording_session", forbidden)
    recovery = importlib.import_module("here.application.recovery").RecoveryService(sessions)
    assert isinstance(recovery.discover(), list)
    recovery.discover()
    assert bad_marker.read_bytes() == original_bad
    assert not good_marker.exists()
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        assert connection.execute("SELECT id,status FROM meetings").fetchall() == [
            (OTHER_MEETING_ID, "completed")
        ]
        assert connection.execute("SELECT COUNT(*) FROM transcript_revisions").fetchone()[0] == 1
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert service.ingest_session(good.directory).status == "unchanged"


_CAPTURE_OWNER = r"""
import pathlib, sys, types
source, data, identity, ready = sys.argv[1:]
package = types.ModuleType("here")
package.__path__ = [source]
sys.modules["here"] = package
from here.memory.service import MemoryService
memory = MemoryService(pathlib.Path(data))
memory.begin_capture(identity)
pathlib.Path(ready).write_text("owner held", encoding="utf-8")
sys.stdin.buffer.read(1)
memory.finish_capture(identity)
"""


def test_live_capture_owner_blocks_discovery_and_stale_candidate_materialization(
    tmp_path,
    monkeypatch,
):
    root = tmp_path / "data"
    sessions = tmp_path / "sessions"
    monkeypatch.setenv("HERE_DATA_DIR", str(root))
    monkeypatch.setenv("HERE_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    service = importlib.import_module("here.memory.service")
    service.MemoryService(root)
    database = importlib.import_module("here.memory.database")
    journal_type = importlib.import_module("here.recording.journal").CaptureJournal
    recovery_type = importlib.import_module("here.application.recovery").RecoveryService
    journal = journal_type.create(sessions)
    writer = journal.open_writer(label="synthetic", sample_rate=8000, channels=1)
    writer.write(np.full((1024, 1), 1234, dtype=np.int16))
    writer.checkpoint()
    writer.close()
    recovery = recovery_type(sessions)
    (stale_candidate,) = recovery.discover()
    assert stale_candidate.can_retry
    before = {
        path.name: path.read_bytes() for path in journal.directory.iterdir() if path.is_file()
    }
    ready = tmp_path / "owner-ready"
    environment = dict(os.environ)
    environment["HERE_ENV_FILE"] = str(tmp_path / "absent.env")
    environment["HERE_DATA_DIR"] = str(root)
    owner = subprocess.Popen(
        [
            sys.executable,
            "-c",
            _CAPTURE_OWNER,
            str(Path(service.__file__).resolve().parents[1]),
            str(root),
            journal.document.capture_id,
            str(ready),
        ],
        env=environment,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )

    def forbidden(*args, **kwargs):
        pytest.fail("Materialization allocated provider/network work")

    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr("here.transcription.client.build_client", forbidden)
    try:
        deadline = time.monotonic() + 15
        while not ready.exists() and owner.poll() is None and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready.is_file() and owner.poll() is None
        live_candidates = recovery.discover()
        with pytest.raises(database.Busy):
            recovery.materialize(stale_candidate)
        assert not any(
            candidate.capture_id == journal.document.capture_id for candidate in live_candidates
        )
        assert {
            path.name: path.read_bytes() for path in journal.directory.iterdir() if path.is_file()
        } == before
        assert not journal.destination.exists()
        _stdout, stderr = owner.communicate(input=b"release", timeout=10)
        assert owner.returncode == 0, stderr.decode()
        candidates = recovery.discover()
        (available,) = [
            candidate
            for candidate in candidates
            if candidate.capture_id == journal.document.capture_id
        ]
        assert available.can_retry
        directory = recovery.materialize(available)
        assert (directory / "audio.wav").is_file()
        assert (directory / "session.json").is_file()
    finally:
        if owner.poll() is None:
            owner.kill()
        owner.communicate(timeout=5)
