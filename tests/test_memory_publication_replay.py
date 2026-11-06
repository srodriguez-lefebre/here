"""An already committed publication replays once; a new retry stays a new revision."""

from __future__ import annotations

import gc
import importlib
import json
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest
from memory_fixtures import MEETING_ID, SEGMENT_TEXTS, TRANSCRIPT_TEXT, make_session

CHANGED_TEXT = "Synthetic changed evidence for one committed publication."


def _rows(root, sql, parameters=()):
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        return connection.execute(sql, parameters).fetchall()


def _revisions(root):
    return _rows(
        root,
        "SELECT id,meeting_id,revision_number,raw_text,content_hash "
        "FROM transcript_revisions ORDER BY revision_number",
    )


def _setup(tmp_path, monkeypatch):
    root = tmp_path / "data"
    monkeypatch.setenv("HERE_DATA_DIR", str(root))
    monkeypatch.setenv("HERE_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-publication-replay-key")
    monkeypatch.setenv("TRANSCRIPTION_MODEL", "synthetic-transcription-model")
    monkeypatch.setenv("CLEANUP_MODEL", "synthetic-cleanup-model")
    monkeypatch.setenv("CLEANUP_ENABLED", "false")

    def forbidden(*args, **kwargs):
        pytest.fail("Local publication recovery invoked provider/network transport")

    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr("here.transcription.client.build_client", forbidden)
    memory = importlib.import_module("here.memory.service").MemoryService(root)
    source = make_session(tmp_path / "sessions" / "recorded", meeting_id=MEETING_ID)
    assert memory.ingest_session(source.directory).status == "inserted"
    return root, memory, source


def _result(identical):
    client = importlib.import_module("here.transcription.client")
    segment = importlib.import_module("here.transcription.segments").TranscriptSegment
    evidence = (
        [
            segment(
                text,
                start=float(index),
                end=float(index + 1),
                speaker="speaker_0",
                chunk_index=index,
                speaker_scope=f"synthetic-request-{index}",
            )
            for index, text in enumerate(SEGMENT_TEXTS)
        ]
        if identical
        else [segment(CHANGED_TEXT)]
    )
    text = TRANSCRIPT_TEXT if identical else CHANGED_TEXT
    return client.TranscriptionResult(raw_text=text, final_text=text, segments=evidence)


def _forbid_recovery_provider(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Local replay allocated another transcription")

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("here.application.processing.transcribe_recording_session", forbidden)
    monkeypatch.setattr("here.transcription.client.build_client", forbidden)


@pytest.mark.parametrize("identical", [False, True], ids=["changed", "identical"])
def test_controller_real_processor_publishes_retry_only_once_without_pending_warning(
    tmp_path, monkeypatch, identical
):
    root, memory, source = _setup(tmp_path, monkeypatch)
    original = _revisions(root)[0]
    audio_before = source.audio.read_bytes()
    processing = importlib.import_module("here.application.processing")
    application = importlib.import_module("here.application")
    calls = []

    def provider(*args, **kwargs):
        calls.append("explicit retry")
        return _result(identical)

    processor = processing.SessionProcessor(
        transcribe=provider, retry_delays=(), memory_service=memory
    )
    controller = application.HereApplicationController(processor=processor, memory_service=memory)
    controller.retry(source.directory)
    terminal = controller.wait_until_terminal(20)
    assert terminal.worker_complete and terminal.state.value == "completed"
    assert terminal.last_error is None
    revisions = _revisions(root)
    assert len(revisions) == 2 and revisions[0] == original
    assert revisions[1][2:4] == (2, TRANSCRIPT_TEXT if identical else CHANGED_TEXT)
    assert _rows(root, "SELECT status,active_revision_id FROM meetings") == [
        ("completed", revisions[1][0])
    ]
    assert terminal.memory_warning is None, "Controller replay falsely reported library pending"
    assert not (source.directory / "memory_pending.json").exists()
    assert source.audio.read_bytes() == audio_before and calls == ["explicit retry"]
    _forbid_recovery_provider(monkeypatch)
    recovery = importlib.import_module("here.application.recovery").RecoveryService(
        source.directory.parent
    )
    recovery.discover()
    recovery.discover()
    assert _revisions(root) == revisions
    assert not (source.directory / "memory_pending.json").exists()


_CRASH_BEFORE_MARKER_UNLINK = r"""
import json, os, pathlib, socket, sqlite3, sys, types
source, tests, data, directory, checkpoint, identical = sys.argv[1:]
package = types.ModuleType('here')
package.__path__ = [source]
sys.modules['here'] = package
sys.path.insert(0, tests)
from test_memory_publication_replay import _result
from here.application.processing import SessionProcessor
from here.memory.service import MemoryService
from here.transcription import client
def forbidden(*args, **kwargs):
    raise AssertionError('Synthetic crash child allocated provider/network transport')
socket.socket.connect = forbidden
client.build_client = forbidden
root, directory = pathlib.Path(data), pathlib.Path(directory)
marker = directory / 'memory_pending.json'
calls = []
with sqlite3.connect(root / 'library.sqlite3') as connection:
    previous = connection.execute('SELECT active_revision_id FROM meetings').fetchone()[0]
original_unlink = pathlib.Path.unlink
def crash_at_committed_marker(path, *args, **kwargs):
    if path.absolute() == marker.absolute():
        with sqlite3.connect(root / 'library.sqlite3') as connection:
            revisions = connection.execute(
                'SELECT id,meeting_id,revision_number,raw_text,content_hash '
                'FROM transcript_revisions ORDER BY revision_number'
            ).fetchall()
            meeting = connection.execute(
                'SELECT id,status,active_revision_id FROM meetings'
            ).fetchone()
        intent = json.loads(marker.read_text(encoding='utf-8'))
        assert len(revisions) == 2 and revisions[0][0] == previous
        assert meeting[1:] == ('completed', revisions[1][0])
        assert intent['revision_of'] == previous and calls == ['explicit retry']
        pathlib.Path(checkpoint).write_text(json.dumps({
            'revisions': revisions, 'meeting': meeting, 'parent': previous,
            'marker': intent,
        }), encoding='utf-8')
        os._exit(73)
    return original_unlink(path, *args, **kwargs)
pathlib.Path.unlink = crash_at_committed_marker
def provider(*args, **kwargs):
    calls.append('explicit retry')
    return _result(identical == 'true')
processor = SessionProcessor(
    transcribe=provider, retry_delays=(), memory_service=MemoryService(root)
)
processor.retry(directory)
raise AssertionError('Actual publication never reached committed source-marker unlink')
"""


@pytest.mark.parametrize("identical", [False, True], ids=["changed", "identical"])
@pytest.mark.parametrize("rebuild", [False, True], ids=["existing-db", "rebuilt-db"])
def test_crash_after_real_commit_before_origin_marker_unlink_replays_locally_once(
    tmp_path, monkeypatch, identical, rebuild
):
    root, memory, source = _setup(tmp_path, monkeypatch)
    original = _revisions(root)[0]
    checkpoint = tmp_path / "committed-before-source-unlink.json"
    package = importlib.import_module("here")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            _CRASH_BEFORE_MARKER_UNLINK,
            str(Path(next(iter(package.__path__))).resolve()),
            str(Path(__file__).parent),
            str(root),
            str(source.directory),
            str(checkpoint),
            "true" if identical else "false",
        ],
        env=dict(os.environ),
        capture_output=True,
        timeout=30,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert result.returncode == 73, result.stdout.decode(errors="replace") + result.stderr.decode(
        errors="replace"
    )
    assert checkpoint.is_file(), "Child must prove R2 committed before its deliberate exit"
    proof = json.loads(checkpoint.read_text(encoding="utf-8"))
    committed = _revisions(root)
    assert committed == [tuple(row) for row in proof["revisions"]]
    assert len(committed) == 2 and committed[0] == original
    assert committed[1][2:4] == (2, TRANSCRIPT_TEXT if identical else CHANGED_TEXT)
    assert _rows(root, "SELECT id,status,active_revision_id FROM meetings") == [
        (MEETING_ID, "completed", committed[1][0])
    ]
    marker = source.directory / "memory_pending.json"
    assert json.loads(marker.read_text(encoding="utf-8")) == proof["marker"]
    assert proof["marker"]["revision_of"] == original[0]
    preserved = {
        path.name: path.read_bytes()
        for path in source.directory.iterdir()
        if path.is_file() and path != marker
    }
    _forbid_recovery_provider(monkeypatch)
    if rebuild:
        with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
            assert connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] == 0
        del memory
        gc.collect()
        assert root.resolve().is_relative_to(tmp_path.resolve())
        for suffix in ("", "-wal", "-shm", "-journal"):
            path = root / ("library.sqlite3" + suffix)
            assert path.absolute().parent == root.absolute() and not path.is_symlink()
            path.unlink(missing_ok=True)
        memory = importlib.import_module("here.memory.service").MemoryService(root)
    memory.reconcile()
    assert _revisions(root) == committed
    recovery = importlib.import_module("here.application.recovery").RecoveryService(
        source.directory.parent
    )
    recovery.discover()
    assert not marker.exists(), "Durably committed source publication remained pending after replay"
    recovery.discover()
    assert _revisions(root) == committed
    assert _rows(root, "SELECT status,active_revision_id FROM meetings") == [
        ("completed", committed[1][0])
    ]
    assert {
        path.name: path.read_bytes() for path in source.directory.iterdir() if path.is_file()
    } == preserved
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_control_arbitrary_stale_parent_cannot_publish_different_source_content(
    tmp_path, monkeypatch
):
    root, memory, source = _setup(tmp_path, monkeypatch)
    original = _revisions(root)[0]
    processing = importlib.import_module("here.application.processing")
    processor = processing.SessionProcessor(
        transcribe=lambda *args, **kwargs: _result(False), retry_delays=(), memory_service=memory
    )
    processor.retry(source.directory)
    before = _revisions(root)
    assert len(before) == 2
    forged = make_session(
        tmp_path / "arbitrary-stale-origin",
        meeting_id=MEETING_ID,
        transcript_text="Synthetic arbitrary content is not a committed publication replay.",
    )
    bytes_before = {path.name: path.read_bytes() for path in forged.directory.iterdir()}
    result = memory.ingest_session(forged.directory, revision_of=original[0])
    assert result.status == "rejected"
    assert _revisions(root) == before
    assert _rows(root, "SELECT active_revision_id FROM meetings") == [(before[1][0],)]
    assert {path.name: path.read_bytes() for path in forged.directory.iterdir()} == bytes_before


def test_control_each_new_identical_controller_retry_adds_one_immutable_revision(
    tmp_path, monkeypatch
):
    root, memory, source = _setup(tmp_path, monkeypatch)
    original = _revisions(root)[0]
    processing = importlib.import_module("here.application.processing")
    application = importlib.import_module("here.application")
    calls = []

    def provider(*args, **kwargs):
        calls.append("new explicit retry")
        return _result(True)

    processor = processing.SessionProcessor(
        transcribe=provider, retry_delays=(), memory_service=memory
    )
    controller = application.HereApplicationController(processor=processor, memory_service=memory)
    for _attempt in range(2):
        controller.retry(source.directory)
        terminal = controller.wait_until_terminal(20)
        assert terminal.worker_complete and terminal.state.value == "completed"
    revisions = _revisions(root)
    assert len(revisions) == 3 and revisions[0] == original
    assert [row[2] for row in revisions] == [1, 2, 3]
    assert len({row[0] for row in revisions}) == 3
    assert all(row[3:] == original[3:] for row in revisions)
    assert _rows(root, "SELECT status,active_revision_id FROM meetings") == [
        ("completed", revisions[-1][0])
    ]
    assert calls == ["new explicit retry", "new explicit retry"]
