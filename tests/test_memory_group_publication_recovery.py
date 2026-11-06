"""Whole-group GR1: final artifact commit already owns durable catalog recovery."""

from __future__ import annotations

import gc
import hashlib
import importlib
import json
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest
from memory_fixtures import MEETING_ID, TRANSCRIPT_TEXT, make_session, write_marked_wav

FINAL_TEXT = "Synthetic saved result after the actual final metadata replacement."


def _rows(root, sql, parameters=()):
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        return connection.execute(sql, parameters).fetchall()


_CRASH_AFTER_METADATA = r"""
import hashlib, json, os, pathlib, socket, sqlite3, sys, types
source, data, target, origin, proof_path, mode, boundary = sys.argv[1:]
package = types.ModuleType('here')
package.__path__ = [source]
sys.modules['here'] = package
from here.application.processing import SessionProcessor
from here.memory.service import MemoryService
from here.transcription import client
from here.transcription.segments import TranscriptSegment
root, target, origin = pathlib.Path(data), pathlib.Path(target), pathlib.Path(origin)
def forbidden(*args, **kwargs):
    raise AssertionError('Synthetic child allocated provider/network transport')
socket.socket.connect = forbidden
client.build_client = forbidden
calls = []
real_replace = pathlib.Path.replace
def observed_replace(path, destination, *args, **kwargs):
    destination = pathlib.Path(destination)
    is_manifest = destination.name == 'session.json' and destination.is_relative_to(target)
    before_final = (
        boundary == 'before-final' and is_manifest
        and json.loads(pathlib.Path(path).read_bytes())['status'] == 'completed'
    )
    result = None if before_final else real_replace(path, destination, *args, **kwargs)
    if not is_manifest or boundary == 'before-final' and not before_final:
        return result
    metadata_bytes = destination.read_bytes()
    metadata = json.loads(metadata_bytes)
    wanted = 'completed' if boundary == 'final' else 'pending'
    if metadata['status'] != wanted:
        return result
    directory = destination.parent
    with sqlite3.connect(root / 'library.sqlite3') as connection:
        meetings = connection.execute(
            'SELECT id,status,active_revision_id FROM meetings'
        ).fetchall()
        revisions = connection.execute(
            'SELECT id,meeting_id,revision_number,raw_text,content_hash '
            'FROM transcript_revisions ORDER BY revision_number'
        ).fetchall()
    marker = directory / 'memory_pending.json'
    marker_bytes = marker.read_bytes() if marker.exists() else None
    raw = directory / 'raw_transcript.txt'
    audio = directory / metadata['recoverable_audio']
    proof = {
        'directory': str(directory), 'meetings': meetings, 'revisions': revisions,
        'metadata_sha256': hashlib.sha256(metadata_bytes).hexdigest(),
        'metadata': metadata, 'provider_calls': len(calls),
        'marker': json.loads(marker_bytes) if marker_bytes is not None else None,
        'raw_text': raw.read_text(encoding='utf-8-sig') if raw.exists() else None,
        'audio_sha256': hashlib.sha256(audio.read_bytes()).hexdigest(),
        'prospective_final': json.loads(pathlib.Path(path).read_bytes()) if before_final else None,
    }
    assert len(meetings) == 1 and meetings[0][1] == 'processing'
    assert len(revisions) == (1 if mode == 'retry' else 0)
    if mode == 'retry':
        assert meetings[0][2] == revisions[0][0]
    else:
        assert meetings[0][2] is None
    if boundary in {'final', 'before-final'}:
        assert len(calls) == 1
        assert proof['raw_text'] == (
            'Synthetic saved result after the actual final metadata replacement.'
        )
    else:
        assert calls == []
    with pathlib.Path(proof_path).open('w', encoding='utf-8') as stream:
        json.dump(proof, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os._exit({'final': 81, 'pending': 82, 'before-final': 83}[boundary])
    return result
pathlib.Path.replace = observed_replace
def provider(*args, **kwargs):
    calls.append('explicit synthetic request')
    text = 'Synthetic saved result after the actual final metadata replacement.'
    return client.TranscriptionResult(raw_text=text, final_text=text,
                                      segments=[TranscriptSegment(text)])
processor = SessionProcessor(
    transcribe=provider, retry_delays=(), memory_service=MemoryService(root)
)
if mode == 'retry':
    processor.retry(origin)
else:
    processor.transcribe_audio_path(origin, target)
raise AssertionError('Actual compatibility metadata replacement never reached the crash boundary')
"""


def _run_crash(tmp_path, monkeypatch, mode, boundary):
    root, sessions = tmp_path / "data", tmp_path / "sessions"
    monkeypatch.setenv("HERE_DATA_DIR", str(root))
    monkeypatch.setenv("HERE_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-group-publication-key")
    monkeypatch.setenv("CLEANUP_ENABLED", "false")
    memory = importlib.import_module("here.memory.service").MemoryService(root)
    previous = []
    if mode == "retry":
        source = make_session(sessions / "canonical", meeting_id=MEETING_ID)
        assert memory.ingest_session(source.directory).status == "inserted"
        origin = source.directory
        input_audio = source.audio
        previous = _rows(
            root,
            "SELECT id,meeting_id,revision_number,raw_text,content_hash FROM transcript_revisions",
        )
    else:
        origin = tmp_path / "external-input.wav"
        write_marked_wav(origin)
        input_audio = origin
    original_input = input_audio.read_bytes()
    proof_path = tmp_path / "actual-metadata-replace-proof.json"
    package = importlib.import_module("here")
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            _CRASH_AFTER_METADATA,
            str(Path(next(iter(package.__path__))).resolve()),
            str(root),
            str(sessions),
            str(origin),
            str(proof_path),
            mode,
            boundary,
        ],
        env=dict(os.environ),
        capture_output=True,
        timeout=30,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert process.returncode == {"final": 81, "pending": 82, "before-final": 83}[boundary], (
        process.stdout.decode(errors="replace") + process.stderr.decode(errors="replace")
    )
    assert proof_path.is_file()
    proof = json.loads(proof_path.read_text(encoding="utf-8"))
    directory = Path(proof["directory"])
    assert directory.parent == sessions
    metadata_bytes = (directory / "session.json").read_bytes()
    assert hashlib.sha256(metadata_bytes).hexdigest() == proof["metadata_sha256"]
    assert json.loads(metadata_bytes) == proof["metadata"]
    assert proof["metadata"]["status"] == ("completed" if boundary == "final" else "pending")
    assert _rows(root, "SELECT id,status,active_revision_id FROM meetings") == [
        tuple(row) for row in proof["meetings"]
    ]
    assert (
        _rows(
            root,
            "SELECT id,meeting_id,revision_number,raw_text,content_hash FROM transcript_revisions",
        )
        == previous
    )
    assert (
        hashlib.sha256(
            (directory / proof["metadata"]["recoverable_audio"]).read_bytes()
        ).hexdigest()
        == proof["audio_sha256"]
    )
    marker = directory / "memory_pending.json"
    assert (json.loads(marker.read_text(encoding="utf-8")) if marker.exists() else None) == proof[
        "marker"
    ]
    assert input_audio.read_bytes() == original_input
    return root, directory, memory, previous, proof


def _keyless(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("Startup recovery invoked another provider/network operation")

    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr("here.transcription.client.build_client", forbidden)
    monkeypatch.setattr("here.application.processing.transcribe_recording_session", forbidden)


@pytest.mark.parametrize("mode", ["fresh-file", "retry"])
@pytest.mark.parametrize("rebuild", [False, True], ids=["existing-db", "rebuilt-db"])
def test_final_metadata_replace_crash_recovers_saved_result_locally_once(
    tmp_path, monkeypatch, mode, rebuild
):
    root, directory, memory, previous, proof = _run_crash(tmp_path, monkeypatch, mode, "final")
    assert proof["provider_calls"] == 1
    assert (directory / "raw_transcript.txt").read_text(encoding="utf-8-sig") == FINAL_TEXT
    saved_audio = (directory / proof["metadata"]["recoverable_audio"]).read_bytes()
    _keyless(monkeypatch)
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
    recovery = importlib.import_module("here.application.recovery").RecoveryService(
        directory.parent
    )
    memory.reconcile()
    assert (
        _rows(
            root,
            "SELECT id,meeting_id,revision_number,raw_text,content_hash "
            "FROM transcript_revisions ORDER BY revision_number",
        )
        == previous
    )
    recovery.discover()
    recovery.discover()
    revisions = _rows(
        root,
        "SELECT id,meeting_id,revision_number,raw_text,content_hash "
        "FROM transcript_revisions ORDER BY revision_number",
    )
    assert len(revisions) == len(previous) + 1, (
        "Saved final evidence was never published on startup"
    )
    assert revisions[: len(previous)] == previous
    assert revisions[-1][2:4] == (len(previous) + 1, FINAL_TEXT)
    identity = proof["meetings"][0][0]
    assert revisions[-1][1] == identity
    assert _rows(root, "SELECT id,status,active_revision_id FROM meetings") == [
        (identity, "completed", revisions[-1][0])
    ]
    assert not (directory / "memory_pending.json").exists()
    assert (directory / proof["metadata"]["recoverable_audio"]).read_bytes() == saved_audio
    memory.reconcile()
    assert _rows(root, "SELECT COUNT(*) FROM transcript_revisions") == [(len(revisions),)]
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


def test_control_pending_attempt_crash_does_not_adopt_previous_transcript_as_new_result(
    tmp_path, monkeypatch
):
    root, directory, memory, previous, proof = _run_crash(tmp_path, monkeypatch, "retry", "pending")
    assert proof["provider_calls"] == 0 and previous[0][3] == TRANSCRIPT_TEXT
    _keyless(monkeypatch)
    recovery = importlib.import_module("here.application.recovery").RecoveryService(
        directory.parent
    )
    memory.reconcile()
    recovery.discover()
    recovery.discover()
    assert (
        _rows(
            root,
            "SELECT id,meeting_id,revision_number,raw_text,content_hash FROM transcript_revisions",
        )
        == previous
    )
    assert _rows(root, "SELECT status,active_revision_id FROM meetings") == [
        ("interrupted", previous[0][0])
    ]
    assert (
        json.loads((directory / "session.json").read_text(encoding="utf-8"))["status"] == "pending"
    )


def test_control_new_local_result_before_final_manifest_commit_is_not_adopted_on_startup(
    tmp_path, monkeypatch
):
    root, directory, memory, previous, proof = _run_crash(
        tmp_path, monkeypatch, "retry", "before-final"
    )
    assert proof["provider_calls"] == 1
    assert proof["metadata"]["status"] == "pending"
    assert proof["prospective_final"]["status"] == "completed"
    assert proof["prospective_final"]["meeting_id"] == MEETING_ID
    assert proof["raw_text"] == FINAL_TEXT
    assert previous[0][3] == TRANSCRIPT_TEXT and previous[0][3] != FINAL_TEXT
    before = {path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()}
    marker = directory / "memory_pending.json"
    marker_before = marker.read_bytes() if marker.exists() else None
    _keyless(monkeypatch)
    recovery = importlib.import_module("here.application.recovery").RecoveryService(
        directory.parent
    )
    memory.reconcile()
    candidates = recovery.discover()
    recovery.discover()
    assert (
        _rows(
            root,
            "SELECT id,meeting_id,revision_number,raw_text,content_hash FROM transcript_revisions",
        )
        == previous
    )
    assert _rows(root, "SELECT status,active_revision_id FROM meetings") == [
        ("interrupted", previous[0][0])
    ]
    assert {
        path.name: path.read_bytes() for path in directory.iterdir() if path.is_file()
    } == before
    if marker_before is not None:
        assert marker.read_bytes() == marker_before
        assert any(
            candidate.session_dir == directory and candidate.error_summary
            for candidate in candidates
        ), "Unmatched final-outcome intent must remain visible as uncertain recovery"
