"""W2: deletion must account for a real post-journal retry publication stage.

All evidence is synthetic. The child cuts production ingestion before its real
rename, after independently checking that the prepared SQL journal committed.
No reconciliation is allowed between that cut and the deletion under test.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from here.application.recovery import RecoveryService
from here.memory import service as service_module
from here.memory.database import open_library
from here.memory.service import MemoryService, NotFound
from here.output.ownership import FileLease
from memory_fixtures import MEETING_ID, make_session, write_json

TOKEN = "w2cobaltatlas"
CANARY = f"{TOKEN} synthetic interrupted trusted retry evidence"

_RETRY_CHILD = r"""
import json, os, socket, sqlite3, sys, types
from contextlib import closing
from pathlib import Path
repo, data, source, sessions, parent, publication = sys.argv[1:]
root = Path(data)
package = types.ModuleType('here')
package.__path__ = [str(Path(repo) / 'src')]
sys.modules['here'] = package
def forbidden(*args, **kwargs):
    raise AssertionError('W2 retry child attempted provider/network work')
socket.socket.connect = forbidden
socket.socket.connect_ex = forbidden
socket.create_connection = forbidden
from here.memory.service import MemoryService
import here.transcription.client as client
client.build_client = forbidden
original_rename = Path.rename
def before_publication_rename(stage, target):
    destination = Path(target)
    if stage.parent == root / '.staging':
        with closing(sqlite3.connect(root / 'library.sqlite3')) as db:
            db.row_factory = sqlite3.Row
            row = db.execute(
                "SELECT * FROM pending_operations WHERE staging_path=? AND phase='prepared'",
                (stage.relative_to(root).as_posix(),),
            ).fetchone()
            assert row is not None, 'W2 harness: rename preceded journal commit'
            payload = json.loads(row['payload_json'])
            plan = json.loads((stage / '.publication.json').read_text(encoding='utf-8'))
            assert row['kind'] == 'publish_revision'
            assert row['id'] == plan['operation_id'] == stage.name
            assert row['destination_path'] == destination.relative_to(root).as_posix()
            assert plan['hashes'] == payload['hashes']
            proof = payload['publication']
            assert proof['publication_id'] == publication
            assert proof['revision_of'] == parent
            assert proof['revision_id'] == payload['revision_id']
            assert proof['source_kind'] == 'session'
            assert not destination.exists(), 'W2 harness: destination already published'
        os._exit(73)
    return original_rename(stage, target)
Path.rename = before_publication_rename
memory = MemoryService(root, sessions_root=Path(sessions))
with memory._publication(publication, 'session'):
    memory.ingest_session(Path(source), revision_of=parent)
raise AssertionError('W2 harness: actual post-journal/pre-rename boundary not reached')
"""


@dataclass(frozen=True)
class InterruptedRetry:
    root: Path
    sessions: Path
    old_source: Path
    retry_source: Path
    origins: tuple[tuple[Path, dict[str, bytes]], ...]
    old_audio: bytes
    retry_audio: bytes
    active_revision: str
    publication_id: str
    operation: dict
    stage: Path
    stage_files: dict[str, bytes]

    def memory(self):
        return MemoryService(self.root, sessions_root=self.sessions)


def _files(directory):
    return {
        path.relative_to(directory).as_posix(): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }


def _rows(root, sql, parameters=()):
    with closing(sqlite3.connect(root / "library.sqlite3")) as db:
        return db.execute(sql, parameters).fetchall()


def _environment(root, sessions):
    # Only OS process-launch variables are inherited, never provider credentials.
    env = {
        name: os.environ[name]
        for name in ("PATH", "SystemRoot", "WINDIR", "TEMP", "TMP", "COMSPEC", "PATHEXT")
        if name in os.environ
    }
    env.update(
        HERE_ENV_FILE=str(root.parent / "absent-w2-child.env"),
        HERE_DATA_DIR=str(root),
        TRANSCRIPTIONS_DIR=str(sessions),
        OPENAI_API_KEY="synthetic-test-key",
        QT_QPA_PLATFORM="offscreen",
        PYTHONDONTWRITEBYTECODE="1",
    )
    assert not Path(env["HERE_ENV_FILE"]).exists()
    return env


@pytest.fixture(autouse=True)
def synthetic_environment(tmp_path, monkeypatch):
    monkeypatch.setenv("HERE_ENV_FILE", str(tmp_path / "absent-w2.env"))
    monkeypatch.setenv("HERE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("TRANSCRIPTIONS_DIR", str(tmp_path / "sessions"))
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")
    monkeypatch.delenv("HERE_PROVIDER_SMOKE_KEY", raising=False)
    monkeypatch.delenv("HERE_WHEEL_PYTHON", raising=False)

    def forbidden(*args, **kwargs):
        pytest.fail("W2 local deletion/recovery attempted provider/network work")

    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr("socket.socket.connect_ex", forbidden)
    monkeypatch.setattr("socket.create_connection", forbidden)
    monkeypatch.setattr("here.transcription.client.build_client", forbidden)


@pytest.fixture
def interrupted_retry(tmp_path):
    root, sessions = tmp_path / "data", tmp_path / "sessions"
    old = make_session(
        sessions / "original",
        meeting_id=MEETING_ID,
        transcript_text=f"{TOKEN} synthetic original evidence",
    )
    retry = make_session(
        sessions / "retry",
        meeting_id=MEETING_ID,
        transcript_text=CANARY,
        segments=[{"text": CANARY, "start": 0.0, "end": 1.0}],
    )
    # Reuse the established WAV fixture, changing one valid PCM sample so retry
    # really copies new audio instead of reusing the active revision's asset.
    audio = bytearray(retry.audio.read_bytes())
    audio[-2:] = b"\x11\x01"
    retry.audio.write_bytes(audio)
    assert retry.audio.read_bytes() != old.audio.read_bytes()
    (retry.directory / "raw_transcript.txt").write_text(CANARY, encoding="utf-8")
    metadata = json.loads(retry.metadata.read_text(encoding="utf-8"))
    metadata["output_files"].append("raw_transcript.txt")
    write_json(retry.metadata, metadata)
    memory = MemoryService(root, sessions_root=sessions)
    first = memory.ingest_session(old.directory)
    assert (first.status, first.meeting_id) == ("inserted", MEETING_ID), first
    active = memory.active_revision(MEETING_ID)
    assert active is not None
    UUID(active)
    origins = tuple((source.directory, _files(source.directory)) for source in (old, retry))
    publication = str(uuid4())
    process = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            _RETRY_CHILD,
            str(Path(__file__).resolve().parents[1]),
            str(root),
            str(retry.directory),
            str(sessions),
            active,
            publication,
        ],
        env=_environment(root, sessions),
        capture_output=True,
        text=True,
        timeout=25,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    assert process.returncode == 73, (
        "W2 harness did not reach the real post-journal/pre-rename cut: "
        + process.stdout
        + process.stderr
    )
    with closing(sqlite3.connect(root / "library.sqlite3")) as db:
        db.row_factory = sqlite3.Row
        operations = db.execute(
            "SELECT * FROM pending_operations WHERE kind='publish_revision' "
            "AND phase='prepared' AND meeting_id=?",
            (MEETING_ID,),
        ).fetchall()
        assert len(operations) == 1, "W2 harness: missing unique persisted retry journal"
        operation = dict(operations[0])
        assert db.execute(
            "SELECT status,active_revision_id FROM meetings WHERE id=?",
            (MEETING_ID,),
        ).fetchone()[:] == ("completed", active)
        assert db.execute("SELECT COUNT(*) FROM transcript_revisions").fetchone()[0] == 1
    payload = json.loads(operation["payload_json"])
    proof = payload["publication"]
    assert proof["publication_id"] == publication
    assert proof["revision_of"] == active and proof["source_kind"] == "session"
    assert Path(proof["origin_path"]).resolve() == retry.directory.resolve()
    assert len(proof["origin_fingerprint"]) == 64
    int(proof["origin_fingerprint"], 16)
    revision = payload["revision_id"]
    UUID(revision)
    assert operation["destination_path"] == f"meetings/{MEETING_ID}/revisions/{revision}"
    assert operation["staging_path"] == f".staging/{operation['id']}"
    stage = root / operation["staging_path"]
    assert not (root / operation["destination_path"]).exists()
    plan = json.loads((stage / ".publication.json").read_text(encoding="utf-8"))
    assert plan == {
        "schema_version": 1,
        "operation_id": operation["id"],
        "hashes": payload["hashes"],
    }
    required = {"session.json", "raw_transcript.txt", "transcript.txt", "audio.wav"}
    assert required <= set(plan["hashes"])
    for name, digest in plan["hashes"].items():
        assert hashlib.sha256((stage / name).read_bytes()).hexdigest() == digest
    assert json.loads((stage / "session.json").read_text())["meeting_id"] == MEETING_ID
    assert (stage / "raw_transcript.txt").read_text(encoding="utf-8") == CANARY
    assert (stage / "audio.wav").read_bytes() == retry.audio.read_bytes()
    # The synchronous producer really ended. Probe its actual publication lock;
    # do not manufacture a native producer or infer ownership from a filename.
    with closing(FileLease(root / ".memory.lock", timeout=0, create=False)):
        pass
    MemoryService.check_owner(root, MEETING_ID)
    assert not (sessions / ".capture_owners").exists()
    for origin, before in origins:
        assert _files(origin) == before
    return InterruptedRetry(
        root,
        sessions,
        old.directory,
        retry.directory,
        origins,
        old.audio.read_bytes(),
        retry.audio.read_bytes(),
        active,
        publication,
        operation,
        stage,
        _files(stage),
    )


def _record(case):
    document = json.loads(
        (case.root / ".deletions" / f"{MEETING_ID}.json").read_text(encoding="utf-8")
    )
    assert set(document) == {"schema_version", "meeting_id", "origins", "phase"}
    assert document["schema_version"] == 1 and document["meeting_id"] == MEETING_ID
    assert document["origins"]
    return document


def _hidden(memory):
    library = open_library(memory.data_dir)
    try:
        with service_module.catalog_read_view(library) as suppression:
            assert MEETING_ID in suppression.suppressed_meeting_ids
    finally:
        library.close()
    assert memory.show(MEETING_ID) is None
    assert memory.active_revision(MEETING_ID) is None
    with pytest.raises(NotFound):
        memory.get_meeting(MEETING_ID)


def _pending(case, result):
    assert (result.status, result.meeting_id, result.error_category) == (
        "pending",
        MEETING_ID,
        "io",
    ), "Identified publication residue must never be reported completed/rejected"
    assert result.reason in {"busy", "hook_failed", "purge_failed", "io", "cleanup_pending"}
    assert result.warnings
    assert _record(case)["phase"] == "pending"
    _hidden(case.memory())
    publication = _rows(
        case.root,
        "SELECT kind,phase,meeting_id,staging_path,destination_path,payload_json "
        "FROM pending_operations WHERE id=?",
        (case.operation["id"],),
    )
    if publication:
        columns = (
            "kind",
            "phase",
            "meeting_id",
            "staging_path",
            "destination_path",
            "payload_json",
        )
        assert publication == [tuple(case.operation[name] for name in columns)]
    else:
        # The existing delete journal may carry the validated inventory instead;
        # do not require retaining a redundant publication row after that transfer.
        inventories = _rows(
            case.root,
            "SELECT payload_json FROM pending_operations "
            "WHERE kind='delete' AND phase='prepared' AND meeting_id=?",
            (MEETING_ID,),
        )
        plan = json.loads(case.stage_files[".publication.json"])
        required = {
            f"{case.operation['staging_path']}/{name}": digest
            for name, digest in plan["hashes"].items()
        }
        required[f"{case.operation['staging_path']}/.publication.json"] = hashlib.sha256(
            case.stage_files[".publication.json"]
        ).hexdigest()
        assert any(
            required.items() <= json.loads(row[0]).get("files", {}).items()
            and case.operation["staging_path"] in json.loads(row[0]).get("directories", [])
            for row in inventories
        ), "A surviving identified stage lost its registered meeting inventory authority"


def _no_residue(case):
    assert _record(case)["phase"] == "complete"
    residue = []
    # Always finish the WHOLE-root scan before asserting, including SQLite,
    # sidecars, managed backups and stages; the meeting subtree is insufficient.
    for path in case.root.rglob("*"):
        if path.is_file():
            contents = path.read_bytes()
            if any(
                TOKEN.encode(encoding) in contents
                for encoding in (
                    "utf-8",
                    "utf-16-le",
                    "utf-16-be",
                )
            ):
                residue.append(("text", path.relative_to(case.root).as_posix()))
            if case.old_audio in contents:
                residue.append(("old-audio", path.relative_to(case.root).as_posix()))
            if case.retry_audio in contents:
                residue.append(("staged-audio", path.relative_to(case.root).as_posix()))
    assert residue == [], f"Completed deletion retained managed meeting content: {residue}"
    assert not case.stage.exists(), "Completed deletion retained its registered publication stage"
    for table in ("meetings", "transcript_revisions", "audio_assets", "revision_audio_assets"):
        column = "id" if table == "meetings" else "meeting_id"
        assert _rows(case.root, f"SELECT 1 FROM {table} WHERE {column}=?", (MEETING_ID,)) == []
    assert _rows(case.root, "SELECT 1 FROM segments") == []
    assert _rows(case.root, "SELECT 1 FROM ingest_sources WHERE meeting_id=?", (MEETING_ID,)) == []
    assert (
        _rows(
            case.root,
            "SELECT 1 FROM pending_operations WHERE kind!='delete' AND meeting_id=?",
            (MEETING_ID,),
        )
        == []
    )
    assert _rows(case.root, "SELECT 1 FROM deleted_origins")
    if _rows(case.root, "SELECT 1 FROM sqlite_master WHERE name='segments_fts'"):
        assert (
            _rows(
                case.root,
                "SELECT 1 FROM segments_fts WHERE meeting_id=?",
                (MEETING_ID,),
            )
            == []
        )
    for origin, before in case.origins:
        assert _files(origin) == before
    _hidden(case.memory())


def _finish_without_resurrection(case):
    memory = case.memory()
    memory.reconcile()
    memory.reconcile()
    result = memory.delete_meeting(MEETING_ID)
    assert (result.status, result.meeting_id, result.error_category) == (
        "completed",
        MEETING_ID,
        None,
    )
    _no_residue(case)
    for source in (case.old_source, case.retry_source):
        rejected = memory.ingest_session(source)
        assert (rejected.status, rejected.error_category) == ("rejected", "conflict")
    rejected = memory.ingest_session(case.retry_source, revision_of=case.active_revision)
    assert (rejected.status, rejected.error_category) == ("rejected", "conflict")
    assert memory.reindex().indexed_segments == 0
    assert RecoveryService(case.sessions).discover() == []
    assert memory.delete_meeting(MEETING_ID).status == "completed"
    _no_residue(case)


def _accept_with_stage_cleanup_denied(case, monkeypatch):
    original_unlink = Path.unlink

    def deny_stage_unlink(path, *args, **kwargs):
        if path.is_relative_to(case.stage):
            raise PermissionError("Synthetic W2 registered-stage cleanup denial")
        return original_unlink(path, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(Path, "unlink", deny_stage_unlink)
        # This is the FIRST service operation after the child's interrupted retry.
        result = case.memory().delete_meeting(MEETING_ID)
    _pending(case, result)
    assert _files(case.stage) == case.stage_files
    return result


def _lose_sql(case, temporary_root):
    assert case.root.resolve().is_relative_to(temporary_root.resolve())
    for suffix in ("", "-wal", "-shm", "-journal"):
        path = case.root / ("library.sqlite3" + suffix)
        assert path.parent == case.root and not path.is_symlink() and not path.is_junction()
        path.unlink(missing_ok=True)


def test_control_real_retry_stage_is_identified_while_old_revision_stays_idle(interrupted_retry):
    case = interrupted_retry
    detail = case.memory().get_meeting(MEETING_ID)
    assert detail.meeting.status == "completed"
    assert detail.meeting.active_revision_id == case.active_revision
    assert len(detail.revisions) == 1
    assert detail.revisions[0].raw_text == f"{TOKEN} synthetic original evidence"
    assert _files(case.stage) == case.stage_files
    assert not (case.root / case.operation["destination_path"]).exists()


def test_delete_accounts_for_real_retry_stage_before_reporting_complete(interrupted_retry):
    case = interrupted_retry
    # No reconcile here: it would erase the evidence before completion is tested.
    result = case.memory().delete_meeting(MEETING_ID)
    assert result.meeting_id == MEETING_ID and result.status in {"completed", "pending"}
    if result.status == "completed":
        _no_residue(case)
    else:
        _pending(case, result)
    _finish_without_resurrection(case)


@pytest.mark.parametrize("database", ["retained", "lost"], ids=["sql-retained", "sql-lost"])
def test_registered_stage_pending_cleanup_survives_restart(
    interrupted_retry,
    monkeypatch,
    tmp_path,
    database,
):
    case = interrupted_retry
    _accept_with_stage_cleanup_denied(case, monkeypatch)
    if database == "lost":
        _lose_sql(case, tmp_path)
    # A fresh service honors durable suppression before receipt reconstruction.
    _hidden(case.memory())
    _finish_without_resurrection(case)


def test_unregistered_stage_child_is_retained_pending_with_authority(
    interrupted_retry,
    monkeypatch,
):
    case = interrupted_retry
    _accept_with_stage_cleanup_denied(case, monkeypatch)
    unknown = case.stage / "unregistered.wav"
    unknown.write_bytes(b"Synthetic unknown is not authorized by UUID, hash or WAV suffix")
    before = unknown.read_bytes()
    memory = case.memory()
    report = memory.reconcile()
    result = memory.delete_meeting(MEETING_ID)
    _pending(case, result)
    assert unknown.read_bytes() == before
    assert report.warnings or report.skipped
    assert (case.stage / ".publication.json").is_file()
    for origin, contents in case.origins:
        assert _files(origin) == contents
    # Only this test removes its own unregistered obstacle; product cleanup must
    # never infer that authority. The original validated proof remains available.
    unknown.unlink()
    _finish_without_resurrection(case)
