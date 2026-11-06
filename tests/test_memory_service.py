"""Synthetic end-to-end ingestion and recovery contracts."""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from uuid import UUID

import pytest
from memory_fixtures import (
    LEGACY_SESSION_ID,
    MEETING_ID,
    OTHER_MEETING_ID,
    TRANSCRIPT_TEXT,
    make_corpus,
    make_session,
    write_json,
)


def _service(root):
    return importlib.import_module("here.memory.service").MemoryService(root)


def _connection(root):
    connection = sqlite3.connect(root / "library.sqlite3")
    connection.row_factory = sqlite3.Row
    return connection


def _revision(root, meeting_id):
    with _connection(root) as connection:
        return dict(
            connection.execute(
                "SELECT r.* FROM transcript_revisions r "
                "JOIN meetings m ON m.active_revision_id=r.id "
                "WHERE m.id=?",
                (meeting_id,),
            ).fetchone()
        )


def _assert_one_publication(root, meeting_id, audio_hash):
    with _connection(root) as connection:
        assert connection.execute("SELECT id FROM meetings").fetchall()[0][0] == meeting_id
        assert connection.execute("SELECT COUNT(*) FROM meetings").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM transcript_revisions").fetchone()[0] == 1
        assert connection.execute("SELECT COUNT(*) FROM audio_assets").fetchone()[0] == 1
        asset = connection.execute("SELECT relative_path, sha256 FROM audio_assets").fetchone()
        assert asset[1] == audio_hash
        assert hashlib.sha256((root / asset[0]).read_bytes()).hexdigest() == audio_hash
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM pending_operations WHERE phase='prepared'"
            ).fetchone()[0]
            == 0
        )
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert _revision(root, meeting_id)["raw_text"] == TRANSCRIPT_TEXT
    assert {path.name for path in (root / "meetings").iterdir()} == {meeting_id}
    assert not [path for path in (root / ".staging").rglob("*") if path.is_file()]


def test_legacy_ingest_assigns_identity_once_and_preserves_origin_without_key(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    source = make_session(tmp_path / "legacy")
    before = (source.audio.read_bytes(), source.metadata.read_bytes())
    root = tmp_path / "data"
    service = _service(root)
    first = service.ingest_session(source.directory)
    second = service.ingest_session(source.directory / ".")
    assert first.status == "inserted"
    UUID(first.meeting_id)
    assert (second.status, second.meeting_id) == ("unchanged", first.meeting_id)
    assert (source.audio.read_bytes(), source.metadata.read_bytes()) == before
    with _connection(root) as connection:
        meeting = connection.execute("SELECT * FROM meetings").fetchone()
        assert meeting["legacy_session_id"] == LEGACY_SESSION_ID
        assert (meeting["duration_ms"], meeting["paused_ms"]) == (2000, 300000)
        origins = connection.execute("SELECT * FROM ingest_sources").fetchall()
        assert len(origins) == 1
        assert Path(origins[0]["origin_path"]).resolve() == source.directory.resolve()
        assert len(origins[0]["origin_fingerprint"]) == 64
    _assert_one_publication(root, first.meeting_id, source.audio_sha256)


def test_conflicting_uuid_is_readable_rejection_without_overwrite(tmp_path):
    corpus = make_corpus(tmp_path)
    root = tmp_path / "data"
    service = _service(root)
    first = service.ingest_session(corpus.identified.directory)
    previous = _revision(root, first.meeting_id)
    rejected = service.ingest_session(corpus.conflicting_identity.directory)
    assert rejected.status == "rejected"
    assert rejected.meeting_id == MEETING_ID
    assert rejected.warnings
    assert any("conflict" in str(warning).lower() for warning in rejected.warnings)
    assert _revision(root, MEETING_ID) == previous
    _assert_one_publication(root, MEETING_ID, corpus.identified.audio_sha256)


def test_identical_audio_and_legacy_labels_do_not_merge_distinct_origins(tmp_path):
    corpus = make_corpus(tmp_path)
    root = tmp_path / "data"
    service = _service(root)
    results = [
        service.ingest_session(item.directory)
        for item in (
            corpus.legacy,
            corpus.copied_legacy,
            corpus.identified,
            corpus.same_audio_other_identity,
        )
    ]
    assert [result.status for result in results] == ["inserted"] * 4
    assert len({result.meeting_id for result in results}) == 4
    assert {MEETING_ID, OTHER_MEETING_ID} <= {result.meeting_id for result in results}
    with _connection(root) as connection:
        assert (
            connection.execute("SELECT COUNT(DISTINCT sha256) FROM audio_assets").fetchone()[0] == 1
        )
        assert connection.execute("SELECT COUNT(*) FROM meetings").fetchone()[0] == 4


def test_retry_keeps_original_revision_and_publishes_new_evidence_atomically(tmp_path):
    corpus = make_corpus(tmp_path)
    root = tmp_path / "data"
    service = _service(root)
    service.ingest_session(corpus.identified.directory)
    previous = _revision(root, MEETING_ID)
    previous_files = {p: p.read_bytes() for p in (root / "meetings").rglob("*") if p.is_file()}
    result = service.ingest_session(
        corpus.conflicting_identity.directory, revision_of=previous["id"]
    )
    assert (result.status, result.meeting_id) == ("revised", MEETING_ID)
    current = _revision(root, MEETING_ID)
    assert current["id"] != previous["id"]
    assert (current["revision_number"], current["raw_text"]) == (2, "Synthetic changed canary.")
    with _connection(root) as connection:
        retained = dict(
            connection.execute(
                "SELECT * FROM transcript_revisions WHERE id=?", (previous["id"],)
            ).fetchone()
        )
        assert retained == previous
        assert connection.execute("SELECT COUNT(*) FROM audio_assets").fetchone()[0] == 1
    assert all(path.read_bytes() == contents for path, contents in previous_files.items())
    stale = service.ingest_session(
        corpus.conflicting_identity.directory, revision_of=previous["id"]
    )
    assert stale.status in {"unchanged", "rejected"}
    assert _revision(root, MEETING_ID)["id"] == current["id"]


def test_failed_retry_preserves_previous_active_revision_and_managed_files(tmp_path, monkeypatch):
    corpus = make_corpus(tmp_path)
    root = tmp_path / "data"
    service = _service(root)
    service.ingest_session(corpus.identified.directory)
    previous = _revision(root, MEETING_ID)
    previous_files = {p: p.read_bytes() for p in (root / "meetings").rglob("*") if p.is_file()}
    repository = importlib.import_module("here.memory.repository")

    def fail_activation(*args, **kwargs):
        raise RuntimeError("synthetic activation failure")

    with monkeypatch.context() as patch:
        patch.setattr(repository.MemoryRepository, "activate_revision", fail_activation)
        failed = service.ingest_session(
            corpus.conflicting_identity.directory, revision_of=previous["id"]
        )
    assert failed.status == "rejected"
    assert failed.warnings
    assert _revision(root, MEETING_ID) == previous
    assert all(path.read_bytes() == contents for path, contents in previous_files.items())
    service.reconcile()
    with _connection(root) as connection:
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    assert _revision(root, MEETING_ID) == previous
    assert {p for p in (root / "meetings").rglob("*") if p.is_file()} == set(previous_files)


_CRASH_SCRIPT = r"""
import os, pathlib, shutil, sys, types
from contextlib import contextmanager
source_root, data, source, boundary = map(pathlib.Path, sys.argv[1:])
boundary = str(boundary)
here = types.ModuleType("here")
here.__path__ = [str(source_root / "src")]
sys.modules["here"] = here
from here.memory.service import MemoryService
from here.memory.repository import MemoryRepository
service = MemoryService(data)
activated = [0]
original_activate = MemoryRepository.activate_revision
def activate(self, *args, **kwargs):
    result = original_activate(self, *args, **kwargs)
    activated[0] += 1
    if boundary == "before-commit": os._exit(73)
    return result
MemoryRepository.activate_revision = activate
original_transaction = MemoryRepository.transaction
@contextmanager
def transaction(self):
    before = activated[0]
    with original_transaction(self) as result:
        yield result
    if boundary == "after-commit" and activated[0] > before: os._exit(73)
MemoryRepository.transaction = transaction
original_transition = MemoryRepository.transition_pending_operation
def transition(self, operation_id, phase):
    if boundary == "before-complete" and phase == "complete": os._exit(73)
    return original_transition(self, operation_id, phase)
MemoryRepository.transition_pending_operation = transition
for name in ("copy2", "copyfile"):
    original = getattr(shutil, name)
    def copy(src, dst, *args, _original=original, **kwargs):
        is_staging = pathlib.Path(dst).absolute().is_relative_to(data / ".staging")
        if boundary == "before-stage" and is_staging:
            os._exit(73)
        return _original(src, dst, *args, **kwargs)
    setattr(shutil, name, copy)
for name in ("rename", "replace"):
    original = getattr(os, name)
    def rename(src, dst, *args, _original=original, **kwargs):
        result = _original(src, dst, *args, **kwargs)
        is_published = pathlib.Path(dst).absolute().is_relative_to(data / "meetings")
        if boundary == "after-rename" and is_published:
            os._exit(73)
        return result
    setattr(os, name, rename)
service.ingest_session(source)
raise AssertionError("publication boundary was not reached: " + boundary)
"""


@pytest.mark.parametrize(
    "boundary",
    [
        "before-stage",
        "after-rename",
        "before-commit",
        "after-commit",
        "before-complete",
    ],
)
def test_process_kill_then_local_reconcile_and_retry_has_one_identity(
    tmp_path, monkeypatch, boundary
):
    root = tmp_path / "data"
    service = _service(root)
    source = make_session(tmp_path / "legacy")
    before = (source.audio.read_bytes(), source.metadata.read_bytes())
    env = dict(os.environ)
    env.pop("OPENAI_API_KEY", None)
    env["HERE_ENV_FILE"] = str(tmp_path / "absent.env")
    env["HERE_DATA_DIR"] = str(root)
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            _CRASH_SCRIPT,
            str(Path(__file__).resolve().parents[1]),
            str(root),
            str(source.directory),
            boundary,
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert process.returncode == 73, process.stdout + process.stderr
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("socket.socket.connect", lambda *args: pytest.fail("Recovery used network"))
    service.reconcile()
    result = service.ingest_session(source.directory)
    assert result.status in {"inserted", "unchanged"}
    UUID(result.meeting_id)
    _assert_one_publication(root, result.meeting_id, source.audio_sha256)
    service.reconcile()
    again = service.ingest_session(source.directory)
    assert (again.status, again.meeting_id) == ("unchanged", result.meeting_id)
    _assert_one_publication(root, result.meeting_id, source.audio_sha256)
    assert (source.audio.read_bytes(), source.metadata.read_bytes()) == before


def test_fts_contains_only_active_complete_revision_and_folds_diacritics(tmp_path):
    module = importlib.import_module("here.memory.service")
    root = tmp_path / "data"
    service = module.MemoryService(root)
    corpus = make_corpus(tmp_path)
    service.ingest_session(corpus.identified.directory)
    previous = _revision(root, MEETING_ID)
    (corpus.conflicting_identity.directory / "transcript.txt").write_text(
        "Decisión ágil.", encoding="utf-8-sig"
    )
    write_json(
        corpus.conflicting_identity.directory / "segments.json",
        {
            "schema_version": 1,
            "segments": [{"text": "Decisión ágil.", "start": 0.0, "end": 1.0}],
        },
    )
    service.ingest_session(corpus.conflicting_identity.directory, revision_of=previous["id"])
    active = _revision(root, MEETING_ID)
    partial = make_session(tmp_path / "partial", meeting_id="33333333-3333-4333-8333-333333333333")
    metadata = json.loads(partial.metadata.read_text(encoding="utf-8"))
    metadata["status"] = "failed"
    write_json(partial.metadata, metadata)
    assert service.ingest_session(partial.directory).status == "inserted"
    with _connection(root) as connection:
        assert (
            connection.execute(
                "SELECT status FROM transcript_revisions WHERE meeting_id=?", (partial.meeting_id,)
            ).fetchone()[0]
            == "partial"
        )
        assert (
            connection.execute(
                "SELECT DISTINCT revision_id FROM segments_fts WHERE segments_fts MATCH 'decision'"
            ).fetchall()[0][0]
            == active["id"]
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM segments_fts WHERE revision_id=?", (previous["id"],)
            ).fetchone()[0]
            == 0
        )
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM segments_fts WHERE segments_fts MATCH 'copper'"
            ).fetchone()[0]
            == 0
        )


def test_missing_fts_capability_is_typed_warning_and_keeps_ingestion_working(tmp_path, monkeypatch):
    module = importlib.import_module("here.memory.service")
    monkeypatch.setattr(module, "fts_available", lambda connection: False)
    root = tmp_path / "data"
    service = module.MemoryService(root)
    source = make_session(tmp_path / "legacy")
    result = service.ingest_session(source.directory)
    assert result.status == "inserted"
    warnings = [
        warning
        for warning in result.warnings
        if getattr(warning, "code", None) == "fts_unavailable"
    ]
    assert warnings and all(isinstance(warning, module.FtsUnavailable) for warning in warnings)
    _assert_one_publication(root, result.meeting_id, source.audio_sha256)


def test_startup_reconcile_marks_abandoned_recording_interrupted_without_network(
    tmp_path, monkeypatch
):
    root = tmp_path / "data"
    service = _service(root)
    database = importlib.import_module("here.memory.database")
    models = importlib.import_module("here.memory.models")
    repository = importlib.import_module("here.memory.repository")
    library = database.open_library(root)
    try:
        repo = repository.MemoryRepository(library)
        with repo.transaction():
            repo.insert_meeting(
                models.Meeting(id=MEETING_ID, status=models.MeetingStatus.RECORDING)
            )
    finally:
        library.close()
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("socket.socket.connect", lambda *args: pytest.fail("Startup used network"))
    service.reconcile()
    service.reconcile()
    with _connection(root) as connection:
        assert (
            connection.execute("SELECT id, status FROM meetings").fetchone()["status"]
            == "interrupted"
        )
        assert connection.execute("SELECT COUNT(*) FROM transcript_revisions").fetchone()[0] == 0
