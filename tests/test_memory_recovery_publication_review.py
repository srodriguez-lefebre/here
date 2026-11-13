"""Recovery replay regressions found while reviewing the R62 writer."""

from __future__ import annotations

import hashlib
import json
from contextlib import closing, contextmanager
from datetime import datetime, timezone
from pathlib import Path

import pytest
from here.application.processing import SessionProcessor
from here.application.recovery import RecoveryService
from here.memory.artifacts import (
    normalized_origin,
    publication_hashes,
    read_pending_intent,
    write_durable_json,
)
from here.memory.repository import MemoryRepository
from here.memory.service import MemoryService, OwnershipBusy
from here.output.ownership import FileLease, LeaseBusy
from here.output.session_writer import read_session_metadata
from here.recording.journal import CaptureJournal
from here.transcription.client import TranscriptionResult
from memory_fixtures import OTHER_MEETING_ID, make_session, write_marked_wav
from test_memory_delete import _files, _rows
from test_memory_recovery_publication import MARKER, _observe_source
from test_memory_recovery_publication import sandbox as sandbox

FIRST_TEXT = "Synthetic first revision."
RETRY_TEXT = "Synthetic retry revision."


class _BeforePublisher(BaseException):
    """Stop after the real producer seals compatibility artifacts."""


def _processor(memory, text, *, clock=None):
    def transcribe(*args, **kwargs):
        return TranscriptionResult(text, text)

    return SessionProcessor(
        transcribe=transcribe, retry_delays=(), clock=clock, memory_service=memory
    )


def _sealed_retry(memory, directory, monkeypatch):
    def stop(*args, **kwargs):
        raise _BeforePublisher

    with monkeypatch.context() as cut:
        cut.setattr(SessionProcessor, "publish_memory", stop)
        with pytest.raises(_BeforePublisher):
            _processor(memory, RETRY_TEXT).retry(directory)


@contextmanager
def _fail_marker_cleanup(monkeypatch, marker):
    real_unlink = Path.unlink

    def unlink(path, *args, **kwargs):
        if path == marker:
            raise OSError("Synthetic marker cleanup failure")
        return real_unlink(path, *args, **kwargs)

    with monkeypatch.context() as fault:
        fault.setattr(Path, "unlink", unlink)
        yield


def _revisions(root):
    return _rows(
        root,
        "SELECT meeting_id,id,revision_number,raw_text FROM transcript_revisions ORDER BY 1,3",
    )


def _receipt(root, identity, revision):
    path = root / "meetings" / identity / "revisions" / revision / "catalog.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _sealed_intent(directory, source_id):
    metadata = read_session_metadata(directory)
    assert metadata is not None and metadata.meeting_id == source_id
    intent = read_pending_intent(directory, source_id)
    assert intent.hashes is not None and intent.publication_id is not None
    assert dict(intent.hashes) == publication_hashes(directory, metadata)
    return intent


def _seal_fingerprint(intent):
    return hashlib.sha256(json.dumps(intent.hashes).encode()).hexdigest()


def test_recovery_immediately_reverts_precommit_failure_under_claimed_owner(sandbox, monkeypatch):
    memory, sessions = sandbox
    root = memory.data_dir
    directory = make_session(sessions / "legacy").directory
    identity, parent = memory.prepare_retry(directory, None)
    assert memory.retry_parent(directory, None) == (identity, parent)
    sibling = make_session(sessions / "sibling", meeting_id=OTHER_MEETING_ID)
    assert memory.ingest_session(sibling.directory).status == "inserted"
    unrelated_stage = root / ".staging" / "unrelated-inventory"
    unrelated_stage.mkdir()
    (unrelated_stage / "retained.txt").write_bytes(b"Synthetic unregistered inventory")

    _sealed_retry(memory, directory, monkeypatch)
    intent = _sealed_intent(directory, None)
    assert intent.revision_of == parent
    native_owner = CaptureJournal.owner_path(sessions, identity)
    MemoryService.check_owner(root, identity)
    revisions = _revisions(root)
    managed_revisions = root / "meetings" / identity / "revisions"
    sibling_managed = root / "meetings" / OTHER_MEETING_ID
    unaffected = (sibling_managed, sibling.directory, unrelated_stage)
    prior_source_files = _files(directory)
    del prior_source_files[MARKER]
    preserved = {path: _files(path) for path in unaffected}
    prior_revision_files = _files(managed_revisions)

    original_activate = MemoryRepository.activate_revision
    attempted = []

    def fail_before_commit(repo, meeting_id, revision_id):
        if meeting_id != identity or revision_id == parent:
            return original_activate(repo, meeting_id, revision_id)
        assert repo.connection.in_transaction
        assert repo.get_meeting(identity).active_revision_id == parent
        assert repo.get_revision(revision_id).raw_text == RETRY_TEXT
        ((operation_id, stage_rel, destination_rel, payload),) = repo.library.execute(
            "SELECT id,staging_path,destination_path,payload_json FROM pending_operations "
            "WHERE meeting_id=? AND kind='publish_revision' AND phase='prepared'",
            (identity,),
        ).fetchall()
        assert json.loads(payload)["revision_id"] == revision_id
        stage, destination = root / stage_rel, root / destination_rel
        assert not stage.exists() and destination.is_dir()
        with pytest.raises(OwnershipBusy):
            MemoryService.check_owner(root, identity)
        with pytest.raises(LeaseBusy), closing(FileLease(native_owner, timeout=0, create=False)):
            pass
        attempted.append((operation_id, stage, destination))
        raise OSError("Synthetic activation failure before the entity commit")

    with monkeypatch.context() as fault:
        fault.setattr(MemoryRepository, "activate_revision", fail_before_commit)
        RecoveryService(sessions, memory_service=memory).discover()

    ((operation_id, stage, destination),) = attempted
    MemoryService.check_owner(root, identity)
    with closing(FileLease(native_owner, timeout=0, create=False)):
        pass
    assert _revisions(root) == revisions
    assert memory.active_revision(identity) == parent
    assert {path: _files(path) for path in unaffected} == preserved
    assert _sealed_intent(directory, None) == intent
    source_files = _files(directory)
    del source_files[MARKER]
    assert source_files == prior_source_files
    for name, data in prior_revision_files.items():
        assert (managed_revisions / name).read_bytes() == data
    phase = _rows(root, "SELECT phase FROM pending_operations WHERE id=?", (operation_id,))
    assert (phase, stage.exists(), destination.exists()) == ([("reverted",)], False, False)
    assert _files(managed_revisions) == prior_revision_files


def test_recovery_replays_exact_live_receipt_at_tombstoned_origin(sandbox, tmp_path, monkeypatch):
    memory, sessions = sandbox
    root = memory.data_dir
    completed = datetime(2026, 1, 1, 12, tzinfo=timezone.utc)
    first = make_session(sessions / "20260101_120000", meeting_id=OTHER_MEETING_ID)
    origin = normalized_origin(first.directory)
    first_publication = SessionProcessor(memory_service=memory).publish_memory(first.directory)
    assert first_publication.status == "inserted"
    ((first_fingerprint,),) = _rows(
        root,
        "SELECT origin_fingerprint FROM ingest_sources WHERE meeting_id=? AND origin_path=?",
        (OTHER_MEETING_ID, origin),
    )
    assert memory.delete_meeting(OTHER_MEETING_ID).status == "completed"
    tombstone = root / ".deletions" / f"{OTHER_MEETING_ID}.json"
    tombstone_before = tombstone.read_bytes()
    deletion = json.loads(tombstone_before)
    assert deletion["meeting_id"] == OTHER_MEETING_ID and deletion["phase"] == "complete"
    assert deletion["origins"] == [{"origin_path": origin, "origin_fingerprint": first_fingerprint}]
    archived = tmp_path / "retired-synthetic-source"
    assert first.directory.resolve().parent == sessions.resolve()
    assert archived.resolve().parent == tmp_path.resolve() and not archived.exists()
    first.directory.rename(archived)

    replacement_audio = tmp_path / "replacement.wav"
    write_marked_wav(replacement_audio)
    marker = first.directory / MARKER
    with _fail_marker_cleanup(monkeypatch, marker):
        replacement = _processor(memory, RETRY_TEXT, clock=lambda: completed)
        artifacts = replacement.transcribe_audio_path(replacement_audio, sessions)

    assert artifacts.session_dir == first.directory
    ((identity, revision, number, text),) = committed = _revisions(root)
    assert identity != OTHER_MEETING_ID and (number, text) == (1, RETRY_TEXT)
    intent = _sealed_intent(first.directory, identity)
    assert intent.revision_of is None
    ((current_origin, fingerprint),) = _rows(
        root, "SELECT origin_path,origin_fingerprint FROM ingest_sources WHERE meeting_id=?",
        (identity,),
    )
    assert current_origin == origin and fingerprint != first_fingerprint
    assert _seal_fingerprint(intent) == fingerprint
    receipt = _receipt(root, identity, revision)
    assert receipt["schema_version"] == 3
    assert receipt["publication"] == {
        "publication_id": intent.publication_id,
        "revision_id": revision,
        "revision_of": None,
        "origin_path": origin,
        "origin_fingerprint": fingerprint,
        "source_kind": "session",
    }
    assert memory.active_revision(identity) == revision
    assert memory.show(OTHER_MEETING_ID) is None
    preserved = _files(first.directory)
    del preserved[MARKER]

    RecoveryService(sessions, memory_service=memory).discover()

    assert memory.show(OTHER_MEETING_ID) is None
    assert tombstone.read_bytes() == tombstone_before
    assert _revisions(root) == committed
    assert memory.active_revision(identity) == revision
    assert not marker.exists(), "Exact committed live receipt should allow marker cleanup"
    assert _files(first.directory) == preserved


def test_recovery_denies_receipt_with_different_seal_before_effects(sandbox, monkeypatch):
    memory, sessions = sandbox
    root = memory.data_dir
    directory = make_session(sessions / "legacy").directory
    identity, parent = memory.prepare_retry(directory, None)
    marker = directory / MARKER
    with _fail_marker_cleanup(monkeypatch, marker):
        _processor(memory, RETRY_TEXT).retry(directory)
    first, active = committed = _revisions(root)
    assert first[:3] == (identity, parent, 1)
    revision = active[1]
    assert active == (identity, revision, 2, RETRY_TEXT)
    assert memory.retry_parent(directory, None) == (identity, revision)

    intent = _sealed_intent(directory, None)
    receipt = _receipt(root, identity, revision)["publication"]
    assert receipt["publication_id"] == intent.publication_id
    assert receipt["origin_path"] == normalized_origin(directory)
    assert receipt["source_kind"] == intent.source_kind
    assert receipt["revision_of"] == intent.revision_of == parent
    assert receipt["origin_fingerprint"] == _seal_fingerprint(intent)
    (directory / "raw_transcript.txt").write_text("Different synthetic words", encoding="utf-8")
    metadata = read_session_metadata(directory)
    assert metadata is not None
    document = json.loads(marker.read_text(encoding="utf-8"))
    document["hashes"] = publication_hashes(directory, metadata)
    write_durable_json(marker, document)
    changed = _sealed_intent(directory, None)
    assert changed.publication_id == intent.publication_id
    assert changed.revision_of == intent.revision_of
    assert changed.source_kind == intent.source_kind
    assert _seal_fingerprint(changed) != receipt["origin_fingerprint"]
    preserved = _files(directory)

    with _observe_source(monkeypatch, directory) as effects:
        RecoveryService(sessions, memory_service=memory).discover()

    assert _revisions(root) == committed
    assert memory.active_revision(identity) == revision
    assert (effects, _files(directory) == preserved) == ([], True)


def test_recovery_replays_audio_only_retry_from_rebuilt_parent(sandbox, tmp_path, monkeypatch):
    memory, sessions = sandbox
    root = memory.data_dir
    directory = sessions / "wave-only"
    directory.mkdir(parents=True)
    write_marked_wav(directory / "audio.wav")
    _processor(memory, FIRST_TEXT).retry(directory)
    ((identity, parent, number, text),) = original = _revisions(root)
    assert (number, text) == (1, FIRST_TEXT)
    assert not (directory / MARKER).exists()
    receipt = _receipt(root, identity, parent)
    assert receipt["schema_version"] == 3
    assert receipt["publication"]["source_kind"] == "legacy_audio_only"
    reservations = "SELECT meeting_id FROM pending_operations WHERE kind='reserve_audio_only'"
    assert _rows(root, reservations) == [(identity,)]

    assert _rows(root, "PRAGMA wal_checkpoint(TRUNCATE)")[0][0] == 0
    retired = tmp_path / "retired-synthetic-db"
    retired.mkdir()
    assert root.resolve().parent == retired.resolve().parent == tmp_path.resolve()
    for suffix in ("", "-wal", "-shm", "-journal"):
        database = root / f"library.sqlite3{suffix}"
        if database.exists():
            assert not database.is_symlink()
            database.rename(retired / database.name)
    assert not (root / "library.sqlite3").exists()
    memory = MemoryService(root, sessions_root=sessions)
    memory.reconcile()
    assert _revisions(root) == original
    assert _rows(root, reservations) == []
    assert memory.retry_parent(directory, None) == (identity, parent)

    _sealed_retry(memory, directory, monkeypatch)
    intent = _sealed_intent(directory, None)
    assert intent.revision_of == parent and intent.source_kind == "legacy_audio_only"
    assert _rows(root, reservations) == []
    assert _revisions(root) == original

    RecoveryService(sessions, memory_service=memory).discover()

    assert not (directory / MARKER).exists(), "The rebuilt parent authorizes this retry"
    first, active = _revisions(root)
    assert first == original[0]
    assert active == (identity, memory.active_revision(identity), 2, RETRY_TEXT)
