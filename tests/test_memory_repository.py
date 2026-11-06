"""Repository boundaries: atomic publication, immutable evidence, and bounded locks."""

from __future__ import annotations

import importlib
import sqlite3
from dataclasses import FrozenInstanceError, replace
from time import monotonic

import pytest


def _api():
    return (
        importlib.import_module("here.memory.database"),
        importlib.import_module("here.memory.models"),
        importlib.import_module("here.memory.repository"),
    )


def _open(tmp_path):
    database, models, repository = _api()
    library = database.open_library(tmp_path)
    database.migrate(library)
    return database, models, library, repository.MemoryRepository(library)


def _entities(models):
    meeting = models.Meeting(
        id="7c23d988-daa7-49b2-bb74-e6c30ac87c48",
        legacy_session_id="20261105_120000",
        title="Synthetic meeting",
        status=models.MeetingStatus.COMPLETED,
    )
    asset = models.AudioAsset(
        id="asset-a",
        meeting_id=meeting.id,
        relative_path=f"meetings/{meeting.id}/audio.wav",
        size_bytes=16044,
        sha256="a" * 64,
        codec="pcm_s16le",
        sample_rate=8000,
        channels=1,
        duration_ms=1000,
        status="available",
    )
    revision = models.TranscriptRevision(
        id="revision-a",
        meeting_id=meeting.id,
        revision_number=1,
        raw_text="Original evidence. Untimed follow-up.",
        provider="synthetic",
        model="fixture",
        origin="import",
        content_hash="b" * 64,
        status="complete",
    )
    segments = (
        models.Segment(
            id="segment-a",
            revision_id=revision.id,
            ordinal=0,
            text="Original evidence.",
            start_ms=0,
            end_ms=1000,
            timing_source="provider",
            speaker_key="request-1:speaker_0",
        ),
        models.Segment(
            id="segment-b",
            revision_id=revision.id,
            ordinal=1,
            text="Untimed follow-up.",
            timing_source="absent",
        ),
    )
    return meeting, asset, revision, segments


def _insert(repo, entities):
    meeting, asset, revision, segments = entities
    repo.insert_meeting(meeting)
    repo.insert_audio_asset(asset)
    repo.insert_revision(revision)
    repo.insert_segments(segments)
    repo.activate_revision(meeting.id, revision.id)


def test_publication_is_visible_only_after_commit_and_round_trips_evidence(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    database, models, library, repo = _open(tmp_path)
    observer = database.open_library(tmp_path)
    observer_repo = importlib.import_module("here.memory.repository").MemoryRepository(observer)
    entities = _entities(models)
    meeting, asset, revision, segments = entities
    try:
        with repo.transaction():
            _insert(repo, entities)
            assert observer_repo.get_meeting(meeting.id) is None
        assert observer_repo.get_meeting(meeting.id).active_revision_id == revision.id
        assert repo.get_meeting_by_legacy_id("20261105_120000").id == meeting.id
        assert repo.get_audio_asset(asset.id) == asset
        assert repo.get_revision(revision.id) == revision
        assert tuple(repo.get_segments(revision.id)) == segments
        assert repo.get_meeting("absent") is None
    finally:
        observer.close()
        library.close()


def test_failed_retry_rolls_back_new_entities_and_keeps_previous_revision_active(tmp_path):
    _, models, library, repo = _open(tmp_path)
    entities = _entities(models)
    meeting, _, revision, _ = entities
    try:
        with repo.transaction():
            _insert(repo, entities)
        next_revision = replace(revision, id="revision-b", revision_number=2, raw_text="Retry")
        with pytest.raises(RuntimeError, match="publication failed"):
            with repo.transaction():
                repo.insert_revision(next_revision)
                repo.insert_segments(
                    [
                        models.Segment(
                            id="retry-segment",
                            revision_id="revision-b",
                            ordinal=0,
                            text="Retry",
                            timing_source="absent",
                        )
                    ]
                )
                repo.activate_revision(meeting.id, next_revision.id)
                raise RuntimeError("publication failed")
        assert repo.get_meeting(meeting.id).active_revision_id == revision.id
        assert repo.get_revision(revision.id) == revision
        assert repo.get_revision("revision-b") is None
        assert tuple(repo.get_segments("revision-b")) == ()
        assert library.connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert library.connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        with repo.transaction():
            repo.insert_revision(next_revision)
            repo.activate_revision(meeting.id, next_revision.id)
        assert repo.get_meeting(meeting.id).active_revision_id == "revision-b"
        assert repo.get_revision(revision.id) == revision
    finally:
        library.close()


def test_failed_initial_publication_leaves_no_partial_meeting_or_assets(tmp_path):
    _, models, library, repo = _open(tmp_path)
    entities = _entities(models)
    try:
        with pytest.raises(RuntimeError, match="disk publication failed"):
            with repo.transaction():
                _insert(repo, entities)
                raise RuntimeError("disk publication failed")
        assert repo.get_meeting(entities[0].id) is None
        assert repo.get_audio_asset("asset-a") is None
        assert repo.get_revision("revision-a") is None
        assert tuple(repo.get_segments("revision-a")) == ()
    finally:
        library.close()


def test_activation_requires_explicit_transaction(tmp_path):
    database, models, library, repo = _open(tmp_path)
    entities = _entities(models)
    try:
        with repo.transaction():
            repo.insert_meeting(entities[0])
            repo.insert_revision(entities[2])
        with pytest.raises(database.IntegrityFailure):
            repo.activate_revision(entities[0].id, "revision-a")
        assert repo.get_meeting(entities[0].id).active_revision_id is None
    finally:
        library.close()


@pytest.mark.parametrize("target", ["missing", "other-meeting"])
def test_activation_rejects_missing_or_foreign_revision(tmp_path, target):
    database, models, library, repo = _open(tmp_path)
    entities = _entities(models)
    try:
        with repo.transaction():
            _insert(repo, entities)
            if target == "other-meeting":
                other = replace(entities[0], id="other-meeting", legacy_session_id="other")
                repo.insert_meeting(other)
                repo.insert_revision(replace(entities[2], id="foreign", meeting_id=other.id))
        with pytest.raises(database.IntegrityFailure):
            with repo.transaction():
                repo.activate_revision(
                    entities[0].id, "missing" if target == "missing" else "foreign"
                )
        assert repo.get_meeting(entities[0].id).active_revision_id == "revision-a"
    finally:
        library.close()


def test_busy_second_writer_is_typed_bounded_and_recovers_after_lock_release(tmp_path):
    database, models, library, first = _open(tmp_path)
    second_library = database.open_library(tmp_path)
    second = importlib.import_module("here.memory.repository").MemoryRepository(second_library)
    second_library.connection.execute("PRAGMA busy_timeout=40")
    try:
        with first.transaction():
            started = monotonic()
            with pytest.raises(database.Busy):
                with second.transaction():
                    pytest.fail("BEGIN IMMEDIATE must reject the second writer before mutations")
            assert monotonic() - started < 1.0
        with second.transaction():
            second.insert_meeting(_entities(models)[0])
        assert first.get_meeting(_entities(models)[0].id) is not None
    finally:
        second_library.close()
        library.close()


@pytest.mark.parametrize(
    "operation_name",
    [
        "migrate",
        "get_meeting",
        "get_meeting_by_legacy_id",
        "get_audio_asset",
        "get_revision",
        "get_segments",
        "get_ingest_source",
        "get_pending_operation",
    ],
)
def test_exclusive_lock_returns_typed_bounded_busy_from_storage_reads(tmp_path, operation_name):
    database, _, library, repo = _open(tmp_path)
    locker = None
    operations = {
        "migrate": lambda: database.migrate(library),
        "get_meeting": lambda: repo.get_meeting("synthetic"),
        "get_meeting_by_legacy_id": lambda: repo.get_meeting_by_legacy_id("synthetic"),
        "get_audio_asset": lambda: repo.get_audio_asset("synthetic"),
        "get_revision": lambda: repo.get_revision("synthetic"),
        "get_segments": lambda: repo.get_segments("synthetic"),
        "get_ingest_source": lambda: repo.get_ingest_source("synthetic", "a" * 64),
        "get_pending_operation": lambda: repo.get_pending_operation("synthetic"),
    }
    try:
        assert library.connection.execute("PRAGMA journal_mode=DELETE").fetchone()[0] == "delete"
        library.connection.execute("PRAGMA busy_timeout=40")
        locker = sqlite3.connect(tmp_path / "library.sqlite3", isolation_level=None)
        locker.execute("BEGIN EXCLUSIVE")
        started = monotonic()
        try:
            with pytest.raises(database.Busy):
                operations[operation_name]()
        finally:
            assert monotonic() - started < 1.0
            locker.rollback()
        assert not library.connection.in_transaction
        operations[operation_name]()
        assert library.connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        if locker is not None:
            locker.close()
        library.close()


def test_ingest_identity_and_operation_phase_survive_reopen(tmp_path):
    database, models, library, repo = _open(tmp_path)
    meeting = _entities(models)[0]
    source = models.IngestSource(
        id="source-a",
        origin_path="synthetic/legacy-session",
        origin_fingerprint="c" * 64,
        meeting_id=meeting.id,
        revision_id=None,
    )
    operation = models.PendingOperation(
        id="operation-a",
        kind="publish_revision",
        phase="prepared",
        meeting_id=meeting.id,
        staging_path=".staging/operation-a",
        destination_path=f"meetings/{meeting.id}/revisions/1",
    )
    try:
        with repo.transaction():
            repo.insert_meeting(meeting)
            repo.insert_ingest_source(source)
            repo.insert_pending_operation(operation)
        with pytest.raises(RuntimeError, match="abort phase"):
            with repo.transaction():
                repo.transition_pending_operation(operation.id, "complete")
                raise RuntimeError("abort phase")
        assert repo.get_pending_operation(operation.id).phase == "prepared"
        with repo.transaction():
            repo.transition_pending_operation(operation.id, "complete")
    finally:
        library.close()
    reopened = database.open_library(tmp_path)
    try:
        persisted = importlib.import_module("here.memory.repository").MemoryRepository(reopened)
        assert persisted.get_ingest_source(source.origin_path, source.origin_fingerprint) == source
        assert persisted.get_pending_operation(operation.id).phase == "complete"
        assert persisted.get_ingest_source(source.origin_path, "d" * 64) is None
    finally:
        reopened.close()


def test_all_persistence_dtos_are_frozen_and_meeting_status_has_only_durable_states():
    _, models, _ = _api()
    meeting, asset, revision, segments = _entities(models)
    values = [
        meeting,
        asset,
        revision,
        *segments,
        models.IngestSource(
            id="source",
            origin_path="synthetic/origin",
            origin_fingerprint="c" * 64,
            meeting_id=meeting.id,
        ),
        models.PendingOperation(
            id="operation",
            kind="publish_revision",
            phase="prepared",
            meeting_id=meeting.id,
            staging_path=".staging/operation",
            destination_path=f"meetings/{meeting.id}/revisions/1",
        ),
    ]
    for value in values:
        with pytest.raises((FrozenInstanceError, AttributeError, TypeError)):
            value.id = "mutated"
    assert {status.value for status in models.MeetingStatus} == {
        "recording",
        "processing",
        "completed",
        "failed",
        "cancelled",
        "interrupted",
        "deleting",
    }
