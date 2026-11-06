"""B4 lifecycle edits use real catalog transactions and immutable evidence."""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import asdict, is_dataclass
from uuid import uuid4

import pytest
from here.memory import models
from here.memory import service as service_module
from here.memory.database import LibraryError
from memory_fixtures import MEETING_ID, OTHER_MEETING_ID, make_session, write_marked_wav


def _document(value):
    return asdict(value) if is_dataclass(value) else value


def _rows(root, sql, parameters=()):
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        return connection.execute(sql, parameters).fetchall()


def _patch(**values):
    # Resolve planned DTOs at the action, so missing B4 never breaks collection.
    return models.MeetingPatch(**values)


def _error(name):
    return getattr(service_module, name)


@pytest.fixture
def catalog(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("A local lifecycle operation attempted provider/network work")

    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr("here.transcription.client.build_client", forbidden)
    root = tmp_path / "data"
    memory = service_module.MemoryService(root)
    source = make_session(tmp_path / "source", meeting_id=MEETING_ID)
    assert memory.ingest_session(source.directory).status == "inserted"
    return root, memory, source


def _files(root):
    return {path: path.read_bytes() for path in (root / "meetings").rglob("*") if path.is_file()}


def _snapshot(root):
    return {
        table: _rows(root, f"SELECT * FROM {table} ORDER BY 1")
        for table in (
            "meetings",
            "transcript_revisions",
            "segments",
            "audio_assets",
            "revision_audio_assets",
            "lifecycle_events",
            "segments_fts",
        )
    }


def test_assignment_is_explicit_request_scoped_and_removal_keeps_original_evidence(catalog):
    root, memory, _source = catalog
    before = memory.show(MEETING_ID)
    revision = before["meeting"]["active_revision_id"]
    keys = [segment["speaker_key"] for segment in before["segments"]]
    assert [json.loads(key) for key in keys] == [
        ["synthetic-request-0", "speaker_0"],
        ["synthetic-request-1", "speaker_0"],
    ]
    assert _rows(root, "SELECT * FROM speaker_assignments") == []
    participant = _document(memory.add_participant("Synthetica"))
    memory.link_participant(MEETING_ID, participant["id"], role="observer")
    memory.assign_speaker(MEETING_ID, keys[0], participant["id"], scope=revision)
    assert _rows(
        root, "SELECT revision_id,speaker_key,participant_id FROM speaker_assignments"
    ) == [
        (revision, keys[0], participant["id"]),
    ]
    assert _rows(root, "SELECT role FROM meeting_participants") == [("observer",)]
    assert memory.show(MEETING_ID)["segments"] == before["segments"]
    memory.assign_speaker(MEETING_ID, keys[0], None, scope=revision)
    assert not _rows(root, "SELECT 1 FROM speaker_assignments WHERE participant_id IS NOT NULL")
    assert _rows(root, "SELECT id,display_name FROM participants") == [
        (participant["id"], "Synthetica"),
    ]
    assert _rows(root, "SELECT participant_id FROM meeting_participants") == [(participant["id"],)]
    assert memory.show(MEETING_ID)["segments"] == before["segments"]


def test_equal_display_names_do_not_merge_manual_participants(catalog):
    root, memory, _source = catalog
    first = _document(memory.add_participant("Same synthetic name"))
    second = _document(memory.add_participant("Same synthetic name"))
    assert first["id"] != second["id"]
    assert _rows(root, "SELECT display_name FROM participants") == [
        ("Same synthetic name",),
        ("Same synthetic name",),
    ]


@pytest.mark.parametrize("display_name", ["", "  \t", None, 7])
def test_invalid_participant_name_creates_no_row(catalog, display_name):
    root, memory, _source = catalog
    with pytest.raises(_error("InvalidInput")):
        memory.add_participant(display_name)
    assert _rows(root, "SELECT * FROM participants") == []


@pytest.mark.parametrize("invalid", ["raw-label", "foreign-revision", "unknown-participant"])
def test_assignment_rejects_unproven_scope_or_identity_without_writing(catalog, tmp_path, invalid):
    root, memory, _source = catalog
    other = make_session(tmp_path / "other", meeting_id=OTHER_MEETING_ID)
    assert memory.ingest_session(other.directory).status == "inserted"
    participant = _document(memory.add_participant("Explicit synthetic identity"))
    memory.link_participant(MEETING_ID, participant["id"])
    shown = memory.show(MEETING_ID)
    key = shown["segments"][0]["speaker_key"]
    revision = shown["meeting"]["active_revision_id"]
    participant_id = participant["id"]
    if invalid == "raw-label":
        key = "speaker_0"
    elif invalid == "foreign-revision":
        revision = memory.active_revision(OTHER_MEETING_ID)
    else:
        participant_id = str(uuid4())
    before = _snapshot(root), _files(root)
    with pytest.raises((_error("InvalidInput"), _error("NotFound"))):
        memory.assign_speaker(MEETING_ID, key, participant_id, scope=revision)
    assert (_snapshot(root), _files(root)) == before
    assert _rows(root, "SELECT * FROM speaker_assignments") == []


def test_edit_updates_title_notes_version_event_and_fts_without_mutating_evidence(catalog):
    root, memory, _source = catalog
    before = memory.show(MEETING_ID)
    files = _files(root)
    event_count = _rows(root, "SELECT COUNT(*) FROM lifecycle_events")[0][0]
    result = _document(
        memory.update_meeting(
            MEETING_ID,
            _patch(title="Zéphyr agenda", notes="Private synthetic notes"),
            expected_version=before["meeting"]["row_version"],
        )
    )
    meeting = result["meeting"]
    assert (meeting["title"], meeting["notes"], meeting["row_version"]) == (
        "Zéphyr agenda",
        "Private synthetic notes",
        before["meeting"]["row_version"] + 1,
    )
    after = memory.show(MEETING_ID)
    for key in ("segments", "revisions", "audio_assets", "revision_audio_assets"):
        assert after[key] == before[key]
    assert all(path.read_bytes() == content for path, content in files.items())
    assert _rows(root, "SELECT COUNT(*) FROM lifecycle_events")[0][0] == event_count + 1
    event = _rows(root, "SELECT occurred_at,recorded_duration_ms,payload FROM lifecycle_events")[-1]
    assert event[0] and event[1] == 2000
    payload = json.loads(event[2])
    assert isinstance(payload, dict)
    assert "Private synthetic notes" not in event[2] and "Zéphyr agenda" not in event[2]
    assert _rows(
        root, "SELECT DISTINCT meeting_id FROM segments_fts WHERE segments_fts MATCH 'zephyr'"
    ) == [(MEETING_ID,)]
    assert _rows(root, "SELECT COUNT(*) FROM segments_fts WHERE title<>?", ("Zéphyr agenda",)) == [
        (0,)
    ]


def test_stale_editor_cannot_change_rows_events_fts_or_durable_states(catalog):
    root, memory, _source = catalog
    version = memory.show(MEETING_ID)["meeting"]["row_version"]
    independent = service_module.MemoryService(root)
    memory.update_meeting(MEETING_ID, _patch(title="First editor"), expected_version=version)
    before = _snapshot(root), _files(root)
    with pytest.raises(_error("Conflict")):
        independent.update_meeting(
            MEETING_ID,
            _patch(title="Stale editor", notes="Must not persist"),
            expected_version=version,
        )
    assert (_snapshot(root), _files(root)) == before


@pytest.mark.parametrize("changes", [{"title": 2}, {"notes": []}])
def test_invalid_patch_does_not_mutate_catalog(catalog, changes):
    root, memory, _source = catalog
    before = _snapshot(root), _files(root)
    with pytest.raises(_error("InvalidInput")):
        memory.update_meeting(MEETING_ID, _patch(**changes), expected_version=2)
    assert (_snapshot(root), _files(root)) == before


def test_missing_meeting_edit_is_typed_not_found(catalog):
    _root, memory, _source = catalog
    with pytest.raises(_error("NotFound")):
        memory.update_meeting(OTHER_MEETING_ID, _patch(title="Absent"), expected_version=1)


def test_edit_rejects_untyped_free_sql_fields_without_changing_status(catalog):
    root, memory, _source = catalog
    before = _snapshot(root), _files(root)
    with pytest.raises(_error("InvalidInput")):
        memory.update_meeting(MEETING_ID, {"status": "deleting"}, expected_version=2)
    assert (_snapshot(root), _files(root)) == before


def test_event_insert_failure_rolls_back_title_version_and_fts_in_same_transaction(catalog):
    root, memory, _source = catalog
    before = _snapshot(root), _files(root)
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        connection.execute(
            "CREATE TRIGGER synthetic_event_failure BEFORE INSERT ON lifecycle_events "
            "BEGIN SELECT RAISE(ABORT, 'synthetic event write failure'); END"
        )
        connection.commit()
    with pytest.raises(LibraryError):
        memory.update_meeting(
            MEETING_ID,
            _patch(title="Atomic zephyr"),
            expected_version=memory.show(MEETING_ID)["meeting"]["row_version"],
        )
    assert (_snapshot(root), _files(root)) == before


def test_postcommit_state_io_failure_reports_committed_edit_with_recovery_warning(
    catalog, monkeypatch
):
    root, memory, _source = catalog
    version = memory.show(MEETING_ID)["meeting"]["row_version"]
    original = service_module.MemoryService._write_state
    failed = []

    def fail_state(self, repo, identity):
        if identity == MEETING_ID and not failed:
            assert _rows(root, "SELECT title,row_version FROM meetings") == [
                ("Committed zephyr", version + 1),
            ]
            failed.append(identity)
            raise OSError("synthetic postcommit state disk failure")
        return original(self, repo, identity)

    with monkeypatch.context() as patch:
        patch.setattr(service_module.MemoryService, "_write_state", fail_state)
        result = _document(
            memory.update_meeting(
                MEETING_ID,
                _patch(title="Committed zephyr"),
                expected_version=version,
            )
        )
    assert failed == [MEETING_ID]
    assert result["meeting"]["title"] == "Committed zephyr" and result["warnings"]
    assert _rows(root, "SELECT title,row_version FROM meetings") == [
        ("Committed zephyr", version + 1)
    ]
    memory.reconcile()
    states = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (root / "meetings" / MEETING_ID / "states").glob("*.json")
    ]
    assert any(state["meeting"]["title"] == "Committed zephyr" for state in states)


def test_edit_works_without_fts_capability(catalog, monkeypatch):
    _root, memory, _source = catalog
    monkeypatch.setattr(service_module, "fts_available", lambda connection: False)
    version = memory.show(MEETING_ID)["meeting"]["row_version"]
    result = _document(
        memory.update_meeting(
            MEETING_ID,
            _patch(title="Locally editable"),
            expected_version=version,
        )
    )
    assert result["meeting"]["title"] == "Locally editable"


@pytest.mark.parametrize("receipt_version", [2, 3])
def test_edited_state_rebuild_keeps_exact_a_b_a_links_and_immutable_receipts(
    tmp_path,
    monkeypatch,
    receipt_version,
):
    root = tmp_path / "data"
    memory = service_module.MemoryService(root)
    sources = [
        make_session(tmp_path / name, meeting_id=MEETING_ID) for name in ("a", "b", "a-again")
    ]
    audio_b = bytearray(sources[1].audio.read_bytes())
    audio_b[-2:] = b"\x34\x12"
    sources[1].audio.write_bytes(audio_b)
    expected_hashes = [hashlib.sha256(source.audio.read_bytes()).hexdigest() for source in sources]
    parent = None
    for source in sources:
        if receipt_version == 3:
            with memory._publication(str(uuid4()), "session"):
                result = memory.ingest_session(source.directory, revision_of=parent)
        else:
            result = memory.ingest_session(source.directory, revision_of=parent)
        assert result.status in {"inserted", "revised"}
        parent = memory.active_revision(MEETING_ID)
    before = memory.show(MEETING_ID)
    revisions = sorted(before["revisions"], key=lambda revision: revision["revision_number"])
    assets = {asset["id"]: asset for asset in before["audio_assets"]}
    links = before["revision_audio_assets"]
    selected = [
        next(link["asset_id"] for link in links if link["revision_id"] == revision["id"])
        for revision in revisions
    ]
    assert selected[0] == selected[2] != selected[1]
    assert [assets[identity]["sha256"] for identity in selected] == expected_hashes
    receipts = {path: path.read_bytes() for path in (root / "meetings").rglob("catalog.json")}
    assert {json.loads(content)["schema_version"] for content in receipts.values()} == {
        receipt_version
    }
    memory.update_meeting(
        MEETING_ID,
        _patch(title="Durable zephyr", notes="Durable notes"),
        expected_version=before["meeting"]["row_version"],
    )
    edited = memory.show(MEETING_ID)
    assert all(path.read_bytes() == content for path, content in receipts.items())
    for source in sources:
        source.directory.rename(source.directory.with_name(source.directory.name + "-unavailable"))
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        assert connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] == 0
    for suffix in ("", "-wal", "-shm", "-journal"):
        path = root / ("library.sqlite3" + suffix)
        assert path.absolute().parent == root.absolute() and not path.is_symlink()
        path.unlink(missing_ok=True)
    monkeypatch.setattr("socket.socket.connect", lambda *args: pytest.fail("Rebuild used network"))
    restored = service_module.MemoryService(root)
    restored.reconcile()
    after = restored.show(MEETING_ID)
    assert after["meeting"] == edited["meeting"]
    for key in ("revisions", "segments", "audio_assets", "revision_audio_assets"):
        assert sorted(after[key], key=lambda item: json.dumps(item, sort_keys=True)) == sorted(
            before[key],
            key=lambda item: json.dumps(item, sort_keys=True),
        )
    assert all(path.read_bytes() == content for path, content in receipts.items())


def test_edit_audio_only_reservation_preserves_null_dates_and_zero_historical_revisions(tmp_path):
    root = tmp_path / "data"
    memory = service_module.MemoryService(root)
    directory = tmp_path / "orphan"
    directory.mkdir()
    write_marked_wav(directory / "audio.wav")
    identity, parent = memory._reserve_audio_only(directory, 2000)
    try:
        assert parent is None
        before = memory.show(identity)
        assert before["revisions"] == []
        result = _document(
            memory.update_meeting(
                identity,
                _patch(title="Unknown capture date", notes="Synthetic orphan"),
                expected_version=before["meeting"]["row_version"],
            )
        )
        assert result["meeting"]["started_at"] is result["meeting"]["completed_at"] is None
        assert memory.show(identity)["revisions"] == []
        assert _rows(root, "SELECT revision_id FROM ingest_sources") == [(None,)]
        assert not list((root / "meetings").rglob("catalog.json"))
    finally:
        memory.finish_capture(identity)


def test_edited_audio_only_publication_rebuild_preserves_null_dates_and_source_kind(tmp_path):
    root = tmp_path / "data"
    memory = service_module.MemoryService(root)
    directory = tmp_path / "orphan"
    directory.mkdir()
    write_marked_wav(directory / "audio.wav")
    identity, parent = memory._reserve_audio_only(directory, 2000)
    try:
        assert parent is None and memory.show(identity)["revisions"] == []
        compatibility = make_session(tmp_path / "compatibility")
        # The accepted fixture supplies only synthetic v1 compatibility output.
        # Its real-looking capture dates must never override reserved unknown dates.
        for path in compatibility.directory.iterdir():
            if path.name != "audio.wav":
                (directory / path.name).write_bytes(path.read_bytes())
        with memory._publication(str(uuid4()), "legacy_audio_only"):
            result = memory.ingest_session(directory)
        assert result.status == "inserted" and result.meeting_id == identity
        before = memory.show(identity)
        assert len(before["revisions"]) == 1
        assert before["meeting"]["started_at"] is before["meeting"]["completed_at"] is None
        memory.update_meeting(
            identity,
            _patch(title="Edited unknown date"),
            expected_version=before["meeting"]["row_version"],
        )
        edited = memory.show(identity)
    finally:
        memory.finish_capture(identity)
    receipts = list((root / "meetings").rglob("catalog.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert receipt["publication"]["source_kind"] == "legacy_audio_only"
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        assert connection.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()[0] == 0
    for suffix in ("", "-wal", "-shm", "-journal"):
        path = root / ("library.sqlite3" + suffix)
        assert path.absolute().parent == root.absolute() and not path.is_symlink()
        path.unlink(missing_ok=True)
    restored = service_module.MemoryService(root)
    restored.reconcile()
    after = restored.show(identity)
    assert after["meeting"] == edited["meeting"]
    assert len(after["revisions"]) == 1
    assert after["revision_audio_assets"] == before["revision_audio_assets"]
    assert after["meeting"]["started_at"] is after["meeting"]["completed_at"] is None
    assert _rows(root, "SELECT COUNT(*) FROM ingest_sources WHERE revision_id IS NULL") == [(1,)]


def test_r48_participant_reads_distinguish_empty_global_and_per_meeting_membership(
    catalog, tmp_path
):
    _root, memory, _source = catalog
    assert memory.list_participants() == ()
    assert memory.list_participants(MEETING_ID) == ()
    unlinked = memory.add_participant("Unlinked synthetic person")
    first = memory.add_participant("First linked synthetic person")
    second = memory.add_participant("Second linked synthetic person")
    other = make_session(tmp_path / "r48-other", meeting_id=OTHER_MEETING_ID)
    assert memory.ingest_session(other.directory).status == "inserted"
    memory.link_participant(MEETING_ID, first.id, role="chair")
    memory.link_participant(OTHER_MEETING_ID, second.id)
    global_people = memory.list_participants()
    assert isinstance(global_people, tuple)
    assert all(isinstance(person, models.Participant) for person in global_people)
    assert {person.id: person.display_name for person in global_people} == {
        unlinked.id: "Unlinked synthetic person",
        first.id: "First linked synthetic person",
        second.id: "Second linked synthetic person",
    }
    assert memory.list_participants(MEETING_ID) == (first,)
    assert memory.list_participants(OTHER_MEETING_ID) == (second,)


def test_r48_get_meeting_preserves_typed_exact_audio_revision_and_empty_attribution(catalog):
    _root, memory, _source = catalog
    before = memory.show(MEETING_ID)
    detail = memory.get_meeting(MEETING_ID)
    assert isinstance(detail, models.MeetingDetail)
    assert isinstance(detail.meeting, models.Meeting)
    assert asdict(detail.meeting) == before["meeting"]
    for name, expected_key, model in (
        ("audio_assets", "audio_assets", models.AudioAsset),
        ("revisions", "revisions", models.TranscriptRevision),
        ("segments", "segments", models.Segment),
        ("revision_audio_assets", "revision_audio_assets", models.RevisionAudioAsset),
    ):
        values = getattr(detail, name)
        assert isinstance(values, tuple) and all(isinstance(value, model) for value in values)
        assert sorted((asdict(value) for value in values), key=lambda value: str(value)) == sorted(
            before[expected_key],
            key=lambda value: str(value),
        )
    assert detail.participants == detail.participant_links == detail.speaker_assignments == ()
    assert isinstance(detail.warnings, tuple)


def test_r48_get_meeting_exposes_roles_and_nullable_request_assignments_as_typed_tuples(catalog):
    root, memory, _source = catalog
    first = memory.add_participant("Explicit first person")
    second = memory.add_participant("Explicit second person")
    memory.add_participant("Unlinked person excluded from meeting detail")
    memory.link_participant(MEETING_ID, first.id, role="chair")
    memory.link_participant(MEETING_ID, second.id)
    before = memory.show(MEETING_ID)
    revision_id = before["meeting"]["active_revision_id"]
    first_key, second_key = [segment["speaker_key"] for segment in before["segments"]]
    memory.assign_speaker(MEETING_ID, first_key, first.id, scope=revision_id)
    # Schema2 permits a stored explicitly nullable assignment; reading cannot
    # turn its equal provider label into the first request's participant.
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        connection.execute(
            "INSERT INTO speaker_assignments VALUES (?,?,?,NULL)",
            (MEETING_ID, revision_id, second_key),
        )
        connection.commit()
    detail = memory.get_meeting(MEETING_ID)
    assert isinstance(detail.participants, tuple)
    assert set(detail.participants) == {first, second}
    assert isinstance(detail.participant_links, tuple)
    link_type = getattr(models, "MeetingParticipant")
    assert all(isinstance(link, link_type) for link in detail.participant_links)
    assert {
        (link.meeting_id, link.participant_id, link.role) for link in detail.participant_links
    } == {
        (MEETING_ID, first.id, "chair"),
        (MEETING_ID, second.id, None),
    }
    assignment_type = getattr(models, "SpeakerAssignment")
    assert isinstance(detail.speaker_assignments, tuple)
    assert all(isinstance(item, assignment_type) for item in detail.speaker_assignments)
    assert {
        (item.meeting_id, item.revision_id, item.speaker_key, item.participant_id)
        for item in detail.speaker_assignments
    } == {
        (MEETING_ID, revision_id, first_key, first.id),
        (MEETING_ID, revision_id, second_key, None),
    }
    assert [asdict(segment) for segment in detail.segments] == before["segments"]


def test_r48_detached_immutable_read_snapshot_survives_subsequent_real_writes(catalog):
    from dataclasses import FrozenInstanceError

    _root, memory, _source = catalog
    person = memory.add_participant("Original explicit person")
    memory.link_participant(MEETING_ID, person.id, role="original role")
    show = memory.show(MEETING_ID)
    revision = show["meeting"]["active_revision_id"]
    key = show["segments"][0]["speaker_key"]
    memory.assign_speaker(MEETING_ID, key, person.id, scope=revision)
    snapshot = memory.get_meeting(MEETING_ID)
    participants = memory.list_participants(MEETING_ID)
    retained = asdict(snapshot)
    memory.link_participant(MEETING_ID, person.id, role="new role")
    memory.assign_speaker(MEETING_ID, key, None, scope=revision)
    memory.update_meeting(
        MEETING_ID,
        models.MeetingPatch(title="After snapshot"),
        expected_version=memory.show(MEETING_ID)["meeting"]["row_version"],
    )
    memory.add_participant("Later global participant")
    assert asdict(snapshot) == retained
    assert participants == (person,)
    assert snapshot.participant_links[0].role == "original role"
    assert snapshot.speaker_assignments[0].participant_id == person.id
    current = memory.get_meeting(MEETING_ID)
    assert current.meeting.title == "After snapshot"
    assert current.participant_links[0].role == "new role"
    assert current.speaker_assignments == ()
    for value, field_name in (
        (snapshot, "participants"),
        (snapshot.meeting, "title"),
        (snapshot.participants[0], "display_name"),
        (snapshot.participant_links[0], "role"),
        (snapshot.speaker_assignments[0], "participant_id"),
    ):
        with pytest.raises(FrozenInstanceError):
            setattr(value, field_name, None)


def test_r48_detail_old_positional_constructor_keeps_warnings_before_new_tuple_defaults(catalog):
    _root, memory, _source = catalog
    existing = memory.link_participant(MEETING_ID, memory.add_participant("Linked person").id)
    warnings = ("Synthetic existing positional warning",)
    detail = models.MeetingDetail(
        existing.meeting,
        existing.audio_assets,
        existing.revisions,
        existing.segments,
        existing.revision_audio_assets,
        warnings,
    )
    assert detail.meeting is existing.meeting
    assert detail.audio_assets == existing.audio_assets
    assert detail.revisions == existing.revisions
    assert detail.segments == existing.segments
    assert detail.revision_audio_assets == existing.revision_audio_assets
    assert detail.warnings == warnings
    assert detail.participants == detail.participant_links == detail.speaker_assignments == ()


def test_r48_missing_meeting_read_raises_canonical_not_found_without_fabricating_rows(catalog):
    root, memory, _source = catalog
    before = _snapshot(root), _files(root)
    with pytest.raises(service_module.NotFound):
        memory.get_meeting(OTHER_MEETING_ID)
    assert (_snapshot(root), _files(root)) == before


@pytest.mark.parametrize("kind", ["link", "assignment"])
def test_r48_new_link_and_assignment_dtos_preserve_default_role_and_nullable_identity(kind):
    from dataclasses import FrozenInstanceError

    if kind == "link":
        link = getattr(models, "MeetingParticipant")(MEETING_ID, OTHER_MEETING_ID)
        assert asdict(link) == {
            "meeting_id": MEETING_ID,
            "participant_id": OTHER_MEETING_ID,
            "role": None,
        }
        with pytest.raises(FrozenInstanceError):
            link.role = "inferred role"
    else:
        assignment = getattr(models, "SpeakerAssignment")(
            MEETING_ID,
            "33333333-3333-4333-8333-333333333333",
            '["synthetic-request-1", "speaker_0"]',
            None,
        )
        assert asdict(assignment) == {
            "meeting_id": MEETING_ID,
            "revision_id": "33333333-3333-4333-8333-333333333333",
            "speaker_key": '["synthetic-request-1", "speaker_0"]',
            "participant_id": None,
        }
        with pytest.raises(FrozenInstanceError):
            assignment.participant_id = OTHER_MEETING_ID


@pytest.fixture
def h2_registry_observer(monkeypatch):
    """Observe actual final-record opens/reads and each live SQLite transaction mode."""
    from pathlib import Path
    from here.memory.database import LibraryConnection

    observations = []
    statements = []
    live = {}
    original_open = service_module.open_library
    original_close = LibraryConnection.close
    original_path_open = Path.open

    def open_library(*args, **kwargs):
        library = original_open(*args, **kwargs)
        state = {"library": library, "mode": None}
        live[id(library)] = state

        def trace(sql):
            statements.append(sql)
            normalized = sql.strip().upper()
            if normalized.startswith("BEGIN"):
                state["mode"] = "write" if "IMMEDIATE" in normalized else "read"
            elif normalized in {"COMMIT", "ROLLBACK"}:
                state["mode"] = None

        library.connection.set_trace_callback(trace)
        return library

    def close_library(library):
        live.pop(id(library), None)
        return original_close(library)

    def observe(action, path):
        observations.append((action, path.name, tuple(
            (state["library"].connection.in_transaction, state["mode"])
            for state in live.values()
        )))

    class RegistryFile:
        def __init__(self, file, path):
            self.file = file
            self.path = path

        def __getattr__(self, name):
            return getattr(self.file, name)

        def __enter__(self):
            self.file.__enter__()
            return self

        def __exit__(self, *args):
            return self.file.__exit__(*args)

        def read(self, *args, **kwargs):
            observe("read", self.path)
            return self.file.read(*args, **kwargs)

    def path_open(path, *args, **kwargs):
        file = original_path_open(path, *args, **kwargs)
        if path.parent.name == ".deletions" and path.suffix == ".json":
            observe("open", path)
            return RegistryFile(file, path)
        return file

    monkeypatch.setattr(service_module, "open_library", open_library)
    monkeypatch.setattr(LibraryConnection, "close", close_library)
    monkeypatch.setattr(Path, "open", path_open)
    return observations, statements


@pytest.mark.parametrize("operation", ["begin-processing", "update-meeting"])
@pytest.mark.parametrize("outcome", ["committed", "event-rollback", "suppressed"])
def test_h2_registry_io_precedes_write_transaction_without_losing_atomicity_or_suppression(
    catalog, h2_registry_observer, operation, outcome
):
    from memory_fixtures import write_json

    root, memory, _source = catalog
    observations, statements = h2_registry_observer
    directory = root / ".deletions"
    directory.mkdir()
    # An unrelated valid final forces actual filesystem work on an ordinary write.
    write_json(directory / f"{OTHER_MEETING_ID}.json", {
        "schema_version": 1, "meeting_id": OTHER_MEETING_ID, "origins": [], "phase": "pending",
    })
    detail = memory.get_meeting(MEETING_ID)
    assert detail.meeting.id == MEETING_ID
    assert any(
        action == "read" and (True, "read") in states
        for action, _name, states in observations
    ), "R50 must still validate records within its coherent BEGIN read snapshot"
    before = _snapshot(root), _files(root)
    pending_before = _rows(root, "SELECT * FROM pending_operations ORDER BY id")
    version = detail.meeting.row_version
    old_status = detail.meeting.status
    if outcome == "suppressed":
        write_json(directory / f"{MEETING_ID}.json", {
            "schema_version": 1, "meeting_id": MEETING_ID, "origins": [], "phase": "pending",
        })
    elif outcome == "event-rollback":
        with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
            connection.execute(
                "CREATE TRIGGER h2_event_failure BEFORE INSERT ON lifecycle_events "
                "BEGIN SELECT RAISE(ABORT, 'synthetic H2 event failure'); END"
            )
            connection.commit()
    observations.clear()
    statements.clear()

    def perform():
        if operation == "begin-processing":
            memory.begin_processing(MEETING_ID)
        else:
            memory.update_meeting(
                MEETING_ID, _patch(title="H2 atomic title"), expected_version=version,
            )

    try:
        if outcome == "suppressed":
            with pytest.raises(service_module.NotFound):
                perform()
        elif outcome == "event-rollback":
            with pytest.raises(LibraryError):
                perform()
        else:
            perform()
    finally:
        if operation == "begin-processing":
            memory.finish_capture(MEETING_ID)

    # Check effects before the I/O boundary assertion, so a RED identifies the
    # forbidden interval rather than hiding whether the real operation completed.
    if outcome == "committed":
        assert _rows(root, "SELECT status,row_version,title FROM meetings WHERE id=?", (MEETING_ID,)) == [
            ("processing" if operation == "begin-processing" else old_status,
             version + 1, detail.meeting.title if operation == "begin-processing" else "H2 atomic title"),
        ]
        events = _rows(root, "SELECT kind,payload FROM lifecycle_events WHERE meeting_id=?", (MEETING_ID,))
        assert len(events) == len(before[0]["lifecycle_events"]) + 1
        expected_kind = "status_changed" if operation == "begin-processing" else "metadata_updated"
        expected_payload = ({"from": old_status, "to": "processing"}
                            if operation == "begin-processing" else {"fields": ["title"]})
        assert sum(kind == expected_kind and json.loads(payload) == expected_payload
                   for kind, payload in events) == 1
        for table in ("transcript_revisions", "segments", "audio_assets", "revision_audio_assets"):
            assert _rows(root, f"SELECT * FROM {table} ORDER BY 1") == before[0][table]
    else:
        assert (_snapshot(root), _files(root)) == before
        assert _rows(root, "SELECT * FROM pending_operations ORDER BY id") == pending_before
    if outcome != "suppressed":
        assert any(sql.strip().upper() == "BEGIN IMMEDIATE" for sql in statements)
    assert {action for action, _name, _states in observations} == {"open", "read"}
    forbidden = [item for item in observations if (True, "write") in item[2]]
    assert forbidden == [], "Deletion registry filesystem I/O occurred under BEGIN IMMEDIATE"
    if outcome == "suppressed":
        assert memory.show(MEETING_ID) is None
