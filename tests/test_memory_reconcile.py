"""B4 adds inspectable local recovery outcomes, events and asset health."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from dataclasses import asdict, is_dataclass
from uuid import uuid4

import pytest
from here.memory.service import MemoryService
from here.output.ownership import FileLease
from memory_fixtures import MEETING_ID, OTHER_MEETING_ID, TRANSCRIPT_TEXT, make_session, write_json


def _rows(root, sql, parameters=()):
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        return connection.execute(sql, parameters).fetchall()


def _outcome(report, category, identity):
    entries = getattr(report, category)
    documents = [asdict(item) if is_dataclass(item) else item for item in entries]
    matches = [item for item in documents if item["identity"] == identity]
    assert len(matches) == 1, (category, identity, documents)
    assert isinstance(matches[0]["reason"], str) and matches[0]["reason"]
    return matches[0]


@pytest.fixture(autouse=True)
def no_provider(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Reconciliation allocated provider/network work")

    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr("here.transcription.client.build_client", forbidden)
    monkeypatch.setattr("here.application.processing.transcribe_recording_session", forbidden)


@pytest.mark.parametrize("transition", ["capture", "processing", "failed", "cancelled", "status"])
def test_explicit_status_transition_records_one_validated_lifecycle_event(tmp_path, transition):
    root = tmp_path / "data"
    memory = MemoryService(root)
    if transition != "capture":
        source = make_session(tmp_path / "source", meeting_id=MEETING_ID)
        assert memory.ingest_session(source.directory).status == "inserted"
    if transition in {"failed", "cancelled"}:
        memory.begin_processing(MEETING_ID)
    count = _rows(root, "SELECT COUNT(*) FROM lifecycle_events")[0][0]
    try:
        if transition == "capture":
            memory.begin_capture(MEETING_ID)
            expected_status = "recording"
        elif transition == "processing":
            memory.begin_processing(MEETING_ID)
            expected_status = "processing"
        elif transition == "status":
            memory.set_status(MEETING_ID, "interrupted")
            expected_status = "interrupted"
        else:
            result = memory.finish_attempt(
                MEETING_ID, transition, revision_of=memory.active_revision(MEETING_ID)
            )
            assert result.status == "unchanged"
            expected_status = transition
        assert _rows(root, "SELECT status FROM meetings") == [(expected_status,)]
        assert _rows(root, "SELECT COUNT(*) FROM lifecycle_events")[0][0] == count + 1
        event = _rows(
            root, "SELECT occurred_at,recorded_duration_ms,payload FROM lifecycle_events"
        )[-1]
        assert event[0] and (event[1] is None or event[1] >= 0)
        assert isinstance(json.loads(event[2]), dict)
    finally:
        memory.finish_capture(MEETING_ID)


@pytest.mark.parametrize("status", ["recording", "processing"])
def test_stale_transition_records_one_event_keeps_audio_and_reports_marked(tmp_path, status):
    root = tmp_path / "data"
    memory = MemoryService(root)
    source = make_session(tmp_path / "source", meeting_id=MEETING_ID)
    assert memory.ingest_session(source.directory).status == "inserted"
    before = memory.show(MEETING_ID)
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        connection.execute("UPDATE meetings SET status=? WHERE id=?", (status, MEETING_ID))
        connection.commit()
    count = _rows(root, "SELECT COUNT(*) FROM lifecycle_events")[0][0]
    report = memory.reconcile()
    after = memory.show(MEETING_ID)
    assert after["meeting"]["status"] == "interrupted"
    assert after["meeting"]["row_version"] == before["meeting"]["row_version"] + 1
    for key in ("audio_assets", "revisions", "revision_audio_assets", "segments"):
        assert after[key] == before[key]
    assert all((root / asset["relative_path"]).is_file() for asset in after["audio_assets"])
    assert _rows(root, "SELECT COUNT(*) FROM lifecycle_events")[0][0] == count + 1
    event = _rows(root, "SELECT occurred_at,recorded_duration_ms,payload FROM lifecycle_events")[-1]
    assert event[0] and event[1] == 2000 and isinstance(json.loads(event[2]), dict)
    _outcome(report, "marked", MEETING_ID)
    second = memory.reconcile()
    assert not second.marked
    assert _rows(root, "SELECT COUNT(*) FROM lifecycle_events")[0][0] == count + 1


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_asset_damage_is_reported_without_deleting_text_or_other_files(tmp_path, damage):
    root = tmp_path / "data"
    memory = MemoryService(root)
    source = make_session(tmp_path / "source", meeting_id=MEETING_ID)
    assert memory.ingest_session(source.directory).status == "inserted"
    before = memory.show(MEETING_ID)
    asset = before["audio_assets"][0]
    path = root / asset["relative_path"]
    original = path.read_bytes()
    if damage == "missing":
        path.unlink()
    else:
        path.write_bytes(original[:-2] + b"\x11\x22")
    retained = {
        entry: entry.read_bytes() for entry in (root / "meetings").rglob("*") if entry.is_file()
    }
    report = memory.reconcile()
    after = memory.show(MEETING_ID)
    assert after["audio_assets"][0]["status"] == damage
    assert after["revisions"][0]["raw_text"] == TRANSCRIPT_TEXT
    assert after["segments"] == before["segments"]
    assert after["meeting"]["status"] == "completed"
    assert all(entry.read_bytes() == contents for entry, contents in retained.items())
    assert path.exists() == (damage == "corrupt")
    _outcome(report, "marked", asset["id"])
    memory.reconcile()
    assert all(entry.read_bytes() == contents for entry, contents in retained.items())
    path.write_bytes(original)
    memory.reconcile()
    assert memory.show(MEETING_ID)["audio_assets"][0]["status"] == "available"


def test_held_owner_skips_status_and_prepared_operation_then_residual_file_allows_recovery(
    tmp_path,
):
    root = tmp_path / "data"
    memory = MemoryService(root)
    source = make_session(tmp_path / "source", meeting_id=MEETING_ID)
    assert memory.ingest_session(source.directory).status == "inserted"
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        connection.execute("UPDATE meetings SET status='processing'")
        connection.execute("UPDATE pending_operations SET phase='prepared'")
        connection.commit()
    operation_id = _rows(root, "SELECT id FROM pending_operations")[0][0]
    owner_path = root / ".owners" / f"{MEETING_ID}.lock"
    owner_path.parent.mkdir(exist_ok=True)
    lease = FileLease(owner_path)
    try:
        report = memory.reconcile()
        assert _rows(root, "SELECT status FROM meetings") == [("processing",)]
        assert _rows(root, "SELECT phase FROM pending_operations") == [("prepared",)]
        _outcome(report, "skipped", MEETING_ID)
    finally:
        lease.close()
    assert owner_path.is_file()  # An unlocked file is deliberately left behind.
    report = memory.reconcile()
    assert _rows(root, "SELECT status FROM meetings") == [("interrupted",)]
    assert _rows(root, "SELECT phase FROM pending_operations") == [("complete",)]
    _outcome(report, "completed", operation_id)
    _outcome(report, "marked", MEETING_ID)


def test_unsafe_owner_path_is_skipped_without_treating_work_as_stale(tmp_path):
    root = tmp_path / "data"
    memory = MemoryService(root)
    source = make_session(tmp_path / "source", meeting_id=MEETING_ID)
    assert memory.ingest_session(source.directory).status == "inserted"
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        connection.execute("UPDATE meetings SET status='processing'")
        connection.commit()
    unsafe = root / ".owners" / f"{MEETING_ID}.lock"
    unsafe.mkdir(parents=True)  # A directory cannot establish absence of an OS owner.
    report = memory.reconcile()
    assert _rows(root, "SELECT status FROM meetings") == [("processing",)]
    assert report.warnings and unsafe.is_dir()
    _outcome(report, "skipped", MEETING_ID)


def test_reconcile_reports_completed_reverted_and_unsafe_sibling_idempotently(tmp_path):
    root = tmp_path / "data"
    memory = MemoryService(root)
    source = make_session(tmp_path / "source", meeting_id=MEETING_ID)
    assert memory.ingest_session(source.directory).status == "inserted"
    complete_id = _rows(root, "SELECT id FROM pending_operations")[0][0]
    reverted_id, revision_id, unsafe_id = (str(uuid4()) for _ in range(3))
    staging = root / ".staging" / reverted_id
    staging.mkdir(parents=True)
    (staging / "transcript.txt").write_text("Synthetic uncommitted evidence", encoding="utf-8")
    external = tmp_path / "external"
    external.mkdir()
    canary = external / "keep.txt"
    canary.write_text("Synthetic external canary", encoding="utf-8")
    with closing(sqlite3.connect(root / "library.sqlite3")) as connection:
        connection.execute("UPDATE pending_operations SET phase='prepared'")
        connection.execute(
            "INSERT INTO pending_operations VALUES (?,?,?, ?,?,?,?)",
            (
                reverted_id,
                "publish_revision",
                "prepared",
                OTHER_MEETING_ID,
                f".staging/{reverted_id}",
                f"meetings/{OTHER_MEETING_ID}/revisions/{revision_id}",
                json.dumps({"revision_id": revision_id, "hashes": {"transcript.txt": "0" * 64}}),
            ),
        )
        connection.execute(
            "INSERT INTO pending_operations VALUES (?,?,?, ?,?,?,?)",
            (
                unsafe_id,
                "publish_revision",
                "prepared",
                OTHER_MEETING_ID,
                "../external",
                str(external),
                "{}",
            ),
        )
        connection.commit()
    report = memory.reconcile()
    assert _rows(root, "SELECT phase FROM pending_operations WHERE id=?", (complete_id,)) == [
        ("complete",)
    ]
    assert _rows(root, "SELECT phase FROM pending_operations WHERE id=?", (reverted_id,)) == [
        ("reverted",)
    ]
    assert _rows(root, "SELECT phase FROM pending_operations WHERE id=?", (unsafe_id,)) == [
        ("prepared",)
    ]
    assert not staging.exists()
    assert canary.read_text(encoding="utf-8") == "Synthetic external canary"
    assert memory.show(MEETING_ID)["revisions"][0]["raw_text"] == TRANSCRIPT_TEXT
    _outcome(report, "completed", complete_id)
    _outcome(report, "reverted", reverted_id)
    _outcome(report, "skipped", unsafe_id)
    assert report.warnings
    second = memory.reconcile()
    assert not second.completed and not second.reverted
    _outcome(second, "skipped", unsafe_id)


@pytest.mark.parametrize(
    "invalid",
    [
        "mismatched-identity",
        "stale-revision",
        "nonstring-intent",
        "nonstring-publication",
        "nonstring-source-kind",
    ],
)
def test_pending_marker_rejects_extra_authority_while_valid_sibling_replays(
    tmp_path,
    monkeypatch,
    invalid,
):
    from here.application.recovery import RecoveryService

    root = tmp_path / "data"
    monkeypatch.setenv("HERE_DATA_DIR", str(root))
    sessions = tmp_path / "sessions"
    bad = make_session(sessions / "a-invalid", meeting_id=MEETING_ID)
    good = make_session(sessions / "z-valid", meeting_id=OTHER_MEETING_ID)
    memory = MemoryService(root)
    assert memory.ingest_session(bad.directory).status == "inserted"
    old = memory.active_revision(MEETING_ID)
    assert memory.ingest_session(bad.directory, revision_of=old).status == "revised"
    before = memory.show(MEETING_ID)
    envelope = {
        "schema_version": 2,
        "meeting_id": MEETING_ID,
        "revision_of": before["meeting"]["active_revision_id"],
        "intent": "ingest",
        "publication_id": str(uuid4()),
        "source_kind": "session",
    }
    if invalid == "mismatched-identity":
        envelope["meeting_id"] = OTHER_MEETING_ID
    elif invalid == "stale-revision":
        envelope["revision_of"] = old
    else:
        field = {
            "nonstring-intent": "intent",
            "nonstring-publication": "publication_id",
            "nonstring-source-kind": "source_kind",
        }[invalid]
        envelope[field] = 42
    marker = bad.directory / "memory_pending.json"
    write_json(marker, envelope)
    write_json(
        good.directory / "memory_pending.json",
        {
            "schema_version": 1,
            "meeting_id": OTHER_MEETING_ID,
            "revision_of": None,
        },
    )
    marker_before = json.loads(marker.read_text(encoding="utf-8"))
    recovery = RecoveryService(sessions)
    candidates = recovery.discover()
    recovery.discover()
    # Retain the exact authority, not incidental JSON whitespace from a retry.
    assert json.loads(marker.read_text(encoding="utf-8")) == marker_before
    assert not (good.directory / "memory_pending.json").exists()
    assert memory.show(MEETING_ID) == before
    assert memory.show(OTHER_MEETING_ID)["revisions"][0]["raw_text"] == TRANSCRIPT_TEXT
    (candidate,) = [item for item in candidates if item.session_dir == bad.directory]
    assert not candidate.can_retry and candidate.error_summary


def test_r48_get_meeting_uses_one_coherent_snapshot_across_external_sqlite_commit(
    tmp_path, monkeypatch
):
    from here.memory.database import LibraryConnection
    from here.memory.models import MeetingPatch

    root = tmp_path / "data"
    memory = MemoryService(root)
    source = make_session(tmp_path / "r48-source", meeting_id=MEETING_ID)
    assert memory.ingest_session(source.directory).status == "inserted"
    memory.update_meeting(
        MEETING_ID,
        MeetingPatch(title="Before coherent read"),
        expected_version=memory.show(MEETING_ID)["meeting"]["row_version"],
    )
    person = memory.add_participant("Before participant")
    memory.link_participant(MEETING_ID, person.id, role="before role")
    shown = memory.show(MEETING_ID)
    revision = shown["meeting"]["active_revision_id"]
    speaker_key = shown["segments"][0]["speaker_key"]
    memory.assign_speaker(MEETING_ID, speaker_key, person.id, scope=revision)
    attempted = []
    committed = []

    def external_commit():
        with closing(sqlite3.connect(root / "library.sqlite3", timeout=0)) as writer:
            writer.execute("BEGIN IMMEDIATE")
            writer.execute(
                "UPDATE meetings SET title='After coherent read',row_version=row_version+1 "
                "WHERE id=?",
                (MEETING_ID,),
            )
            writer.execute(
                "UPDATE participants SET display_name='After participant' WHERE id=?", (person.id,)
            )
            writer.execute(
                "UPDATE meeting_participants SET role='after role' WHERE meeting_id=?",
                (MEETING_ID,),
            )
            writer.execute(
                "UPDATE speaker_assignments SET participant_id=NULL WHERE meeting_id=?",
                (MEETING_ID,),
            )
            writer.execute(
                "UPDATE audio_assets SET status='corrupt' WHERE meeting_id=?", (MEETING_ID,)
            )
            writer.commit()
        committed.append(True)

    execute = LibraryConnection.execute

    def interpose_read(library, sql, parameters=()):
        cursor = execute(library, sql, parameters)
        statement = " ".join(sql.lower().split())
        domain_tables = (
            "meetings",
            "participants",
            "meeting_participants",
            "speaker_assignments",
            "audio_assets",
            "transcript_revisions",
            "segments",
            "revision_audio_assets",
        )
        if (
            not attempted
            and statement.startswith("select ")
            and any(
                f"from {table}" in statement or f"join {table}" in statement
                for table in domain_tables
            )
        ):
            attempted.append(True)
            try:
                external_commit()
            except sqlite3.OperationalError as error:
                # BEGIN IMMEDIATE is also coherent: it may serialize the writer.
                # Do not require a nonblocking implementation the contract never promised.
                assert "locked" in str(error).lower(), str(error)
        return cursor

    with monkeypatch.context() as patch:
        patch.setattr(LibraryConnection, "execute", interpose_read)
        detail = memory.get_meeting(MEETING_ID)
    assert attempted, "External SQL fault must be interposed at an actual domain read"
    if not committed:
        external_commit()
    assert _rows(root, "SELECT title FROM meetings") == [("After coherent read",)]
    assert (
        len(detail.participants)
        == len(detail.participant_links)
        == len(detail.speaker_assignments)
        == 1
    )
    observed = (
        detail.meeting.title,
        detail.participants[0].display_name,
        detail.participant_links[0].role,
        detail.speaker_assignments[0].participant_id,
        detail.audio_assets[0].status,
    )
    assert observed in {
        ("Before coherent read", "Before participant", "before role", person.id, "available"),
        ("After coherent read", "After participant", "after role", None, "corrupt"),
    }, f"get_meeting returned a torn catalog snapshot: {observed}"


@pytest.mark.parametrize("injected", [False, True], ids=["default-root", "injected-custom-root"])
def test_r48_controller_exposes_read_only_actual_memory_service_and_real_catalog(
    tmp_path,
    monkeypatch,
    injected,
):
    from here.application.controller import HereApplicationController
    from here.application.models import ApplicationState

    default_root = tmp_path / "default-root"
    custom_root = tmp_path / "custom-root"
    monkeypatch.setenv("HERE_DATA_DIR", str(default_root))

    def forbidden(*args, **kwargs):
        pytest.fail("Reading controller memory_service started capture or live processing")

    supplied = MemoryService(custom_root) if injected else None
    controller = HereApplicationController(
        memory_service=supplied, capture_factory=forbidden, live_factory=forbidden
    )
    exposed = controller.memory_service
    assert isinstance(exposed, MemoryService)
    if injected:
        assert exposed is supplied
    assert controller.memory_service is exposed
    source = make_session(tmp_path / "r48-source", meeting_id=MEETING_ID)
    assert exposed.ingest_session(source.directory).status == "inserted"
    expected_root = custom_root if injected else default_root
    assert exposed.data_dir == expected_root.absolute()
    assert _rows(expected_root, "SELECT id FROM meetings") == [(MEETING_ID,)]
    assert exposed.get_meeting(MEETING_ID).meeting.id == MEETING_ID
    person = exposed.add_participant("Controller-visible explicit person")
    exposed.link_participant(MEETING_ID, person.id, role="observer")
    assert exposed.list_participants(MEETING_ID) == (person,)
    assert controller.snapshot.state == ApplicationState.IDLE
    assert not controller.snapshot.has_active_work
    with pytest.raises(AttributeError):
        controller.memory_service = exposed
    assert controller.memory_service is exposed
    if injected:
        assert not (default_root / "library.sqlite3").exists()
