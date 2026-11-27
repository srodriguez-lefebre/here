from __future__ import annotations

import json
from datetime import datetime, timedelta

import here.cli as cli
import here.memory.search as search_module
import pytest
from here.application import ApplicationState, HereApplicationController
from here.application.models import ApplicationSnapshot
from here.application.processing import SessionProcessor
from here.application.recovery import RecoveryService
from here.memory.database import Busy, open_library
from here.memory.models import MeetingQuery, SearchQuery
from here.memory.search import SearchService
from here.memory.service import Conflict, InvalidInput, MemoryService
from here.ui import playback
from memory_fixtures import MEETING_ID, OTHER_MEETING_ID, make_session, write_json
from PySide6.QtCore import QObject
from typer.testing import CliRunner

MISSING_ID = "99999999-9999-4999-8999-999999999999"
AMBIGUOUS_ID = "11111111-2222-4222-8222-222222222222"
runner = CliRunner()


def invoke(*arguments, exit_code=0):
    result = runner.invoke(cli.app, ["meetings", *arguments, "--json"])
    assert result.exit_code == exit_code, result.output + repr(result.exception)
    return json.loads(result.stdout)


@pytest.fixture
def library(tmp_path, monkeypatch):
    root = tmp_path / "data"
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    monkeypatch.setenv("HERE_DATA_DIR", str(root))
    monkeypatch.setenv("HERE_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.setenv("TRANSCRIPTIONS_DIR", str(sessions))
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")

    def forbidden(*args, **kwargs):
        pytest.fail("CLI invoked provider or network work")

    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr("here.transcription.client.build_client", forbidden)
    memory = MemoryService(root, sessions_root=sessions)
    for identity in (MEETING_ID, OTHER_MEETING_ID):
        source = make_session(tmp_path / identity, meeting_id=identity)
        assert memory.ingest_session(source.directory).status == "inserted"
    return memory


def test_help_registers_meetings_and_preserves_memory():
    result = runner.invoke(cli.app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "meetings" in result.output
    assert "memory" in result.output


def test_list_show_and_search_preserve_order_full_identity_and_utc(library):
    listing = invoke(
        "list", "--from", "2026-01-01T00:00:00+00:00", "--to", "2026-01-02T00:00:00+00:00"
    )
    assert [item["meeting"]["id"] for item in listing["items"]] == [
        MEETING_ID,
        OTHER_MEETING_ID,
    ]
    shown = invoke("show", MEETING_ID[:8])
    assert shown["meeting"]["id"] == MEETING_ID
    started_at = datetime.fromisoformat(shown["meeting"]["started_at"])
    assert started_at.utcoffset() == timedelta(0)
    assert started_at == datetime.fromisoformat("2026-01-01T12:00:00+00:00")
    ranked = invoke("search", "canary", "--meeting", OTHER_MEETING_ID[:8])
    assert len(ranked["items"]) == 2
    assert {item["meeting_id"] for item in ranked["items"]} == {OTHER_MEETING_ID}
    empty = invoke("search", "  ", "--meeting", OTHER_MEETING_ID[:8])
    assert [item["meeting"]["id"] for item in empty["items"]] == [OTHER_MEETING_ID]


@pytest.mark.parametrize("route", ["list", "empty", "ranked", "literal"])
def test_meeting_scope_precedes_limit_and_is_bound_to_cursor(library, monkeypatch, route):
    if route == "literal":
        monkeypatch.setattr(search_module, "fts_available", lambda connection: False)
    service = SearchService(lambda: open_library(library.data_dir))
    if route == "list":
        query, fetch = MeetingQuery, service.list_meetings
        options = {}
    else:
        query, fetch = SearchQuery, service.search
        options = {"text": " " if route == "empty" else "canary"}

    page = fetch(query(**options, limit=1))
    assert page.next_cursor
    scoped = fetch(query(**options, limit=1, meeting_id=OTHER_MEETING_ID))
    assert len(scoped.items) == 1
    item = scoped.items[0]
    identity = item.meeting_id if route in {"ranked", "literal"} else item.meeting.id
    assert identity == OTHER_MEETING_ID
    with pytest.raises(InvalidInput):
        fetch(query(**options, limit=1, meeting_id=OTHER_MEETING_ID, cursor=page.next_cursor))
    if route in {"ranked", "literal"}:
        assert scoped.next_cursor
        with pytest.raises(InvalidInput):
            fetch(query(**options, limit=1, meeting_id=MEETING_ID, cursor=scoped.next_cursor))


def test_rename_and_participant_use_catalog_versions_and_speaker(library):
    before = library.show(MEETING_ID)["meeting"]
    result = invoke("rename", MEETING_ID[:8], "Agenda sintética")
    assert result["meeting"]["row_version"] == before["row_version"] + 1
    assert library.show(MEETING_ID)["meeting"]["title"] == "Agenda sintética"
    speaker = library.show(MEETING_ID)["segments"][0]["speaker_key"]
    invoke("participant", MEETING_ID[:8], "Persona sintética", "--speaker", speaker)
    detail = library.show(MEETING_ID)
    participant = next(
        item for item in detail["participants"] if item["display_name"] == "Persona sintética"
    )
    assert detail["speaker_assignments"][0]["participant_id"] == participant["id"]
    filtered = invoke("list", "--status", "completed", "--participant", participant["id"])
    assert [item["meeting"]["id"] for item in filtered["items"]] == [MEETING_ID]


def test_import_and_reindex_use_real_catalog(library, tmp_path):
    source = make_session(tmp_path / "import", meeting_id=AMBIGUOUS_ID)
    imported = invoke("import-session", str(source.directory))
    assert imported["status"] == "inserted"
    assert imported["meeting_id"] == AMBIGUOUS_ID
    indexed = invoke("reindex")
    assert indexed["indexed_meetings"] == 3 and indexed["indexed_segments"] == 6


def test_ambiguous_abbreviation_returns_full_candidates(library, tmp_path):
    source = make_session(tmp_path / "ambiguous", meeting_id=AMBIGUOUS_ID)
    assert library.ingest_session(source.directory).status == "inserted"
    result = invoke("show", MEETING_ID[:8], exit_code=2)
    assert result["error_category"] == "invalid_input"
    assert set(result["candidates"]) == {MEETING_ID, AMBIGUOUS_ID}


def test_delete_requires_confirmation_and_refreshes_scoped_search(library):
    original = library.show(MEETING_ID)
    result = runner.invoke(cli.app, ["meetings", "delete", MEETING_ID])
    assert result.exit_code == 2, result.output
    assert library.show(MEETING_ID) == original
    assert not (library.data_dir / ".deletions" / f"{MEETING_ID}.json").exists()
    deleted = invoke("delete", MEETING_ID, "--yes")
    assert deleted["status"] == "completed" and deleted["meeting_id"] == MEETING_ID
    assert library.show(MEETING_ID) is None
    invoke("search", "canary", "--meeting", MEETING_ID, exit_code=3)


def test_delete_reports_durable_pending_as_io(library):
    unknown = library.data_dir / "meetings" / MEETING_ID / "keep-unknown.txt"
    unknown.write_text("unregistered synthetic material", encoding="utf-8")
    result = invoke("delete", MEETING_ID, "--yes", exit_code=4)
    assert result["status"] == "pending" and result["meeting_id"] == MEETING_ID
    assert result["reason"] and result["error_category"] == "io"
    assert unknown.read_text(encoding="utf-8") == "unregistered synthetic material"


@pytest.mark.parametrize(
    "error,code,category",
    [(Busy("busy"), 4, "io"), (OSError("io"), 4, "io"), (Conflict("stale"), 3, "conflict")],
)
def test_rename_preserves_canonical_error_categories(library, monkeypatch, error, code, category):
    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(MemoryService, "update_meeting", fail)
    result = invoke("rename", MEETING_ID, "Cambio", exit_code=code)
    assert result["error_category"] == category


def test_search_preserves_structured_corpus_changed_reason(library, monkeypatch):
    def stale(service, query):
        raise InvalidInput("Synthetic message", reason="corpus_changed")

    monkeypatch.setattr(SearchService, "search", stale)
    result = invoke("search", "canary", exit_code=2)
    assert result["error_category"] == "invalid_input"
    assert result["reason"] == "corpus_changed"


def test_play_resolves_authoritative_target_before_multimedia_failure(library, monkeypatch, qapp):
    original = library.show(MEETING_ID)
    segment = original["segments"][0]
    resolved = []
    unavailable_checks = []
    resolve = MemoryService.resolve_citation

    def track(service, citation):
        target = resolve(service, citation)
        resolved.append(citation)
        return target

    class UnavailablePlayer(QObject):
        def isAvailable(self):
            assert len(resolved) == 1
            assert resolved[0].segment_id == segment["id"]
            unavailable_checks.append(True)
            return False

    def forbidden_audio_output(*args, **kwargs):
        pytest.fail("Unavailable multimedia must not allocate an audio output")

    monkeypatch.setattr(MemoryService, "resolve_citation", track)
    monkeypatch.setattr(playback, "QMediaPlayer", UnavailablePlayer)
    monkeypatch.setattr(playback, "QAudioOutput", forbidden_audio_output)

    result = runner.invoke(
        cli.app, ["meetings", "play", MEETING_ID[:8], "--segment", segment["id"], "--json"]
    )
    assert result.exit_code == 4, result.output + repr(result.exception)
    assert unavailable_checks == [True]
    assert "multimedia" in result.stderr.lower()
    target = json.loads(result.stdout)
    assert target["meeting_id"] == MEETING_ID
    assert target["revision_id"] == segment["revision_id"]
    assert target["segment_id"] == segment["id"]
    assert target["start_ms"] == 0


@pytest.mark.parametrize("identity", [MEETING_ID, MISSING_ID])
def test_retry_without_eligible_candidate_never_runs_recovery_discovery(
    library, monkeypatch, identity
):
    def forbidden(*args, **kwargs):
        pytest.fail("Retry ran broad discovery or launched the application")

    monkeypatch.setattr(RecoveryService, "discover", forbidden)
    monkeypatch.setattr(cli, "create_default_controller", forbidden)
    monkeypatch.setenv("OPENAI_API_KEY", "")
    result = invoke("retry", identity, exit_code=3)
    assert result["error_category"] == "conflict"


@pytest.fixture
def retry_source(library, tmp_path, monkeypatch):
    source = make_session(tmp_path / "sessions" / "failed", meeting_id=AMBIGUOUS_ID)
    metadata = json.loads(source.metadata.read_text(encoding="utf-8"))
    metadata.update(status="failed", failure_stage="transcription")
    write_json(source.metadata, metadata)
    assert library.ingest_session(source.directory).status == "inserted"

    def forbidden(*args, **kwargs):
        pytest.fail("Targeted retry called mutating discovery")

    monkeypatch.setattr(RecoveryService, "discover", forbidden)
    return source


def test_retry_only_launches_selected_materialized_session_and_waits(retry_source, monkeypatch):
    calls = []
    materialize = RecoveryService.materialize

    def track(recovery, candidate):
        calls.append(("materialize", candidate.session_dir))
        return materialize(recovery, candidate)

    def retry(controller, session_dir=None):
        calls.append(("retry", session_dir))

    def wait(controller, timeout=None):
        calls.append(("wait", retry_source.directory))
        return ApplicationSnapshot(
            state=ApplicationState.COMPLETED, session_dir=retry_source.directory
        )

    monkeypatch.setattr(RecoveryService, "materialize", track)
    monkeypatch.setattr(HereApplicationController, "retry", retry)
    monkeypatch.setattr(HereApplicationController, "wait_until_terminal", wait)
    invoke("retry", retry_source.meeting_id)
    assert calls == [
        ("materialize", retry_source.directory),
        ("retry", retry_source.directory),
        ("wait", retry_source.directory),
    ]


def test_retry_reports_missing_provider_from_eligible_application_path(retry_source, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "")
    result = invoke("retry", retry_source.meeting_id, exit_code=5)
    assert result["error_category"] == "provider"


def test_retry_keeps_terminal_storage_failure_distinct_from_provider(retry_source, monkeypatch):
    def fail(processor, session_dir, *, cancel_event=None):
        assert session_dir == retry_source.directory
        raise Busy("Synthetic storage busy")

    monkeypatch.setattr(SessionProcessor, "retry", fail)
    result = invoke("retry", retry_source.meeting_id, exit_code=4)
    assert result["error_category"] == "io"
