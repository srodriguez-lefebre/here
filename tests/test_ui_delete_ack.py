"""Deliver real pre-delete DTOs after B accepts a durable deletion."""

from __future__ import annotations

import json
from threading import Event, get_ident

import pytest
from here.application import HereApplicationController
from here.memory.models import DeletionResult, MeetingQuery, SearchQuery
from here.memory.service import MemoryService, NotFound
from here.ui.app import HereDesktop
from here.ui.contract import ApplicationUiAdapter
from here.ui.library import LibraryView
from memory_fixtures import MEETING_ID, SEGMENT_TEXTS, make_session
from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QTabWidget
from test_ui_library import _select_first_meeting, _text

pytestmark = pytest.mark.usefixtures("owned_desktops")


@pytest.mark.parametrize("delayed_read", ["page", "detail"])
def test_accepted_delete_prevents_late_library_response_from_resurfacing(
    tmp_path, monkeypatch, qapp, qtbot, delayed_read
):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    memory = MemoryService(tmp_path / "catalog", sessions_root=sessions)
    source = make_session(tmp_path / "mercurycanary", meeting_id=MEETING_ID)
    publication = memory.ingest_session(source.directory)
    assert publication.status == "inserted" and publication.meeting_id == MEETING_ID
    core = HereApplicationController(memory_service=memory)
    assert core.memory_service is memory
    adapter = ApplicationUiAdapter(core, sessions)
    before = adapter.get_meeting(MEETING_ID)
    title = before.meeting.title
    assert title and before.segments[0].text == SEGMENT_TEXTS[0]
    query = SearchQuery(text=title)
    before_search = adapter.search(query)
    assert not before_search.degraded
    assert {hit.meeting_id for hit in before_search.items} == {MEETING_ID}

    # This unowned file leaves an accepted B deletion pending before physical purge.
    unrelated = memory.data_dir / "meetings" / MEETING_ID / "unknown.txt"
    unrelated.write_bytes(b"retain unowned synthetic file")
    desktop = HereDesktop(
        qapp,
        adapter,
        settings=QSettings(str(tmp_path / "delete-ui.ini"), QSettings.Format.IniFormat),
    )
    desktop.show()
    qtbot.waitUntil(lambda: not desktop.jobs.busy)
    tabs = desktop.main_window.findChild(QTabWidget)
    view = desktop.main_window.findChild(LibraryView)
    assert tabs is not None and view is not None
    tab_index = tabs.indexOf(view)
    assert tab_index >= 0

    operation_name = "list_meetings" if delayed_read == "page" else "get_meeting"
    original_read = getattr(adapter, operation_name)
    obtained = Event()
    deliver = Event()
    captured = []
    qt_thread = get_ident()

    def hold_one_response(argument):
        result = original_read(argument)
        if not obtained.is_set():
            assert get_ident() != qt_thread
            captured.append(result)
            obtained.set()
            assert deliver.wait(20), "Timed out waiting to deliver the pre-delete DTO"
        return result

    monkeypatch.setattr(adapter, operation_name, hold_one_response)
    deliveries = []
    desktop.jobs.finished.connect(lambda *result: deliveries.append(result))
    try:
        tabs.setCurrentIndex(tab_index)
        if delayed_read == "detail":
            qtbot.waitUntil(lambda: title in _text(view))
            qtbot.waitUntil(lambda: not desktop.jobs.busy)
            _select_first_meeting(view, qtbot)
        qtbot.waitUntil(obtained.is_set)
        stale = captured[0]
        if delayed_read == "page":
            assert [item.meeting.id for item in stale.items] == [MEETING_ID]
            assert title not in _text(view)
        else:
            assert stale.meeting.id == MEETING_ID
            assert stale.segments[0].text == SEGMENT_TEXTS[0]
            assert SEGMENT_TEXTS[0] not in _text(view)

        # Let B invoke its actual registered callbacks. The harness adds none.
        deletion_request = desktop.jobs.submit(
            "test-delete", lambda: memory.delete_meeting(MEETING_ID)
        )
        assert deletion_request is not None
        qtbot.waitUntil(lambda: any(item[0] == deletion_request for item in deliveries))
        _, _, deleted, error = next(item for item in deliveries if item[0] == deletion_request)
        assert error is None
        assert isinstance(deleted, DeletionResult)
        assert (deleted.meeting_id, deleted.status) == (MEETING_ID, "pending")
        registry = memory.data_dir / ".deletions" / f"{MEETING_ID}.json"
        assert json.loads(registry.read_text(encoding="utf-8"))["phase"] == "pending"
        assert unrelated.read_bytes() == b"retain unowned synthetic file"

        fresh_adapter = ApplicationUiAdapter(core, sessions)
        assert not fresh_adapter.list_meetings(MeetingQuery()).items
        assert not fresh_adapter.search(query).items
        with pytest.raises(NotFound):
            fresh_adapter.get_meeting(MEETING_ID)
    finally:
        deliver.set()
        qtbot.waitUntil(lambda: not desktop.jobs.busy)

    assert any(value is stale and error is None for _, _, value, error in deliveries)
    rendered = _text(view)
    assert SEGMENT_TEXTS[0] not in rendered
    assert title not in rendered
