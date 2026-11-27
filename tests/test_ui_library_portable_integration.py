"""Qt portability uses the services bound to the configured meeting catalog."""

from __future__ import annotations

from pathlib import Path

import pytest
from here.memory.models import MeetingPatch
from here.memory.portability import PortabilityService
from here.memory.service import MemoryService, NotFound
from here.ui.app import HereDesktop
from here.ui.contract import ApplicationUiAdapter
from here.ui.fake_controller import FakeApplicationController
from memory_fixtures import MEETING_ID, OTHER_MEETING_ID, make_session
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import QApplication
from pytestqt.qtbot import QtBot
from test_ui_library_export import action, feedback, picker, show_library, table

_IMPORTED_TITLE = "Reunión transferida"
_IMPORTED_TEXT = "La biblioteca de destino recibe esta evidencia"
_SOURCE_ONLY_TITLE = "Título cambiado en la biblioteca de origen"
_RESIDENT_TITLE = "Reunión ya guardada"
_RESIDENT_TEXT = "Esta evidencia pertenece a la otra reunión"


def test_qt_import_and_markdown_use_configured_catalog(
    qapp: QApplication,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    owned_desktops: None,
) -> None:
    source_sessions = tmp_path / "source-sessions"
    source_fixture = make_session(
        source_sessions / _IMPORTED_TITLE,
        meeting_id=MEETING_ID,
        transcript_text=_IMPORTED_TEXT,
        segments=[{"text": _IMPORTED_TEXT}],
    )
    source = MemoryService(tmp_path / "source-catalog", sessions_root=source_sessions)
    assert source.ingest_session(source_fixture.directory).status == "inserted"
    package = tmp_path / "transfer.here.zip"
    PortabilityService(source).export(
        (MEETING_ID,),
        package,
        include_audio=False,
        progress=lambda _: None,
        cancellation=lambda: False,
    )
    source_meeting = source.get_meeting(MEETING_ID).meeting
    source.update_meeting(
        MEETING_ID, MeetingPatch(title=_SOURCE_ONLY_TITLE), source_meeting.row_version
    )

    configured_sessions = tmp_path / "configured-sessions"
    resident = make_session(
        configured_sessions / _RESIDENT_TITLE,
        meeting_id=OTHER_MEETING_ID,
        transcript_text=_RESIDENT_TEXT,
        segments=[{"text": _RESIDENT_TEXT}],
    )
    configured_root = tmp_path / "configured-catalog"
    configured = MemoryService(configured_root, sessions_root=configured_sessions)
    assert configured.ingest_session(resident.directory).status == "inserted"
    with pytest.raises(NotFound):
        configured.get_meeting(MEETING_ID)

    core = FakeApplicationController()
    controller = ApplicationUiAdapter(core, configured_sessions, memory_service=configured)
    settings = QSettings(str(tmp_path / "desktop.ini"), QSettings.Format.IniFormat)
    desktop = HereDesktop(qapp, controller, settings=settings)
    view = show_library(desktop, qtbot)
    rows = table(view)
    assert rows.rowCount() == 1
    assert rows.item(0, 0).text() == _RESIDENT_TITLE

    picker(monkeypatch, "getOpenFileName", package)
    action(view, r"importar paquete").click()
    qtbot.waitUntil(lambda: rows.rowCount() == 2 and not desktop.jobs.busy, timeout=15000)
    assert f"{MEETING_ID} · Importada" in feedback(view)
    assert configured.get_meeting(MEETING_ID).meeting.title == _IMPORTED_TITLE
    with pytest.raises(NotFound):
        source.get_meeting(OTHER_MEETING_ID)

    (row,) = (i for i in range(rows.rowCount()) if rows.item(i, 0).text() == _IMPORTED_TITLE)
    item = rows.item(row, 0)
    rows.scrollToItem(item)
    position = rows.visualItemRect(item).center()
    qtbot.mouseClick(rows.viewport(), Qt.MouseButton.LeftButton, pos=position)
    qtbot.mouseDClick(rows.viewport(), Qt.MouseButton.LeftButton, pos=position)
    qtbot.mouseRelease(rows.viewport(), Qt.MouseButton.LeftButton, pos=position)
    qtbot.waitUntil(lambda: _IMPORTED_TEXT in feedback(view) and not desktop.jobs.busy)

    markdown = action(view, r"exportar markdown")
    refused = configured_root / "inside-catalog.md"
    picker(monkeypatch, "getSaveFileName", refused)
    markdown.click()
    qtbot.waitUntil(
        lambda: (
            "No se pudo exportar el documento Markdown." in feedback(view) and not desktop.jobs.busy
        )
    )
    assert not refused.exists()

    destination = tmp_path / "imported.md"
    picker(monkeypatch, "getSaveFileName", destination)
    markdown.click()
    qtbot.waitUntil(
        lambda: "Exportación Markdown finalizada." in feedback(view) and not desktop.jobs.busy
    )
    document = destination.read_bytes().decode("utf-8")
    assert _IMPORTED_TITLE in document
    assert _SOURCE_ONLY_TITLE not in document
    assert MEETING_ID in document
    assert _IMPORTED_TEXT in document
    assert OTHER_MEETING_ID not in document
    assert _RESIDENT_TEXT not in document
    assert core.command_log == []
