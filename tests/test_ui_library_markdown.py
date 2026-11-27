"""Single-meeting Markdown export through the real desktop and library detail."""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

import pytest
from here.memory.portability import ExportCancelled
from here.memory.service import NotFound
from here.ui.app import HereDesktop
from here.ui.contract import ApplicationUiAdapter
from here.ui.fake_controller import FakeApplicationController, FakeLibraryBackend
from here.ui.library import LibraryView
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (
    QAbstractButton,
    QApplication,
    QFileDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QTableWidget,
    QTabWidget,
    QWidget,
)
from pytestqt.qtbot import QtBot

Outcome = Literal["saved", "saved_warning", "cancelled", "unavailable", "write_error"]
MARKDOWN_ACTION = r"markdown|\.md\b"
FRESH_TRANSCRIPT = "La revisión vigente contiene una síntesis nueva."
FRESH_TITLE = "Reunión de revisión"
CLEANUP_WARNING = "Limpieza pendiente tras guardar el documento."


class MarkdownController(ApplicationUiAdapter):
    def __init__(self, root: Path) -> None:
        self.library = FakeLibraryBackend()
        self.core = FakeApplicationController()
        self.calls: list[tuple[str, Path, int]] = []
        self.entered = threading.Event()
        self.release = threading.Event()
        self.outcome: Outcome = "saved"
        self.cancellation_seen: bool | None = None
        super().__init__(
            self.core,
            root,
            memory_service=self.library,
            search_service=self.library,
            portability_service=self.library,
        )

    # Proposed UI seam. The fake supplies the worker outcome, not the cached detail.
    def export_meeting_markdown(
        self, meeting_id: str, destination: Path, cancellation: Callable[[], bool]
    ) -> SimpleNamespace:
        self.calls.append((meeting_id, destination, threading.get_ident()))
        self.entered.set()
        assert self.release.wait(5), "Test did not release the export worker"
        self.cancellation_seen = cancellation()
        if self.cancellation_seen:
            raise ExportCancelled("Synthetic cancellation")
        if self.outcome == "unavailable":
            raise NotFound("Synthetic meeting is no longer available")
        if self.outcome == "write_error":
            raise OSError("Synthetic destination write failed")
        if self.outcome == "cancelled":
            raise AssertionError("The cancellation did not reach the export worker")
        destination.write_text(f"# {FRESH_TITLE}\n\n{FRESH_TRANSCRIPT}\n", encoding="utf-8")
        warnings = (CLEANUP_WARNING,) if self.outcome == "saved_warning" else ()
        return SimpleNamespace(path=destination, warnings=warnings)


@pytest.fixture
def library(
    qapp: QApplication, qtbot: QtBot, tmp_path: Path, owned_desktops: None
) -> tuple[HereDesktop, MarkdownController, LibraryView]:
    controller = MarkdownController(tmp_path / "sessions")
    settings = QSettings(str(tmp_path / "desktop.ini"), QSettings.Format.IniFormat)
    desktop = HereDesktop(qapp, controller, settings=settings)
    desktop.show()
    tabs = desktop.main_window.findChild(QTabWidget)
    view = desktop.main_window.findChild(LibraryView)
    assert tabs is not None and view is not None
    tabs.setCurrentIndex(next(i for i in range(tabs.count()) if tabs.tabText(i) == "Biblioteca"))
    qtbot.waitUntil(lambda: results(view).rowCount() == 1 and not desktop.jobs.busy)
    return desktop, controller, view


def results(view: LibraryView) -> QTableWidget:
    [table] = view.findChildren(QTableWidget)
    return table


def buttons(view: LibraryView, pattern: str) -> list[QAbstractButton]:
    return [
        button
        for button in view.findChildren(QAbstractButton)
        if button.isVisible()
        and re.search(pattern, f"{button.text()} {button.accessibleName()}", re.I)
    ]


def markdown_action(view: LibraryView) -> QAbstractButton:
    matches = buttons(view, MARKDOWN_ACTION)
    assert len(matches) == 1, "Expected a dedicated Markdown export action in the meeting detail"
    assert matches[0].isEnabled()
    return matches[0]


def detail_state(view: LibraryView) -> tuple[tuple[object, ...], str, tuple[str, ...]]:
    [title] = [
        editor
        for editor in view.findChildren(QLineEdit)
        if editor.isVisible() and re.search(r"título", editor.accessibleName(), re.I)
    ]
    [transcript] = [
        items
        for items in view.findChildren(QListWidget)
        if items.isVisible() and re.search(r"transcripci", items.accessibleName(), re.I)
    ]
    selected = tuple(index.data() for index in results(view).selectionModel().selectedRows())
    return (
        selected,
        title.text(),
        tuple(transcript.item(row).text() for row in range(transcript.count())),
    )


def open_detail(view: LibraryView, qtbot: QtBot, desktop: HereDesktop) -> None:
    table = results(view)
    item = table.item(0, 0)
    table.scrollToItem(item)
    position = table.visualItemRect(item).center()
    qtbot.mouseClick(table.viewport(), Qt.MouseButton.LeftButton, pos=position)
    qtbot.mouseDClick(table.viewport(), Qt.MouseButton.LeftButton, pos=position)
    qtbot.mouseRelease(table.viewport(), Qt.MouseButton.LeftButton, pos=position)
    qtbot.waitUntil(lambda: bool(buttons(view, r"cerrar.*detalle")) and not desktop.jobs.busy)


def install_picker(
    monkeypatch: pytest.MonkeyPatch, destination: Path | None
) -> list[tuple[str, int]]:
    calls: list[tuple[str, int]] = []

    def get_save_file_name(
        parent: QWidget | None = None,
        caption: str = "",
        dir: str = "",
        filter: str = "",
        selectedFilter: str = "",
        options: QFileDialog.Option = QFileDialog.Option(0),
    ) -> tuple[str, str]:
        calls.append((filter, threading.get_ident()))
        return (str(destination), filter) if destination is not None else ("", "")

    monkeypatch.setattr(QFileDialog, "getSaveFileName", staticmethod(get_save_file_name))
    return calls


@pytest.fixture
def notices(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    messages: list[str] = []

    def show(
        parent: QWidget | None, title: str, text: str, *args: object, **kwargs: object
    ) -> QMessageBox.StandardButton:
        messages.append(text)
        return QMessageBox.StandardButton.Ok

    for method in ("information", "warning", "critical", "question"):
        monkeypatch.setattr(QMessageBox, method, staticmethod(show))
    return messages


def test_markdown_picker_cancel_keeps_open_detail_idle(
    library, qtbot: QtBot, monkeypatch: pytest.MonkeyPatch, tmp_path: Path, notices: list[str]
) -> None:
    desktop, controller, view = library
    assert buttons(view, r"export.*selecci")
    assert not any(button.isEnabled() for button in buttons(view, MARKDOWN_ACTION))
    open_detail(view, qtbot, desktop)
    action = markdown_action(view)
    before = detail_state(view)
    assert before[0]
    paths_before = set(tmp_path.rglob("*.md"))
    picks = install_picker(monkeypatch, None)

    action.click()

    [(file_filter, picker_thread)] = picks
    assert "*.md" in file_filter.lower()
    assert picker_thread == threading.get_ident()
    assert controller.calls == []
    assert not desktop.jobs.busy
    assert set(tmp_path.rglob("*.md")) == paths_before
    assert detail_state(view) == before
    assert action.isEnabled()
    assert not notices
    assert controller.core.command_log == []
    [close] = buttons(view, r"cerrar.*detalle")
    close.click()
    assert not any(button.isEnabled() for button in buttons(view, MARKDOWN_ACTION))


@pytest.mark.parametrize(
    "outcome", ["saved", "saved_warning", "cancelled", "unavailable", "write_error"]
)
def test_markdown_export_delegates_current_identity_and_preserves_detail(
    library,
    qtbot: QtBot,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    notices: list[str],
    outcome: Outcome,
) -> None:
    desktop, controller, view = library
    open_detail(view, qtbot, desktop)
    action = markdown_action(view)
    before = detail_state(view)
    assert before[0]
    assert controller.library.segment.text in " ".join(before[2])
    controller.outcome = outcome
    destination = tmp_path / "meeting.md"
    # A failed overwrite must preserve content already owned by the user.
    previous = b"existing user document\r\n"
    if outcome in {"cancelled", "write_error"}:
        destination.write_bytes(previous)
    picks = install_picker(monkeypatch, destination)
    ui_thread = threading.get_ident()
    labels_before = {label.text() for label in view.findChildren(QLabel) if label.isVisible()}

    try:
        action.click()
        qtbot.waitUntil(controller.entered.is_set)
        assert desktop.jobs.busy
        [(meeting_id, path, worker_thread)] = controller.calls
        assert (meeting_id, path) == (controller.library.meeting.id, destination)
        assert worker_thread != ui_thread
        [(file_filter, picker_thread)] = picks
        assert "*.md" in file_filter.lower()
        assert picker_thread == ui_thread
        if outcome == "cancelled":
            [cancel] = buttons(view, r"cancel")
            cancel.click()
    finally:
        controller.release.set()
        qtbot.waitUntil(lambda: not desktop.jobs.busy, timeout=6000)

    assert detail_state(view) == before
    assert action.isEnabled()
    assert controller.core.command_log == []
    assert controller.cancellation_seen is (outcome == "cancelled")
    if outcome in {"saved", "saved_warning"}:
        exported = destination.read_text(encoding="utf-8")
        assert FRESH_TITLE in exported
        assert FRESH_TRANSCRIPT in exported
        assert controller.library.segment.text not in exported
    elif outcome == "unavailable":
        # The detail was rendered first. Fresh worker refusal must win over the DTO.
        assert not destination.exists()
    else:
        assert destination.read_bytes() == previous
    if outcome == "saved_warning":
        visible = "\n".join(
            label.text() for label in view.findChildren(QLabel) if label.isVisible()
        )
        assert re.search(r"finalizad|guardad[oa]|exportad[oa]", visible, re.I)
        assert CLEANUP_WARNING in visible
        assert not notices
    elif outcome in {"unavailable", "write_error"}:
        labels_after = {label.text() for label in view.findChildren(QLabel) if label.isVisible()}
        assert notices or any(text.strip() for text in labels_after - labels_before)
    else:
        assert not notices
