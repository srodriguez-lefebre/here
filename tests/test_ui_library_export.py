"""Portability actions through the real LibraryView and desktop job owner."""

from __future__ import annotations

import re
import threading
from dataclasses import replace

import pytest
from here.memory.models import MeetingPage, MeetingSummary
from here.ui.app import HereDesktop
from here.ui.contract import ApplicationUiAdapter
from here.ui.fake_controller import FakeApplicationController, FakeLibraryBackend
from here.ui.library import LibraryView
from PySide6.QtCore import QSettings, Qt
from PySide6.QtWidgets import (
    QAbstractButton,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QLabel,
    QListWidget,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QTableWidget,
    QTabWidget,
)

IDS = (
    "00000000-0000-0000-0000-000000000101",
    "00000000-0000-0000-0000-000000000201",
)
COUNTS = (137, 421)


class Recovery:
    def discover(self):
        return ()


class Controller(ApplicationUiAdapter):
    def __init__(self, root):
        self.core = FakeApplicationController()
        library = FakeLibraryBackend()
        self.meetings = (
            library.meeting,
            replace(
                library.meeting, id=IDS[1], title="Synthetic follow-up", active_revision_id=None
            ),
        )
        self.queries = []
        self.export_operation = None
        self.import_operation = None
        super().__init__(
            self.core,
            root,
            memory_service=library,
            search_service=library,
            recovery_service=Recovery(),
        )

    def list_meetings(self, query):
        self.queries.append(query)
        return MeetingPage(items=tuple(MeetingSummary(meeting) for meeting in self.meetings))

    # Proposed additive presentation commands. Production delegates to the
    # configured PortabilityService; these widget tests control its timing.
    def export_meetings(self, ids, destination, include_audio, progress, cancellation):
        assert self.export_operation is not None, "Unexpected export"
        return self.export_operation(ids, destination, include_audio, progress, cancellation)

    def import_meetings(self, source, progress, cancellation):
        assert self.import_operation is not None, "Unexpected import"
        return self.import_operation(source, progress, cancellation)


def show_library(desktop, qtbot):
    desktop.show()
    tabs = desktop.main_window.findChild(QTabWidget)
    view = desktop.main_window.findChild(LibraryView)
    assert tabs is not None and view is not None
    tabs.setCurrentIndex(next(i for i in range(tabs.count()) if tabs.tabText(i) == "Biblioteca"))
    qtbot.waitUntil(lambda: table(view).rowCount() > 0 and not desktop.jobs.busy)
    return view


@pytest.fixture
def portability_library(qtbot, qapp, tmp_path, owned_desktops):
    controller = Controller(tmp_path / "sessions")
    settings = QSettings(str(tmp_path / "portable.ini"), QSettings.Format.IniFormat)
    desktop = HereDesktop(qapp, controller, settings=settings)
    return desktop, controller, show_library(desktop, qtbot)


@pytest.fixture
def notices(monkeypatch):
    output = []

    def show(parent, title, message, *args, **kwargs):
        output.extend((title, message))
        return QMessageBox.StandardButton.Ok

    for name in ("information", "warning", "critical"):
        monkeypatch.setattr(QMessageBox, name, staticmethod(show))
    return output


def semantic(widgets, pattern):
    return [
        widget
        for widget in widgets
        if widget.isVisible()
        and re.search(pattern, f"{widget.accessibleName()} {widget.text()}", re.I)
    ]


def action(root, pattern):
    matches = semantic(root.findChildren(QPushButton), pattern)
    assert len(matches) == 1, f"Expected one visible library action matching {pattern}"
    return matches[0]


def table(view):
    matches = [
        widget
        for widget in view.findChildren(QTableWidget)
        if re.search(r"resultado.*reuni", widget.accessibleName(), re.I)
    ]
    assert len(matches) == 1
    return matches[0]


def select(view, qtbot, row=0, modifier=Qt.KeyboardModifier.NoModifier):
    widget = table(view)
    item = widget.item(row, 0)
    widget.scrollToItem(item)
    qtbot.mouseClick(
        widget.viewport(),
        Qt.MouseButton.LeftButton,
        modifier=modifier,
        pos=widget.visualItemRect(item).center(),
    )


def selection(view):
    rows = table(view).selectionModel().selectedRows()
    return tuple(index.data() for index in sorted(rows, key=lambda index: index.row()))


def filter_status(view, desktop, controller, qtbot):
    combo = next(
        widget
        for widget in view.findChildren(QComboBox)
        if re.search(r"filtrar.*estado", widget.accessibleName(), re.I)
    )
    status = controller.meetings[0].status
    index = combo.findData(status)
    assert index > 0
    prior = len(controller.queries)
    combo.setCurrentIndex(index)
    qtbot.waitUntil(lambda: len(controller.queries) > prior and not desktop.jobs.busy)
    assert controller.queries[-1].statuses == (status,)
    return combo


def picker(monkeypatch, method, path):
    calls = []

    def choose(*args, **kwargs):
        calls.append(threading.get_ident())
        return ("" if path is None else str(path)), ""

    monkeypatch.setattr(QFileDialog, method, staticmethod(choose))
    return calls


def feedback(view, notices=()):
    root = view.window()
    output = list(notices)
    output.extend(widget.text() for widget in root.findChildren(QLabel) if widget.isVisible())
    output.extend(
        widget.toPlainText()
        for widget in root.findChildren(QPlainTextEdit)
        if widget.isVisible()
    )
    for widget in root.findChildren(QListWidget):
        if widget.isVisible():
            output.extend(widget.item(row).text() for row in range(widget.count()))
    for widget in root.findChildren(QTableWidget):
        if widget.isVisible():
            output.extend(
                " ".join(
                    widget.item(row, col).text()
                    for col in range(widget.columnCount())
                    if widget.item(row, col) is not None
                )
                for row in range(widget.rowCount())
            )
    return "\n".join(output).splitlines()


def _progress_visible(view, count):
    bars = view.window().findChildren(QProgressBar)
    return any(bar.isVisible() and bar.value() == count for bar in bars) or any(
        re.search(rf"\b{count}\s*(?:B\b|bytes?\b)", line, re.I) for line in feedback(view)
    )


class ProgressGate:
    def __init__(self):
        self.qt_thread = threading.get_ident()
        self.advance = threading.Event()
        self.release = threading.Event()
        self.cancelled = None

    def report(self, progress, cancellation):
        assert threading.get_ident() != self.qt_thread
        assert not cancellation()
        progress(COUNTS[0])
        assert self.advance.wait(5), "GUI did not acknowledge first progress"
        progress(COUNTS[1])
        assert self.release.wait(5), "GUI did not release the operation"
        self.cancelled = cancellation()

    def observe(self, qtbot, desktop, view):
        qtbot.waitUntil(lambda: _progress_visible(view, COUNTS[0]))
        assert desktop.jobs.busy
        self.advance.set()
        qtbot.waitUntil(lambda: _progress_visible(view, COUNTS[1]))
        assert desktop.jobs.busy

    def drain(self, qtbot, desktop):
        self.advance.set()
        self.release.set()
        qtbot.waitUntil(lambda: not desktop.jobs.busy, timeout=6000)


def cancel_work(view):
    matches = semantic(view.window().findChildren(QAbstractButton), r"cancel")
    assert len(matches) == 1, "Expected one visible cancel control during the operation"
    matches[0].click()


def start_export(view, qtbot, monkeypatch, include_audio, cancelled_picker=None):
    export = action(view, r"export")
    handled_modally = False
    unexpected_dialogs = []

    def options(dialog):
        return semantic(dialog.findChildren(QCheckBox), r"audio")

    def accept(dialog):
        choices = options(dialog)
        assert len(choices) == 1
        choices[0].setChecked(include_audio)
        accepts = [
            button
            for box in dialog.findChildren(QDialogButtonBox)
            for button in box.buttons()
            if box.buttonRole(button)
            in (QDialogButtonBox.ButtonRole.AcceptRole, QDialogButtonBox.ButtonRole.YesRole)
        ]
        if not accepts:
            accepts = [action(dialog, r"export|guardar|aceptar|continuar")]
        assert len(accepts) == 1
        accepts[0].click()

    def run_modal(dialog):
        nonlocal handled_modally
        handled_modally = True
        if not any(
            re.search(r"audio", f"{box.accessibleName()} {box.text()}", re.I)
            for box in dialog.findChildren(QCheckBox)
        ):
            unexpected_dialogs.append(type(dialog).__name__)
            return QDialog.DialogCode.Rejected
        dialog.show()
        try:
            accept(dialog)
            return dialog.result()
        finally:
            dialog.hide()

    def visible_options():
        return [
            dialog
            for dialog in view.window().findChildren(QDialog)
            if dialog.isVisible() and options(dialog)
        ]

    with monkeypatch.context() as patch:
        patch.setattr(QDialog, "exec", run_modal)
        export.click()
        qtbot.waitUntil(lambda: bool(handled_modally or visible_options() or cancelled_picker))
        if not handled_modally and (dialogs := visible_options()):
            assert len(dialogs) == 1
            accept(dialogs[0])
        assert not unexpected_dialogs, f"Unexpected modal dialogs: {unexpected_dialogs}"


@pytest.mark.parametrize("include_audio", [True, False])
def test_export_selected_passes_ids_destination_and_audio_choice(
    portability_library, qtbot, monkeypatch, tmp_path, notices, include_audio
):
    desktop, controller, view = portability_library
    export = action(view, r"export")
    assert not export.isEnabled()
    select(view, qtbot)
    select(view, qtbot, 1, Qt.KeyboardModifier.ControlModifier)
    titles = tuple(meeting.title for meeting in controller.meetings)
    assert selection(view) == titles
    destination = tmp_path / "synthetic-export.here.zip"
    picks = picker(monkeypatch, "getSaveFileName", destination)
    calls = []
    gate = ProgressGate()

    def run(ids, path, audio, progress, cancellation):
        from here.memory.portability import ExportResult

        calls.append((ids, path, audio))
        gate.report(progress, cancellation)
        return ExportResult(
            path=path,
            included_answer_ids=(),
            omitted_answer_ids=(),
            payload_bytes=COUNTS[-1],
            archive_bytes=60,
        )

    controller.export_operation = run
    try:
        start_export(view, qtbot, monkeypatch, include_audio)
        gate.observe(qtbot, desktop, view)
        [(ids, path, audio)] = calls
        assert len(ids) == len(IDS) and set(ids) == set(IDS)
        assert (path, audio) == (destination, include_audio)
        assert picks == [gate.qt_thread]
    finally:
        gate.drain(qtbot, desktop)
    assert gate.cancelled is False
    assert selection(view) == titles
    assert controller.core.command_log == []


@pytest.mark.parametrize("stop", ["picker", "worker", "space"])
def test_export_interruption_keeps_filter_selection_and_application_state(
    portability_library, qtbot, monkeypatch, tmp_path, notices, stop
):
    desktop, controller, view = portability_library
    action(view, r"export")
    combo = filter_status(view, desktop, controller, qtbot)
    select(view, qtbot)
    original = combo.currentData(), selection(view), controller.snapshot
    path = None if stop == "picker" else tmp_path / "export.here.zip"
    picks = picker(monkeypatch, "getSaveFileName", path)
    gate = ProgressGate()
    calls = []

    def run(ids, destination, include_audio, progress, cancellation):
        from here.memory.portability import ExportCancelled, InsufficientSpace

        calls.append(destination)
        if stop == "space":
            raise InsufficientSpace("opaque failure")
        gate.report(progress, cancellation)
        raise ExportCancelled("opaque cancellation")

    controller.export_operation = run
    try:
        start_export(view, qtbot, monkeypatch, False, picks if path is None else None)
        if stop == "worker":
            gate.observe(qtbot, desktop, view)
            cancel_work(view)
            assert desktop.jobs.busy
    finally:
        gate.drain(qtbot, desktop)
    assert picks == [gate.qt_thread]
    assert calls == ([] if path is None else [path])
    if stop == "worker":
        assert gate.cancelled is True
    if stop == "space":
        assert any(re.search(r"espacio", line, re.I) for line in feedback(view, notices))
    assert original == (combo.currentData(), selection(view), controller.snapshot)
    assert action(view, r"export").isEnabled()
