"""Portable import outcomes and cancellation through the Qt library."""

from __future__ import annotations

import re
import socket
import threading
from dataclasses import replace
from types import SimpleNamespace

import pytest
from here.ui import contract, preview
from here.ui.app import HereDesktop
from PySide6.QtCore import QSettings
from test_ui_library_export import (
    IDS,
    ProgressGate,
    action,
    cancel_work,
    feedback,
    filter_status,
    notices as notices,
    picker,
    portability_library as portability_library,
    select,
    selection,
    show_library,
    start_export,
    table,
)

NEW_ID = "00000000-0000-0000-0000-000000000302"
REJECTED_ID = "00000000-0000-0000-0000-000000000301"
IMPORTED_TITLE = "Synthetic imported meeting"


def _report(*rows):
    from here.memory.portability import ImportReport

    return ImportReport(
        meetings=tuple(
            SimpleNamespace(meeting_id=identity, status=status, reason=reason)
            for identity, status, reason in rows
        )
    )


def _assert_outcome(lines, identity, status, *, title=None, reason=None):
    # UI may show a name, an abbreviated ID or the distinguishing reason.
    # These fixture IDs differ at the end, never at the beginning.
    id_pattern = rf"{identity[-6:]}(?![0-9a-f])"
    matches = [
        line
        for line in lines
        if re.search(status, line, re.I)
        and (
            re.search(id_pattern, line, re.I)
            or (title is not None and title in line)
            or (reason is not None and re.search(reason, line, re.I))
        )
    ]
    assert matches, f"Expected a report entry for {identity} matching {status}"


@pytest.mark.parametrize("pending", [False, True])
def test_partial_import_reports_committed_and_rejected(
    portability_library, qtbot, monkeypatch, tmp_path, notices, pending
):
    desktop, controller, view = portability_library
    import_package = action(view, r"import")
    combo = filter_status(view, desktop, controller, qtbot)
    select(view, qtbot)
    status = combo.currentData()
    original_selection = selection(view)
    original_query_count = len(controller.queries)
    snapshot = controller.snapshot
    source = tmp_path / "synthetic-import.here.zip"
    picks = picker(monkeypatch, "getOpenFileName", source)
    gate = ProgressGate()
    calls = []

    def run(path, progress, cancellation):
        calls.append(path)
        gate.report(progress, cancellation)
        assert gate.cancelled is (not pending)
        controller.meetings += (
            replace(controller.meetings[1], id=NEW_ID, title=IMPORTED_TITLE),
        )
        return _report(
            (NEW_ID, "inserted", "local_recovery_pending" if pending else ""),
            (REJECTED_ID, "rejected", "batch_stopped"),
        )

    controller.import_operation = run
    try:
        import_package.click()
        gate.observe(qtbot, desktop, view)
        if not pending:
            cancel_work(view)
        assert desktop.jobs.busy
        assert table(view).rowCount() == 2
    finally:
        gate.drain(qtbot, desktop)
    assert picks == [gate.qt_thread]
    assert calls == [source]
    assert gate.cancelled is (not pending)
    qtbot.waitUntil(lambda: table(view).rowCount() == 3 and not desktop.jobs.busy)
    assert controller.queries[original_query_count:]
    assert all(query.statuses == (status,) for query in controller.queries[original_query_count:])
    assert combo.currentData() == status
    assert selection(view) == original_selection
    lines = feedback(view, notices)
    _assert_outcome(lines, NEW_ID, r"insertad|importad|inserted", title=IMPORTED_TITLE)
    _assert_outcome(lines, REJECTED_ID, r"rechazad|rejected", reason=r"batch_stopped|lote")
    if pending:
        assert any(re.search(r"recuperaci[oó]n|recovery", line, re.I) for line in lines)
    assert any(line.startswith(IMPORTED_TITLE) for line in lines)
    assert controller.snapshot == snapshot
    assert controller.core.command_log == []


@pytest.mark.parametrize("outcome", ["picker", "worker", "invalid", "conflict", "suppressed"])
def test_import_without_a_commit_preserves_filter_selection_and_application_state(
    portability_library, qtbot, monkeypatch, tmp_path, notices, outcome
):
    desktop, controller, view = portability_library
    import_package = action(view, r"import")
    combo = filter_status(view, desktop, controller, qtbot)
    select(view, qtbot)
    original = combo.currentData(), selection(view), controller.snapshot
    source = None if outcome == "picker" else tmp_path / "synthetic-import.here.zip"
    picks = picker(monkeypatch, "getOpenFileName", source)
    gate = ProgressGate()
    calls = []

    def run(path, progress, cancellation):
        from here.memory.portability import ImportCancelled, InvalidPackage

        calls.append(path)
        assert threading.get_ident() != gate.qt_thread
        if outcome == "invalid":
            raise InvalidPackage("opaque failure")
        if outcome == "worker":
            gate.report(progress, cancellation)
            raise ImportCancelled("opaque cancellation")
        if outcome == "suppressed":
            return _report(
                (REJECTED_ID, "rejected", "suppressed"),
                (NEW_ID, "rejected", "batch_stopped"),
            )
        return _report(
            (IDS[1], "unchanged", ""),
            (IDS[0], "conflict", "different_meeting"),
            (REJECTED_ID, "rejected", "batch_stopped"),
        )

    controller.import_operation = run
    try:
        import_package.click()
        if outcome == "worker":
            gate.observe(qtbot, desktop, view)
            cancel_work(view)
            assert desktop.jobs.busy
    finally:
        gate.drain(qtbot, desktop)
    assert picks == [gate.qt_thread]
    assert calls == ([] if source is None else [source])
    if outcome == "worker":
        assert gate.cancelled is True
    lines = feedback(view, notices)
    if outcome == "invalid":
        error = r"inv[aá]lid|dañad|formato|no se pudo import"
        assert any(re.search(error, line, re.I) for line in lines)
    if outcome == "suppressed":
        _assert_outcome(lines, REJECTED_ID, r"rechazad|rejected", reason=r"suprimid|suppressed")
        _assert_outcome(lines, NEW_ID, r"rechazad|rejected", reason=r"batch_stopped|lote")
    if outcome == "conflict":
        _assert_outcome(lines, IDS[1], r"sin cambios|unchanged", title=controller.meetings[1].title)
        _assert_outcome(lines, IDS[0], r"conflict", title=controller.meetings[0].title)
        _assert_outcome(lines, REJECTED_ID, r"rechazad|rejected", reason=r"batch_stopped|lote")
    assert original == (combo.currentData(), selection(view), controller.snapshot)
    assert action(view, r"import").isEnabled()
    assert controller.core.command_log == []


@pytest.mark.parametrize("operation", ["export", "import"])
def test_preview_uses_synthetic_portability(
    qtbot, qapp, tmp_path, monkeypatch, owned_desktops, notices, operation
):
    from here import live_processing
    from here.memory.service import MemoryService
    from here.transcription import client, service

    desktops = []
    forbidden_calls = []

    def desktop_for_preview(application, controller):
        settings = QSettings(str(tmp_path / "preview.ini"), QSettings.Format.IniFormat)
        desktop = HereDesktop(application, controller, settings=settings)
        desktops.append(desktop)
        return desktop

    def forbidden(*args, **kwargs):
        forbidden_calls.append(threading.get_ident())
        raise AssertionError("Preview reached a real catalog, archive or provider boundary")

    def drive_preview():
        (desktop,) = desktops
        view = show_library(desktop, qtbot)
        selected_action = action(view, operation)
        from here.memory.portability import ExportResult, ImportReport, PortabilityService

        for name in ("inspect", "import_archive", "export"):
            monkeypatch.setattr(PortabilityService, name, forbidden)
        archive = tmp_path / "preview.here.zip"
        jobs = []
        desktop.jobs.finished.connect(lambda *result: jobs.append(result))
        if operation == "export":
            picks = picker(monkeypatch, "getSaveFileName", archive)
            select(view, qtbot)
            start_export(view, qtbot, monkeypatch, False)
            expected = ExportResult
        else:
            picks = picker(monkeypatch, "getOpenFileName", archive)
            selected_action.click()
            expected = ImportReport
        qtbot.waitUntil(lambda: bool(jobs) and not desktop.jobs.busy)
        assert any(isinstance(value, expected) and error is None for _, _, value, error in jobs)
        assert picks == [threading.get_ident()]
        assert not archive.exists()
        assert not (tmp_path / "data").exists()
        assert not (tmp_path / "transcriptions").exists()
        return 0

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(MemoryService, "__init__", forbidden)
    monkeypatch.setattr(contract, "require_provider_key", forbidden)
    for module in (client, service, live_processing):
        monkeypatch.setattr(module, "build_client", forbidden)
    for method in ("connect", "connect_ex"):
        monkeypatch.setattr(socket.socket, method, forbidden)
    monkeypatch.setattr(preview, "HereDesktop", desktop_for_preview)
    monkeypatch.setattr(qapp, "exec", drive_preview)
    assert preview.main() == 0
    assert not forbidden_calls
