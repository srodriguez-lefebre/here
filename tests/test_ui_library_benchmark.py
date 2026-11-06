"""Opt-in real guarded 1k/100k LibraryView heartbeat, including render and drain.

Requires an exclusive quiet-host grant and HERE_RUN_SEARCH_BENCHMARK=1. The
shared frozen builder owns corpus/registry setup outside the measured window.
The existing fake-latency UI test remains a separate contract test.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import sys
from pathlib import Path
from time import perf_counter_ns

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.usefixtures("owned_desktops"),
    pytest.mark.skipif(
        os.environ.get("HERE_RUN_SEARCH_BENCHMARK") != "1",
        reason="Real guarded UI heartbeat requires an explicit exclusive host grant",
    ),
]


def _control(view, widget_type, pattern):
    matches = [
        widget
        for widget in view.findChildren(widget_type)
        if widget.isVisibleTo(view) and re.search(pattern, widget.accessibleName(), re.IGNORECASE)
    ]
    assert len(matches) == 1, f"Expected one accessible library control: {pattern}"
    return matches[0]


def _source_hashes():
    root = Path(__file__).resolve().parents[1]
    paths = sorted((root / "src").rglob("*.py"))
    paths += [
        root / "pyproject.toml",
        root / "tests" / "search_fixtures.py",
        root / "tests" / "search_performance_fixtures.py",
        Path(__file__).resolve(),
    ]
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


@pytest.fixture
def guarded_ui_catalog(tmp_path, monkeypatch):
    from search_performance_fixtures import build_guarded_scale_catalog

    def forbidden(*args, **kwargs):
        pytest.fail("Guarded UI benchmark attempted provider/network/audio work")

    monkeypatch.setenv("HERE_DATA_DIR", str(tmp_path / "unused-default"))
    monkeypatch.setenv("HERE_ENV_FILE", str(tmp_path / "absent.env"))
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-test-key")
    monkeypatch.setattr("socket.socket.connect", forbidden)
    monkeypatch.setattr("here.transcription.client.build_client", forbidden)
    # Builder reuses immutable scale_corpus and validates real SQL/FTS/registry,
    # alias resolution and complete public paging before any heartbeat starts.
    return build_guarded_scale_catalog(tmp_path / "guarded-catalog"), forbidden


def _rendered(table, page):
    model = table.model()
    if model is None or not page.items or model.rowCount() != len(page.items):
        return False
    return page.items[0].snippet in str(model.index(0, 0).data() or "") and page.items[
        -1
    ].snippet in str(model.index(len(page.items) - 1, 0).data() or "")


def test_real_guarded_library_query_render_and_worker_drain_keep_all_gaps_below_fifty_ms(
    guarded_ui_catalog, tmp_path, qapp, qtbot, record_property
):
    import PySide6
    from here.application import HereApplicationController
    from here.memory.models import SearchPage, SearchQuery
    from here.ui.app import HereDesktop
    from here.ui.contract import ApplicationUiAdapter
    from here.ui.library import LibraryView
    from PySide6.QtCore import QSettings, Qt, QTimer, qVersion
    from PySide6.QtWidgets import QAbstractItemView, QCheckBox, QLineEdit, QTabWidget

    catalog, forbidden = guarded_ui_catalog
    assert (catalog.physical_meetings, catalog.physical_segments) == (1000, 100000)
    assert len(catalog.registry_record_ids) == 220
    assert len(catalog.suppressed_meeting_ids) == 221
    assert len(catalog.expected_segment_ids) == 44
    assert catalog.alias_meeting_id in catalog.suppressed_meeting_ids
    assert qapp.platformName() == "offscreen"
    metadata = {
        "machine": platform.platform(),
        "uname": platform.uname()._asdict(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": sys.version,
        "python_executable": sys.executable,
        "pyside": PySide6.__version__,
        "qt": qVersion(),
        "qt_platform": qapp.platformName(),
        "source_raw_sha256": _source_hashes(),
        "catalog_root": str(catalog.data_dir),
        "setup_seconds": catalog.setup_seconds,
        "physical_meetings": catalog.physical_meetings,
        "physical_segments": catalog.physical_segments,
        "registry_record_ids": catalog.registry_record_ids,
        "suppressed_meeting_ids": catalog.suppressed_meeting_ids,
        "alias_meeting_id": catalog.alias_meeting_id,
        "alias_origin": catalog.alias_origin,
        "alias_fingerprint": catalog.alias_fingerprint,
        "query_text": catalog.query.text,
        "include_partial": True,
        "ui_limit": SearchQuery().limit,
        "repetitions": 8,
        "timer_interval_ms": 5,
        "gap_budget_ms": 50,
    }
    record_property("guarded_ui_context", json.dumps(metadata, sort_keys=True))
    controller = HereApplicationController(
        memory_service=catalog.memory, capture_factory=forbidden, live_factory=forbidden
    )
    # No memory/query override: the real adapter obtains the actual controller's
    # same injected MemoryService and constructs its production query facade.
    adapter = ApplicationUiAdapter(controller, tmp_path / "empty-sessions")
    desktop = HereDesktop(
        qapp,
        adapter,
        settings=QSettings(str(tmp_path / "performance.ini"), QSettings.Format.IniFormat),
    )
    desktop.show()
    qtbot.waitUntil(lambda: not desktop.jobs.busy, timeout=30000)
    tabs = desktop.main_window.findChild(QTabWidget)
    assert tabs is not None
    view = desktop.main_window.findChild(LibraryView)
    assert view is not None
    library_index = next(index for index in range(tabs.count()) if tabs.widget(index) is view)
    responses = []
    callback_times = []
    errors = []

    def delivered(request, kind, value, error):
        if error is not None:
            errors.append((request, kind, type(error).__name__, str(error)))
        if isinstance(value, SearchPage):
            responses.append((request, value))
            callback_times.append((request, perf_counter_ns()))

    desktop.jobs.finished.connect(delivered)
    ticks = []
    timer = QTimer(view)
    timer.setTimerType(Qt.TimerType.PreciseTimer)
    timer.setInterval(5)
    timer.timeout.connect(lambda: ticks.append(perf_counter_ns()))
    inputs = []
    dispatch_times = []
    completed = []
    rendered_drains = []
    drained = False
    started = perf_counter_ns()
    timer.start()
    try:
        # First tab display/guarded list dispatch is inside the same window as
        # public input, actual query callbacks, DTO rendering and final drain.
        tabs.setCurrentIndex(library_index)
        partial = _control(view, QCheckBox, r"(parcial)")
        if not partial.isChecked():
            qtbot.mouseClick(partial, Qt.MouseButton.LeftButton)
        assert partial.isChecked()
        field = _control(view, QLineEdit, r"busca")
        results = _control(view, QAbstractItemView, r"(resultado|reuni[oó]n|reuniones)")
        last_request = 0
        for repetition in range(8):
            previous_responses = len(responses)
            raw_text = catalog.query.text + (" " if repetition % 2 else "")
            inputs.append(raw_text)
            dispatch_times.append(perf_counter_ns())
            field.setFocus()
            field.selectAll()
            qtbot.keyClicks(field, raw_text)
            qtbot.waitUntil(
                lambda: bool(errors) or len(responses) > previous_responses, timeout=30000
            )
            assert not errors, f"Real background operation failed: {errors}"
            request, page = responses[-1]
            assert request > last_request
            last_request = request
            assert not page.degraded
            assert tuple(hit.segment_id for hit in page.items) == catalog.expected_segment_ids
            assert all(hit.meeting_id not in catalog.suppressed_meeting_ids for hit in page.items)
            assert page.next_cursor is None  # 44 authored survivors fit the real UI's limit50.
            qtbot.waitUntil(lambda: _rendered(results, page), timeout=30000)
            qtbot.waitUntil(lambda: not desktop.jobs.busy, timeout=30000)
            # Actual visible repaint and queued layout/paint delivery remain
            # measured; model contents alone are not a painting certificate.
            desktop.main_window.repaint()
            qapp.processEvents()
            completed.append(request)
            rendered_drains.append((request, perf_counter_ns()))
    finally:
        try:
            qtbot.waitUntil(lambda: not desktop.jobs.busy, timeout=30000)
            desktop.main_window.repaint()
            qapp.processEvents()
            drained = not desktop.jobs.busy
        finally:
            ended = perf_counter_ns()
            timer.stop()
            # Include leading/trailing edges so a stall before the first tick,
            # or during final render/drain after the last tick, cannot disappear.
            points = [started, *ticks, ended]
            gaps_ns = [right - left for left, right in zip(points, points[1:])]
            maximum_ns = max(gaps_ns)
            record_property("heartbeat_points_ns", json.dumps(points))
            record_property("heartbeat_all_gaps_ns", json.dumps(gaps_ns))
            record_property("heartbeat_tick_count", len(ticks))
            record_property("heartbeat_max_gap_ms", maximum_ns / 1_000_000)
            record_property("measured_window_ms", (ended - started) / 1_000_000)
            record_property("public_inputs", json.dumps(inputs))
            record_property("public_input_dispatches_ns", json.dumps(dispatch_times))
            record_property("observed_search_callbacks_ns", json.dumps(callback_times))
            record_property("render_and_drain_completions_ns", json.dumps(rendered_drains))
            record_property("completed_search_request_ids", json.dumps(completed))
            record_property("delivered_search_count", len(responses))
            record_property("background_errors", json.dumps(errors))
            record_property("owned_workers_drained", drained)
    assert drained
    assert not errors
    assert len(completed) == 8
    assert len(ticks) >= 50, "Insufficient actual heartbeat observations; no acceptance claim"
    assert all(gap < 50_000_000 for gap in gaps_ns), (
        f"Observed Qt gap {maximum_ns / 1_000_000:.3f} ms is not below 50 ms"
    )
    assert maximum_ns < 50_000_000
    assert not (tmp_path / "unused-default" / "library.sqlite3").exists()
