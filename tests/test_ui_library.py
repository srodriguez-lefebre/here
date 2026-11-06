"""M3 library behavior through the real desktop and real BackgroundJobs.

Only the controller is synthetic. Canonical DTOs are resolved in fixtures so a
missing feature produces an explanatory test failure, rather than aborting
collection of unrelated lifecycle tests. These tests have NOT yet been run RED.
"""

from __future__ import annotations

import importlib
import re
import threading
from dataclasses import replace
from time import perf_counter

import pytest
from here.memory.database import LibraryError
from here.memory.models import Meeting, MeetingStatus
from here.ui.app import HereDesktop
from here.ui.contract import ApplicationUiAdapter
from here.ui.fake_controller import FakeApplicationController
from PySide6.QtCore import QSettings, Qt, QTimer
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QComboBox,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QTabWidget,
    QTextEdit,
    QWidget,
)

pytestmark = pytest.mark.usefixtures("owned_desktops")


def _feature(module_name, symbol):
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name != module_name:
            raise
        pytest.fail(f"M3 feature missing: {module_name}.{symbol}")
    assert hasattr(module, symbol), f"M3 contract missing: {module_name}.{symbol}"
    return getattr(module, symbol)


@pytest.fixture
def dto():
    # Do not create shadow query/page/hit/target DTOs in this test module.
    names = (
        "MeetingQuery",
        "SearchQuery",
        "MeetingSummary",
        "MeetingPage",
        "SearchPage",
        "SearchHit",
    )
    return {name: _feature("here.memory.models", name) for name in names}


def _meeting(title="Planning", meeting_id="00000000-0000-0000-0000-000000000001"):
    return Meeting(
        id=meeting_id,
        title=title,
        status=MeetingStatus.COMPLETED,
        started_at="2026-11-05T12:00:00+00:00",
        duration_ms=60000,
        row_version=7,
    )


def _page(dto, *meetings, cursor=None):
    items = tuple(dto["MeetingSummary"](meeting=meeting) for meeting in meetings)
    return dto["MeetingPage"](items=items, next_cursor=cursor)


def _hits(
    dto,
    text,
    *,
    cursor=None,
    degraded=False,
    segment_id="00000000-0000-0000-0000-000000000003",
):
    hit = dto["SearchHit"](
        meeting_id="00000000-0000-0000-0000-000000000001",
        revision_id="00000000-0000-0000-0000-000000000002",
        segment_id=segment_id,
        snippet=text,
        title="Planning",
        started_at=None,
        start_ms=None,
        end_ms=None,
        timing_source="absent",
        partial=False,
    )
    return dto["SearchPage"](items=(hit,), next_cursor=cursor, degraded=degraded)


class LibraryController(ApplicationUiAdapter):
    """Controller double: blocking gates model service latency, never Qt delivery."""

    def __init__(self, root, dto):
        self.core = FakeApplicationController()
        super().__init__(self.core, root)
        self.dto = dto
        self.calls = []
        self.list_handler = lambda query: _page(dto)
        self.search_handler = lambda query: dto["SearchPage"](items=(), next_cursor=None)
        self.detail_handler = lambda meeting_id: None

    def discover_recovery(self):
        return ()

    def list_meetings(self, query):
        assert isinstance(query, self.dto["MeetingQuery"])
        self.calls.append(("list", query, threading.get_ident()))
        return self.list_handler(query)

    def search(self, query):
        assert isinstance(query, self.dto["SearchQuery"])
        self.calls.append(("search", query, threading.get_ident()))
        return self.search_handler(query)

    def get_meeting(self, meeting_id):
        self.calls.append(("detail", meeting_id, threading.get_ident()))
        return self.detail_handler(meeting_id)

    def update_meeting(self, meeting_id, patch, expected_version):
        raise AssertionError("Real title-edit acceptance waits for the accepted B4 adapter")


def _named(view, widget_type, pattern):
    matches = [
        widget
        for widget in view.findChildren(widget_type)
        if widget.isVisibleTo(view)
        and re.search(pattern, widget.accessibleName(), re.IGNORECASE)
    ]
    assert len(matches) == 1, f"Expected one accessible {widget_type.__name__}: {pattern}"
    return matches[0]


def _text(view):
    """Read the rendered UI, including item models; never controller state."""
    parts = [label.text() for label in view.findChildren(QLabel) if label.isVisibleTo(view)]
    parts += [edit.toPlainText() for edit in view.findChildren(QTextEdit) if edit.isVisibleTo(view)]
    parts += [
        edit.toPlainText() for edit in view.findChildren(QPlainTextEdit) if edit.isVisibleTo(view)
    ]
    for widget in view.findChildren(QAbstractItemView):
        if not widget.isVisibleTo(view) or isinstance(widget.parentWidget(), QComboBox):
            continue
        model = widget.model()
        if model is not None:
            parts += [
                str(model.index(row, column).data() or "")
                for row in range(model.rowCount())
                for column in range(model.columnCount())
            ]
    return "\n".join(parts)


def _results(view):
    matches = [
        widget
        for widget in view.findChildren(QAbstractItemView)
        if widget.isVisibleTo(view)
        and not isinstance(widget.parentWidget(), QComboBox)
        and re.search(r"(reunion|reuni[oó]n|resultado|biblioteca)", widget.accessibleName(), re.I)
    ]
    assert len(matches) == 1, "Library results need a uniquely named item view"
    return matches[0]


def _continue(view, qtbot):
    # Paging is a user operation. Support either a named continuation action or
    # reaching the end of the results, without calling private view methods.
    buttons = [
        button
        for button in view.findChildren(QPushButton)
        if button.isVisibleTo(view)
        and re.search(r"(siguiente|m[aá]s)", button.accessibleName(), re.I)
    ]
    if buttons:
        assert len(buttons) == 1
        qtbot.mouseClick(buttons[0], Qt.MouseButton.LeftButton)
    else:
        results = _results(view)
        results.setFocus()
        qtbot.keyClick(results, Qt.Key.Key_End)
        results.scrollToBottom()


@pytest.fixture
def library(qtbot, tmp_path, dto, qapp):
    library_type = _feature("here.ui.library", "LibraryView")
    controller = LibraryController(tmp_path, dto)
    settings = QSettings(str(tmp_path / "library.ini"), QSettings.Format.IniFormat)
    desktop = HereDesktop(qapp, controller, settings=settings)
    desktop.show()
    tabs = desktop.main_window.findChildren(QTabWidget)
    assert len(tabs) == 1, "MainWindow must expose capture and Biblioteca tabs"
    indexes = [i for i in range(tabs[0].count()) if tabs[0].tabText(i) == "Biblioteca"]
    assert len(indexes) == 1, "Biblioteca tab missing"
    tabs[0].setCurrentIndex(indexes[0])
    view = desktop.main_window.findChild(library_type)
    assert view is not None, "Biblioteca must contain the real LibraryView"
    qtbot.waitUntil(lambda: not desktop.jobs.busy)
    yield desktop, controller, view
    # Release barriers in individual tests before owned_desktops drains real jobs.


def test_empty_library_offers_recording_action(library, qtbot):
    desktop, controller, view = library
    qtbot.waitUntil(lambda: "Aún no hay reuniones" in _text(view))
    button = _named(view, QPushButton, r"(iniciar|nueva|grabar).*grabaci|grabar")
    qtbot.mouseClick(button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: controller.core.command_log == ["start"])
    assert desktop.controller.snapshot.has_active_work


def test_no_matches_action_clears_search_and_lists_again(library, qtbot):
    _, controller, view = library
    controller.list_handler = lambda query: _page(controller.dto, _meeting())
    search = _named(view, QLineEdit, r"busca")
    qtbot.keyClicks(search, "missing")
    qtbot.waitUntil(lambda: "No hay coincidencias" in _text(view))
    qtbot.mouseClick(_named(view, QPushButton, r"limpiar"), Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: "Planning" in _text(view))
    assert search.text() == ""
    assert controller.calls[-1][0] == "list"
    assert controller.core.command_log == []


def test_five_characters_inside_debounce_issue_one_search(library, qtbot):
    _, controller, view = library
    search = _named(view, QLineEdit, r"busca")
    started = perf_counter()
    qtbot.keyClicks(search, "hello")
    assert (perf_counter() - started) < 0.2, "Typing fixture exceeded debounce window"
    qtbot.wait(100)
    assert not [call for call in controller.calls if call[0] == "search"]
    qtbot.waitUntil(lambda: any(call[0] == "search" for call in controller.calls))
    qtbot.wait(250)
    searches = [call for call in controller.calls if call[0] == "search"]
    assert len(searches) == 1
    assert searches[0][1].text == "hello"
    assert searches[0][2] != threading.get_ident()


def test_delayed_older_worker_cannot_replace_current_search(library, qtbot):
    desktop, controller, view = library
    entered, release = threading.Event(), threading.Event()

    def search_page(query):
        if query.text == "old":
            entered.set()
            assert release.wait(5), "Test did not release old search worker"
            return _hits(controller.dto, "STALE_RESULT")
        return _hits(controller.dto, "CURRENT_RESULT")

    controller.search_handler = search_page
    search = _named(view, QLineEdit, r"busca")
    try:
        search.setText("old")
        qtbot.waitUntil(entered.is_set)
        search.setText("new")
        qtbot.waitUntil(lambda: "CURRENT_RESULT" in _text(view))
    finally:
        release.set()
        qtbot.waitUntil(lambda: not desktop.jobs.busy)
    assert "CURRENT_RESULT" in _text(view)
    assert "STALE_RESULT" not in _text(view)
    assert controller.core.command_log == []


def test_superseded_error_cannot_replace_new_results_or_start_recording(library, qtbot):
    desktop, controller, view = library
    entered, release = threading.Event(), threading.Event()

    def search_page(query):
        if query.text == "old":
            entered.set()
            assert release.wait(5)
            raise LibraryError("STALE_ERROR")
        return _hits(controller.dto, "CURRENT_RESULT")

    controller.search_handler = search_page
    search = _named(view, QLineEdit, r"busca")
    try:
        search.setText("old")
        qtbot.waitUntil(entered.is_set)
        search.setText("new")
        qtbot.waitUntil(lambda: "CURRENT_RESULT" in _text(view))
    finally:
        release.set()
        qtbot.waitUntil(lambda: not desktop.jobs.busy)
    assert "STALE_ERROR" not in _text(view)
    assert "CURRENT_RESULT" in _text(view)
    assert controller.core.command_log == []


def test_corpus_changed_continuation_restarts_page_one_replacing_old_hits(library, qtbot):
    desktop, controller, view = library
    invalid_input = _feature("here.memory.service", "InvalidInput")
    first_page_count = 0

    def search_page(query):
        nonlocal first_page_count
        if query.cursor is not None:
            assert query.cursor == "generation-one-continuation"
            raise invalid_input("Localized opaque diagnostic", reason="corpus_changed")
        first_page_count += 1
        if first_page_count == 1:
            return _hits(controller.dto, "OLD_GENERATION", cursor="generation-one-continuation")
        return _hits(controller.dto, "CURRENT_GENERATION")

    controller.search_handler = search_page
    _named(view, QLineEdit, r"busca").setText("same query")
    qtbot.waitUntil(lambda: "OLD_GENERATION" in _text(view))
    _continue(view, qtbot)
    qtbot.waitUntil(lambda: "CURRENT_GENERATION" in _text(view))
    qtbot.waitUntil(lambda: not desktop.jobs.busy)
    searches = [call[1] for call in controller.calls if call[0] == "search"]
    assert [query.cursor for query in searches] == [None, "generation-one-continuation", None]
    assert [query.text for query in searches] == ["same query"] * 3
    assert "OLD_GENERATION" not in _text(view)
    assert controller.core.command_log == []


def test_continuation_appends_visible_hits_from_the_same_generation(library, qtbot):
    desktop, controller, view = library

    def search_page(query):
        if query.cursor is None:
            return _hits(controller.dto, "FIRST_VISIBLE_HIT", cursor="same-generation")
        assert query.cursor == "same-generation"
        return _hits(
            controller.dto,
            "SECOND_VISIBLE_HIT",
            segment_id="00000000-0000-0000-0000-000000000004",
        )

    controller.search_handler = search_page
    _named(view, QLineEdit, r"busca").setText("query")
    qtbot.waitUntil(lambda: "FIRST_VISIBLE_HIT" in _text(view))
    _continue(view, qtbot)
    qtbot.waitUntil(lambda: "SECOND_VISIBLE_HIT" in _text(view))
    qtbot.waitUntil(lambda: not desktop.jobs.busy)
    assert "FIRST_VISIBLE_HIT" in _text(view)
    assert "SECOND_VISIBLE_HIT" in _text(view)


def test_audio_only_unknown_date_is_not_replaced_with_a_processing_date(library, qtbot):
    _, controller, view = library
    reserved = Meeting(
        id="00000000-0000-0000-0000-000000000005",
        title="Reserved audio only",
        started_at=None,
        duration_ms=None,
        active_revision_id=None,
        status=MeetingStatus.INTERRUPTED,
        created_at="2026-11-06T10:00:00+00:00",
        updated_at="2026-11-06T10:01:00+00:00",
    )
    controller.list_handler = lambda query: _page(controller.dto, reserved)
    search = _named(view, QLineEdit, r"busca")
    search.setText("missing")
    qtbot.waitUntil(lambda: "No hay coincidencias" in _text(view))
    search.clear()
    qtbot.waitUntil(lambda: "Reserved audio only" in _text(view))
    assert "2026" not in _text(view)
    assert "00:00" not in _text(view)


def test_superseded_corpus_changed_cannot_restart_or_append_old_query(library, qtbot):
    desktop, controller, view = library
    invalid_input = _feature("here.memory.service", "InvalidInput")
    entered, release = threading.Event(), threading.Event()

    def search_page(query):
        if query.text == "old" and query.cursor:
            entered.set()
            assert release.wait(5)
            raise invalid_input(
                "No machine-readable reason in this message", reason="corpus_changed"
            )
        if query.text == "old":
            return _hits(controller.dto, "OLD_PAGE", cursor="old-cursor")
        return _hits(controller.dto, "NEW_QUERY_RESULT")

    controller.search_handler = search_page
    search = _named(view, QLineEdit, r"busca")
    try:
        search.setText("old")
        qtbot.waitUntil(lambda: "OLD_PAGE" in _text(view))
        _continue(view, qtbot)
        qtbot.waitUntil(entered.is_set)
        search.setText("new")
        qtbot.waitUntil(lambda: "NEW_QUERY_RESULT" in _text(view))
    finally:
        release.set()
        qtbot.waitUntil(lambda: not desktop.jobs.busy)
    # Allow a wrongly scheduled restart to emerge after its debounce.
    qtbot.wait(250)
    searches = [call[1] for call in controller.calls if call[0] == "search"]
    assert [(query.text, query.cursor) for query in searches] == [
        ("old", None),
        ("old", "old-cursor"),
        ("new", None),
    ]
    assert "NEW_QUERY_RESULT" in _text(view)
    assert "OLD_PAGE" not in _text(view)
    assert controller.core.command_log == []


def test_warning_prose_does_not_trigger_corpus_restart(library, qtbot):
    desktop, controller, view = library
    invalid_input = _feature("here.memory.service", "InvalidInput")

    def search_page(query):
        if query.cursor:
            raise invalid_input("Diagnostic mentions corpus_changed", reason="invalid_input")
        return _hits(controller.dto, "FIRST_PAGE", cursor="continuation")

    controller.search_handler = search_page
    _named(view, QLineEdit, r"busca").setText("query")
    qtbot.waitUntil(lambda: "FIRST_PAGE" in _text(view))
    _continue(view, qtbot)
    qtbot.waitUntil(lambda: len([call for call in controller.calls if call[0] == "search"]) == 2)
    qtbot.waitUntil(lambda: not desktop.jobs.busy)
    qtbot.wait(250)
    assert len([call for call in controller.calls if call[0] == "search"]) == 2
    assert "corpus_changed" in _text(view)
    assert controller.core.command_log == []


def test_closing_detail_discards_its_delayed_failure(library, qtbot):
    desktop, controller, view = library
    entered, release = threading.Event(), threading.Event()
    controller.list_handler = lambda query: _page(controller.dto, _meeting())

    def get_detail(meeting_id):
        entered.set()
        assert release.wait(5)
        raise LibraryError("STALE_DETAIL_ERROR")

    controller.detail_handler = get_detail
    try:
        # Clear a nonempty search to request the populated listing.
        search = _named(view, QLineEdit, r"busca")
        search.setText("missing")
        qtbot.waitUntil(lambda: "No hay coincidencias" in _text(view))
        search.clear()
        qtbot.waitUntil(lambda: "Planning" in _text(view))
        results = _results(view)
        index = results.model().index(0, 0)
        results.setCurrentIndex(index)
        qtbot.mouseDClick(
            results.viewport(),
            Qt.MouseButton.LeftButton,
            pos=results.visualRect(index).center(),
        )
        qtbot.waitUntil(entered.is_set)
        qtbot.mouseClick(_named(view, QPushButton, r"cerrar.*detalle"), Qt.MouseButton.LeftButton)
    finally:
        release.set()
        qtbot.waitUntil(lambda: not desktop.jobs.busy)
    assert "STALE_DETAIL_ERROR" not in _text(view)
    assert "Planning" in _text(view)
    assert controller.core.command_log == []


def test_keyboard_reaches_search_date_status_participant_filters_and_results(library, qtbot, qapp):
    _, _, view = library
    expected = [
        _named(view, QLineEdit, r"busca"),
        _named(view, QAbstractSpinBox, r"(desde|inicio)"),
        _named(view, QAbstractSpinBox, r"(hasta|fin)"),
        _named(view, QComboBox, r"estado"),
        _named(view, QComboBox, r"participante"),
        _results(view),
    ]
    expected[0].setFocus()
    qtbot.waitUntil(lambda: expected[0].hasFocus())
    visited = [expected[0]]
    for _ in range(40):
        focused = qapp.focusWidget()
        assert focused is not None
        qtbot.keyClick(focused, Qt.Key.Key_Tab)
        focused = qapp.focusWidget()
        for control in expected:
            if focused is control or (focused is not None and control.isAncestorOf(focused)):
                if control not in visited:
                    visited.append(control)
        if expected[-1] in visited:
            break
    assert visited == expected, "Keyboard must traverse the library query controls in visual order"


def test_snippet_is_plain_text_and_absent_timing_is_not_zero(library, qtbot):
    _, controller, view = library
    literal = "<b>source & evidence</b>"
    controller.search_handler = lambda query: _hits(controller.dto, literal, degraded=True)
    _named(view, QLineEdit, r"busca").setText("evidence")
    qtbot.waitUntil(lambda: literal in _text(view))
    for label in view.findChildren(QLabel):
        if literal in label.text():
            assert label.textFormat() == Qt.TextFormat.PlainText
    text = _text(view)
    assert "00:00" not in text
    assert re.search(r"(degrad|limitad|sin FTS|literal)", text, re.IGNORECASE)


def test_slow_search_keeps_qt_heartbeat_under_fifty_ms(library, qtbot):
    desktop, controller, view = library
    entered, release = threading.Event(), threading.Event()
    ticks = []
    timer = QTimer(view)
    timer.setInterval(5)
    timer.timeout.connect(lambda: ticks.append(perf_counter()))

    def search_page(query):
        entered.set()
        assert release.wait(5)
        return _hits(controller.dto, "RESPONSIVE_RESULT")

    controller.search_handler = search_page
    try:
        timer.start()
        _named(view, QLineEdit, r"busca").setText("slow")
        qtbot.waitUntil(entered.is_set)
        ticks.clear()
        qtbot.wait(250)
        assert len(ticks) >= 10, "Qt event loop stopped during the outstanding query"
        assert max(b - a for a, b in zip(ticks, ticks[1:])) < 0.05
    finally:
        timer.stop()
        release.set()
        qtbot.waitUntil(lambda: not desktop.jobs.busy)
    qtbot.waitUntil(lambda: "RESPONSIVE_RESULT" in _text(view))


def _luminance(color):
    def linear(value):
        return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4

    return sum(
        weight * linear(channel)
        for weight, channel in zip((0.2126, 0.7152, 0.0722), color.getRgbF()[:3])
    )


@pytest.mark.parametrize("dark", [False, True], ids=["light", "dark"])
def test_library_controls_have_accessible_names_and_readable_contrast(library, dark):
    _, _, view = library
    # Inject OS light/dark palettes; no product theme selector is invented.
    palette = QPalette(view.palette())
    background, foreground = ("#202020", "#ffffff") if dark else ("#ffffff", "#202020")
    for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Base, QPalette.ColorRole.Button):
        palette.setColor(role, QColor(background))
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        palette.setColor(role, QColor(foreground))
    view.setPalette(palette)
    controls = [
        child
        for child in view.findChildren(QWidget)
        if isinstance(child, (QLineEdit, QPushButton, QComboBox, QAbstractItemView, QTextEdit))
        and child.isVisibleTo(view)
        and child.focusPolicy() != Qt.FocusPolicy.NoFocus
        and not isinstance(child.parentWidget(), (QComboBox, QAbstractSpinBox))
    ]
    assert controls
    for control in controls:
        assert control.accessibleName().strip(), f"Unnamed control: {type(control).__name__}"
        actual = control.palette()
        role = (
            QPalette.ColorRole.ButtonText
            if isinstance(control, QPushButton)
            else control.foregroundRole()
        )
        background_role = (
            QPalette.ColorRole.Button
            if isinstance(control, QPushButton)
            else control.backgroundRole()
        )
        values = sorted(
            (_luminance(actual.color(role)), _luminance(actual.color(background_role)))
        )
        assert (values[1] + 0.05) / (values[0] + 0.05) >= 4.5


def _detail(*, start_ms=0, end_ms=1500, timing_source="provider", text=None):
    """Build only canonical frozen values; controller latency remains the sole double."""
    detail_type = _feature("here.memory.models", "MeetingDetail")
    revision_type = _feature("here.memory.models", "TranscriptRevision")
    segment_type = _feature("here.memory.models", "Segment")
    revision_id = "00000000-0000-0000-0000-000000000002"
    meeting = replace(
        _meeting(),
        active_revision_id=revision_id,
        started_at=None,
        duration_ms=None,
    )
    revision = revision_type(
        id=revision_id,
        meeting_id=meeting.id,
        revision_number=1,
        raw_text="RAW_REVISION_NOT_A_REPLACEMENT_FOR_ORIGINAL_SEGMENTS",
        presentation_text="CLEANED_PRESENTATION_NOT_ORIGINAL_EVIDENCE",
        provider="synthetic",
        model="synthetic",
        origin="synthetic-fixture",
        content_hash="a" * 64,
    )
    segment = segment_type(
        id="00000000-0000-0000-0000-000000000003",
        revision_id=revision_id,
        ordinal=0,
        text=text if text is not None else "<b>ORIGINAL_SEGMENT & source evidence</b>",
        timing_source=timing_source,
        start_ms=start_ms,
        end_ms=end_ms,
    )
    return detail_type(
        meeting=meeting,
        audio_assets=(),
        revisions=(revision,),
        segments=(segment,),
        revision_audio_assets=(),
        warnings=(),
    )


def _load_listing(controller, view, qtbot, *meetings):
    controller.list_handler = lambda query: _page(controller.dto, *meetings)
    search = _named(view, QLineEdit, r"busca")
    search.setText("missing")
    qtbot.waitUntil(lambda: "No hay coincidencias" in _text(view))
    search.clear()
    qtbot.waitUntil(lambda: meetings[0].title in _text(view))


def _select_first_meeting(view, qtbot):
    results = _results(view)
    index = results.model().index(0, 0)
    results.setCurrentIndex(index)
    qtbot.mouseDClick(
        results.viewport(),
        Qt.MouseButton.LeftButton,
        pos=results.visualRect(index).center(),
    )


def _open_detail(library, qtbot, detail):
    desktop, controller, view = library
    controller.detail_handler = lambda meeting_id: detail
    _load_listing(controller, view, qtbot, detail.meeting)
    _select_first_meeting(view, qtbot)
    qtbot.waitUntil(lambda: detail.segments[0].text in _text(view))
    qtbot.waitUntil(lambda: not desktop.jobs.busy)
    return view


def _transcript(view):
    # The required original transcript may be a text widget or segment item view.
    # Its accessible role distinguishes it from the meeting results and editor.
    matches = [
        control
        for control in view.findChildren(QWidget)
        if isinstance(control, (QTextEdit, QPlainTextEdit, QAbstractItemView))
        and control.isVisibleTo(view)
        and re.search(r"(transcrip|segment)", control.accessibleName(), re.IGNORECASE)
    ]
    assert len(matches) == 1, "Original transcript needs a unique accessible reading control"
    return matches[0]


def test_detail_renders_original_literal_segments_instead_of_cleaned_presentation(library, qtbot):
    detail = _detail()
    view = _open_detail(library, qtbot, detail)
    text = _text(view)
    assert "<b>ORIGINAL_SEGMENT & source evidence</b>" in text
    assert "CLEANED_PRESENTATION_NOT_ORIGINAL_EVIDENCE" not in text
    for label in view.findChildren(QLabel):
        if "ORIGINAL_SEGMENT" in label.text():
            assert label.textFormat() == Qt.TextFormat.PlainText


@pytest.mark.parametrize("measured", [True, False], ids=["measured-zero", "absent"])
def test_detail_distinguishes_measured_zero_from_absent_segment_timing(library, qtbot, measured):
    detail = _detail(
        start_ms=0 if measured else None,
        end_ms=1500 if measured else None,
        timing_source="provider" if measured else "absent",
        text="TIMING_EVIDENCE_SEGMENT",
    )
    view = _open_detail(library, qtbot, detail)
    # No meeting date/duration is present in this fixture. A zero-time indicator
    # therefore belongs to the measured segment, never a fabricated source date.
    has_zero_time = re.search(r"(?<!\d)0{1,2}:00(?!\d)", _text(view)) is not None
    assert has_zero_time is measured


def test_delayed_successful_detail_cannot_reappear_after_close(library, qtbot):
    desktop, controller, view = library
    entered, release = threading.Event(), threading.Event()
    detail = _detail(text="LATE_SUCCESSFUL_DETAIL_MUST_STAY_CLOSED")

    def delayed_detail(meeting_id):
        entered.set()
        assert release.wait(5), "Test did not release the detail worker"
        return detail

    controller.detail_handler = delayed_detail
    try:
        _load_listing(controller, view, qtbot, detail.meeting)
        _select_first_meeting(view, qtbot)
        qtbot.waitUntil(entered.is_set)
        qtbot.mouseClick(_named(view, QPushButton, r"cerrar.*detalle"), Qt.MouseButton.LeftButton)
    finally:
        release.set()
        qtbot.waitUntil(lambda: not desktop.jobs.busy)
    assert "LATE_SUCCESSFUL_DETAIL_MUST_STAY_CLOSED" not in _text(view)
    assert "Planning" in _text(view)
    assert controller.core.command_log == []


def test_keyboard_continues_from_results_to_open_detail_and_title_editor(library, qtbot, qapp):
    view = _open_detail(library, qtbot, _detail())
    transcript = _transcript(view)
    editor = _named(view, QLineEdit, r"t[ií]tulo")
    assert editor.text() == "Planning"
    results = _results(view)
    results.setFocus()
    qtbot.waitUntil(results.hasFocus)
    visited = []
    for _ in range(40):
        focused = qapp.focusWidget()
        assert focused is not None
        qtbot.keyClick(focused, Qt.Key.Key_Tab)
        focused = qapp.focusWidget()
        for control in (transcript, editor):
            if focused is control or (focused is not None and control.isAncestorOf(focused)):
                if control not in visited:
                    visited.append(control)
        if editor in visited:
            break
    assert visited == [transcript, editor], (
        "Original detail and title edit must be keyboard reachable"
    )


@pytest.mark.parametrize("dark", [False, True], ids=["light", "dark"])
def test_open_detail_controls_have_accessible_names_and_contrast(library, qtbot, dark):
    view = _open_detail(library, qtbot, _detail())
    transcript = _transcript(view)
    editor = _named(view, QLineEdit, r"t[ií]tulo")
    palette = QPalette(view.palette())
    background, foreground = ("#202020", "#ffffff") if dark else ("#ffffff", "#202020")
    for role in (QPalette.ColorRole.Window, QPalette.ColorRole.Base, QPalette.ColorRole.Button):
        palette.setColor(role, QColor(background))
    for role in (
        QPalette.ColorRole.WindowText,
        QPalette.ColorRole.Text,
        QPalette.ColorRole.ButtonText,
    ):
        palette.setColor(role, QColor(foreground))
    view.setPalette(palette)
    controls = [
        control
        for control in view.findChildren(QWidget)
        if isinstance(
            control,
            (
                QLineEdit,
                QPushButton,
                QComboBox,
                QAbstractSpinBox,
                QAbstractItemView,
                QTextEdit,
                QPlainTextEdit,
            ),
        )
        and control.isVisibleTo(view)
        and control.focusPolicy() != Qt.FocusPolicy.NoFocus
        and not isinstance(control.parentWidget(), (QComboBox, QAbstractSpinBox))
    ]
    assert transcript in controls
    assert editor in controls
    for control in controls:
        assert control.accessibleName().strip(), f"Unnamed control: {type(control).__name__}"
        actual = control.palette()
        role = (
            QPalette.ColorRole.ButtonText
            if isinstance(control, QPushButton)
            else control.foregroundRole()
        )
        background_role = (
            QPalette.ColorRole.Button
            if isinstance(control, QPushButton)
            else control.backgroundRole()
        )
        values = sorted(
            (_luminance(actual.color(role)), _luminance(actual.color(background_role)))
        )
        assert (values[1] + 0.05) / (values[0] + 0.05) >= 4.5


def test_status_filter_transports_canonical_status_and_changes_visible_listing(library, qtbot):
    _, controller, view = library
    completed = _meeting(title="Completed visible meeting")
    failed = replace(
        _meeting(
            title="Failed visible meeting",
            meeting_id="00000000-0000-0000-0000-000000000006",
        ),
        status=MeetingStatus.FAILED,
    )
    _load_listing(controller, view, qtbot, completed, failed)
    assert "Failed visible meeting" in _text(view)

    def filtered_page(query):
        assert all(isinstance(value, MeetingStatus) for value in query.statuses)
        if query.statuses == (MeetingStatus.FAILED,):
            return _page(controller.dto, failed)
        return _page(controller.dto, completed, failed)

    controller.list_handler = filtered_page
    status = _named(view, QComboBox, r"estado")
    failed_options = [
        index
        for index in range(status.count())
        if re.search(r"(fall|error)", status.itemText(index), re.IGNORECASE)
    ]
    assert len(failed_options) == 1, "The status filter must offer failed meetings"
    status.setCurrentIndex(failed_options[0])
    qtbot.waitUntil(lambda: "Completed visible meeting" not in _text(view))
    assert "Failed visible meeting" in _text(view)
