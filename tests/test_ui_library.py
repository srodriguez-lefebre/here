"""M3 library behavior through the real desktop and real BackgroundJobs.

Independent widget tests use synthetic controller operations; R48 consumer tests
use real SQLite, services, controller and adapter. Canonical DTOs are resolved in
fixtures so missing features fail meaningfully without aborting collection.
Runtime evidence is recorded separately from this test preparation.
"""

from __future__ import annotations

import importlib
import re
import threading
from dataclasses import replace
from time import perf_counter

import pytest
from here.application import ApplicationEvent, ApplicationState, EventKind
from here.memory.database import Busy, LibraryError
from here.memory.models import Meeting, MeetingStatus
from here.ui.app import HereDesktop
from here.ui.contract import ApplicationUiAdapter
from here.ui.fake_controller import FakeApplicationController
from PySide6.QtCore import QModelIndex, QSettings, Qt, QTimer
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractSpinBox,
    QComboBox,
    QLabel,
    QLineEdit,
    QListWidget,
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
        participant_type = _feature("here.memory.models", "Participant")
        self.participant_options = (
            participant_type(
                id="00000000-0000-0000-0000-000000000010", display_name="Alice Synthetic"
            ),
            participant_type(
                id="00000000-0000-0000-0000-000000000011", display_name="Bob Synthetic"
            ),
        )

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

    def list_participants(self, meeting_id=None):
        # Explicit R48 read contract. This does not model persistence/relations.
        self.calls.append(("participants", meeting_id, threading.get_ident()))
        return self.participant_options

    def update_meeting(self, meeting_id, patch, expected_version):
        raise AssertionError("Real title-edit acceptance waits for the accepted B4 adapter")


def _named(view, widget_type, pattern):
    matches = [
        widget
        for widget in view.findChildren(widget_type)
        if widget.isVisibleTo(view) and re.search(pattern, widget.accessibleName(), re.IGNORECASE)
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
        if isinstance(widget, QListWidget):
            parts += [widget.item(row).text() for row in range(widget.count())]
            continue
        model = widget.model()
        if model is not None:
            parent = QModelIndex()
            parts += [
                str(model.index(row, column, parent).data() or "")
                for row in range(model.rowCount(parent))
                for column in range(model.columnCount(parent))
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
        position = results.visualRect(index).center()
        qtbot.mouseClick(results.viewport(), Qt.MouseButton.LeftButton, pos=position)
        qtbot.mouseDClick(
            results.viewport(),
            Qt.MouseButton.LeftButton,
            pos=position,
        )
        qtbot.mouseRelease(results.viewport(), Qt.MouseButton.LeftButton, pos=position)
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
        values = sorted((_luminance(actual.color(role)), _luminance(actual.color(background_role))))
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
    position = results.visualRect(index).center()
    qtbot.mouseClick(results.viewport(), Qt.MouseButton.LeftButton, pos=position)
    qtbot.mouseDClick(
        results.viewport(),
        Qt.MouseButton.LeftButton,
        pos=position,
    )
    qtbot.mouseRelease(results.viewport(), Qt.MouseButton.LeftButton, pos=position)


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
        values = sorted((_luminance(actual.color(role)), _luminance(actual.color(background_role))))
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
    qtbot.waitUntil(lambda: failed.title in _text(view) and completed.title not in _text(view))
    assert "Failed visible meeting" in _text(view)


def _option_index(selector, pattern):
    matches = [
        index
        for index in range(selector.count())
        if re.search(pattern, selector.itemText(index), re.IGNORECASE)
    ]
    assert len(matches) == 1, f"Expected one visible filter option: {pattern}"
    return matches[0]


def test_participant_choices_load_off_thread_and_id_selection_changes_visible_listing(
    library, qtbot
):
    _, controller, view = library
    selected = _meeting(title="Alice assigned meeting")
    other = _meeting(
        title="Other participant meeting",
        meeting_id="00000000-0000-0000-0000-000000000012",
    )
    _load_listing(controller, view, qtbot, selected, other)
    selector = _named(view, QComboBox, r"participante")
    alice_index = _option_index(selector, r"Alice Synthetic")
    _option_index(selector, r"Bob Synthetic")
    loads = [call for call in controller.calls if call[0] == "participants"]
    assert loads, "Participant names must be loaded through the declared read contract"
    assert all(meeting_id is None for _, meeting_id, _ in loads)
    assert all(thread_id != threading.get_ident() for _, _, thread_id in loads)

    def filtered_page(query):
        if query.participant_id == "00000000-0000-0000-0000-000000000010":
            return _page(controller.dto, selected)
        return _page(controller.dto, selected, other)

    controller.list_handler = filtered_page
    selector.setCurrentIndex(alice_index)
    qtbot.waitUntil(lambda: "Other participant meeting" not in _text(view))
    qtbot.waitUntil(lambda: selected.title in _text(view) and other.title not in _text(view))
    assert "Alice assigned meeting" in _text(view)


def test_no_matches_clear_action_resets_text_status_participant_and_restores_listing(
    library, qtbot
):
    desktop, controller, view = library
    restored = _meeting(title="Unfiltered library restored")

    def unfiltered_page(query):
        if (
            query.text == ""
            and query.statuses == ()
            and query.participant_id is None
            and query.from_utc is None
            and query.to_utc is None
        ):
            return _page(controller.dto, restored)
        return _page(controller.dto)

    controller.list_handler = unfiltered_page
    search = _named(view, QLineEdit, r"busca")
    status = _named(view, QComboBox, r"estado")
    participant = _named(view, QComboBox, r"participante")
    default_status, default_participant = status.currentIndex(), participant.currentIndex()
    search.setText("prime")
    qtbot.waitUntil(lambda: "No hay coincidencias" in _text(view))
    search.clear()
    qtbot.waitUntil(lambda: "Unfiltered library restored" in _text(view))

    status.setCurrentIndex(_option_index(status, r"(fall|error)"))
    participant.setCurrentIndex(_option_index(participant, r"Alice Synthetic"))
    search.setText("missing")
    qtbot.waitUntil(
        lambda: any(
            kind == "search" and query.text == "missing" for kind, query, _ in controller.calls
        )
    )
    qtbot.waitUntil(lambda: not desktop.jobs.busy)
    assert "No hay coincidencias" in _text(view)
    assert "Unfiltered library restored" not in _text(view)
    qtbot.mouseClick(_named(view, QPushButton, r"limpiar"), Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: "Unfiltered library restored" in _text(view))
    assert search.text() == ""
    assert status.currentIndex() == default_status
    assert participant.currentIndex() == default_participant
    assert "No hay coincidencias" not in _text(view)


@pytest.fixture
def real_catalog_library(tmp_path, monkeypatch, qtbot, qapp):
    """Real R48 reads over one disposable catalog; no service/adapter read doubles."""
    library_type = _feature("here.ui.library", "LibraryView")
    open_library = _feature("here.memory.database", "open_library")
    migrate = _feature("here.memory.database", "migrate")
    repository_type = _feature("here.memory.repository", "MemoryRepository")
    revision_type = _feature("here.memory.models", "TranscriptRevision")
    segment_type = _feature("here.memory.models", "Segment")
    participant_type = _feature("here.memory.models", "Participant")
    link_type = _feature("here.memory.models", "MeetingParticipant")
    assignment_type = _feature("here.memory.models", "SpeakerAssignment")
    custom_root, default_root = tmp_path / "injected-catalog", tmp_path / "unused-default"
    default_root.mkdir()
    monkeypatch.setenv("HERE_DATA_DIR", str(default_root))
    meeting = _meeting(title="REAL_R48_CUSTOM_ROOT_MEETING")
    revision = revision_type(
        id="00000000-0000-0000-0000-000000000020",
        meeting_id=meeting.id,
        revision_number=1,
        raw_text="FIRST_REQUEST_ORIGINAL\nSECOND_REQUEST_ORIGINAL",
        provider="synthetic",
        model="synthetic",
        origin="synthetic-sql-fixture",
        content_hash="e" * 64,
    )
    segments = (
        segment_type(
            id="00000000-0000-0000-0000-000000000021",
            revision_id=revision.id,
            ordinal=0,
            text="FIRST_REQUEST_ORIGINAL",
            start_ms=0,
            end_ms=1000,
            timing_source="provider",
            speaker_key='["request:one", "speaker_0"]',
        ),
        segment_type(
            id="00000000-0000-0000-0000-000000000022",
            revision_id=revision.id,
            ordinal=1,
            text="SECOND_REQUEST_ORIGINAL",
            timing_source="absent",
            speaker_key='["request:two", "speaker_0"]',
        ),
    )
    participants = (
        participant_type("00000000-0000-0000-0000-000000000023", "Alice R48 Explicit"),
        participant_type("00000000-0000-0000-0000-000000000024", "Bob R48 Roster Only"),
    )
    assignment = assignment_type(
        meeting.id, revision.id, segments[0].speaker_key, participants[0].id
    )
    library = open_library(custom_root)
    try:
        migrate(library)
        repository = repository_type(library)
        with repository.transaction():
            repository.insert_meeting(meeting)
            repository.insert_revision(revision)
            repository.insert_segments(segments)
            repository.activate_revision(meeting.id, revision.id)
            for participant in participants:
                link = link_type(meeting.id, participant.id)
                library.execute(
                    "INSERT INTO participants(id,display_name) VALUES (?,?)",
                    (participant.id, participant.display_name),
                )
                library.execute(
                    "INSERT INTO meeting_participants(meeting_id,participant_id,role) "
                    "VALUES (?,?,?)",
                    (link.meeting_id, link.participant_id, link.role),
                )
            library.execute(
                "INSERT INTO speaker_assignments"
                "(meeting_id,revision_id,speaker_key,participant_id) VALUES (?,?,?,?)",
                (
                    assignment.meeting_id,
                    assignment.revision_id,
                    assignment.speaker_key,
                    assignment.participant_id,
                ),
            )
    finally:
        library.close()
    memory = _feature("here.memory.service", "MemoryService")(custom_root)
    controller = _feature("here.application", "HereApplicationController")(memory_service=memory)
    # No memory_service/search_service override: exercise the real same-instance
    # getter and the adapter's own query/read routing from the injected controller.
    adapter = ApplicationUiAdapter(controller, tmp_path / "empty-sessions")
    settings = QSettings(str(tmp_path / "real-catalog.ini"), QSettings.Format.IniFormat)
    desktop = HereDesktop(qapp, adapter, settings=settings)
    desktop.show()
    tabs = desktop.main_window.findChildren(QTabWidget)
    assert len(tabs) == 1
    indexes = [index for index in range(tabs[0].count()) if tabs[0].tabText(index) == "Biblioteca"]
    assert len(indexes) == 1
    tabs[0].setCurrentIndex(indexes[0])
    view = desktop.main_window.findChild(library_type)
    assert view is not None
    qtbot.waitUntil(lambda: meeting.title in _text(view))
    qtbot.waitUntil(lambda: not desktop.jobs.busy)
    return desktop, view, meeting, revision, segments, participants, default_root


def test_real_r48_custom_catalog_meeting_and_detail_participants_use_injected_backend(
    real_catalog_library, qtbot
):
    desktop, view, meeting, _, segments, participants, default_root = real_catalog_library
    assert meeting.title in _text(view)
    selector = _named(view, QComboBox, r"participante")
    _option_index(selector, r"Alice R48 Explicit")
    _option_index(selector, r"Bob R48 Roster Only")
    assert participants[1].display_name not in _text(view)
    _select_first_meeting(view, qtbot)
    qtbot.waitUntil(lambda: segments[0].text in _text(view))
    qtbot.waitUntil(lambda: not desktop.jobs.busy)
    assert participants[0].display_name in _text(view)
    # Bob has no speaker assignment: his name proves detail participants are
    # consumed, rather than passing through a mapped transcript speaker alone.
    assert participants[1].display_name in _text(view)
    assert not (default_root / "library.sqlite3").exists()


def test_real_r48_scoped_assignments_control_visible_speakers_and_passive_segment_targets(
    real_catalog_library, qtbot
):
    desktop, view, meeting, revision, segments, participants, default_root = real_catalog_library
    target_type = _feature("here.memory.models", "CitationTarget")
    targets = []
    view.segment_selected.connect(targets.append)
    _select_first_meeting(view, qtbot)
    qtbot.waitUntil(lambda: segments[1].text in _text(view))
    qtbot.waitUntil(lambda: not desktop.jobs.busy)
    transcript = _transcript(view)
    model = transcript.model()
    assert model.rowCount() == 2
    assert participants[0].display_name in str(model.index(0, 0).data())
    assert participants[0].display_name not in str(model.index(1, 0).data())
    for row in (0, 1):
        transcript.setCurrentIndex(model.index(row, 0))
        transcript.setFocus()
        qtbot.keyClick(transcript, Qt.Key.Key_Return)
        qtbot.waitUntil(lambda: len(targets) == row + 1)
    assert all(isinstance(target, target_type) for target in targets)
    assert targets[0].speaker == "Alice R48 Explicit"
    assert targets[1].speaker != "Alice R48 Explicit"
    assert [target.meeting_id for target in targets] == [meeting.id, meeting.id]
    assert [target.revision_id for target in targets] == [revision.id, revision.id]
    assert [target.segment_id for target in targets] == [segment.id for segment in segments]
    assert [target.text for target in targets] == [
        "FIRST_REQUEST_ORIGINAL",
        "SECOND_REQUEST_ORIGINAL",
    ]
    assert [(target.start_ms, target.end_ms, target.timing_source) for target in targets] == [
        (0, 1000, "provider"),
        (None, None, "absent"),
    ]
    assert all(target.audio_path is None and target.status == "current" for target in targets)
    assert not (default_root / "library.sqlite3").exists()


def _publish_memory_ingested(controller, meeting_id):
    controller.core.publish(
        ApplicationEvent(
            kind=EventKind.MEMORY_INGESTED,
            state=controller.core.snapshot.state,
            details={"status": "inserted", "meeting_id": meeting_id},
        )
    )


def test_recording_ingestion_refreshes_library_on_return_without_query_edit(library, qtbot):
    desktop, controller, view = library
    qtbot.waitUntil(lambda: "Aún no hay reuniones" in _text(view))
    search = _named(view, QLineEdit, r"busca")
    qtbot.mouseClick(
        _named(view, QPushButton, r"(iniciar|nueva|grabar).*grabaci|grabar"),
        Qt.MouseButton.LeftButton,
    )
    tabs = desktop.main_window.findChild(QTabWidget)
    assert tabs is not None
    assert tabs.tabText(tabs.currentIndex()) == "Captura"
    controller.core.set_state(ApplicationState.PROCESSING)
    meeting = _meeting(title="INGESTED_LIBRARY_MEETING")
    controller.list_handler = lambda query: _page(controller.dto, meeting)
    _publish_memory_ingested(controller, meeting.id)
    controller.core.complete()
    controller.core.publish(
        ApplicationEvent(kind=EventKind.WORKER_COMPLETED, state=ApplicationState.COMPLETED)
    )
    library_index = next(index for index in range(tabs.count()) if tabs.widget(index) is view)
    tabs.setCurrentIndex(library_index)
    qtbot.waitUntil(lambda: "INGESTED_LIBRARY_MEETING" in _text(view))
    assert search.text() == ""
    assert "Aún no hay reuniones" not in _text(view)
    assert controller.core.command_log == ["start"]


def test_ingestion_invalidates_an_older_inflight_query_without_changing_search(library, qtbot):
    desktop, controller, view = library
    entered, release = threading.Event(), threading.Event()

    def before_ingest(query):
        entered.set()
        assert release.wait(10), "Test did not release the pre-ingestion query"
        return _hits(controller.dto, "STALE_BEFORE_INGESTION")

    controller.search_handler = before_ingest
    search = _named(view, QLineEdit, r"busca")
    try:
        search.setText("evidence")
        qtbot.waitUntil(entered.is_set)
        controller.search_handler = lambda query: _hits(controller.dto, "AFTER_INGESTION")
        _publish_memory_ingested(controller, "00000000-0000-0000-0000-000000000030")
        qtbot.waitUntil(lambda: "AFTER_INGESTION" in _text(view))
    finally:
        release.set()
        qtbot.waitUntil(lambda: not desktop.jobs.busy)
    assert "AFTER_INGESTION" in _text(view)
    assert "STALE_BEFORE_INGESTION" not in _text(view)
    assert search.text() == "evidence"
    assert controller.core.command_log == []


def test_participant_busy_stays_visible_and_retry_recovers_replaceable_options(
    tmp_path, dto, qapp, qtbot
):
    library_type = _feature("here.ui.library", "LibraryView")
    controller = LibraryController(tmp_path, dto)
    read_participants = controller.list_participants
    attempts = 0
    release_page = threading.Event()
    meeting = _meeting(title="PAGE_SUCCESS_DURING_PARTICIPANT_BUSY")

    def transient_participants(meeting_id=None):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            controller.calls.append(("participants", meeting_id, threading.get_ident()))
            raise Busy("SYNTHETIC_PARTICIPANTS_BUSY")
        return read_participants(meeting_id)

    def list_page(query):
        assert release_page.wait(10), "Test did not release the successful page worker"
        return _page(dto, meeting)

    controller.list_participants = transient_participants
    controller.list_handler = list_page
    desktop = HereDesktop(
        qapp,
        controller,
        settings=QSettings(str(tmp_path / "participant-busy.ini"), QSettings.Format.IniFormat),
    )
    try:
        desktop.show()
        tabs = desktop.main_window.findChild(QTabWidget)
        assert tabs is not None
        view = desktop.main_window.findChild(library_type)
        assert view is not None
        library_index = next(index for index in range(tabs.count()) if tabs.widget(index) is view)
        tabs.setCurrentIndex(library_index)
        qtbot.waitUntil(lambda: "SYNTHETIC_PARTICIPANTS_BUSY" in _text(view))
        release_page.set()
        qtbot.waitUntil(lambda: "PAGE_SUCCESS_DURING_PARTICIPANT_BUSY" in _text(view))
        qtbot.waitUntil(lambda: not desktop.jobs.busy)
        assert "SYNTHETIC_PARTICIPANTS_BUSY" in _text(view)
        qtbot.mouseClick(
            _named(view, QPushButton, r"(reintentar|recargar)"), Qt.MouseButton.LeftButton
        )
        selector = _named(view, QComboBox, r"participante")
        qtbot.waitUntil(
            lambda: any("Alice Synthetic" in selector.itemText(i) for i in range(selector.count()))
        )
        qtbot.waitUntil(lambda: not desktop.jobs.busy)
        assert "SYNTHETIC_PARTICIPANTS_BUSY" not in _text(view)
        selector.setCurrentIndex(_option_index(selector, r"Alice Synthetic"))
        qtbot.waitUntil(
            lambda: any(
                kind == "list" and query.participant_id == controller.participant_options[0].id
                for kind, query, _ in controller.calls
            )
        )
        qtbot.waitUntil(lambda: not desktop.jobs.busy)
        for _ in range(2):
            previous_attempts = attempts
            _publish_memory_ingested(controller, meeting.id)
            qtbot.waitUntil(lambda: attempts > previous_attempts)
            qtbot.waitUntil(lambda: not desktop.jobs.busy)
            assert selector.currentData() == "00000000-0000-0000-0000-000000000010"
            options = [selector.itemText(i) for i in range(selector.count())]
            assert options.count("Alice Synthetic") == 1
            assert options.count("Bob Synthetic") == 1
    finally:
        release_page.set()
        qtbot.waitUntil(lambda: not desktop.jobs.busy)


def test_slow_library_query_disables_capture_actions_until_shared_worker_finishes(library, qtbot):
    desktop, controller, view = library
    entered, release = threading.Event(), threading.Event()

    def slow_query(query):
        entered.set()
        assert release.wait(10), "Test did not release the shared library job"
        return _hits(controller.dto, "FINISHED_SHARED_LIBRARY_JOB")

    controller.search_handler = slow_query
    controls = [
        desktop.main_window.findChild(QPushButton, name)
        for name in ("startButton", "devicesButton", "testMicrophoneButton", "testSystemButton")
    ]
    assert all(control is not None and control.isEnabled() for control in controls)
    try:
        _named(view, QLineEdit, r"busca").setText("slow")
        qtbot.waitUntil(entered.is_set)
        tabs = desktop.main_window.findChild(QTabWidget)
        assert tabs is not None
        capture_index = next(
            index for index in range(tabs.count()) if tabs.tabText(index) == "Captura"
        )
        tabs.setCurrentIndex(capture_index)
        assert all(not control.isEnabled() for control in controls)
    finally:
        release.set()
        qtbot.waitUntil(lambda: not desktop.jobs.busy)
    assert all(control.isEnabled() for control in controls)
    assert controller.core.command_log == []


def _hit_identities(page):
    return {(hit.meeting_id, hit.revision_id, hit.segment_id) for hit in page.items}


@pytest.fixture
def real_title_library(tmp_path, monkeypatch, qapp, qtbot):
    """Publish with accepted A ingest so title tests exercise actual complete FTS."""
    from memory_fixtures import MEETING_ID, make_session

    library_type = _feature("here.ui.library", "LibraryView")
    search_query = _feature("here.memory.models", "SearchQuery")
    default_root = tmp_path / "unused-title-default"
    default_root.mkdir()
    monkeypatch.setenv("HERE_DATA_DIR", str(default_root))
    memory = _feature("here.memory.service", "MemoryService")(tmp_path / "title-catalog")
    source = make_session(tmp_path / "oldcinnabar", meeting_id=MEETING_ID)
    publication = memory.ingest_session(source.directory)
    assert publication.status == "inserted"
    assert publication.meeting_id is not None
    controller = _feature("here.application", "HereApplicationController")(memory_service=memory)
    adapter = ApplicationUiAdapter(controller, tmp_path / "empty-title-sessions")
    before = adapter.get_meeting(publication.meeting_id)
    assert before.meeting.title.strip()
    assert len(before.segments) == 2
    expected_hits = {
        (before.meeting.id, segment.revision_id, segment.id) for segment in before.segments
    }
    original_page = adapter.search(search_query(text=before.meeting.title))
    assert not original_page.degraded, "Title integration requires the real complete FTS route"
    assert _hit_identities(original_page) == expected_hits
    desktop = HereDesktop(
        qapp,
        adapter,
        settings=QSettings(str(tmp_path / "title-ui.ini"), QSettings.Format.IniFormat),
    )
    desktop.show()
    tabs = desktop.main_window.findChild(QTabWidget)
    assert tabs is not None
    view = desktop.main_window.findChild(library_type)
    assert view is not None
    library_index = next(index for index in range(tabs.count()) if tabs.widget(index) is view)
    tabs.setCurrentIndex(library_index)
    qtbot.waitUntil(lambda: before.meeting.title in _text(view))
    qtbot.waitUntil(lambda: not desktop.jobs.busy)
    _select_first_meeting(view, qtbot)
    qtbot.waitUntil(lambda: before.segments[0].text in _text(view))
    qtbot.waitUntil(lambda: not desktop.jobs.busy)
    return desktop, view, before, expected_hits, default_root


def test_real_title_editor_current_version_updates_catalog_fts_and_keeps_original_evidence(
    real_title_library, qtbot
):
    desktop, view, before, expected_hits, default_root = real_title_library
    adapter = desktop.controller
    search_query = _feature("here.memory.models", "SearchQuery")
    editor = _named(view, QLineEdit, r"t[ií]tulo")
    assert editor.text() == before.meeting.title
    editor.selectAll()
    qtbot.keyClicks(editor, "Published saffron agenda")
    qtbot.mouseClick(_named(view, QPushButton, r"guardar.*t[ií]tulo"), Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: "Published saffron agenda" in _text(view))
    qtbot.waitUntil(lambda: not desktop.jobs.busy)
    after = adapter.get_meeting(before.meeting.id)
    assert after.meeting.title == "Published saffron agenda"
    assert after.meeting.row_version == before.meeting.row_version + 1
    assert after.meeting.active_revision_id == before.meeting.active_revision_id
    assert after.revisions == before.revisions
    assert after.segments == before.segments
    updated_page = adapter.search(search_query(text="saffron"))
    assert not updated_page.degraded
    assert _hit_identities(updated_page) == expected_hits
    old_page = adapter.search(search_query(text=before.meeting.title))
    assert not old_page.degraded
    assert old_page.items == ()
    assert not (default_root / "library.sqlite3").exists()


def test_real_title_editor_stale_version_shows_conflict_without_overwrite_or_retry(
    real_title_library, qtbot
):
    desktop, view, before, expected_hits, default_root = real_title_library
    adapter = desktop.controller
    patch = _feature("here.memory.models", "MeetingPatch")
    search_query = _feature("here.memory.models", "SearchQuery")
    editor = _named(view, QLineEdit, r"t[ií]tulo")
    assert editor.text() == before.meeting.title
    concurrent = adapter.update_meeting(
        before.meeting.id,
        patch(title="Concurrent coriander agenda"),
        before.meeting.row_version,
    )
    assert concurrent.meeting.row_version == before.meeting.row_version + 1
    # No application event is published for this independent writer: the editor
    # must submit the version it actually displayed and handle the real Conflict.
    assert editor.text() == before.meeting.title
    editor.selectAll()
    qtbot.keyClicks(editor, "Attempted tamarind agenda")
    qtbot.mouseClick(_named(view, QPushButton, r"guardar.*t[ií]tulo"), Qt.MouseButton.LeftButton)
    qtbot.waitUntil(
        lambda: (
            editor.text() == "Concurrent coriander agenda"
            and re.search(r"(cambi[oó]|conflicto|versi[oó]n actual)", _text(view), re.IGNORECASE)
            is not None
        )
    )
    qtbot.waitUntil(lambda: not desktop.jobs.busy)
    assert "Concurrent coriander agenda" in _text(view)
    after = adapter.get_meeting(before.meeting.id)
    assert after.meeting.title == "Concurrent coriander agenda"
    assert after.meeting.row_version == concurrent.meeting.row_version
    assert after.meeting.active_revision_id == before.meeting.active_revision_id
    assert after.revisions == before.revisions
    assert after.segments == before.segments
    current_page = adapter.search(search_query(text="coriander"))
    assert not current_page.degraded
    assert _hit_identities(current_page) == expected_hits
    attempted_page = adapter.search(search_query(text="tamarind"))
    assert not attempted_page.degraded
    assert attempted_page.items == ()
    assert not (default_root / "library.sqlite3").exists()
