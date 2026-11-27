"""Exercise evidence navigation, player accessibility, and desktop deletion acknowledgement."""

from __future__ import annotations

import re
from dataclasses import replace
from threading import Event, get_ident

import pytest
import test_ui_library as library_tests
import test_ui_playback as playback_tests
from here.application import ApplicationState, HereApplicationController
from here.memory.models import Citation, CitationTarget, SearchHit, SearchPage
from here.memory.service import MemoryService
from here.ui.app import HereDesktop
from here.ui.contract import ApplicationUiAdapter
from here.ui.playback import PlaybackController
from here.ui.playback_widget import PlaybackWidget
from memory_fixtures import MEETING_ID, OTHER_MEETING_ID, make_session, write_marked_wav
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QAccessible, QColor, QPalette
from PySide6.QtWidgets import (
    QAbstractButton,
    QComboBox,
    QLabel,
    QLineEdit,
    QListWidget,
    QSlider,
    QWidget,
)
from test_ui_library import (
    _detail,
    _load_listing,
    _luminance,
    _named,
    _select_first_meeting,
    _text,
)
from test_ui_playback import (
    _player_for,
    _ready,
)

dto = library_tests.dto
navigation_library = library_tests.library
fake_media = playback_tests.playback

pytestmark = pytest.mark.usefixtures("owned_desktops")

PLAY_FROM = r"Reproducir desde 00:00"


@pytest.fixture
def evidence(navigation_library, tmp_path, monkeypatch):
    desktop, controller, _ = navigation_library
    desktop.main_window.resize(1050, 850)
    original = _detail(text="Selected source evidence")
    selected = replace(original.segments[0], ordinal=1)
    before = replace(
        selected,
        id="00000000-0000-0000-0000-000000000011",
        ordinal=0,
        text="Previous source evidence",
    )
    after = replace(
        selected,
        id="00000000-0000-0000-0000-000000000012",
        ordinal=2,
        text="Next source evidence",
    )
    detail = replace(original, segments=(before, selected, after))
    path = tmp_path / "selected.wav"
    write_marked_wav(path)
    target = CitationTarget(
        meeting_id=detail.meeting.id,
        revision_id=selected.revision_id,
        segment_id=selected.id,
        title=detail.meeting.title,
        text=selected.text,
        start_ms=selected.start_ms,
        end_ms=selected.end_ms,
        timing_source=selected.timing_source,
        audio_path=path,
    )
    requests = []

    def resolve(citation):
        requests.append((citation, get_ident()))
        assert isinstance(citation, Citation)
        assert (citation.meeting_id, citation.revision_id, citation.segment_id) == (
            target.meeting_id,
            target.revision_id,
            target.segment_id,
        )
        return target

    controller.detail_handler = lambda meeting_id: detail
    monkeypatch.setattr(controller, "resolve_citation", resolve, raising=False)
    return detail, target, requests


def _buttons(view, pattern):
    return [
        button
        for button in view.findChildren(QAbstractButton)
        if button.isVisibleTo(view)
        and re.search(pattern, f"{button.text()} {button.accessibleName()}", re.IGNORECASE)
    ]


def _transcript(view):
    return _named(view, QListWidget, r"transcrip|segment")


def _passive(target):
    return CitationTarget(
        meeting_id=target.meeting_id,
        revision_id=target.revision_id,
        segment_id=target.segment_id,
    )


@pytest.fixture
def ready_player(fake_media, navigation_library, evidence, qtbot):
    _, _, view = navigation_library
    _, target, _ = evidence
    _, players, actions = fake_media
    view.open_target(_passive(target))
    qtbot.waitUntil(lambda: bool(players))
    media = _player_for(players, target)
    _ready(media)
    widget = view.findChild(PlaybackWidget)
    assert widget is not None and widget.isVisibleTo(view)
    return view, widget, media, actions


def _accessible_name(control):
    interface = QAccessible.queryAccessibleInterface(control)
    assert interface is not None
    return interface.text(QAccessible.Text.Name)


def test_player_keyboard_reaches_and_operates_play_pause_seek_and_volume(ready_player, qtbot, qapp):
    view, widget, media, actions = ready_player
    (play,) = _buttons(widget, PLAY_FROM)
    seek = _named(widget, QSlider, r"posici[oó]n")
    volume = _named(widget, QSlider, r"volumen")
    speed = _named(widget, QComboBox, r"velocidad")
    transcript = _transcript(view)
    transcript.setFocus()
    qtbot.waitUntil(transcript.hasFocus)

    pending = {play, seek, volume, speed}
    path = media.source().toLocalFile()
    for _ in range(40):
        focused = qapp.focusWidget()
        assert focused is not None
        qtbot.keyClick(focused, Qt.Key.Key_Tab)
        focused = qapp.focusWidget()
        if focused not in pending:
            continue
        pending.remove(focused)
        if focused is play:
            actions.clear()
            qtbot.keyClick(play, Qt.Key.Key_Space)
            assert ("play", path) in actions
            assert re.search(r"pausar", _accessible_name(play), re.IGNORECASE)
            qtbot.keyClick(play, Qt.Key.Key_Space)
            assert [action for action in actions if action[0] in {"play", "pause"}] == [
                ("play", path),
                ("pause", path),
            ]
        elif focused is seek:
            expected = media.position() + seek.singleStep()
            actions.clear()
            qtbot.keyClick(seek, Qt.Key.Key_Right)
            assert ("seek", path, expected) in actions
            assert media.position() == expected
        elif focused is volume:
            actions.clear()
            qtbot.keyClick(volume, Qt.Key.Key_Left)
            assert ("volume", pytest.approx(0.99)) in actions
        elif focused is speed:
            actions.clear()
            qtbot.keyClick(speed, Qt.Key.Key_Down)
            assert ("rate", 1.25) in actions
        if not pending:
            break
    assert not pending, "Tab must reach play, position, volume and speed from the transcript"


@pytest.mark.parametrize("dark", [False, True], ids=["light", "dark"])
def test_player_accessible_names_and_palette_contrast(ready_player, dark):
    view, widget, _, _ = ready_player
    palette = QPalette(view.palette())
    background, foreground, accent = (
        ("#202020", "#ffffff", "#76baff") if dark else ("#ffffff", "#202020", "#0059b3")
    )
    roles = QPalette.ColorRole
    for role in (roles.Window, roles.Base, roles.Button):
        palette.setColor(role, QColor(background))
    for role in (roles.WindowText, roles.Text, roles.ButtonText):
        palette.setColor(role, QColor(foreground))
    palette.setColor(roles.Highlight, QColor(accent))
    view.setPalette(palette)

    controls = [
        control
        for control in widget.findChildren(QWidget)
        if isinstance(control, (QAbstractButton, QSlider, QComboBox))
        and control.isVisibleTo(widget)
    ]
    assert len(controls) == 4
    names = [_accessible_name(control).strip() for control in controls]
    assert all(names) and len(set(names)) == len(controls)
    (clock,) = [label for label in widget.findChildren(QLabel) if label.isVisibleTo(widget)]
    for control in [*controls, clock]:
        if isinstance(control, QSlider):
            text_role, background_role, minimum = roles.Highlight, roles.Window, 3.0
        elif isinstance(control, (QAbstractButton, QComboBox)):
            text_role, background_role, minimum = roles.ButtonText, roles.Button, 4.5
        else:
            text_role, background_role, minimum = roles.WindowText, roles.Window, 4.5
        actual = control.palette()
        low, high = sorted(
            (_luminance(actual.color(text_role)), _luminance(actual.color(background_role)))
        )
        assert (high + 0.05) / (low + 0.05) >= minimum, control.metaObject().className()


@pytest.mark.parametrize("route", ["list-segment", "search-hit", "external-citation"])
def test_navigation_routes_resolve_and_select_the_same_evidence(
    navigation_library, evidence, fake_media, qtbot, route
):
    _, controller, view = navigation_library
    detail, target, requests = evidence
    _, players, actions = fake_media
    qt_thread = get_ident()
    if route == "list-segment":
        _load_listing(controller, view, qtbot, detail.meeting)
        _select_first_meeting(view, qtbot)
        qtbot.waitUntil(lambda: target.text in _text(view))
        assert not requests and not players
        transcript = _transcript(view)
        item = next(
            transcript.item(index)
            for index in range(transcript.count())
            if target.text in transcript.item(index).text()
        )
        transcript.setCurrentItem(item)
        qtbot.keyClick(transcript, Qt.Key.Key_Return)
    elif route == "search-hit":
        snippet = "Search snippet carries no authority"
        controller.search_handler = lambda query: SearchPage(
            items=(SearchHit(target.meeting_id, target.revision_id, target.segment_id, snippet),)
        )
        _named(view, QLineEdit, r"busca").setText("evidence")
        qtbot.waitUntil(lambda: snippet in _text(view))
        _select_first_meeting(view, qtbot)
    else:
        # The external caller supplies only the IDs carried by a passive citation.
        view.open_target(_passive(target))

    qtbot.waitUntil(lambda: bool(requests) and bool(players))
    qtbot.waitUntil(lambda: target.text in _text(view))
    citation, worker_thread = requests[-1]
    assert citation == Citation(target.meeting_id, target.revision_id, target.segment_id)
    assert worker_thread != qt_thread
    transcript = _transcript(view)
    selected = transcript.currentItem()
    assert selected is not None and target.text in selected.text()
    row = transcript.row(selected)
    for offset, neighbor in ((-1, detail.segments[0]), (1, detail.segments[2])):
        item = transcript.item(row + offset)
        assert item is not None and neighbor.text in item.text()
        assert transcript.viewport().rect().intersects(transcript.visualItemRect(item))

    player = _player_for(players, target)
    _ready(player)
    assert not any(action[0] in {"play", "seek"} for action in actions)
    buttons = _buttons(view, PLAY_FROM)
    assert len(buttons) == 1 and buttons[0].isEnabled()
    qtbot.mouseClick(buttons[0], Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: ("seek", player.source().toLocalFile(), 0) in actions)


@pytest.mark.parametrize("case", ["missing-audio", "absent-timing"])
def test_unplayable_evidence_explains_why_without_offering_play_from_zero(
    navigation_library, evidence, fake_media, monkeypatch, qtbot, case
):
    _, controller, view = navigation_library
    _, target, _ = evidence
    _, _, actions = fake_media
    if case == "missing-audio":
        resolved = replace(target, status="missing", audio_path=None)
        message = r"(?:audio.*(?:no |falta|ausente)|(?:falta|ausente).*audio)"
    else:
        resolved = replace(target, start_ms=None, end_ms=None, timing_source="absent")
        message = r"Sin marca de tiempo"
    monkeypatch.setattr(controller, "resolve_citation", lambda citation: resolved)

    view.open_target(_passive(target))

    qtbot.waitUntil(
        lambda: any(
            label.isVisibleTo(view) and re.search(message, label.text(), re.IGNORECASE)
            for label in view.findChildren(QLabel)
        )
    )
    assert not any(button.isEnabled() for button in _buttons(view, r"Reproducir desde"))
    assert not any(action[0] in {"play", "seek"} for action in actions)


def test_historical_target_offers_a_route_back_to_the_current_revision(
    navigation_library, evidence, fake_media, monkeypatch, qtbot
):
    _, controller, view = navigation_library
    _, target, _ = evidence
    _, players, actions = fake_media
    active = _detail(text="Current source evidence")
    historical_revision = replace(
        active.revisions[0],
        id="00000000-0000-0000-0000-000000000021",
        revision_number=1,
    )
    old = replace(
        active.segments[0],
        id="00000000-0000-0000-0000-000000000022",
        revision_id=historical_revision.id,
        text="Historical source evidence",
    )
    target = replace(
        target,
        revision_id=historical_revision.id,
        segment_id=old.id,
        text=old.text,
        status="historical",
    )
    controller.detail_handler = lambda meeting_id: replace(
        active,
        revisions=(historical_revision, replace(active.revisions[0], revision_number=2)),
        segments=(old, *active.segments),
    )
    current = replace(
        target,
        revision_id=active.revisions[0].id,
        segment_id=active.segments[0].id,
        text=active.segments[0].text,
        status="current",
    )
    monkeypatch.setattr(
        controller,
        "resolve_citation",
        lambda citation: target if citation.revision_id == target.revision_id else current,
    )
    view.open_target(_passive(target))
    qtbot.waitUntil(lambda: bool(players) and old.text in _text(view))
    assert re.search(r"históric", _text(view), re.IGNORECASE)
    assert old.text in _transcript(view).currentItem().text()
    link_pattern = r"revisión actual"
    buttons = _buttons(view, link_pattern)
    links = [
        label
        for label in view.findChildren(QLabel)
        if label.isVisibleTo(view)
        and "<a " in label.text()
        and re.search(link_pattern, label.text(), re.IGNORECASE)
    ]
    assert len(buttons) + len(links) == 1
    if buttons:
        qtbot.mouseClick(buttons[0], Qt.MouseButton.LeftButton)
    else:
        href = re.search(r"""href=['"]([^'"]+)""", links[0].text())
        assert href is not None
        links[0].linkActivated.emit(href[1])

    def selected_current_revision():
        item = _transcript(view).currentItem()
        return item is not None and current.text in item.text()

    qtbot.waitUntil(selected_current_revision)
    assert not any(action[0] == "play" for action in actions)


@pytest.mark.parametrize(
    "state", [ApplicationState.PREPARING, ApplicationState.RECORDING, ApplicationState.PAUSED]
)
def test_capture_snapshot_disables_playback_controls(
    navigation_library, evidence, fake_media, qtbot, state
):
    desktop, _, view = navigation_library
    _, target, _ = evidence
    _, players, actions = fake_media
    view.open_target(_passive(target))
    qtbot.waitUntil(lambda: bool(players))
    _ready(_player_for(players, target))
    qtbot.waitUntil(lambda: bool(_buttons(view, PLAY_FROM)))
    (button,) = _buttons(view, PLAY_FROM)
    assert button.isEnabled()

    window = desktop.main_window
    window.set_snapshot(replace(window.snapshot, state=state))

    assert not button.isEnabled()
    assert re.search(
        r"grabaci|captura",
        "\n".join(label.text() for label in view.findChildren(QLabel) if label.isVisibleTo(view)),
        re.IGNORECASE,
    )
    assert not any(action[0] in {"play", "seek"} for action in actions)
    window.set_snapshot(replace(window.snapshot, state=ApplicationState.COMPLETED))
    assert button.isEnabled()


def test_opening_another_meeting_clears_the_previous_source(
    navigation_library, evidence, fake_media, qtbot
):
    _, controller, view = navigation_library
    detail, target, _ = evidence
    _, players, actions = fake_media
    view.open_target(_passive(target))
    qtbot.waitUntil(lambda: bool(players))
    player = _player_for(players, target)
    source = player.source().toLocalFile()

    other = replace(detail.meeting, id=OTHER_MEETING_ID, title="Other synthetic meeting")
    controller.detail_handler = lambda meeting_id: replace(detail, meeting=other, segments=())
    _load_listing(controller, view, qtbot, other)
    _select_first_meeting(view, qtbot)
    qtbot.waitUntil(lambda: ("source", None) in actions)
    assert actions.index(("stop", source)) < actions.index(("source", None))
    assert not any(action[0] == "play" for action in actions)


def test_closing_detail_discards_a_resolver_result_already_in_flight(
    navigation_library, evidence, fake_media, qtbot, monkeypatch
):
    desktop, controller, view = navigation_library
    _, target, _ = evidence
    _, players, actions = fake_media
    obtained, deliver = Event(), Event()

    def delayed_resolve(citation):
        obtained.set()
        assert deliver.wait(5), "Timed out waiting to release the synthetic resolver"
        return target

    monkeypatch.setattr(controller, "resolve_citation", delayed_resolve)
    try:
        view.open_target(_passive(target))
        qtbot.waitUntil(obtained.is_set)
        view.close_detail()
    finally:
        deliver.set()
        qtbot.waitUntil(lambda: not desktop.jobs.busy)

    assert not players
    assert not any(action[0] in {"play", "seek"} for action in actions)
    assert target.text not in _text(view)


@pytest.mark.parametrize("affected", [True, False], ids=["matching", "unrelated"])
def test_tracked_delete_ack_releases_only_the_affected_media(
    tmp_path, qapp, qtbot, monkeypatch, fake_media, affected
):
    sessions = tmp_path / "sessions"
    sessions.mkdir()
    memory = MemoryService(tmp_path / "catalog", sessions_root=sessions)
    session = make_session(tmp_path / "delete-ack", meeting_id=MEETING_ID)
    assert memory.ingest_session(session.directory).meeting_id == MEETING_ID
    unknown = memory.data_dir / "meetings" / MEETING_ID / "unknown.txt"
    unknown.write_bytes(b"retain unowned synthetic file")
    adapter = ApplicationUiAdapter(HereApplicationController(memory_service=memory), sessions)
    desktop = HereDesktop(
        qapp,
        adapter,
        settings=QSettings(str(tmp_path / "playback.ini"), QSettings.Format.IniFormat),
    )
    desktop.show()
    view = desktop.main_window.library
    desktop.main_window.tabs.setCurrentWidget(view)
    qtbot.waitUntil(lambda: not desktop.jobs.busy)
    controller = view.findChild(PlaybackController)
    assert controller is not None, "The desktop library must own a playback controller"
    _, players, actions = fake_media
    path = tmp_path / "ack-media.wav"
    write_marked_wav(path)
    target = CitationTarget(
        meeting_id=MEETING_ID if affected else OTHER_MEETING_ID,
        revision_id="33333333-3333-4333-8333-333333333333",
        segment_id="44444444-4444-4444-8444-444444444444",
        start_ms=1250,
        timing_source="provider",
        audio_path=path,
    )
    controller.load(target)
    player = _player_for(players, target)
    source = player.source().toLocalFile()
    if affected:
        controller.seek_ms(1250)
    else:
        _ready(player)
        controller.play()

    # Keep the retired fake alive long enough to deliver a late backend event.
    delete_later = player.deleteLater
    monkeypatch.setattr(player, "deleteLater", lambda: None)
    observed = []

    def observe_after_qt_ack(meeting_id):
        observed.append((meeting_id, tuple(actions)))

    memory.before_delete.append(observe_after_qt_ack)
    finished = []
    desktop.jobs.finished.connect(lambda *result: finished.append(result))

    try:
        request = desktop.jobs.submit("test-delete", lambda: memory.delete_meeting(MEETING_ID))
        assert request is not None
        qtbot.waitUntil(lambda: any(result[0] == request for result in finished))
        _, _, deletion, error = next(result for result in finished if result[0] == request)
        assert error is None
        assert (deletion.meeting_id, deletion.status) == (MEETING_ID, "pending")
        assert len(observed) == 1 and observed[0][0] == MEETING_ID
        at_ack = observed[0][1]
        if affected:
            assert at_ack.index(("stop", source)) < at_ack.index(("source", None))
            _ready(player)
            assert player.source().isEmpty()
            assert not any(action[0] in {"seek", "play"} for action in actions)
        else:
            assert ("play", source) in at_ack
            assert not any(action[0] == "stop" or action == ("source", None) for action in actions)
            assert player.source().toLocalFile() == source
            assert player.playbackState() == player.PlaybackState.PlayingState
    finally:
        controller.stop_and_unload()
        delete_later()
