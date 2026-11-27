"""Exercise the standalone Qt player and its meetings CLI boundary."""

from __future__ import annotations

import json
from dataclasses import replace

import here.cli as cli
import pytest
import test_m3_library_cli as meetings_tests
import test_ui_playback as playback_tests
from here.ui import playback_widget
from here.ui.playback import PlaybackController
from memory_fixtures import MEETING_ID
from PySide6.QtCore import QEventLoop, Qt, QTimer
from PySide6.QtWidgets import QPushButton
from test_ui_playback import _player_for, _ready
from typer.testing import CliRunner

library = meetings_tests.library
fake_media = playback_tests.playback
marked_target = playback_tests.target


@pytest.mark.parametrize(
    "seek_from_segment", [False, True], ids=["untimed-meeting", "timed-segment"]
)
def test_standalone_player_waits_for_readiness_and_explicit_play(
    qapp, qtbot, monkeypatch, fake_media, marked_target, seek_from_segment
):
    target = replace(marked_target, title="Standalone synthetic evidence")
    if not seek_from_segment:
        target = replace(target, start_ms=None, end_ms=None, timing_source="absent")
    _, players, actions = fake_media
    timed_out = False
    exercised = False

    class GuardedLoop(QEventLoop):
        def exec(self):
            def timeout():
                nonlocal timed_out
                timed_out = True
                self.quit()

            watchdog = QTimer(self)
            watchdog.setSingleShot(True)
            watchdog.timeout.connect(timeout)
            watchdog.start(5000)
            try:
                return super().exec()
            finally:
                watchdog.stop()

    def exercise_window():
        nonlocal exercised
        (window,) = (
            widget for widget in qapp.topLevelWidgets() if widget.windowTitle() == target.title
        )
        try:
            assert window.findChild(PlaybackController) is not None
            buttons = window.findChildren(QPushButton)
            (play,) = (button for button in buttons if button.text().startswith("Reproducir"))
            player = _player_for(players, target)
            source = player.source().toLocalFile()
            assert not play.isEnabled()
            qtbot.mouseClick(play, Qt.MouseButton.LeftButton)
            _ready(player)
            assert play.isEnabled()
            assert player.playbackState() != player.PlaybackState.PlayingState
            assert not any(action[0] in {"seek", "play"} for action in actions)

            qtbot.mouseClick(play, Qt.MouseButton.LeftButton)

            expected = [("play", source)]
            if seek_from_segment:
                expected.insert(0, ("seek", source, marked_target.start_ms))
            assert [action for action in actions if action[0] in {"seek", "play"}] == expected
            assert player.playbackState() == player.PlaybackState.PlayingState
            exercised = True
        finally:
            window.close()

    monkeypatch.setattr(playback_widget, "QEventLoop", GuardedLoop)
    start = QTimer()
    start.setSingleShot(True)
    start.timeout.connect(exercise_window)
    with qtbot.captureExceptions() as exceptions:
        start.start(0)
        try:
            status = playback_widget.run_playback_window(
                target, seek_from_segment=seek_from_segment
            )
        finally:
            start.stop()
            if timed_out:
                for window in qapp.topLevelWidgets():
                    if window.windowTitle() == target.title:
                        window.close()
    assert not exceptions, exceptions
    assert not timed_out, "The standalone playback window did not close"
    assert exercised and status == 0


def test_playback_window_io_status_explains_failure_and_preserves_target_json(library, monkeypatch):
    segment = library.show(MEETING_ID)["segments"][0]
    launches = []

    def unavailable(target, *, seek_from_segment):
        launches.append((target, seek_from_segment))
        return 4

    monkeypatch.setattr(playback_widget, "run_playback_window", unavailable)
    result = CliRunner().invoke(
        cli.app, ["meetings", "play", MEETING_ID[:8], "--segment", segment["id"], "--json"]
    )

    assert result.exit_code == 4, result.output + repr(result.exception)
    (launch,) = launches
    target, seek_from_segment = launch
    assert seek_from_segment is True
    payload = json.loads(result.stdout)
    assert payload["meeting_id"] == target.meeting_id == MEETING_ID
    assert payload["revision_id"] == target.revision_id == segment["revision_id"]
    assert payload["segment_id"] == target.segment_id == segment["id"]
    assert target.audio_path is not None
    assert payload["audio_path"] == str(target.audio_path)
    message = result.stderr.strip()
    assert len(message.splitlines()) == 1
    assert any(word in message.casefold() for word in ("multimedia", "playback", "reproduc"))
