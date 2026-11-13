"""Exercise the Qt playback adapter without opening a real audio device."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest
from here.application import ApplicationState
from here.memory.models import CitationTarget
from memory_fixtures import MEETING_ID, OTHER_MEETING_ID, write_marked_wav
from PySide6.QtCore import QObject, QUrl, Signal
from PySide6.QtMultimedia import QMediaPlayer


class _FakeMediaPlayer(QObject):
    MediaStatus = QMediaPlayer.MediaStatus
    PlaybackState = QMediaPlayer.PlaybackState
    Error = QMediaPlayer.Error

    mediaStatusChanged = Signal(QMediaPlayer.MediaStatus)
    seekableChanged = Signal(bool)
    positionChanged = Signal(int)
    durationChanged = Signal(int)
    playbackStateChanged = Signal(QMediaPlayer.PlaybackState)
    errorOccurred = Signal(QMediaPlayer.Error, str)
    sourceChanged = Signal(QUrl)

    def __init__(self, actions: list[tuple], parent: QObject | None = None):
        super().__init__(parent)
        self._actions = actions
        self._source = QUrl()
        self._media_status = self.MediaStatus.NoMedia
        self._seekable = False
        self._position = 0
        self._duration = 0
        self._playback_state = self.PlaybackState.StoppedState
        self._audio_output = None
        self._rate = 1.0

    def setAudioOutput(self, output):
        self._audio_output = output

    def audioOutput(self):
        return self._audio_output

    def setSource(self, source: QUrl):
        self._actions.append(("source", source.toLocalFile() if not source.isEmpty() else None))
        self._source = source
        self._position = 0
        self._duration = 0
        self._seekable = False
        self._playback_state = self.PlaybackState.StoppedState
        self.sourceChanged.emit(source)
        self.report_status(self.MediaStatus.NoMedia if source.isEmpty() else self.MediaStatus.LoadingMedia)

    def source(self) -> QUrl:
        return self._source

    def mediaStatus(self):
        return self._media_status

    def isSeekable(self) -> bool:
        return self._seekable

    def position(self) -> int:
        return self._position

    def duration(self) -> int:
        return self._duration

    def playbackState(self):
        return self._playback_state

    def setPosition(self, position: int):
        self._actions.append(("seek", self._source.toLocalFile(), position))
        self._position = position
        self.positionChanged.emit(position)

    def play(self):
        self._actions.append(("play", self._source.toLocalFile()))
        self._playback_state = self.PlaybackState.PlayingState
        self.playbackStateChanged.emit(self._playback_state)

    def pause(self):
        self._actions.append(("pause", self._source.toLocalFile()))
        self._playback_state = self.PlaybackState.PausedState
        self.playbackStateChanged.emit(self._playback_state)

    def stop(self):
        self._actions.append(("stop", self._source.toLocalFile()))
        self._playback_state = self.PlaybackState.StoppedState
        self.playbackStateChanged.emit(self._playback_state)

    def setPlaybackRate(self, rate: float):
        self._actions.append(("rate", rate))
        self._rate = rate

    def playbackRate(self) -> float:
        return self._rate

    def report_status(self, status: QMediaPlayer.MediaStatus):
        self._media_status = status
        self.mediaStatusChanged.emit(status)

    def report_seekable(self):
        self._seekable = True
        self.seekableChanged.emit(True)

    def report_duration(self, duration: int):
        self._duration = duration
        self.durationChanged.emit(duration)


class _FakeAudioOutput(QObject):
    def __init__(self, actions: list[tuple], parent: QObject | None = None):
        super().__init__(parent)
        self._actions = actions
        self._volume = 1.0

    def setVolume(self, volume: float):
        self._actions.append(("volume", volume))
        self._volume = volume

    def volume(self) -> float:
        return self._volume


@pytest.fixture
def playback(qapp, monkeypatch):
    from here.ui import playback as module

    actions: list[tuple] = []
    players: list[_FakeMediaPlayer] = []

    class FakeMediaPlayer(_FakeMediaPlayer):
        def __init__(self, parent=None):
            super().__init__(actions, parent)
            players.append(self)

    class FakeAudioOutput(_FakeAudioOutput):
        def __init__(self, parent=None):
            super().__init__(actions, parent)

    monkeypatch.setattr(module, "QMediaPlayer", FakeMediaPlayer)
    monkeypatch.setattr(module, "QAudioOutput", FakeAudioOutput)
    controller = module.PlaybackController(qapp)
    try:
        yield controller, players, actions
    finally:
        controller.stop_and_unload()
        controller.deleteLater()


@pytest.fixture
def target(tmp_path: Path) -> CitationTarget:
    path = tmp_path / "marked.wav"
    write_marked_wav(path)
    return CitationTarget(
        meeting_id=MEETING_ID,
        revision_id="33333333-3333-4333-8333-333333333333",
        segment_id="44444444-4444-4444-8444-444444444444",
        start_ms=1250,
        end_ms=1500,
        timing_source="provider",
        audio_path=path,
    )


def _player_for(players: list[_FakeMediaPlayer], target: CitationTarget) -> _FakeMediaPlayer:
    expected = QUrl.fromLocalFile(str(target.audio_path))
    return next(player for player in reversed(players) if player.source() == expected)


def _ready(player: _FakeMediaPlayer) -> None:
    player.report_status(player.MediaStatus.LoadedMedia)
    player.report_duration(2000)
    player.report_seekable()


@pytest.mark.parametrize("first_event", ["loaded", "seekable"])
def test_pending_seek_waits_for_both_media_signals_and_applies_once(playback, target, first_event):
    controller, players, actions = playback
    controller.load(target)
    controller.seek_ms(1250)
    player = _player_for(players, target)
    loaded = lambda: player.report_status(player.MediaStatus.LoadedMedia)
    first, second = (loaded, player.report_seekable)
    if first_event == "seekable":
        first, second = second, first

    first()
    assert not any(action[0] == "seek" for action in actions)
    second()
    first()
    second()

    assert [action for action in actions if action[0] == "seek"] == [
        ("seek", player.source().toLocalFile(), 1250)
    ]


def test_selecting_and_loading_evidence_never_plays_without_an_explicit_action(playback, target):
    controller, players, actions = playback
    controller.load(target)
    player = _player_for(players, target)
    _ready(player)
    controller.seek_ms(1250)
    assert not any(action[0] == "play" for action in actions)

    controller.play()
    controller.pause()
    assert [action[0] for action in actions if action[0] in {"play", "pause"}] == ["play", "pause"]


def test_meeting_switch_releases_old_source_and_invalidates_pending_seek(playback, target, tmp_path):
    controller, players, actions = playback
    replacement_path = tmp_path / "other-marked.wav"
    write_marked_wav(replacement_path)
    replacement = replace(target, meeting_id=OTHER_MEETING_ID, start_ms=1500, audio_path=replacement_path)
    controller.load(target)
    old = _player_for(players, target)
    old_path = old.source().toLocalFile()
    controller.seek_ms(1250)
    actions.clear()

    controller.load(replacement)
    new = _player_for(players, replacement)
    new_path = new.source().toLocalFile()
    assert actions.index(("stop", old_path)) < actions.index(("source", None))
    assert actions.index(("source", None)) < actions.index(("source", new_path))
    controller.seek_ms(1500)

    if old is not new:
        _ready(old)
    else:
        old.mediaStatusChanged.emit(old.MediaStatus.LoadedMedia)
        old.seekableChanged.emit(True)
    assert not any(action[0] == "seek" for action in actions)

    _ready(new)
    assert [action for action in actions if action[0] == "seek"] == [("seek", new_path, 1500)]
    assert not any(action[0] == "play" for action in actions)


def test_explicit_unload_clears_queued_seek_before_late_media_events(playback, target):
    controller, players, actions = playback
    controller.load(target)
    player = _player_for(players, target)
    controller.seek_ms(1250)

    controller.stop_and_unload()
    _ready(player)
    assert player.source().isEmpty()
    assert not any(action[0] in {"seek", "play"} for action in actions)


def test_loading_untimed_evidence_does_not_invent_a_seek_to_zero(playback, target):
    controller, players, actions = playback
    target = replace(target, start_ms=None, end_ms=None, timing_source="absent")
    controller.load(target)
    if any(not player.source().isEmpty() for player in players):
        _ready(_player_for(players, target))

    assert not any(action[0] in {"seek", "play"} for action in actions)


def test_measured_zero_is_a_valid_explicit_seek(playback, target):
    controller, players, actions = playback
    target = replace(target, start_ms=0)
    controller.load(target)
    controller.seek_ms(0)
    player = _player_for(players, target)
    _ready(player)

    assert [action for action in actions if action[0] == "seek"] == [
        ("seek", player.source().toLocalFile(), 0)
    ]


def test_volume_and_moderate_rate_reach_the_audio_output_and_player(playback, target):
    controller, players, _ = playback
    controller.load(target)
    player = _player_for(players, target)
    _ready(player)
    controller.set_volume(0.35)
    controller.set_rate(1.25)

    assert player.audioOutput().volume() == pytest.approx(0.35)
    assert player.playbackRate() == pytest.approx(1.25)


def test_missing_local_file_reports_an_error_without_opening_a_source(playback, target, tmp_path):
    controller, _, actions = playback
    messages: list[str] = []
    controller.error.connect(messages.append)
    missing = tmp_path / "no-such-media.wav"
    controller.load(replace(target, audio_path=missing))

    assert messages and messages[-1].strip()
    assert not any(action[0] == "source" and action[1] is not None for action in actions)
    assert not any(action[0] == "play" for action in actions)


def test_codec_failure_reports_text_and_does_not_start_playback(playback, target):
    controller, players, actions = playback
    messages: list[str] = []
    controller.error.connect(messages.append)
    controller.load(target)
    player = _player_for(players, target)
    player.errorOccurred.emit(player.Error.FormatError, "Unsupported synthetic codec")

    assert messages and messages[-1].strip()
    assert not any(action[0] == "play" for action in actions)


@pytest.mark.parametrize(
    "state", [ApplicationState.PREPARING, ApplicationState.RECORDING, ApplicationState.PAUSED]
)
def test_capture_disables_play_and_seek_until_completion(playback, target, state):
    controller, players, actions = playback
    controller.load(target)
    _ready(_player_for(players, target))
    controller.set_application_state(state)
    actions.clear()

    controller.play()
    controller.seek_ms(1500)
    assert not any(action[0] in {"play", "seek"} for action in actions)

    controller.set_application_state(ApplicationState.COMPLETED)
    controller.load(target)
    _ready(_player_for(players, target))
    assert not any(action[0] == "play" for action in actions)
    assert not any(action[0] == "seek" and action[-1] == 1500 for action in actions)
    controller.play()
    assert any(action[0] == "play" for action in actions)
