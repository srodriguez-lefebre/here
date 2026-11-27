"""Qt playback of local evidence selected by the citation resolver."""

from __future__ import annotations

import math

from here.application import ApplicationState
from here.memory.models import CitationTarget
from PySide6.QtCore import QObject, QUrl, Signal
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer

PLAYBACK_RATES = (0.75, 1.0, 1.25, 1.5)
_CAPTURE_STATES = {
    ApplicationState.PREPARING,
    ApplicationState.RECORDING,
    ApplicationState.PAUSED,
}
_READY_STATUSES = {
    QMediaPlayer.MediaStatus.LoadedMedia,
    QMediaPlayer.MediaStatus.BufferingMedia,
    QMediaPlayer.MediaStatus.BufferedMedia,
    QMediaPlayer.MediaStatus.StalledMedia,
    QMediaPlayer.MediaStatus.EndOfMedia,
}


class PlaybackController(QObject):
    position_changed = Signal(int)
    duration_changed = Signal(int)
    state_changed = Signal(str)
    error = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._player: QMediaPlayer | None = None
        self._meeting_id: str | None = None
        self._pending_seek: int | None = None
        self._capture_active = False
        self._volume = 1.0
        self._rate = 1.0
        self._state = "empty"

    @property
    def meeting_id(self) -> str | None:
        return self._meeting_id

    @property
    def state(self) -> str:
        return self._state

    @property
    def disabled_reason(self) -> str:
        if self._capture_active:
            return "La reproducción está deshabilitada durante la captura."
        return ""

    @property
    def can_play(self) -> bool:
        return not self._capture_active and self._state in {"ready", "playing", "paused"}

    def set_application_state(self, state: ApplicationState) -> None:
        self._capture_active = state in _CAPTURE_STATES
        if self._capture_active:
            self._pending_seek = None
            self.pause()
        self._update_state()

    def load(self, target: CitationTarget, *, require_segment_timing: bool = True) -> None:
        self.stop_and_unload()
        if target.status not in {"current", "historical"}:
            return
        if require_segment_timing and (
            target.start_ms is None or target.start_ms < 0 or target.timing_source == "absent"
        ):
            return

        path = target.audio_path
        try:
            valid_path = path is not None and path.is_absolute() and path.is_file()
        except (OSError, ValueError):
            valid_path = False
        if path is None or not valid_path:
            self._fail("El archivo de audio local no está disponible.")
            return

        # A fresh backend scopes delayed multimedia events even when the same
        # path is selected twice. Retire the previous player before setting the URL.
        player = QMediaPlayer(self)
        if not player.isAvailable():
            player.deleteLater()
            self._fail("La reproducción de audio local no está disponible.")
            return

        output = QAudioOutput(player)
        output.setVolume(self._volume)
        player.setAudioOutput(output)
        player.setPlaybackRate(self._rate)
        self._player = player
        self._meeting_id = target.meeting_id

        player.mediaStatusChanged.connect(lambda status: self._media_status(player, status))
        player.seekableChanged.connect(lambda _: self._refresh_player(player))
        player.playbackStateChanged.connect(lambda _: self._refresh_player(player))
        player.positionChanged.connect(lambda value: self._position(player, value))
        player.durationChanged.connect(lambda value: self._duration(player, value))
        player.errorOccurred.connect(lambda code, _: self._media_error(player, code))
        self._update_state()
        if self._player is player:
            player.setSource(QUrl.fromLocalFile(str(path)))

    def play(self) -> None:
        player = self._player
        if player is not None and self.can_play and self._pending_seek is None:
            player.play()

    def pause(self) -> None:
        player = self._player
        if player is not None and player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            player.pause()

    def seek_ms(self, position_ms: int) -> None:
        if self._capture_active or self._player is None or position_ms < 0:
            return
        self._pending_seek = position_ms
        self._apply_pending_seek()

    def set_volume(self, volume: float) -> None:
        if self._capture_active or not math.isfinite(volume):
            return
        self._volume = min(1.0, max(0.0, volume))
        if self._player is not None:
            output = self._player.audioOutput()
            if output is not None:
                output.setVolume(self._volume)

    def set_rate(self, rate: float) -> None:
        if self._capture_active or rate not in PLAYBACK_RATES:
            return
        self._rate = rate
        if self._player is not None:
            self._player.setPlaybackRate(rate)

    def stop_and_unload(self) -> None:
        self._release_player()
        self._update_state()

    def _release_player(self) -> None:
        player = self._player
        self._player = None
        self._meeting_id = None
        self._pending_seek = None
        if player is not None:
            player.stop()
            player.setSource(QUrl())
            player.setAudioOutput(None)
            player.deleteLater()
            self.position_changed.emit(0)
            self.duration_changed.emit(0)

    def _media_status(self, player: QMediaPlayer, status: QMediaPlayer.MediaStatus) -> None:
        if self._player is player and status == QMediaPlayer.MediaStatus.InvalidMedia:
            self._fail("No se pudo reproducir el archivo de audio local.")
        else:
            self._refresh_player(player)

    def _media_error(self, player: QMediaPlayer, code: QMediaPlayer.Error) -> None:
        if self._player is player and code != QMediaPlayer.Error.NoError:
            self._fail("No se pudo reproducir el archivo de audio local.")

    def _fail(self, message: str) -> None:
        self._release_player()
        self._set_state("blocked" if self._capture_active else "error")
        self.error.emit(message)

    def _position(self, player: QMediaPlayer, position_ms: int) -> None:
        if self._player is player:
            self.position_changed.emit(position_ms)

    def _duration(self, player: QMediaPlayer, duration_ms: int) -> None:
        if self._player is player:
            self.duration_changed.emit(duration_ms)
            self._refresh_player(player)

    def _refresh_player(self, player: QMediaPlayer) -> None:
        if self._player is not player:
            return
        self._apply_pending_seek()
        if self._player is player:
            self._update_state()

    def _apply_pending_seek(self) -> None:
        player = self._player
        position_ms = self._pending_seek
        if self._capture_active or player is None or position_ms is None or not self._ready(player):
            return
        self._pending_seek = None
        duration_ms = player.duration()
        if duration_ms > 0 and position_ms > duration_ms:
            self._fail("La marca de tiempo supera la duración del audio.")
            return
        player.setPosition(position_ms)

    @staticmethod
    def _ready(player: QMediaPlayer) -> bool:
        return player.mediaStatus() in _READY_STATUSES and player.isSeekable()

    def _update_state(self) -> None:
        player = self._player
        if self._capture_active:
            state = "blocked"
        elif player is None:
            state = "empty"
        elif not self._ready(player):
            state = "loading"
        elif player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            state = "playing"
        elif player.playbackState() == QMediaPlayer.PlaybackState.PausedState:
            state = "paused"
        else:
            state = "ready"
        self._set_state(state)

    def _set_state(self, state: str) -> None:
        if self._state != state:
            self._state = state
            self.state_changed.emit(state)
