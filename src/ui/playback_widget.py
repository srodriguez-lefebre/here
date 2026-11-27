"""Shared local-audio controls for the library and standalone playback window."""

from __future__ import annotations

from here.memory.models import CitationTarget
from here.ui.contract import ApplicationState
from PySide6.QtCore import QEventLoop, Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)


def _clock(milliseconds: int) -> str:
    seconds = milliseconds // 1000
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _plain_label() -> QLabel:
    label = QLabel()
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(True)
    return label


class PlaybackWidget(QWidget):
    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._target: CitationTarget | None = None
        self._segment_start: int | None = None
        self._notice = ""
        self._error = ""
        self._position = self._duration = 0
        self._started = False
        self._message = _plain_label()
        self._time = _plain_label()
        self._play = QPushButton("Reproducir")
        self._slider = QSlider(Qt.Orientation.Horizontal)
        self._slider.setAccessibleName("Posición de reproducción")
        self._volume = QSlider(Qt.Orientation.Horizontal)
        self._volume.setAccessibleName("Volumen")
        self._volume.setRange(0, 100)
        self._volume.setValue(100)
        self._speed = QComboBox()
        self._speed.setAccessibleName("Velocidad de reproducción")
        controls = QHBoxLayout()
        for widget in (self._play, self._slider, self._time, self._volume, self._speed):
            controls.addWidget(widget)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._message)
        layout.addLayout(controls)
        try:
            from here.ui.playback import PLAYBACK_RATES, PlaybackController
        except (ImportError, OSError):
            self._controller = None
        else:
            self._controller = PlaybackController(self)
            self._controller.state_changed.connect(self._update_controls)
            self._controller.error.connect(self._show_error)
            self._controller.position_changed.connect(self._set_position)
            self._controller.duration_changed.connect(self._set_duration)
            for rate in PLAYBACK_RATES:
                self._speed.addItem(f"{rate:g}×", rate)
            self._speed.setCurrentIndex(self._speed.findData(1.0))
        self._play.clicked.connect(self._toggle_play)
        self._slider.sliderMoved.connect(self._seek)
        self._slider.actionTriggered.connect(self._slider_action)
        self._volume.valueChanged.connect(self._set_volume)
        self._speed.currentIndexChanged.connect(self._set_speed)
        self._update_clock()
        self._update_controls()

    @property
    def meeting_id(self) -> str | None:
        if self._controller is not None and self._controller.meeting_id is not None:
            return self._controller.meeting_id
        return self._target.meeting_id if self._target is not None else None

    @property
    def failed(self) -> bool:
        return bool(self._error or self._notice)

    def set_application_state(self, state: ApplicationState) -> None:
        if self._controller is not None:
            self._controller.set_application_state(state)
            self._update_controls()

    def load(
        self,
        target: CitationTarget,
        *,
        seek_from_segment: bool = True,
        revision_available: bool | None = None,
    ) -> None:
        self.clear()
        self._target = target
        untimed = seek_from_segment and (
            target.timing_source == "absent" or target.start_ms is None or target.start_ms < 0
        )
        if target.status == "missing":
            if revision_available is False:
                self._notice = "La revisión ya no está disponible."
            else:
                self._notice = "El archivo de audio no está disponible."
                if untimed:
                    self._notice += "\nSin marca de tiempo"
        elif untimed:
            self._notice = "Sin marca de tiempo"
        elif self._controller is None:
            self._error = "La reproducción de audio local no está disponible."
        else:
            self._segment_start = target.start_ms if seek_from_segment else None
            self._controller.load(target, require_segment_timing=seek_from_segment)
        self._update_controls()

    def clear(self) -> None:
        self._target = None
        self._segment_start = None
        self._started = False
        self._error = ""
        self._notice = ""
        if self._controller is not None:
            self._controller.stop_and_unload()
        self._update_controls()

    def _show_error(self, message: str) -> None:
        self._error = message
        self._update_controls()

    def _update_controls(self, _state: str = "") -> None:
        controller = self._controller
        state = controller.state if controller is not None else "empty"
        enabled = bool(controller and controller.can_play)
        controls_visible = self._target is not None and not self._notice and not self._error
        message = self._error or self._notice or (controller.disabled_reason if controller else "")
        if not message and state == "loading":
            message = "Cargando audio…"
        self._message.setText(message)
        self._message.setVisible(bool(message) and self._target is not None)
        self._play.setText(
            "Pausar"
            if state == "playing"
            else (
                f"Reproducir desde {_clock(self._segment_start)}"
                if not self._started and self._segment_start is not None
                else "Reproducir"
            )
        )
        for widget in (self._play, self._slider, self._time, self._volume, self._speed):
            widget.setVisible(controls_visible)
            widget.setEnabled(enabled)

    def _toggle_play(self) -> None:
        controller = self._controller
        if controller is None or not controller.can_play:
            return
        if controller.state == "playing":
            controller.pause()
            return
        if not self._started and self._segment_start is not None:
            controller.seek_ms(self._segment_start)
        self._started = True
        controller.play()
        self._update_controls()

    def _set_position(self, milliseconds: int) -> None:
        self._position = milliseconds
        if not self._slider.isSliderDown():
            self._slider.setValue(milliseconds)
        self._update_clock()

    def _set_duration(self, milliseconds: int) -> None:
        self._duration = milliseconds
        self._slider.setRange(0, milliseconds)
        self._update_clock()

    def _update_clock(self) -> None:
        self._time.setText(f"{_clock(self._position)} / {_clock(self._duration)}")

    def _slider_action(self, action: int) -> None:
        if action != QSlider.SliderAction.SliderMove.value:
            self._seek(self._slider.sliderPosition())

    def _seek(self, milliseconds: int) -> None:
        if self._controller is not None and self._controller.can_play:
            self._started = True
            self._controller.seek_ms(milliseconds)
            self._update_controls()

    def _set_volume(self, percent: int) -> None:
        if self._controller is not None:
            self._controller.set_volume(percent / 100)

    def _set_speed(self, _index: int) -> None:
        if self._controller is not None:
            self._controller.set_rate(self._speed.currentData())


def run_playback_window(target: CitationTarget, *, seek_from_segment: bool) -> int:
    """Show a player that waits for the user's explicit playback action."""
    application = QApplication.instance() or QApplication([])
    loop = QEventLoop()
    application.aboutToQuit.connect(loop.quit)
    window = QWidget()
    window.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
    window.setWindowTitle(target.title or "Reproducción")
    window.destroyed.connect(loop.quit)
    layout = QVBoxLayout(window)
    player = PlaybackWidget(window)
    failure = [False]
    if player._controller is not None:
        player._controller.error.connect(lambda _message: failure.__setitem__(0, True))
    player.load(target, seek_from_segment=seek_from_segment)
    if player.failed:
        window.deleteLater()
        return 4
    close = QPushButton("Cerrar")
    close.clicked.connect(window.close)
    layout.addWidget(player)
    layout.addWidget(close)
    window.resize(600, 100)
    window.show()
    loop.exec()
    return 4 if failure[0] else 0
