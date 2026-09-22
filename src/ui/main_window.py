"""Deliberately small Qt Widgets shell for the Windows application."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QCloseEvent, QColor, QPalette
from PySide6.QtWidgets import (
    QColorDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .contract import ApplicationSnapshot, ApplicationState, VisualController
from .preferences import VisualPreferences

STATE_LABELS = {
    ApplicationState.IDLE: "Listo para grabar",
    ApplicationState.PREPARING: "Preparando dispositivos…",
    ApplicationState.RECORDING: "Grabando micrófono y audio del sistema",
    ApplicationState.PAUSED: "Grabación pausada",
    ApplicationState.STOPPING: "Finalizando captura…",
    ApplicationState.PROCESSING: "Procesando sesión…",
    ApplicationState.COMPLETED: "Sesión completada",
    ApplicationState.FAILED: "La sesión terminó con un error",
    ApplicationState.CANCELLED: "Operación cancelada",
}


class MainWindow(QMainWindow):
    """Minimal controls backed exclusively by the shared application contract."""

    idleCloseRequested = Signal()

    def __init__(
        self,
        controller: VisualController,
        preferences: VisualPreferences,
    ) -> None:
        super().__init__()
        self._controller = controller
        self._preferences = preferences
        self._snapshot = controller.snapshot

        self.setWindowTitle("here")
        self.setMinimumWidth(430)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)

        self._state_label = QLabel()
        self._state_label.setObjectName("stateLabel")
        self._state_label.setStyleSheet("font-size: 18px; font-weight: 600")

        self._detail_label = QLabel()
        self._detail_label.setObjectName("detailLabel")
        self._detail_label.setWordWrap(True)

        self._source_label = QLabel(f"Fuentes: {controller.source_label}")
        self._source_label.setObjectName("sourceLabel")
        self._destination_label = QLabel(f"Destino: {controller.output_dir}")
        self._destination_label.setObjectName("destinationLabel")
        self._destination_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )

        self._level = QProgressBar()
        self._level.setObjectName("combinedAudioLevel")
        self._level.setRange(0, 100)
        self._level.setTextVisible(False)
        self._level.setAccessibleName("Nivel de audio combinado")

        self._start_button = QPushButton("Iniciar grabación")
        self._start_button.setObjectName("startButton")
        self._start_button.clicked.connect(self._controller.start_recording)

        self._pause_button = QPushButton("Pausar")
        self._pause_button.setObjectName("pauseButton")
        self._pause_button.clicked.connect(self._toggle_pause)

        self._stop_button = QPushButton("Detener y guardar")
        self._stop_button.setObjectName("stopButton")
        self._stop_button.clicked.connect(self._controller.stop_and_process)

        self._cancel_button = QPushButton("Cancelar")
        self._cancel_button.setObjectName("cancelButton")
        self._cancel_button.clicked.connect(self._confirm_cancel)

        self._color_button = QPushButton("Color del indicador")
        self._color_button.setObjectName("accentButton")
        self._color_button.clicked.connect(self._choose_accent)

        actions = QHBoxLayout()
        actions.addWidget(self._start_button)
        actions.addWidget(self._pause_button)
        actions.addWidget(self._stop_button)
        actions.addWidget(self._cancel_button)

        layout = QVBoxLayout()
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(14)
        layout.addWidget(self._state_label)
        layout.addWidget(self._detail_label)
        layout.addWidget(self._source_label)
        layout.addWidget(self._destination_label)
        layout.addWidget(self._level)
        layout.addLayout(actions)
        layout.addWidget(self._color_button, alignment=Qt.AlignmentFlag.AlignLeft)

        central = QWidget()
        central.setLayout(layout)
        self.setCentralWidget(central)

        self._preferences.accentChanged.connect(self._update_accent_button)
        self._update_accent_button(self._preferences.accent)
        self.set_snapshot(self._snapshot)

    @property
    def snapshot(self) -> ApplicationSnapshot:
        return self._snapshot

    @Slot(object)
    def set_snapshot(self, snapshot: ApplicationSnapshot) -> None:
        self._snapshot = snapshot
        state = snapshot.state
        self._state_label.setText(STATE_LABELS[state])

        detail = ""
        if snapshot.last_error is not None:
            detail = snapshot.last_error.message
        elif snapshot.recoverable and state in {
            ApplicationState.FAILED,
            ApplicationState.CANCELLED,
        }:
            detail = "El audio está guardado y la sesión se puede reintentar."
        self._detail_label.setText(detail)
        self._detail_label.setVisible(bool(detail))

        recording = state in {ApplicationState.RECORDING, ApplicationState.PAUSED}
        self._start_button.setVisible(state is ApplicationState.IDLE)
        self._pause_button.setVisible(recording)
        self._pause_button.setText(
            "Reanudar" if state is ApplicationState.PAUSED else "Pausar"
        )
        self._stop_button.setVisible(recording)
        self._cancel_button.setVisible(
            state
            in {
                ApplicationState.PREPARING,
                ApplicationState.RECORDING,
                ApplicationState.PAUSED,
                ApplicationState.PROCESSING,
            }
        )
        self._level.setVisible(state is ApplicationState.RECORDING)
        if state is not ApplicationState.RECORDING:
            self._level.setValue(0)

    @Slot(float)
    def set_audio_level(self, level: float) -> None:
        if self._snapshot.state is ApplicationState.RECORDING:
            self._level.setValue(round(min(1.0, max(0.0, level)) * 100))

    @Slot()
    def _toggle_pause(self) -> None:
        if self._snapshot.state is ApplicationState.PAUSED:
            self._controller.resume_recording()
        elif self._snapshot.state is ApplicationState.RECORDING:
            self._controller.pause_recording()

    @Slot()
    def _confirm_cancel(self) -> None:
        if self._snapshot.state is ApplicationState.PROCESSING:
            answer = QMessageBox.question(
                self,
                "Cancelar procesamiento",
                "Se detendrá el procesamiento. El audio y la sesión recuperable se conservarán.",
                QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Yes,
                QMessageBox.StandardButton.Cancel,
            )
            if answer is QMessageBox.StandardButton.Yes:
                self._controller.cancel_processing()
            return

        answer = QMessageBox.question(
            self,
            "Cancelar grabación",
            "Se perderá el audio capturado y no se creará una sesión. ¿Continuar?",
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Yes,
            QMessageBox.StandardButton.Cancel,
        )
        if answer is QMessageBox.StandardButton.Yes:
            self._controller.cancel_recording()

    @Slot()
    def _choose_accent(self) -> None:
        selected = QColorDialog.getColor(
            self._preferences.accent,
            self,
            "Color del indicador",
        )
        if selected.isValid():
            self._preferences.set_accent(selected)

    @Slot(QColor)
    def _update_accent_button(self, color: QColor) -> None:
        palette = self._color_button.palette()
        palette.setColor(QPalette.ColorRole.Button, color)
        palette.setColor(
            QPalette.ColorRole.ButtonText,
            Qt.GlobalColor.white if color.lightnessF() < 0.55 else Qt.GlobalColor.black,
        )
        self._color_button.setPalette(palette)
        self._color_button.setAutoFillBackground(True)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 (Qt override)
        if self._snapshot.has_active_work:
            event.ignore()
            self.hide()
            return
        event.accept()
        self.idleCloseRequested.emit()
