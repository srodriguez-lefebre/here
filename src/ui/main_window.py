"""Deliberately small Qt Widgets shell for the Windows application."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtGui import QCloseEvent, QColor, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QColorDialog,
    QComboBox,
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
    exitRequested = Signal()
    diagnosticsRequested = Signal(str)
    recoveryRequested = Signal(object)

    def __init__(
        self,
        controller: VisualController,
        preferences: VisualPreferences,
    ) -> None:
        super().__init__()
        self._controller = controller
        self._preferences = preferences
        self._snapshot = controller.snapshot
        self._background_busy = False
        self._exiting = False

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
        self._start_button.clicked.connect(lambda: self._invoke(self._controller.start_recording))

        self._pause_button = QPushButton("Pausar")
        self._pause_button.setObjectName("pauseButton")
        self._pause_button.clicked.connect(self._toggle_pause)

        self._stop_button = QPushButton("Detener y guardar")
        self._stop_button.setObjectName("stopButton")
        self._stop_button.clicked.connect(self._controller.stop_and_process)

        self._cancel_button = QPushButton("Cancelar")
        self._cancel_button.setObjectName("cancelButton")
        self._cancel_button.clicked.connect(self._confirm_cancel)

        self._retry_button = QPushButton("Reintentar procesamiento")
        self._retry_button.setObjectName("retryButton")
        self._retry_button.clicked.connect(self._controller.retry_processing)

        self._exit_button = QPushButton("Salir")
        self._exit_button.setObjectName("exitButton")
        self._exit_button.clicked.connect(self.exitRequested)
        self._configuration_button = QPushButton("Configuración")
        self._configuration_button.setObjectName("configurationButton")
        self._configuration_button.clicked.connect(self._configure)
        self._diagnostics_label = QLabel()
        self._diagnostics_label.setObjectName("diagnosticsLabel")
        self._diagnostics_label.setWordWrap(True)
        self._diagnostic_buttons = []
        diagnostic_actions = QHBoxLayout()
        for text, name, source in [
            ("Ver dispositivos", "devicesButton", "devices"),
            ("Probar micrófono", "testMicrophoneButton", "microphone"),
            ("Probar sistema", "testSystemButton", "system audio"),
        ]:
            button = QPushButton(text)
            button.setObjectName(name)
            button.clicked.connect(
                lambda checked=False, source=source: self.diagnosticsRequested.emit(source)
            )
            self._diagnostic_buttons.append(button)
            diagnostic_actions.addWidget(button)
        self._recovery_selector = QComboBox()
        self._recovery_selector.setObjectName("recoverySelector")
        self._recover_button = QPushButton("Recuperar y reintentar")
        self._recover_button.setObjectName("recoverButton")
        self._recover_button.clicked.connect(
            lambda: self.recoveryRequested.emit(self._recovery_selector.currentData())
        )
        self._recovery_selector.currentIndexChanged.connect(lambda: self._update_enabled())
        self._recovery_label = QLabel()
        self._recovery_label.setObjectName("recoveryLabel")
        self._recovery_label.setWordWrap(True)

        self._color_button = QPushButton("Color del indicador")
        self._color_button.setObjectName("accentButton")
        self._color_button.clicked.connect(self._choose_accent)

        actions = QHBoxLayout()
        actions.addWidget(self._start_button)
        actions.addWidget(self._pause_button)
        actions.addWidget(self._stop_button)
        actions.addWidget(self._cancel_button)
        actions.addWidget(self._retry_button)

        layout = QVBoxLayout()
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(14)
        layout.addWidget(self._state_label)
        layout.addWidget(self._detail_label)
        layout.addWidget(self._source_label)
        layout.addWidget(self._destination_label)
        layout.addWidget(self._level)
        layout.addLayout(diagnostic_actions)
        layout.addWidget(self._diagnostics_label)
        layout.addWidget(self._recovery_label)
        layout.addWidget(self._recovery_selector)
        layout.addWidget(self._recover_button)
        layout.addLayout(actions)
        layout.addWidget(self._color_button, alignment=Qt.AlignmentFlag.AlignLeft)
        layout.addWidget(self._configuration_button)
        layout.addWidget(self._exit_button, alignment=Qt.AlignmentFlag.AlignRight)

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
        source_text = (
            " · ".join(f"{item.label}: {item.device_name}" for item in snapshot.opened_sources)
            or self._controller.source_label
        )
        self._source_label.setText(f"Fuentes: {source_text}")
        self._state_label.setText(STATE_LABELS[state])

        detail = ""
        if snapshot.last_error is not None:
            from here.diagnostics import explain_error

            detail = explain_error(
                snapshot.last_error.message,
                snapshot.last_error.error_type,
                recoverable=snapshot.recoverable,
            )
            if snapshot.session_dir is None and state is ApplicationState.FAILED:
                self._state_label.setText("No se pudo iniciar la operación")
        elif snapshot.recoverable and state in {
            ApplicationState.FAILED,
            ApplicationState.CANCELLED,
        }:
            detail = "El audio está guardado y la sesión se puede reintentar."
        self._detail_label.setText(detail)
        self._detail_label.setStyleSheet(
            "color: #e89624; font-weight: 600" if snapshot.last_error else ""
        )
        self._detail_label.setVisible(bool(detail))

        recording = state in {ApplicationState.RECORDING, ApplicationState.PAUSED}
        self._start_button.setVisible(
            state
            in {
                ApplicationState.IDLE,
                ApplicationState.COMPLETED,
                ApplicationState.FAILED,
                ApplicationState.CANCELLED,
            }
        )
        self._start_button.setText(
            "Iniciar grabación" if state is ApplicationState.IDLE else "Nueva grabación"
        )
        self._pause_button.setVisible(recording)
        self._pause_button.setText("Reanudar" if state is ApplicationState.PAUSED else "Pausar")
        self._stop_button.setVisible(recording)
        self._cancel_button.setVisible(
            state
            in {
                ApplicationState.PREPARING,
                ApplicationState.RECORDING,
                ApplicationState.PAUSED,
                ApplicationState.STOPPING,
                ApplicationState.PROCESSING,
            }
        )
        self._retry_button.setVisible(
            snapshot.recoverable and state in {ApplicationState.FAILED, ApplicationState.CANCELLED}
        )
        self._level.setVisible(state is ApplicationState.RECORDING)
        if state is not ApplicationState.RECORDING:
            self._level.setValue(0)
        self._update_enabled()

    def _invoke(self, operation):
        try:
            operation()
        except Exception as exc:
            self._show_error("command", exc)

    def _show_error(self, stage, error):
        from here.diagnostics import explain_error, record_error

        record_error(stage, error)
        self._detail_label.setText(explain_error(str(error), type(error).__name__))
        self._detail_label.setStyleSheet("color: #e89624; font-weight: 600")
        self._detail_label.show()
        self.setWindowState(self.windowState() & ~Qt.WindowState.WindowMinimized)
        self.show()
        self.raise_()
        QApplication.alert(self)

    def _configure(self):
        from .configuration import ConfigurationDialog

        if self._controller.snapshot.has_active_work:
            return
        dialog = ConfigurationDialog(self._preferences, self)
        if dialog.exec() == dialog.DialogCode.Accepted:
            try:
                self._controller.refresh_configuration()
            except Exception as error:
                self._show_error("configuration_reload", error)
            else:
                self._destination_label.setText(f"Destino: {self._controller.output_dir}")
                self._detail_label.setText(
                    "Configuración guardada. Podés iniciar una nueva grabación."
                )
                self._detail_label.setStyleSheet("")
                self._detail_label.show()

    def set_background_busy(self, busy, *, exiting=False):
        self._background_busy = busy
        self._exiting = exiting
        self._update_enabled()

    def _update_enabled(self):
        idle = (
            not self._snapshot.has_active_work
            and self._snapshot.worker_complete
            and not self._background_busy
            and not self._exiting
        )
        for button in [self._start_button, self._retry_button, *self._diagnostic_buttons]:
            button.setEnabled(idle)
        self._configuration_button.setEnabled(idle)
        self._recovery_selector.setEnabled(idle)
        candidate = self._recovery_selector.currentData()
        self._recover_button.setEnabled(idle and candidate is not None and candidate.can_retry)
        if self._exiting:
            self._cancel_button.setEnabled(False)
            self._pause_button.setEnabled(False)
            self._detail_label.setText("Guardando y cerrando…")
            self._detail_label.show()

    def set_diagnostics(self, value, error=None):
        if error:
            self._show_error("audio_diagnostics", RuntimeError(error))
            text = f"Error de audio: {error}"
        elif isinstance(value, tuple):
            text = " · ".join(f"{item.source}: {item.name}" for item in value)
        else:
            signal = "Señal detectada" if value.has_signal else "Sin señal"
            text = f"{value.device.name}: {signal} ({value.duration_seconds:g} s)"
        self._diagnostics_label.setText(text)

    def set_recovery_error(self, error):
        """A failed explicit retry leaves the current selection available to try again."""
        self._recovery_label.setText(error)
        self._show_error("recovery", RuntimeError(error))

    def set_recovery(self, candidates, error=None):
        if error:
            self._show_error("recovery_discovery", RuntimeError(error))
        self._recovery_selector.clear()
        for item in candidates or []:
            text = f"{item.display_id} · {item.status} · {item.recorded_duration_seconds:.1f} s"
            if item.error_summary:
                text += f" · {item.error_summary}"
            self._recovery_selector.addItem(text, item)
        self._recovery_label.setText(
            error
            or (
                "Sesiones recuperables"
                if candidates
                else "No hay sesiones pendientes de recuperación."
            )
        )
        self._update_enabled()

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
        if self._snapshot.state in {
            ApplicationState.STOPPING,
            ApplicationState.PROCESSING,
        }:
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
        if self._controller.snapshot.has_active_work:
            event.ignore()
            self.hide()
            return
        event.accept()
        self.idleCloseRequested.emit()
