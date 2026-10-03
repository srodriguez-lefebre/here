"""Local application configuration and persistent indicator appearance."""

from __future__ import annotations

import os

from here.config.editor import EDITABLE_KEYS, read_configuration, save_configuration
from here.config.paths import get_data_dir, get_env_file
from here.diagnostics import explain_error, record_error, record_event
from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSlider,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .preferences import VisualPreferences


class ConfigurationDialog(QDialog):
    def __init__(self, preferences: VisualPreferences, parent=None):
        super().__init__(parent)
        self._preferences = preferences
        self.setWindowTitle("Configuración de here")
        self.setMinimumWidth(500)
        self._inputs = {}
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        layout.addWidget(tabs)
        account = QWidget()
        form = QFormLayout(account)
        tabs.addTab(account, "Cuenta y sesiones")
        file_label = QLabel(f"Archivo local: {get_env_file()}")
        file_label.setWordWrap(True)
        file_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        form.addRow(file_label)
        self._status = QLabel()
        self._status.setObjectName("configurationStatusLabel")
        self._status.setWordWrap(True)
        try:
            values = read_configuration()
        except (OSError, UnicodeError) as error:
            values = {}
            self._status.setText(explain_error(str(error), type(error).__name__))
            record_error("configuration_read", error)
        defaults = {
            "TRANSCRIPTION_MODEL": "gpt-4o-transcribe-diarize",
            "ALT_TRANSCRIPTION_MODEL": "gpt-4o-transcribe",
            "CLEANUP_MODEL": "gpt-4.1-mini",
        }
        names = {
            "OPENAI_API_KEY": "Clave de OpenAI",
            "TRANSCRIPTION_MODEL": "Modelo de transcripción",
            "ALT_TRANSCRIPTION_MODEL": "Modelo alternativo",
            "CLEANUP_MODEL": "Modelo de limpieza",
            "TRANSCRIPTIONS_DIR": "Carpeta de sesiones",
        }
        for key, label in names.items():
            field = QLineEdit(values.get(key, defaults.get(key, "")))
            field.setObjectName("apiKeyInput" if key == "OPENAI_API_KEY" else key)
            self._inputs[key] = field
            if key == "OPENAI_API_KEY":
                field.setEchoMode(QLineEdit.EchoMode.Password)
            if key == "TRANSCRIPTIONS_DIR":
                field.setPlaceholderText(str(get_data_dir() / "sessions"))
                row = QHBoxLayout()
                row.addWidget(field)
                browse = QPushButton("Elegir…")
                browse.clicked.connect(self._choose_folder)
                row.addWidget(browse)
                form.addRow(label, row)
            else:
                form.addRow(label, field)
        reveal = QCheckBox("Mostrar clave")
        reveal.toggled.connect(
            lambda visible: self._inputs["OPENAI_API_KEY"].setEchoMode(
                QLineEdit.EchoMode.Normal if visible else QLineEdit.EchoMode.Password
            )
        )
        form.addRow(reveal)
        self._cleanup = QCheckBox("Limpiar la transcripción al terminar")
        self._cleanup.setChecked(
            values.get("CLEANUP_ENABLED", "false").lower() in {"true", "1", "yes"}
        )
        form.addRow(self._cleanup)
        inherited = [key for key in EDITABLE_KEYS if key in os.environ]
        if inherited:
            warning = QLabel(
                "Estas variables del entorno tienen prioridad sobre el archivo: "
                + ", ".join(inherited)
            )
            warning.setWordWrap(True)
            form.addRow(warning)
        appearance = QWidget()
        visual = QFormLayout(appearance)
        tabs.addTab(appearance, "Indicador")
        self._size = QSpinBox()
        self._size.setObjectName("indicatorSizeInput")
        self._size.setRange(48, 208)
        self._size.setSuffix(" px")
        self._size.setValue(preferences.size)
        visual.addRow("Tamaño", self._size)
        self._transparency = QSlider(Qt.Orientation.Horizontal)
        self._transparency.setObjectName("indicatorTransparencyInput")
        self._transparency.setRange(0, 80)
        self._transparency.setValue(round((1 - preferences.opacity) * 100))
        amount = QLabel(f"{self._transparency.value()} %")
        self._transparency.valueChanged.connect(lambda value: amount.setText(f"{value} %"))
        visual.addRow("Transparencia", self._transparency)
        visual.addRow(amount)
        visual.addRow(QLabel("0 %: opaco. 80 %: más transparente. Se conserva al volver a abrir."))
        logs = QPushButton("Abrir carpeta de registros")
        logs.clicked.connect(self._open_logs)
        layout.addWidget(logs)
        layout.addWidget(self._status)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("Guardar")
        buttons.button(QDialogButtonBox.StandardButton.Save).setObjectName(
            "saveConfigurationButton"
        )
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Cancelar")
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _choose_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Carpeta de sesiones", self._inputs["TRANSCRIPTIONS_DIR"].text()
        )
        if folder:
            self._inputs["TRANSCRIPTIONS_DIR"].setText(folder)

    def _open_logs(self):
        try:
            path = get_data_dir() / "logs"
            path.mkdir(parents=True, exist_ok=True)
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))
        except OSError as error:
            self._status.setText(explain_error(str(error), type(error).__name__))

    def _save(self):
        values = {key: field.text() for key, field in self._inputs.items()}
        values["CLEANUP_ENABLED"] = "true" if self._cleanup.isChecked() else "false"
        # An empty output field means the per-user default, not Path('').
        values["TRANSCRIPTIONS_DIR"] = values["TRANSCRIPTIONS_DIR"].strip() or str(
            get_data_dir() / "sessions"
        )
        try:
            save_configuration(values)
        except (OSError, ValueError, UnicodeError) as error:
            record_error("configuration_save", error)
            self._status.setText(explain_error(str(error), type(error).__name__))
            return
        self._preferences.set_size(self._size.value())
        self._preferences.set_opacity(1 - self._transparency.value() / 100)
        record_event("configuration", "saved")
        self.accept()
