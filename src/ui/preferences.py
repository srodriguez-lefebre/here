"""Presentation preferences that are independent from recording state."""

from __future__ import annotations

from PySide6.QtCore import QObject, QSettings, Signal
from PySide6.QtGui import QColor

DEFAULT_ACCENT = "#6c63ff"


class VisualPreferences(QObject):
    accentChanged = Signal(QColor)

    def __init__(self, settings: QSettings | None = None) -> None:
        super().__init__()
        self._settings = settings or QSettings("here", "here")
        stored = str(self._settings.value("visual/accent", DEFAULT_ACCENT, type=str))
        self._accent = self._valid_color(stored)

    @property
    def accent(self) -> QColor:
        return QColor(self._accent)

    def set_accent(self, color: QColor | str) -> None:
        candidate = self._valid_color(color)
        if candidate == self._accent:
            return
        self._accent = candidate
        self._settings.setValue("visual/accent", candidate.name())
        self.accentChanged.emit(QColor(candidate))

    @staticmethod
    def _valid_color(color: QColor | str) -> QColor:
        candidate = QColor(color)
        if not candidate.isValid():
            return QColor(DEFAULT_ACCENT)
        return candidate
