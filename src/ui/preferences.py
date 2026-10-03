"""Presentation preferences that are independent from recording state."""

from __future__ import annotations

import math

from PySide6.QtCore import QObject, QSettings, Signal
from PySide6.QtGui import QColor

DEFAULT_ACCENT = "#6c63ff"


class VisualPreferences(QObject):
    accentChanged = Signal(QColor)
    sizeChanged = Signal(int)
    opacityChanged = Signal(float)

    def __init__(self, settings: QSettings | None = None) -> None:
        super().__init__()
        self._settings = settings or QSettings("here", "here")
        stored = str(self._settings.value("visual/accent", DEFAULT_ACCENT, type=str))
        self._accent = self._valid_color(stored)
        self._size = self._bounded(self._settings.value("visual/size", 104), 48, 208, 104)
        self._opacity = self._bounded(self._settings.value("visual/opacity", 1.0), 0.2, 1.0, 1.0)

    @staticmethod
    def _bounded(value, minimum, maximum, default):
        try:
            number = float(value)
            if not math.isfinite(number):
                return default
            return type(default)(min(maximum, max(minimum, number)))
        except (TypeError, ValueError):
            return default

    @property
    def size(self) -> int:
        return self._size

    @property
    def opacity(self) -> float:
        return self._opacity

    def set_size(self, value: int) -> None:
        candidate = self._bounded(value, 48, 208, 104)
        if candidate != self._size:
            self._size = candidate
            self._settings.setValue("visual/size", candidate)
            self.sizeChanged.emit(candidate)

    def set_opacity(self, value: float) -> None:
        candidate = self._bounded(value, 0.2, 1.0, 1.0)
        if candidate != self._opacity:
            self._opacity = candidate
            self._settings.setValue("visual/opacity", candidate)
            self.opacityChanged.emit(candidate)

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
