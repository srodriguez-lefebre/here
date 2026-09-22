"""Transparent, frameless live-logo overlay."""

from __future__ import annotations

import math

from here.ui.contract import ApplicationSnapshot, ApplicationState, VisualController
from here.ui.preferences import VisualPreferences
from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, QTimer, Signal, Slot
from PySide6.QtGui import (
    QAction,
    QColor,
    QContextMenuEvent,
    QMouseEvent,
    QPainter,
    QPainterPath,
    QPen,
)
from PySide6.QtWidgets import QApplication, QMenu, QMessageBox, QStyle, QWidget

TERMINAL_DURATION_MS = 3000
CANCELLED_DURATION_MS = 1800


class LiveLogoOverlay(QWidget):
    """One visual object that reflects capture, processing, and outcomes."""

    restoreRequested = Signal()
    terminalDisplayFinished = Signal()

    def __init__(
        self,
        controller: VisualController,
        preferences: VisualPreferences,
    ) -> None:
        super().__init__(None)
        self._controller = controller
        self._preferences = preferences
        self._snapshot = controller.snapshot
        self._level = 0.0
        self._target_level = 0.0
        self._phase = 0.0
        self._press_global: QPoint | None = None
        self._window_at_press: QPoint | None = None
        self._dragged = False

        self.setObjectName("liveLogoOverlay")
        self.setFixedSize(104, 104)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setAccessibleName("Indicador de estado de here")

        self._animation_timer = QTimer(self)
        self._animation_timer.setInterval(33)
        self._animation_timer.timeout.connect(self._animate)

        self._terminal_timer = QTimer(self)
        self._terminal_timer.setSingleShot(True)
        self._terminal_timer.timeout.connect(self._finish_terminal_display)

        self._preferences.accentChanged.connect(lambda _color: self.update())
        self.set_snapshot(self._snapshot)

    @property
    def snapshot(self) -> ApplicationSnapshot:
        return self._snapshot

    @property
    def animation_timer(self) -> QTimer:
        return self._animation_timer

    @property
    def terminal_timer(self) -> QTimer:
        return self._terminal_timer

    @Slot(object)
    def set_snapshot(self, snapshot: ApplicationSnapshot) -> None:
        previous = self._snapshot.state
        self._snapshot = snapshot
        state = snapshot.state

        if not _is_overlay_state(previous) and _is_active_overlay_state(state):
            self.reset_position()

        self._terminal_timer.stop()
        if _is_active_overlay_state(state):
            self.show()
            self._animation_timer.start()
        elif state in {
            ApplicationState.COMPLETED,
            ApplicationState.FAILED,
            ApplicationState.CANCELLED,
        }:
            self.show()
            self._animation_timer.start()
            duration = (
                CANCELLED_DURATION_MS
                if state is ApplicationState.CANCELLED
                else TERMINAL_DURATION_MS
            )
            self._terminal_timer.start(duration)
        else:
            self._animation_timer.stop()
            self.hide()

        if state is not ApplicationState.RECORDING:
            self._level = 0.0
            self._target_level = 0.0
        self.update()

    @Slot(float)
    def set_audio_level(self, level: float) -> None:
        if self._snapshot.state is ApplicationState.RECORDING:
            self._target_level = min(1.0, max(0.0, level))

    def reset_position(self) -> None:
        """Place every new session at the lower-right; never persist drag position."""

        app = QApplication.instance()
        if app is None:
            return
        screen = QApplication.screenAt(self.cursor().pos()) or QApplication.primaryScreen()
        if screen is None:
            return
        available = screen.availableGeometry()
        margin = 24
        self.move(
            available.right() - self.width() - margin + 1,
            available.bottom() - self.height() - margin + 1,
        )

    def build_context_menu(self) -> QMenu:
        """Build the state-specific menu; exposed to enable headless verification."""

        menu = QMenu(self)
        state = self._snapshot.state
        style = self.style()
        if state in {ApplicationState.RECORDING, ApplicationState.PAUSED}:
            stop_action = QAction(
                style.standardIcon(QStyle.StandardPixmap.SP_MediaStop),
                "Detener y guardar",
                menu,
            )
            stop_action.setObjectName("stopAndSaveAction")
            stop_action.triggered.connect(self._controller.stop_and_process)
            menu.addAction(stop_action)

            if state is ApplicationState.PAUSED:
                pause_action = QAction(
                    style.standardIcon(QStyle.StandardPixmap.SP_MediaPlay),
                    "Reanudar",
                    menu,
                )
                pause_action.triggered.connect(self._controller.resume_recording)
            else:
                pause_action = QAction(
                    style.standardIcon(QStyle.StandardPixmap.SP_MediaPause),
                    "Pausar",
                    menu,
                )
                pause_action.triggered.connect(self._controller.pause_recording)
            pause_action.setObjectName("pauseResumeAction")
            menu.addAction(pause_action)

            cancel_action = QAction(
                style.standardIcon(QStyle.StandardPixmap.SP_TrashIcon),
                "Cancelar grabación",
                menu,
            )
            cancel_action.setObjectName("cancelRecordingAction")
            cancel_action.triggered.connect(self._confirm_cancel_recording)
            menu.addAction(cancel_action)
        elif state is ApplicationState.PROCESSING:
            cancel_action = QAction(
                style.standardIcon(QStyle.StandardPixmap.SP_DialogCancelButton),
                "Cancelar procesamiento",
                menu,
            )
            cancel_action.setObjectName("cancelProcessingAction")
            cancel_action.triggered.connect(self._confirm_cancel_processing)
            menu.addAction(cancel_action)
        return menu

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:  # noqa: N802
        menu = self.build_context_menu()
        if menu.actions():
            menu.exec(event.globalPos())
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() is Qt.MouseButton.LeftButton:
            self._press_global = event.globalPosition().toPoint()
            self._window_at_press = self.pos()
            self._dragged = False
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if (
            self._press_global is not None
            and self._window_at_press is not None
            and event.buttons() & Qt.MouseButton.LeftButton
        ):
            delta = event.globalPosition().toPoint() - self._press_global
            if delta.manhattanLength() >= QApplication.startDragDistance():
                self._dragged = True
                self.move(self._window_at_press + delta)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802
        if event.button() is Qt.MouseButton.LeftButton and self._press_global is not None:
            if not self._dragged:
                self.restoreRequested.emit()
            self._press_global = None
            self._window_at_press = None
            self._dragged = False
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def paintEvent(self, _event: object) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        state = self._snapshot.state
        if state is ApplicationState.RECORDING:
            self._paint_recording(painter)
        elif state is ApplicationState.PAUSED:
            self._paint_paused(painter)
        elif state in {
            ApplicationState.PREPARING,
            ApplicationState.STOPPING,
            ApplicationState.PROCESSING,
        }:
            self._paint_processing(painter)
        elif state is ApplicationState.COMPLETED:
            self._paint_completed(painter)
        elif state is ApplicationState.FAILED:
            self._paint_failed(painter)
        elif state is ApplicationState.CANCELLED:
            self._paint_cancelled(painter)

    @Slot()
    def _animate(self) -> None:
        self._phase = (self._phase + 0.045) % 1.0
        self._level += (self._target_level - self._level) * 0.24
        self.update()

    @Slot()
    def _finish_terminal_display(self) -> None:
        self._animation_timer.stop()
        self.hide()
        self.terminalDisplayFinished.emit()

    @Slot()
    def _confirm_cancel_recording(self) -> None:
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
    def _confirm_cancel_processing(self) -> None:
        answer = QMessageBox.question(
            self,
            "Cancelar procesamiento",
            "Se detendrá el procesamiento. El audio y la sesión recuperable se conservarán.",
            QMessageBox.StandardButton.Cancel | QMessageBox.StandardButton.Yes,
            QMessageBox.StandardButton.Cancel,
        )
        if answer is QMessageBox.StandardButton.Yes:
            self._controller.cancel_processing()

    def _paint_recording(self, painter: QPainter) -> None:
        color = self._preferences.accent
        center = QPointF(self.width() / 2, self.height() / 2)
        path = QPainterPath()
        count = 72
        for index in range(count + 1):
            fraction = index / count
            angle = fraction * math.tau
            ripple = math.sin(angle * 5 + self._phase * math.tau * 2)
            radius = 31 + ripple * (2 + self._level * 8)
            point = QPointF(
                center.x() + math.cos(angle) * radius,
                center.y() + math.sin(angle) * radius,
            )
            if index == 0:
                path.moveTo(point)
            else:
                path.lineTo(point)
        painter.setPen(QPen(color, 5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawPath(path)
        self._paint_center(painter, color)

    def _paint_paused(self, painter: QPainter) -> None:
        color = self._preferences.accent
        painter.setPen(QPen(color, 5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawEllipse(QRectF(20, 20, 64, 64))
        painter.drawLine(QPointF(43, 39), QPointF(43, 65))
        painter.drawLine(QPointF(61, 39), QPointF(61, 65))

    def _paint_processing(self, painter: QPainter) -> None:
        color = self._preferences.accent
        painter.setPen(QPen(color, 5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        rotation = self._phase * math.tau
        center = QPointF(52, 52)
        path = QPainterPath()
        for index in range(90):
            fraction = index / 89
            angle = rotation + fraction * math.tau * 1.7
            radius = 9 + fraction * 28
            point = QPointF(
                center.x() + math.cos(angle) * radius,
                center.y() + math.sin(angle) * radius,
            )
            if index == 0:
                path.moveTo(point)
            else:
                path.lineTo(point)
        painter.drawPath(path)

    def _paint_completed(self, painter: QPainter) -> None:
        painter.setPen(
            QPen(QColor("#21b36b"), 8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        )
        path = QPainterPath(QPointF(27, 53))
        path.lineTo(45, 70)
        path.lineTo(79, 33)
        painter.drawPath(path)

    def _paint_failed(self, painter: QPainter) -> None:
        painter.setPen(
            QPen(QColor("#e5484d"), 8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap)
        )
        painter.drawLine(QPointF(31, 31), QPointF(73, 73))
        painter.drawLine(QPointF(73, 31), QPointF(31, 73))

    def _paint_cancelled(self, painter: QPainter) -> None:
        color = QColor("#80868b")
        painter.setPen(QPen(color, 5, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawEllipse(QRectF(20, 20, 64, 64))
        painter.drawLine(QPointF(36, 52), QPointF(68, 52))

    @staticmethod
    def _paint_center(painter: QPainter, color: QColor) -> None:
        painter.setPen(Qt.PenStyle.NoPen)
        fill = QColor(color)
        fill.setAlpha(210)
        painter.setBrush(fill)
        painter.drawEllipse(QPointF(52, 52), 9, 9)


def _is_active_overlay_state(state: ApplicationState) -> bool:
    return state in {
        ApplicationState.PREPARING,
        ApplicationState.RECORDING,
        ApplicationState.PAUSED,
        ApplicationState.STOPPING,
        ApplicationState.PROCESSING,
    }


def _is_overlay_state(state: ApplicationState) -> bool:
    return _is_active_overlay_state(state) or state in {
        ApplicationState.COMPLETED,
        ApplicationState.FAILED,
        ApplicationState.CANCELLED,
    }
