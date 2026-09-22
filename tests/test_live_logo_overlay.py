from pathlib import Path

import pytest
from here.application import ApplicationSnapshot, ApplicationState
from here.ui.contract import ApplicationUiAdapter
from here.ui.fake_controller import FakeApplicationController
from here.ui.overlay import (
    CANCELLED_DURATION_MS,
    TERMINAL_DURATION_MS,
    LiveLogoOverlay,
)
from here.ui.preferences import VisualPreferences
from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QColor
from PySide6.QtTest import QSignalSpy
from PySide6.QtWidgets import QApplication, QMessageBox
from pytestqt.qtbot import QtBot


@pytest.fixture
def overlay_parts(tmp_path: Path, qtbot: QtBot) -> tuple[
    FakeApplicationController,
    ApplicationUiAdapter,
    LiveLogoOverlay,
]:
    settings = QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat)
    core = FakeApplicationController()
    adapter = ApplicationUiAdapter(core, tmp_path)
    overlay = LiveLogoOverlay(adapter, VisualPreferences(settings))
    qtbot.addWidget(overlay)
    return core, adapter, overlay


def _menu_labels(overlay: LiveLogoOverlay) -> list[str]:
    menu = overlay.build_context_menu()
    return [action.text() for action in menu.actions()]


def test_overlay_is_transparent_frameless_and_always_on_top(
    overlay_parts: tuple[FakeApplicationController, ApplicationUiAdapter, LiveLogoOverlay],
) -> None:
    _, _, overlay = overlay_parts

    assert overlay.testAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
    assert overlay.windowFlags() & Qt.WindowType.FramelessWindowHint
    assert overlay.windowFlags() & Qt.WindowType.WindowStaysOnTopHint


@pytest.mark.parametrize(
    ("state", "labels"),
    [
        (
            ApplicationState.RECORDING,
            ["Detener y guardar", "Pausar", "Cancelar grabación"],
        ),
        (
            ApplicationState.PAUSED,
            ["Detener y guardar", "Reanudar", "Cancelar grabación"],
        ),
        (ApplicationState.PROCESSING, ["Cancelar procesamiento"]),
        (ApplicationState.IDLE, []),
    ],
)
def test_context_menu_matches_application_state_and_every_action_has_an_icon(
    overlay_parts: tuple[FakeApplicationController, ApplicationUiAdapter, LiveLogoOverlay],
    state: ApplicationState,
    labels: list[str],
) -> None:
    _, _, overlay = overlay_parts
    overlay.set_snapshot(ApplicationSnapshot(state=state))

    menu = overlay.build_context_menu()

    assert [action.text() for action in menu.actions()] == labels
    assert all(not action.icon().isNull() for action in menu.actions())


def test_recording_menu_dispatches_stop_pause_and_confirmed_cancel(
    overlay_parts: tuple[FakeApplicationController, ApplicationUiAdapter, LiveLogoOverlay],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core, _, overlay = overlay_parts
    core.set_state(ApplicationState.RECORDING)
    overlay.set_snapshot(core.snapshot)
    menu = overlay.build_context_menu()

    menu.actions()[0].trigger()
    core.set_state(ApplicationState.RECORDING)
    overlay.set_snapshot(core.snapshot)
    menu = overlay.build_context_menu()
    menu.actions()[1].trigger()
    core.set_state(ApplicationState.RECORDING)
    overlay.set_snapshot(core.snapshot)
    menu = overlay.build_context_menu()
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Yes,
    )
    menu.actions()[2].trigger()

    assert core.command_log == ["stop", "pause", "cancel"]
    assert core.snapshot.recoverable is False


def test_processing_cancel_is_confirmed_and_recoverable(
    overlay_parts: tuple[FakeApplicationController, ApplicationUiAdapter, LiveLogoOverlay],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    core, _, overlay = overlay_parts
    core.set_state(ApplicationState.PROCESSING, recoverable=True)
    overlay.set_snapshot(core.snapshot)
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Yes,
    )

    overlay.build_context_menu().actions()[0].trigger()

    assert core.command_log == ["cancel"]
    assert core.snapshot.recoverable is True


def test_primary_click_restores_but_a_drag_does_not(
    overlay_parts: tuple[FakeApplicationController, ApplicationUiAdapter, LiveLogoOverlay],
    qtbot: QtBot,
) -> None:
    _, _, overlay = overlay_parts
    overlay.set_snapshot(ApplicationSnapshot(state=ApplicationState.RECORDING))
    spy = QSignalSpy(overlay.restoreRequested)

    qtbot.mouseClick(overlay, Qt.MouseButton.LeftButton, pos=overlay.rect().center())
    assert spy.count() == 1

    qtbot.mousePress(overlay, Qt.MouseButton.LeftButton, pos=overlay.rect().center())
    overlay._dragged = True
    qtbot.mouseRelease(overlay, Qt.MouseButton.LeftButton, pos=overlay.rect().center())
    assert spy.count() == 1


@pytest.mark.parametrize(
    ("state", "duration"),
    [
        (ApplicationState.COMPLETED, TERMINAL_DURATION_MS),
        (ApplicationState.FAILED, TERMINAL_DURATION_MS),
        (ApplicationState.CANCELLED, CANCELLED_DURATION_MS),
    ],
)
def test_terminal_states_use_bounded_timers_and_hide_without_sleeping(
    overlay_parts: tuple[FakeApplicationController, ApplicationUiAdapter, LiveLogoOverlay],
    qtbot: QtBot,
    state: ApplicationState,
    duration: int,
) -> None:
    _, _, overlay = overlay_parts
    finished = QSignalSpy(overlay.terminalDisplayFinished)

    overlay.set_snapshot(
        ApplicationSnapshot(
            state=state,
            recoverable=state is ApplicationState.CANCELLED,
        )
    )
    qtbot.waitUntil(overlay.isVisible)

    assert overlay.terminal_timer.isActive()
    assert overlay.terminal_timer.interval() == duration
    overlay.terminal_timer.timeout.emit()
    assert not overlay.isVisible()
    assert finished.count() == 1


def test_destructive_recording_cancellation_disappears_without_terminal_symbol(
    overlay_parts: tuple[FakeApplicationController, ApplicationUiAdapter, LiveLogoOverlay],
) -> None:
    _, _, overlay = overlay_parts
    finished = QSignalSpy(overlay.terminalDisplayFinished)
    overlay.set_snapshot(ApplicationSnapshot(state=ApplicationState.RECORDING))
    assert overlay.isVisible()

    overlay.set_snapshot(
        ApplicationSnapshot(state=ApplicationState.CANCELLED, recoverable=False)
    )

    assert not overlay.isVisible()
    assert not overlay.terminal_timer.isActive()
    assert finished.count() == 1


def test_recoverable_processing_cancellation_shows_neutral_terminal_symbol(
    overlay_parts: tuple[FakeApplicationController, ApplicationUiAdapter, LiveLogoOverlay],
    qtbot: QtBot,
) -> None:
    _, _, overlay = overlay_parts
    overlay.set_snapshot(
        ApplicationSnapshot(state=ApplicationState.CANCELLED, recoverable=True)
    )
    qtbot.waitUntil(overlay.isVisible)

    assert overlay.terminal_timer.isActive()
    assert overlay.terminal_timer.interval() == CANCELLED_DURATION_MS


def test_combined_audio_level_is_ignored_while_paused(
    overlay_parts: tuple[FakeApplicationController, ApplicationUiAdapter, LiveLogoOverlay],
) -> None:
    _, _, overlay = overlay_parts
    overlay.set_snapshot(ApplicationSnapshot(state=ApplicationState.RECORDING))
    overlay.set_audio_level(0.8)
    assert overlay._target_level == pytest.approx(0.8)

    overlay.set_snapshot(ApplicationSnapshot(state=ApplicationState.PAUSED))
    overlay.set_audio_level(0.9)
    assert overlay._target_level == 0.0
    assert not overlay.animation_timer.isActive()


def test_new_session_resets_to_lower_right_but_pause_keeps_drag_position(
    overlay_parts: tuple[FakeApplicationController, ApplicationUiAdapter, LiveLogoOverlay],
) -> None:
    _, _, overlay = overlay_parts
    screen = QApplication.primaryScreen()
    assert screen is not None
    available = screen.availableGeometry()
    expected = (
        available.right() - overlay.width() - 24 + 1,
        available.bottom() - overlay.height() - 24 + 1,
    )

    overlay.move(3, 7)
    overlay.set_snapshot(ApplicationSnapshot(state=ApplicationState.RECORDING))
    assert (overlay.x(), overlay.y()) == expected

    overlay.move(15, 19)
    overlay.set_snapshot(ApplicationSnapshot(state=ApplicationState.PAUSED))
    assert (overlay.x(), overlay.y()) == (15, 19)

    overlay.set_snapshot(ApplicationSnapshot(state=ApplicationState.IDLE))
    overlay.set_snapshot(ApplicationSnapshot(state=ApplicationState.RECORDING))
    assert (overlay.x(), overlay.y()) == expected


@pytest.mark.parametrize(
    ("state", "semantic_color"),
    [
        (ApplicationState.COMPLETED, QColor("#21b36b")),
        (ApplicationState.FAILED, QColor("#e5484d")),
    ],
)
def test_terminal_semantic_colors_do_not_follow_the_accent(
    overlay_parts: tuple[FakeApplicationController, ApplicationUiAdapter, LiveLogoOverlay],
    qtbot: QtBot,
    state: ApplicationState,
    semantic_color: QColor,
) -> None:
    _, _, overlay = overlay_parts
    overlay.set_snapshot(ApplicationSnapshot(state=state))
    qtbot.waitUntil(overlay.isVisible)
    image = overlay.grab().toImage()

    colored_pixels = sum(
        1
        for x in range(image.width())
        for y in range(image.height())
        if QColor(image.pixelColor(x, y)).name() == semantic_color.name()
    )

    assert colored_pixels > 20
