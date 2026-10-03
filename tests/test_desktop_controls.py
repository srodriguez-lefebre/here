import threading
from dataclasses import replace

import pytest
from here.application import ApplicationSnapshot, ApplicationState
from here.application.models import OpenedSource
from here.application.recovery import RecoveryCandidate
from here.recording.diagnostics import AudioDeviceInfo, SignalTestResult
from here.ui.app import HereDesktop
from here.ui.contract import ApplicationUiAdapter
from here.ui.fake_controller import FakeApplicationController
from PySide6.QtWidgets import QApplication


def desktop(qtbot, tmp_path, **kwargs):
    core = FakeApplicationController()
    adapter = ApplicationUiAdapter(core, tmp_path, **kwargs)
    result = HereDesktop(QApplication.instance(), adapter)
    qtbot.addWidget(result.main_window)
    qtbot.addWidget(result.overlay)
    return core, result


def test_actual_opened_names_replace_generic_sources(qtbot, tmp_path):
    _, ui = desktop(qtbot, tmp_path)
    ui.main_window.set_snapshot(
        ApplicationSnapshot(
            state=ApplicationState.RECORDING,
            opened_sources=(OpenedSource("microphone", "Actual USB", 3, 48000, 2),),
        )
    )
    assert "Actual USB" in ui.main_window.findChild(object, "sourceLabel").text()


def test_background_signal_truthful_and_ui_remains_responsive(qtbot, tmp_path):
    entered, release = threading.Event(), threading.Event()
    main_thread = threading.get_ident()

    class Diagnostics:
        def test_signal(self, source):
            assert threading.get_ident() != main_thread
            entered.set()
            assert release.wait(3)
            return SignalTestResult(
                source, AudioDeviceInfo(source, "Mic", 0, 48000, 1), 3.0, 0.0, 0.0, False
            )

    _, ui = desktop(qtbot, tmp_path, diagnostics_service=Diagnostics())
    qtbot.waitUntil(lambda: not ui.jobs.busy)
    ui.main_window.findChild(object, "testMicrophoneButton").click()
    try:
        qtbot.waitUntil(entered.is_set)
        assert not ui.main_window.findChild(object, "startButton").isEnabled()
        assert ui.main_window.findChild(object, "exitButton").isEnabled()
    finally:
        release.set()
    qtbot.waitUntil(lambda: not ui.jobs.busy)
    assert "Sin señal" in ui.main_window.findChild(object, "diagnosticsLabel").text()


def test_recovery_discovery_is_local_and_retry_requires_selection(qtbot, tmp_path):
    calls = []
    candidate = RecoveryCandidate(
        tmp_path / "recovered", "Interrupted", 4, "interrupted", None, True
    )

    class Recovery:
        def discover(self):
            calls.append("discover")
            return [candidate]

        def materialize(self, item):
            calls.append("materialize")
            assert item == candidate
            return item.session_dir

    core, ui = desktop(qtbot, tmp_path, recovery_service=Recovery())
    qtbot.waitUntil(lambda: not ui.jobs.busy)
    assert calls == ["discover"]
    assert core.command_log == []
    ui.main_window.findChild(object, "recoverButton").click()
    qtbot.waitUntil(lambda: "retry" in core.command_log)
    assert calls == ["discover", "materialize"]


def test_exit_does_not_quit_until_terminal_and_completion(qtbot, tmp_path, monkeypatch):
    core, ui = desktop(qtbot, tmp_path)
    qtbot.waitUntil(lambda: not ui.jobs.busy)
    quit_calls = []
    monkeypatch.setattr(ui.application, "quit", lambda: quit_calls.append(True))
    core.set_state(ApplicationState.PREPARING)
    core._snapshot = replace(core.snapshot, worker_complete=False)
    ui.request_exit()
    ui.request_exit()
    assert core.command_log == ["stop"]
    core.set_state(ApplicationState.COMPLETED, worker_complete=False)
    ui._check_exit()
    assert quit_calls == []
    core._snapshot = replace(core.snapshot, worker_complete=True)
    ui._check_exit()
    assert quit_calls == [True]


@pytest.mark.parametrize(
    "state",
    [
        ApplicationState.RECORDING,
        ApplicationState.PAUSED,
        ApplicationState.STOPPING,
        ApplicationState.PROCESSING,
        ApplicationState.FAILED,
    ],
)
def test_exit_every_state_preserves_stop_save_semantics(qtbot, tmp_path, monkeypatch, state):
    core, ui = desktop(qtbot, tmp_path)
    qtbot.waitUntil(lambda: not ui.jobs.busy)
    calls = []
    monkeypatch.setattr(ui.application, "quit", lambda: calls.append("quit"))
    core.set_state(state, worker_complete=False)
    ui.request_exit()
    assert core.command_log == (
        ["stop"] if state in {ApplicationState.RECORDING, ApplicationState.PAUSED} else []
    )
    assert calls == []
    core.set_state(ApplicationState.FAILED, worker_complete=True)
    qtbot.waitUntil(lambda: calls == ["quit"])


@pytest.mark.parametrize("shutdown", [False, True])
def test_stale_diagnostic_delivery_ignored_after_capture_or_shutdown(
    qtbot, tmp_path, monkeypatch, shutdown
):
    entered, release = threading.Event(), threading.Event()

    class Diagnostics:
        def test_signal(self, source):
            entered.set()
            assert release.wait(3)
            raise RuntimeError("late result")

    core, ui = desktop(qtbot, tmp_path, diagnostics_service=Diagnostics())
    qtbot.waitUntil(lambda: not ui.jobs.busy)
    calls = []
    monkeypatch.setattr(ui.application, "quit", lambda: calls.append(True))
    ui._diagnose("microphone")
    try:
        qtbot.waitUntil(entered.is_set)
        if shutdown:
            ui.request_exit()
            assert calls == []
        else:
            core.set_state(ApplicationState.RECORDING)
            qtbot.waitUntil(lambda: ui._epoch > 0)
    finally:
        release.set()
    qtbot.waitUntil(lambda: not ui.jobs.busy)
    assert "late result" not in ui.main_window.findChild(object, "diagnosticsLabel").text()
    assert calls == ([True] if shutdown else [])


def test_selected_retry_missing_key_preserves_recovery_before_materialization(
    qtbot, tmp_path, monkeypatch
):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    calls = []
    journal = tmp_path / "keep-journal.json"
    journal.write_text("preserve")
    candidate = RecoveryCandidate(tmp_path, "Interrupted", 3, "interrupted", None, True)
    other = RecoveryCandidate(tmp_path / "other", "Other", 2, "failed", None, True)

    class Recovery:
        def discover(self):
            return [other, candidate]

        def materialize(self, item):
            assert item == candidate
            calls.append("materialize")
            return tmp_path

    core, ui = desktop(qtbot, tmp_path, recovery_service=Recovery())
    qtbot.waitUntil(lambda: not ui.jobs.busy)
    selector = ui.main_window.findChild(object, "recoverySelector")
    selector.setCurrentIndex(1)
    ui.main_window.findChild(object, "recoverButton").click()
    qtbot.waitUntil(lambda: not ui.jobs.busy)
    assert calls == []
    assert core.command_log == []
    assert journal.read_text() == "preserve"
    assert "OPENAI_API_KEY" in ui.main_window.findChild(object, "recoveryLabel").text()
    assert selector.count() == 2
    assert selector.currentData() == candidate
    assert selector.isEnabled()
    assert ui.main_window.findChild(object, "recoverButton").isEnabled()

    # Correct the effective file; the same window must permit a new explicit operation.
    from here.config.paths import get_env_file

    get_env_file().write_text("OPENAI_API_KEY=synthetic-test-only\n")
    assert calls == [] and core.command_log == []
    ui.main_window.findChild(object, "recoverButton").click()
    qtbot.waitUntil(lambda: not ui.jobs.busy)
    assert calls == ["materialize"]
    assert core.command_log == ["retry"]
    assert journal.read_text() == "preserve"


def test_fast_terminal_worker_event_refreshes_recovery(qtbot, tmp_path):
    from here.application import ApplicationEvent, EventKind

    calls = []

    class Recovery:
        def discover(self):
            calls.append(True)
            return []

    core, ui = desktop(qtbot, tmp_path, recovery_service=Recovery())
    qtbot.waitUntil(lambda: not ui.jobs.busy)
    # A job may finish before Qt delivers even its first state snapshot.
    core._snapshot = ApplicationSnapshot(state=ApplicationState.COMPLETED, worker_complete=True)
    core.publish(
        ApplicationEvent(kind=EventKind.WORKER_COMPLETED, state=ApplicationState.COMPLETED)
    )
    qtbot.waitUntil(lambda: len(calls) == 2, timeout=1000)
