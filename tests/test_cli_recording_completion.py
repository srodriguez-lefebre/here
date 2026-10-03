import threading
from types import SimpleNamespace

import here.cli as cli
import pytest
import typer
from here.application import ApplicationState, EventKind, HereApplicationController
from here.recording.models import RecordingSession


@pytest.fixture
def recording_owner(tmp_path, monkeypatch):
    capture_done = threading.Event()
    cleanup_entered = threading.Event()
    release_cleanup = threading.Event()
    events = []

    class Capture:
        error = None
        cancelled = False

        def wait(self):
            assert capture_done.wait(5)
            if self.error is not None:
                raise self.error
            return RecordingSession(sources=[])

        def stop(self):
            capture_done.set()

        def cancel(self):
            self.cancelled = True
            capture_done.set()

    class Live:
        def abort(self):
            pass

        def cleanup(self):
            pass

        def wait_closed(self):
            cleanup_entered.set()
            assert release_cleanup.wait(5)

    capture = Capture()
    core = HereApplicationController(
        capture_factory=lambda *args: capture,
        live_factory=lambda *args: Live(),
        processor=SimpleNamespace(process=lambda *a, **kw: SimpleNamespace(session_dir=tmp_path)),
    )
    core.subscribe(events.append)
    monkeypatch.setattr(cli, "create_default_controller", lambda: core)
    owner = SimpleNamespace(
        core=core,
        capture=capture,
        capture_done=capture_done,
        cleanup_entered=cleanup_entered,
        release_cleanup=release_cleanup,
        events=events,
    )
    yield owner
    capture_done.set()
    release_cleanup.set()
    core.wait_until_terminal(5)


def run_cli(tmp_path):
    done = threading.Event()
    results = []

    def run():
        try:
            cli._run_recording(cli.record_mic_until_enter, tmp_path, expected_source_count=1)
        except BaseException as exc:
            results.append(exc)
        else:
            results.append(None)
        finally:
            done.set()

    thread = threading.Thread(target=run)
    thread.start()
    return thread, done, results


@pytest.mark.parametrize("branch", ["startup", "runtime", "generic", "interrupt"])
def test_failed_cli_waits_for_actual_worker_cleanup(tmp_path, monkeypatch, recording_owner, branch):
    owner = recording_owner
    failure = OSError("original capture failure")
    owner.capture.error = failure
    input_errors = {
        "runtime": RuntimeError("input runtime failure"),
        "generic": EOFError("input closed"),
        "interrupt": KeyboardInterrupt(),
    }
    if branch == "startup":

        def fail_open(*args):
            raise failure

        owner.core._capture_factory = fail_open
    else:

        def failed_input():
            owner.capture_done.set()
            assert owner.cleanup_entered.wait(3)
            raise input_errors[branch]

        monkeypatch.setattr("builtins.input", failed_input)

    thread, done, results = run_cli(tmp_path)
    try:
        assert owner.cleanup_entered.wait(3)
        assert owner.core.snapshot.state is ApplicationState.FAILED
        assert not owner.core.snapshot.worker_complete
        assert not done.wait(0.15), "CLI returned while its recording worker still owned cleanup"
    finally:
        owner.release_cleanup.set()
        thread.join(5)
    assert not thread.is_alive()
    (exit_error,) = results
    assert isinstance(exit_error, typer.Exit)
    assert exit_error.exit_code == (130 if branch == "interrupt" else 1)
    if branch == "startup":
        assert str(exit_error.__cause__) == "original capture failure"
    elif branch != "interrupt":
        assert exit_error.__cause__ is input_errors[branch]
    assert owner.core.snapshot.state is ApplicationState.FAILED
    assert owner.core.snapshot.last_error.message == "original capture failure"
    assert not owner.capture.cancelled
    assert owner.core.snapshot.worker_complete
    assert owner.events[-1].kind is EventKind.WORKER_COMPLETED
    assert not owner.core._worker.is_alive()
    assert not owner.core._completion.is_alive()


@pytest.mark.parametrize("error", [RuntimeError("input failed"), EOFError(), KeyboardInterrupt()])
def test_active_cli_error_cancels_then_awaits_cleanup(
    tmp_path, monkeypatch, recording_owner, error
):
    owner = recording_owner

    def fail_input():
        raise error

    monkeypatch.setattr("builtins.input", fail_input)
    thread, done, results = run_cli(tmp_path)
    try:
        assert owner.cleanup_entered.wait(3)
        assert owner.capture.cancelled
        assert not done.wait(0.15)
    finally:
        owner.release_cleanup.set()
        thread.join(5)
    (exit_error,) = results
    assert isinstance(exit_error, typer.Exit)
    assert exit_error.exit_code == (130 if isinstance(error, KeyboardInterrupt) else 1)
    assert owner.core.snapshot.state is ApplicationState.CANCELLED
    assert owner.core.snapshot.worker_complete
    assert owner.events[-1].kind is EventKind.WORKER_COMPLETED


def test_successful_cli_waits_for_cleanup_before_return(tmp_path, monkeypatch, recording_owner):
    owner = recording_owner
    monkeypatch.setattr("builtins.input", lambda: "")
    thread, done, results = run_cli(tmp_path)
    try:
        assert owner.cleanup_entered.wait(3)
        assert owner.core.snapshot.state is ApplicationState.COMPLETED
        assert not done.wait(0.15)
    finally:
        owner.release_cleanup.set()
        thread.join(5)
    assert results == [None]
    assert owner.core.snapshot.worker_complete
    assert not owner.capture.cancelled


def test_cli_preflight_failure_without_worker_keeps_original_error(tmp_path, monkeypatch):
    from here.recording.reservation import audio_reservation

    core = HereApplicationController()
    monkeypatch.setattr(cli, "create_default_controller", lambda: core)
    audio_reservation.acquire()
    try:
        with pytest.raises(typer.Exit) as caught:
            cli._run_recording(cli.record_mic_until_enter, tmp_path, expected_source_count=1)
    finally:
        audio_reservation.release()
    assert caught.value.exit_code == 1
    assert "ocupados" in str(caught.value.__cause__)
    assert core.snapshot.state is ApplicationState.IDLE
    assert core.snapshot.worker_complete
    assert core._worker is None


@pytest.mark.parametrize("error", [RuntimeError("input failed"), EOFError(), KeyboardInterrupt()])
def test_cli_cancellation_race_still_awaits_completion_and_keeps_exit_code(
    tmp_path, monkeypatch, recording_owner, error
):
    owner = recording_owner
    owner.capture.error = OSError("capture failed during cancellation")
    cancel = owner.core.cancel

    def racing_cancel():
        # CLI observed RECORDING, but failure reaches its final cleanup before cancel.
        owner.capture_done.set()
        assert owner.cleanup_entered.wait(3)
        cancel()

    def fail_input():
        raise error

    monkeypatch.setattr(owner.core, "cancel", racing_cancel)
    monkeypatch.setattr("builtins.input", fail_input)
    messages = []
    log_sink = cli.logger.add(lambda message: messages.append(str(message)))
    thread, done, results = run_cli(tmp_path)
    try:
        assert owner.cleanup_entered.wait(3)
        assert not done.wait(0.15), "Cancellation failure skipped the worker completion fence"
    finally:
        owner.release_cleanup.set()
        thread.join(5)
        cli.logger.remove(log_sink)
    (exit_error,) = results
    assert isinstance(exit_error, typer.Exit)
    assert exit_error.exit_code == (130 if isinstance(error, KeyboardInterrupt) else 1)
    assert any("Failed to clean up" in message for message in messages)
    assert owner.core.snapshot.state is ApplicationState.FAILED
    assert owner.core.snapshot.worker_complete
    assert not owner.capture.cancelled
    assert owner.events[-1].kind is EventKind.WORKER_COMPLETED


def test_interrupt_logs_completion_error_and_keeps_exit_code(
    tmp_path, monkeypatch, recording_owner
):
    owner = recording_owner
    wait = owner.core.wait_until_terminal

    def wait_with_error():
        wait()
        raise OSError("completion reporting failed")

    def interrupt():
        raise KeyboardInterrupt()

    monkeypatch.setattr(owner.core, "wait_until_terminal", wait_with_error)
    monkeypatch.setattr("builtins.input", interrupt)
    messages = []
    log_sink = cli.logger.add(lambda message: messages.append(str(message)))
    thread, done, results = run_cli(tmp_path)
    try:
        assert owner.cleanup_entered.wait(3)
        assert not done.wait(0.15)
    finally:
        owner.release_cleanup.set()
        thread.join(5)
        monkeypatch.setattr(owner.core, "wait_until_terminal", wait)
        cli.logger.remove(log_sink)
    (exit_error,) = results
    assert isinstance(exit_error, typer.Exit)
    assert exit_error.exit_code == 130
    assert any("completion reporting failed" in message for message in messages)
    assert any("Failed to clean up" in message for message in messages)
    assert not any("Saved to" in message for message in messages)
    assert owner.core.snapshot.worker_complete
