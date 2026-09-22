from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import here.application.controller as controller_module
import numpy as np
import pytest
from here.application import (
    ApplicationState,
    EventKind,
    HereApplicationController,
    InvalidApplicationCommand,
    StartRequest,
)
from here.application.processing import ProcessingCancelled, SessionProcessingFailed
from here.recording.models import RecordingSession


class FakeCapture:
    def __init__(self, session: RecordingSession | None = None) -> None:
        self.session = session or RecordingSession(sources=[])
        self.done = threading.Event()
        self.paused = False
        self.cancelled = False

    def pause(self) -> None:
        self.paused = True

    def resume(self) -> None:
        self.paused = False

    def stop(self) -> None:
        self.done.set()

    def cancel(self) -> None:
        self.cancelled = True
        self.done.set()

    def wait(self, timeout: float | None = None) -> RecordingSession:
        assert self.done.wait(timeout or 2)
        if self.cancelled:
            raise RuntimeError("cancelled")
        return self.session


class DelayedStopCapture(FakeCapture):
    def __init__(self) -> None:
        super().__init__()
        self.release = threading.Event()

    def wait(self, timeout: float | None = None) -> RecordingSession:
        assert self.done.wait(timeout or 2)
        assert self.release.wait(timeout or 2)
        return self.session


class FakeLive:
    def __init__(self) -> None:
        self.blocks: list[tuple[object, ...]] = []
        self.aborted = False
        self.cleaned = False

    def submit_block(self, *args: object) -> None:
        self.blocks.append(args)

    def abort(self) -> None:
        self.aborted = True

    def cleanup(self) -> None:
        self.cleaned = True


class FakeProcessor:
    def __init__(self, session_dir: Path) -> None:
        self.session_dir = session_dir
        self.called = threading.Event()
        self.release = threading.Event()
        self.cancel_seen = False

    def process(self, session: object, target_dir: Path, **kwargs: object) -> object:
        del session, target_dir
        self.called.set()
        cancel_event = kwargs["cancel_event"]
        assert isinstance(cancel_event, threading.Event)
        while not self.release.wait(0.01):
            if cancel_event.is_set():
                self.cancel_seen = True
                raise ProcessingCancelled(self.session_dir)
        return SimpleNamespace(session_dir=self.session_dir)

    def retry(self, session_dir: Path, **kwargs: object) -> object:
        del kwargs
        return SimpleNamespace(session_dir=session_dir)


def wait_for_state(
    controller: HereApplicationController,
    state: ApplicationState,
    timeout: float = 2,
) -> None:
    deadline = time.monotonic() + timeout
    while controller.snapshot.state is not state and time.monotonic() < deadline:
        time.sleep(0.005)
    assert controller.snapshot.state is state


def build_controller(
    tmp_path: Path,
) -> tuple[HereApplicationController, FakeCapture, FakeLive, FakeProcessor]:
    capture = FakeCapture()
    live = FakeLive()
    processor = FakeProcessor(tmp_path / "session")
    controller = HereApplicationController(
        capture_factory=lambda request, sink: capture,
        live_factory=lambda count, alt: live,
        processor=processor,  # type: ignore[arg-type]
    )
    return controller, capture, live, processor


def test_start_stop_and_complete_emit_ordered_states(tmp_path: Path) -> None:
    controller, capture, _, processor = build_controller(tmp_path)
    states: list[ApplicationState] = []
    controller.subscribe(
        lambda event: states.append(event.state) if event.kind is EventKind.STATE_CHANGED else None
    )

    controller.start(StartRequest(output_dir=tmp_path))
    wait_for_state(controller, ApplicationState.RECORDING)
    controller.stop()
    assert capture.done.is_set()
    assert processor.called.wait(2)
    processor.release.set()

    snapshot = controller.wait_until_terminal(2)

    assert snapshot.state is ApplicationState.COMPLETED
    assert snapshot.session_dir == tmp_path / "session"
    assert states == [
        ApplicationState.PREPARING,
        ApplicationState.RECORDING,
        ApplicationState.STOPPING,
        ApplicationState.PROCESSING,
        ApplicationState.COMPLETED,
    ]


def test_pause_resume_controls_capture_and_records_wall_duration(tmp_path: Path) -> None:
    controller, capture, _, processor = build_controller(tmp_path)

    controller.start(StartRequest(output_dir=tmp_path))
    wait_for_state(controller, ApplicationState.RECORDING)
    controller.pause()

    assert controller.snapshot.state is ApplicationState.PAUSED
    assert capture.paused
    time.sleep(0.01)
    controller.resume()
    assert not capture.paused
    assert controller.snapshot.total_paused_seconds > 0
    controller.stop()
    assert processor.called.wait(2)
    processor.release.set()
    controller.wait_until_terminal(2)


def test_cancel_during_recording_is_destructive_and_has_no_session(tmp_path: Path) -> None:
    controller, capture, live, processor = build_controller(tmp_path)

    controller.start(StartRequest(output_dir=tmp_path))
    wait_for_state(controller, ApplicationState.RECORDING)
    controller.cancel()
    snapshot = controller.wait_until_terminal(2)

    assert capture.cancelled
    assert live.aborted
    assert snapshot.state is ApplicationState.CANCELLED
    assert snapshot.session_dir is None
    assert not snapshot.recoverable
    assert not processor.called.is_set()


def test_cancel_during_processing_preserves_recoverable_session(tmp_path: Path) -> None:
    controller, _, _, processor = build_controller(tmp_path)
    events = []
    controller.subscribe(events.append)

    controller.start(StartRequest(output_dir=tmp_path))
    wait_for_state(controller, ApplicationState.RECORDING)
    controller.stop()
    assert processor.called.wait(2)
    wait_for_state(controller, ApplicationState.PROCESSING)
    controller.cancel()
    snapshot = controller.wait_until_terminal(2)

    assert processor.cancel_seen
    assert snapshot.state is ApplicationState.CANCELLED
    assert snapshot.recoverable
    assert snapshot.session_dir == tmp_path / "session"
    assert any(event.kind is EventKind.SESSION_PERSISTED for event in events)


def test_cancel_while_stopping_aborts_live_before_audio_materialization(tmp_path: Path) -> None:
    capture = DelayedStopCapture()
    live = FakeLive()
    processor = FakeProcessor(tmp_path / "session")
    controller = HereApplicationController(
        capture_factory=lambda request, sink: capture,
        live_factory=lambda count, alt: live,
        processor=processor,  # type: ignore[arg-type]
    )
    controller.start(StartRequest(output_dir=tmp_path))
    wait_for_state(controller, ApplicationState.RECORDING)

    controller.stop()
    assert controller.snapshot.state is ApplicationState.STOPPING
    controller.cancel()

    assert live.aborted
    capture.release.set()
    snapshot = controller.wait_until_terminal(2)
    assert snapshot.state is ApplicationState.CANCELLED
    assert snapshot.recoverable


def test_audio_telemetry_is_combined_normalized_and_not_emitted_while_paused(
    tmp_path: Path,
) -> None:
    captured_sink = None
    capture = FakeCapture()
    live = FakeLive()
    processor = FakeProcessor(tmp_path / "session")

    def capture_factory(request: object, sink: object) -> FakeCapture:
        nonlocal captured_sink
        del request
        captured_sink = sink
        return capture

    controller = HereApplicationController(
        capture_factory=capture_factory,  # type: ignore[arg-type]
        live_factory=lambda count, alt: live,
        processor=processor,  # type: ignore[arg-type]
        monotonic=iter([1.0, 2.0, 3.0]).__next__,
    )
    levels = []
    controller.subscribe(
        lambda event: (
            levels.append(event.audio_level) if event.kind is EventKind.AUDIO_LEVEL else None
        )
    )
    controller.start(StartRequest(output_dir=tmp_path))
    wait_for_state(controller, ApplicationState.RECORDING)
    assert callable(captured_sink)

    captured_sink("microphone", np.array([[16384]], dtype=np.int16), 16000, 1)
    controller.pause()
    captured_sink("microphone", np.array([[32767]], dtype=np.int16), 16000, 1)

    assert len(levels) == 1
    assert levels[0].source == "combined"
    assert levels[0].peak == pytest.approx(0.5)
    controller.cancel()
    controller.wait_until_terminal(2)


def test_audio_telemetry_rechecks_state_before_emitting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    controller, _, _, _ = build_controller(tmp_path)
    levels = []
    controller.subscribe(
        lambda event: levels.append(event) if event.kind is EventKind.AUDIO_LEVEL else None
    )
    controller.start(StartRequest(output_dir=tmp_path))
    wait_for_state(controller, ApplicationState.RECORDING)
    original_max = controller_module.np.max
    paused = False

    def pause_during_level_calculation(*args: object, **kwargs: object) -> object:
        nonlocal paused
        if not paused:
            paused = True
            controller.pause()
        return original_max(*args, **kwargs)

    monkeypatch.setattr(controller_module.np, "max", pause_during_level_calculation)

    controller._block_sink("microphone", np.array([[16384]], dtype=np.int16), 16000, 1)

    assert controller.snapshot.state is ApplicationState.PAUSED
    assert levels == []
    controller.cancel()
    controller.wait_until_terminal(2)


def test_invalid_commands_are_rejected_without_changing_state(tmp_path: Path) -> None:
    controller, _, _, _ = build_controller(tmp_path)

    with pytest.raises(InvalidApplicationCommand, match="Pause"):
        controller.pause()

    assert controller.snapshot.state is ApplicationState.IDLE


def test_processing_failure_exposes_recoverable_session_for_retry(tmp_path: Path) -> None:
    controller, _, _, processor = build_controller(tmp_path)

    def fail_process(*args: object, **kwargs: object) -> object:
        del args, kwargs
        processor.called.set()
        raise SessionProcessingFailed("provider down", processor.session_dir, recoverable=True)

    processor.process = fail_process  # type: ignore[method-assign]
    controller.start(StartRequest(output_dir=tmp_path))
    wait_for_state(controller, ApplicationState.RECORDING)
    controller.stop()

    snapshot = controller.wait_until_terminal(2)

    assert snapshot.state is ApplicationState.FAILED
    assert snapshot.recoverable
    assert snapshot.session_dir == processor.session_dir
    assert snapshot.last_error is not None


def test_successful_retry_clears_the_previous_error(tmp_path: Path) -> None:
    controller, _, _, processor = build_controller(tmp_path)

    def fail_process(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise SessionProcessingFailed("provider down", processor.session_dir, recoverable=True)

    processor.process = fail_process  # type: ignore[method-assign]
    controller.start(StartRequest(output_dir=tmp_path))
    wait_for_state(controller, ApplicationState.RECORDING)
    controller.stop()
    failed = controller.wait_until_terminal(2)
    assert failed.last_error is not None

    controller.retry()
    recovered = controller.wait_until_terminal(2)

    assert recovered.state is ApplicationState.COMPLETED
    assert recovered.last_error is None


def test_finishing_worker_does_not_clear_a_new_recording(tmp_path: Path) -> None:
    first_capture = FakeCapture()
    second_capture = FakeCapture()
    captures = iter([first_capture, second_capture])

    class ImmediateProcessor(FakeProcessor):
        def process(self, session: object, target_dir: Path, **kwargs: object) -> object:
            del session, target_dir, kwargs
            return SimpleNamespace(session_dir=self.session_dir)

    controller = HereApplicationController(
        capture_factory=lambda request, sink: next(captures),
        live_factory=lambda count, alt: FakeLive(),
        processor=ImmediateProcessor(tmp_path / "session"),  # type: ignore[arg-type]
    )
    replacement_started = threading.Event()

    def start_replacement(event: object) -> None:
        if getattr(event, "state", None) is not ApplicationState.COMPLETED:
            return
        controller.start(StartRequest(output_dir=tmp_path))
        wait_for_state(controller, ApplicationState.RECORDING)
        replacement_started.set()

    controller.subscribe(start_replacement)
    controller.start(StartRequest(output_dir=tmp_path))
    wait_for_state(controller, ApplicationState.RECORDING)
    controller.stop()
    assert replacement_started.wait(2)
    time.sleep(0.01)

    controller.pause()

    assert second_capture.paused
    controller.cancel()
    controller.wait_until_terminal(2)


def test_capture_start_failure_aborts_and_cleans_live_pipeline(tmp_path: Path) -> None:
    live = FakeLive()
    controller = HereApplicationController(
        capture_factory=lambda request, sink: (_ for _ in ()).throw(
            RuntimeError("device unavailable")
        ),
        live_factory=lambda count, alt: live,
        processor=FakeProcessor(tmp_path / "session"),  # type: ignore[arg-type]
    )

    controller.start(StartRequest(output_dir=tmp_path))
    snapshot = controller.wait_until_terminal(2)

    assert snapshot.state is ApplicationState.FAILED
    assert live.aborted
    assert live.cleaned
    assert snapshot.session_dir is None
    assert not snapshot.recoverable


def test_first_capture_blocks_reach_live_pipeline_while_devices_finish_preparing(
    tmp_path: Path,
) -> None:
    capture = FakeCapture()
    live = FakeLive()

    def capture_factory(request: object, sink: object) -> FakeCapture:
        del request
        assert callable(sink)
        sink("microphone", np.array([[1000]], dtype=np.int16), 16000, 1)
        return capture

    controller = HereApplicationController(
        capture_factory=capture_factory,  # type: ignore[arg-type]
        live_factory=lambda count, alt: live,
        processor=FakeProcessor(tmp_path / "session"),  # type: ignore[arg-type]
    )
    levels = []
    controller.subscribe(
        lambda event: levels.append(event) if event.kind is EventKind.AUDIO_LEVEL else None
    )

    controller.start(StartRequest(output_dir=tmp_path))
    wait_for_state(controller, ApplicationState.RECORDING)

    assert len(live.blocks) == 1
    assert levels == []
    controller.cancel()
    controller.wait_until_terminal(2)
