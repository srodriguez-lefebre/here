from pathlib import Path

from here.application import ApplicationState, SourceMode
from here.ui.contract import ApplicationUiAdapter
from here.ui.fake_controller import FakeApplicationController


def test_visual_adapter_translates_commands_to_application_contract(tmp_path: Path) -> None:
    core = FakeApplicationController()
    adapter = ApplicationUiAdapter(core, tmp_path)

    adapter.start_recording()
    adapter.pause_recording()
    adapter.resume_recording()
    adapter.stop_and_process()
    adapter.cancel_processing()
    adapter.retry_processing()

    assert core.command_log == ["start", "pause", "resume", "stop", "cancel", "retry"]
    assert core.last_request is not None
    assert core.last_request.output_dir == tmp_path
    assert core.last_request.source_mode is SourceMode.BOTH
    assert adapter.snapshot.state is ApplicationState.PROCESSING
    assert adapter.snapshot.recoverable is True


def test_recording_cancellation_is_destructive_in_fake_contract(tmp_path: Path) -> None:
    core = FakeApplicationController()
    adapter = ApplicationUiAdapter(core, tmp_path)

    adapter.start_recording()
    adapter.cancel_recording()

    assert adapter.snapshot.state is ApplicationState.CANCELLED
    assert adapter.snapshot.recoverable is False
