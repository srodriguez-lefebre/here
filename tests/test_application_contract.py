from datetime import datetime, timezone
from pathlib import Path

import pytest
from here.application import (
    ApplicationEvent,
    ApplicationSnapshot,
    ApplicationState,
    AudioLevel,
    EventKind,
    SourceMode,
    StartRequest,
)


def test_all_visual_lifecycle_states_are_stable_string_values() -> None:
    assert [state.value for state in ApplicationState] == [
        "idle",
        "preparing",
        "recording",
        "paused",
        "stopping",
        "processing",
        "completed",
        "failed",
        "cancelled",
    ]


@pytest.mark.parametrize("state", list(ApplicationState))
def test_snapshot_reports_only_working_states_as_active(state: ApplicationState) -> None:
    snapshot = ApplicationSnapshot(state=state)

    assert snapshot.has_active_work is (
        state
        in {
            ApplicationState.PREPARING,
            ApplicationState.RECORDING,
            ApplicationState.PAUSED,
            ApplicationState.STOPPING,
            ApplicationState.PROCESSING,
        }
    )


def test_start_request_defaults_to_combined_capture(tmp_path: Path) -> None:
    request = StartRequest(output_dir=tmp_path)

    assert request.source_mode is SourceMode.BOTH
    assert request.use_alt_transcription_model is False


def test_audio_level_accepts_normalized_aggregates() -> None:
    captured_at = datetime.now(timezone.utc)

    level = AudioLevel(source="microphone", peak=0.75, rms=0.25, captured_at=captured_at)

    assert level.captured_at is captured_at


@pytest.mark.parametrize(("peak", "rms"), [(-0.1, 0.1), (1.1, 0.1), (0.1, -0.1), (0.1, 1.1)])
def test_audio_level_rejects_non_normalized_values(peak: float, rms: float) -> None:
    with pytest.raises(ValueError):
        AudioLevel(source="microphone", peak=peak, rms=rms)


def test_application_event_carries_state_transition_and_small_details() -> None:
    event = ApplicationEvent(
        kind=EventKind.STATE_CHANGED,
        state=ApplicationState.RECORDING,
        previous_state=ApplicationState.PREPARING,
        details={"source_count": 2},
    )

    assert event.previous_state is ApplicationState.PREPARING
    assert event.details == {"source_count": 2}
