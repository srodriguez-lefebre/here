import sys
from collections.abc import Callable

import numpy as np
from here.recording.control import ControllableRecording
from here.recording.linux import record_mic_linux, record_os_linux
from here.recording.models import RecordingSession
from here.recording.windows import (
    record_both_windows,
    record_mic_windows,
    record_os_windows,
    start_windows_recording,
)

BlockSink = Callable[[str, np.ndarray, int, int], None]


def start_recording(
    mode: str = "both",
    *,
    block_sink: BlockSink | None = None,
    microphone_device_id: int | None = None,
    system_device_id: int | None = None,
) -> ControllableRecording:
    """Start a programmatically controlled recording for the Windows application."""

    if sys.platform != "win32":
        raise RuntimeError(
            "Programmatic application capture is currently supported on Windows only."
        )
    return start_windows_recording(
        mode,
        block_sink=block_sink,
        microphone_device_id=microphone_device_id,
        system_device_id=system_device_id,
    )


def record_mic_until_enter(
    sample_rate: int = 16000,
    *,
    block_sink: BlockSink | None = None,
) -> RecordingSession:
    """
    Record audio from the microphone until the user presses Enter.

    On Windows: uses PyAudioWPatch and ignores the requested sample_rate,
    capturing at the device default sample rate.
    On Linux/WSL2: uses sounddevice and honors the requested sample_rate.
    """
    if sys.platform == "win32":
        return record_mic_windows(block_sink=block_sink)
    return record_mic_linux(sample_rate, block_sink=block_sink)


def record_os_until_enter(
    sample_rate: int = 16000,
    *,
    block_sink: BlockSink | None = None,
) -> RecordingSession:
    """
    Record system audio until the user presses Enter.

    On Windows: uses WASAPI loopback via PyAudioWPatch and ignores the
    requested sample_rate, capturing at the device default sample rate.
    On Linux/WSL2: uses the PulseAudio monitor source and honors the
    requested sample_rate.
    """
    if sys.platform == "win32":
        return record_os_windows(block_sink=block_sink)
    return record_os_linux(sample_rate, block_sink=block_sink)


def record_both_until_enter(*, block_sink: BlockSink | None = None) -> RecordingSession:
    """Record microphone and system audio together until the user presses Enter."""
    if sys.platform != "win32":
        raise RuntimeError(
            "`record` currently supports combined mic + system capture only on Windows."
        )
    return record_both_windows(block_sink=block_sink)
