import sys
from collections.abc import Callable

import numpy as np

from here.recording.linux import record_mic_linux, record_os_linux
from here.recording.models import RecordingSession
from here.recording.windows import record_both_windows, record_mic_windows, record_os_windows

BlockSink = Callable[[str, np.ndarray, int, int], None]


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
        return record_mic_windows(block_sink=block_sink) if block_sink is not None else record_mic_windows()
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
        return record_os_windows(block_sink=block_sink) if block_sink is not None else record_os_windows()
    return record_os_linux(sample_rate, block_sink=block_sink)


def record_both_until_enter(*, block_sink: BlockSink | None = None) -> RecordingSession:
    """Record microphone and system audio together until the user presses Enter."""
    if sys.platform != "win32":
        raise RuntimeError(
            "`record` currently supports combined mic + system capture only on Windows."
        )
    return record_both_windows(block_sink=block_sink) if block_sink is not None else record_both_windows()
