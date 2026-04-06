import sys

from here.recording.linux import record_mic_linux, record_os_linux
from here.recording.models import RecordingSession
from here.recording.windows import record_both_windows, record_mic_windows, record_os_windows


def record_mic_until_enter(sample_rate: int = 16000) -> RecordingSession:
    """
    Record audio from the microphone until the user presses Enter.

    On Windows: uses PyAudioWPatch.
    On Linux/WSL2: uses sounddevice.
    """
    if sys.platform == "win32":
        return record_mic_windows()
    return record_mic_linux(sample_rate)


def record_os_until_enter(sample_rate: int = 16000) -> RecordingSession:
    """
    Record system audio until the user presses Enter.

    On Windows: uses WASAPI loopback via PyAudioWPatch.
    On Linux/WSL2: uses the PulseAudio monitor source.
    """
    if sys.platform == "win32":
        return record_os_windows()
    return record_os_linux(sample_rate)


def record_both_until_enter() -> RecordingSession:
    """Record microphone and system audio together until the user presses Enter."""
    if sys.platform != "win32":
        raise RuntimeError(
            "`record` currently supports combined mic + system capture only on Windows."
        )
    return record_both_windows()
