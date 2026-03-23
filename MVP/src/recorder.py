import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from loguru import logger

from here.config.settings import get_settings


def _setup_pulse() -> None:
    """Set PULSE_SERVER in the environment if configured."""
    pulse_server = get_settings().PULSE_SERVER
    if pulse_server:
        os.environ["PULSE_SERVER"] = pulse_server


def _get_monitor_source_name() -> str | None:
    """Find the PulseAudio monitor source name using pactl."""
    try:
        result = subprocess.run(
            ["pactl", "list", "sources", "short"],
            capture_output=True,
            text=True,
            env=os.environ,
        )
        for line in result.stdout.splitlines():
            if ".monitor" in line:
                parts = line.split()
                if len(parts) >= 2:
                    return parts[1]
    except Exception as exc:
        logger.error("Failed to query PulseAudio sources: {exc}", exc=exc)
    return None


def _get_default_source() -> str:
    """Return the current PulseAudio default source name."""
    result = subprocess.run(
        ["pactl", "get-default-source"],
        capture_output=True,
        text=True,
        env=os.environ,
    )
    return result.stdout.strip()


def _set_default_source(source: str) -> None:
    """Set the PulseAudio default source."""
    subprocess.run(["pactl", "set-default-source", source], env=os.environ)


def _record_until_enter(sample_rate: int, label: str) -> Path:
    """Core recording loop — shared by mic and os recorders."""
    chunks: list[np.ndarray] = []

    def callback(
        indata: np.ndarray,
        frames: int,
        time: object,
        status: sd.CallbackFlags,
    ) -> None:
        if status:
            logger.warning("Audio status: {status}", status=status)
        chunks.append(indata.copy())

    logger.info("Recording {label}... Press Enter to stop.", label=label)

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", callback=callback):
        input()

    logger.info("Recording stopped. {n} chunks captured.", n=len(chunks))

    audio = np.concatenate(chunks, axis=0)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, sample_rate)
    return Path(tmp.name)


def record_mic_until_enter(sample_rate: int = 16000) -> Path:
    """
    Records audio from the microphone until the user presses Enter.

    Args:
        sample_rate: Sample rate in Hz. Defaults to 16000 (optimal for Whisper).

    Returns:
        Path to a temporary WAV file with the recorded audio.
    """
    _setup_pulse()
    return _record_until_enter(sample_rate, label="mic")


def record_os_until_enter(sample_rate: int = 16000) -> Path:
    """
    Records system audio (OS output) until the user presses Enter.
    Uses the PulseAudio monitor source of the default sink.

    Args:
        sample_rate: Sample rate in Hz. Defaults to 16000 (optimal for Whisper).

    Returns:
        Path to a temporary WAV file with the recorded audio.
    """
    _setup_pulse()

    monitor = _get_monitor_source_name()
    if not monitor:
        raise RuntimeError("No PulseAudio monitor source found. Is PulseAudio running?")

    logger.info("Using monitor source: {monitor}", monitor=monitor)

    previous_source = _get_default_source()
    _set_default_source(monitor)

    try:
        return _record_until_enter(sample_rate, label="system audio")
    finally:
        _set_default_source(previous_source)
