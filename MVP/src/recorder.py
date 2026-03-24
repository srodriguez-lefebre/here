import os
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

import numpy as np
import sounddevice as sd
import soundfile as sf
from loguru import logger

from here.config.settings import get_settings


# ── Shared ────────────────────────────────────────────────────────────────────


def _record_until_enter(sample_rate: int, channels: int, label: str) -> Path:
    """Core recording loop using sounddevice — shared by mic and Linux OS recorders."""
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

    with sd.InputStream(samplerate=sample_rate, channels=channels, dtype="float32", callback=callback):
        input()

    logger.info("Recording stopped. {n} chunks captured.", n=len(chunks))

    audio = np.concatenate(chunks, axis=0)
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio, sample_rate)
    return Path(tmp.name)


# ── Linux / WSL2 (PulseAudio) ─────────────────────────────────────────────────


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


def _record_os_linux(sample_rate: int) -> Path:
    _setup_pulse()

    monitor = _get_monitor_source_name()
    if not monitor:
        raise RuntimeError("No PulseAudio monitor source found. Is PulseAudio running?")

    logger.info("Using monitor source: {monitor}", monitor=monitor)

    previous_source = _get_default_source()
    _set_default_source(monitor)

    try:
        return _record_until_enter(sample_rate, channels=1, label="system audio")
    finally:
        _set_default_source(previous_source)


# ── Windows (WASAPI loopback) ─────────────────────────────────────────────────


def _record_os_windows() -> Path:
    import pyaudiowpatch as pyaudio

    p = pyaudio.PyAudio()
    try:
        device = p.get_default_wasapi_loopback()
        logger.info("Using WASAPI loopback: {name}", name=device["name"])

        CHUNK = 1024
        channels = device["maxInputChannels"]
        sample_rate = int(device["defaultSampleRate"])

        stream = p.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            input=True,
            input_device_index=device["index"],
            frames_per_buffer=CHUNK,
        )

        frames: list[bytes] = []
        stop_event = threading.Event()

        def _record() -> None:
            while not stop_event.is_set():
                data = stream.read(CHUNK, exception_on_overflow=False)
                frames.append(data)

        t = threading.Thread(target=_record, daemon=True)
        t.start()

        logger.info("Recording system audio... Press Enter to stop.")
        input()

        stop_event.set()
        t.join()
        stream.stop_stream()
        stream.close()
    finally:
        p.terminate()

    logger.info("Recording stopped. {n} chunks captured.", n=len(frames))

    audio_int16 = np.frombuffer(b"".join(frames), dtype=np.int16)
    audio_float = audio_int16.astype(np.float32) / 32768.0
    audio_float = audio_float.reshape(-1, channels)

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    sf.write(tmp.name, audio_float, sample_rate)
    return Path(tmp.name)


# ── Public API ────────────────────────────────────────────────────────────────


def record_mic_until_enter(sample_rate: int = 16000) -> Path:
    """
    Records audio from the microphone until the user presses Enter.

    Args:
        sample_rate: Sample rate in Hz. Defaults to 16000 (optimal for Whisper).

    Returns:
        Path to a temporary WAV file with the recorded audio.
    """
    if sys.platform != "win32":
        _setup_pulse()
    return _record_until_enter(sample_rate, channels=1, label="mic")


def record_os_until_enter(sample_rate: int = 16000) -> Path:
    """
    Records system audio (OS output) until the user presses Enter.

    On Windows: uses WASAPI loopback via pyaudiowpatch.
    On Linux/WSL2: uses the PulseAudio monitor source.

    Args:
        sample_rate: Sample rate in Hz. Used on Linux; on Windows the device's
                     native rate is used instead.

    Returns:
        Path to a temporary WAV file with the recorded audio.
    """
    if sys.platform == "win32":
        return _record_os_windows()
    return _record_os_linux(sample_rate)
