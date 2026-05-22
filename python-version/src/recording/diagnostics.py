from __future__ import annotations

import math
import sys
import time
from dataclasses import dataclass

import numpy as np

from here.recording.windows import (
    WINDOWS_CAPTURE_CHUNK,
    _get_default_windows_input_device,
    _get_default_windows_loopback_device,
    _open_windows_input_stream,
    _safe_close_stream,
)

SIGNAL_PEAK_THRESHOLD = 0.01


@dataclass(slots=True)
class AudioDeviceInfo:
    source: str
    name: str
    index: int
    sample_rate: int
    channels: int


@dataclass(slots=True)
class SignalTestResult:
    source: str
    device: AudioDeviceInfo
    duration_seconds: float
    peak: float
    rms: float
    has_signal: bool


def _ensure_windows_diagnostics() -> None:
    if sys.platform != "win32":
        raise RuntimeError("Audio diagnostics currently support Windows only.")


def _device_info(source: str, device: dict[str, object]) -> AudioDeviceInfo:
    return AudioDeviceInfo(
        source=source,
        name=str(device["name"]),
        index=int(device["index"]),
        sample_rate=int(float(device["defaultSampleRate"])),
        channels=int(device["maxInputChannels"]),
    )


def get_windows_audio_devices() -> list[AudioDeviceInfo]:
    _ensure_windows_diagnostics()
    return [
        _device_info("microphone", _get_default_windows_input_device()),
        _device_info("system audio", _get_default_windows_loopback_device()),
    ]


def _get_windows_device_for_source(source: str) -> dict[str, object]:
    if source == "microphone":
        return _get_default_windows_input_device()
    if source == "system audio":
        return _get_default_windows_loopback_device()
    raise ValueError(f"Unsupported diagnostic source: {source}")


def test_windows_audio_signal(
    source: str,
    *,
    duration_seconds: float = 3.0,
) -> SignalTestResult:
    _ensure_windows_diagnostics()

    import pyaudiowpatch as pyaudio

    device = _get_windows_device_for_source(source)
    device_info = _device_info(source, device)
    duration_seconds = max(0.1, duration_seconds)
    chunk = WINDOWS_CAPTURE_CHUNK
    p = pyaudio.PyAudio()
    stream: object | None = None
    samples: list[np.ndarray] = []

    try:
        stream, sample_rate, channels = _open_windows_input_stream(p, pyaudio, device, chunk)
        chunks_to_read = max(1, int(math.ceil(duration_seconds * sample_rate / chunk)))
        for _ in range(chunks_to_read):
            data = stream.read(chunk, exception_on_overflow=False)
            audio = np.frombuffer(data, dtype=np.int16)
            usable_samples = (audio.size // channels) * channels
            if usable_samples > 0:
                samples.append(audio[:usable_samples].astype(np.float32) / 32768.0)
            time.sleep(chunk / sample_rate)
    finally:
        _safe_close_stream(stream)
        p.terminate()

    if samples:
        combined = np.concatenate(samples)
        peak = float(np.max(np.abs(combined))) if combined.size else 0.0
        rms = float(np.sqrt(np.mean(np.square(combined), dtype=np.float32))) if combined.size else 0.0
    else:
        peak = 0.0
        rms = 0.0

    return SignalTestResult(
        source=source,
        device=device_info,
        duration_seconds=duration_seconds,
        peak=peak,
        rms=rms,
        has_signal=peak >= SIGNAL_PEAK_THRESHOLD,
    )
