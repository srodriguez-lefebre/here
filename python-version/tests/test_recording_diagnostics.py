from __future__ import annotations

import sys
import types

import numpy as np
import pytest

import here.recording.diagnostics as diagnostics


def _device(name: str = "Device") -> dict[str, object]:
    return {
        "name": name,
        "index": 7,
        "defaultSampleRate": 16000.0,
        "maxInputChannels": 1,
    }


def test_get_windows_audio_devices_returns_default_microphone_and_loopback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(diagnostics.sys, "platform", "win32")
    monkeypatch.setattr(diagnostics, "_get_default_windows_input_device", lambda: _device("Mic"))
    monkeypatch.setattr(diagnostics, "_get_default_windows_loopback_device", lambda: _device("Loopback"))

    devices = diagnostics.get_windows_audio_devices()

    assert [device.source for device in devices] == ["microphone", "system audio"]
    assert [device.name for device in devices] == ["Mic", "Loopback"]
    assert devices[0].sample_rate == 16000
    assert devices[0].channels == 1


def test_get_windows_audio_devices_rejects_non_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diagnostics.sys, "platform", "linux")

    with pytest.raises(RuntimeError, match="Windows only"):
        diagnostics.get_windows_audio_devices()


def test_test_windows_audio_signal_reports_peak_and_rms(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FakePyAudio:
        def terminate(self) -> None:
            return None

    class _FakeStream:
        def read(self, chunk: int, exception_on_overflow: bool = False) -> bytes:
            del chunk, exception_on_overflow
            return np.array([0, 16384, -16384, 0], dtype=np.int16).tobytes()

        def stop_stream(self) -> None:
            return None

        def close(self) -> None:
            return None

    fake_pyaudio = types.ModuleType("pyaudiowpatch")
    fake_pyaudio.PyAudio = _FakePyAudio
    fake_pyaudio.paInt16 = object()

    monkeypatch.setattr(diagnostics.sys, "platform", "win32")
    monkeypatch.setitem(sys.modules, "pyaudiowpatch", fake_pyaudio)
    monkeypatch.setattr(diagnostics, "_get_default_windows_input_device", lambda: _device("Mic"))
    monkeypatch.setattr(
        diagnostics,
        "_open_windows_input_stream",
        lambda p, pyaudio, device, chunk: (_FakeStream(), 16000, 1),
    )
    monkeypatch.setattr(diagnostics.time, "sleep", lambda seconds: None)

    result = diagnostics.test_windows_audio_signal("microphone", duration_seconds=0.1)

    assert result.source == "microphone"
    assert result.device.name == "Mic"
    assert result.peak == 0.5
    assert result.rms > 0
    assert result.has_signal
