from __future__ import annotations

import here.recording.service as service_module
import pytest


def test_record_mic_until_enter_uses_windows_backend_on_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = object()
    monkeypatch.setattr(service_module.sys, "platform", "win32")
    monkeypatch.setattr(service_module, "record_mic_windows", lambda **kwargs: (expected, kwargs))

    result = service_module.record_mic_until_enter(22050, block_sink=object())

    assert result[0] is expected
    assert "block_sink" in result[1]


def test_record_mic_until_enter_uses_linux_backend_elsewhere(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = object()
    monkeypatch.setattr(service_module.sys, "platform", "linux")
    monkeypatch.setattr(
        service_module,
        "record_mic_linux",
        lambda sample_rate, **kwargs: (expected, sample_rate, kwargs),
    )

    result = service_module.record_mic_until_enter(22050, block_sink=object())

    assert result[0] is expected
    assert result[1] == 22050
    assert "block_sink" in result[2]


def test_record_os_until_enter_dispatches_by_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service_module.sys, "platform", "linux")
    monkeypatch.setattr(
        service_module,
        "record_os_linux",
        lambda sample_rate, **kwargs: ("linux", sample_rate, kwargs),
    )

    assert service_module.record_os_until_enter(44100, block_sink=object())[:2] == ("linux", 44100)

    monkeypatch.setattr(service_module.sys, "platform", "win32")
    monkeypatch.setattr(service_module, "record_os_windows", lambda **kwargs: ("windows", kwargs))

    assert service_module.record_os_until_enter(44100, block_sink=object())[0] == "windows"


def test_record_both_until_enter_raises_on_non_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service_module.sys, "platform", "linux")

    with pytest.raises(RuntimeError, match="combined mic \\+ system capture only on Windows"):
        service_module.record_both_until_enter()


def test_record_both_until_enter_uses_windows_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service_module.sys, "platform", "win32")
    monkeypatch.setattr(
        service_module, "record_both_windows", lambda **kwargs: ("captured", kwargs)
    )

    result = service_module.record_both_until_enter(block_sink=object())
    assert result[0] == "captured"
    assert "block_sink" in result[1]


def test_start_recording_dispatches_programmatic_windows_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = object()
    monkeypatch.setattr(service_module.sys, "platform", "win32")
    monkeypatch.setattr(
        service_module,
        "start_windows_recording",
        lambda mode, **kwargs: (expected, mode, kwargs),
    )

    result = service_module.start_recording(
        "both",
        block_sink=object(),
        microphone_device_id=3,
        system_device_id=5,
    )

    assert result[0] is expected
    assert result[1] == "both"
    assert result[2]["microphone_device_id"] == 3


def test_start_recording_rejects_non_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service_module.sys, "platform", "linux")

    with pytest.raises(RuntimeError, match="supported on Windows only"):
        service_module.start_recording()
