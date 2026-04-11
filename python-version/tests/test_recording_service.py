from __future__ import annotations

import pytest

import here.recording.service as service_module


def test_record_mic_until_enter_uses_windows_backend_on_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = object()
    monkeypatch.setattr(service_module.sys, "platform", "win32")
    monkeypatch.setattr(service_module, "record_mic_windows", lambda: expected)

    result = service_module.record_mic_until_enter(22050)

    assert result is expected


def test_record_mic_until_enter_uses_linux_backend_elsewhere(monkeypatch: pytest.MonkeyPatch) -> None:
    expected = object()
    monkeypatch.setattr(service_module.sys, "platform", "linux")
    monkeypatch.setattr(service_module, "record_mic_linux", lambda sample_rate: (expected, sample_rate))

    result = service_module.record_mic_until_enter(22050)

    assert result == (expected, 22050)


def test_record_os_until_enter_dispatches_by_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service_module.sys, "platform", "linux")
    monkeypatch.setattr(service_module, "record_os_linux", lambda sample_rate: ("linux", sample_rate))

    assert service_module.record_os_until_enter(44100) == ("linux", 44100)

    monkeypatch.setattr(service_module.sys, "platform", "win32")
    monkeypatch.setattr(service_module, "record_os_windows", lambda: "windows")

    assert service_module.record_os_until_enter(44100) == "windows"


def test_record_both_until_enter_raises_on_non_windows(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service_module.sys, "platform", "linux")

    with pytest.raises(RuntimeError, match="combined mic \\+ system capture only on Windows"):
        service_module.record_both_until_enter()


def test_record_both_until_enter_uses_windows_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service_module.sys, "platform", "win32")
    monkeypatch.setattr(service_module, "record_both_windows", lambda: "captured")

    assert service_module.record_both_until_enter() == "captured"
