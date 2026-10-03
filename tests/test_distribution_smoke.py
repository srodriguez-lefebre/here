"""Exercise packaging behavior through a real noneditable wheel, outside the repo."""

import json
import os
import subprocess

import pytest


@pytest.mark.parametrize("entry", ["cli_main", "gui_main"])
def test_wheel_smoke_opens_isolated_desktop_and_reaps_helper(tmp_path, entry):
    interpreter = os.environ.get("HERE_WHEEL_PYTHON")
    if not interpreter:
        pytest.skip("Set HERE_WHEEL_PYTHON to the noneditable wheel acceptance environment")
    report = tmp_path / "report with spaces.json"
    environment = os.environ.copy()
    environment.pop("OPENAI_API_KEY", None)
    environment.pop("PYTHONPATH", None)
    # A synthetic existing data root must remain untouched by smoke.
    real_root = tmp_path / "existing user data"
    real_root.mkdir()
    canary = real_root / "canary.txt"
    canary.write_text("preserve-existing-data", encoding="utf-8")
    environment["HERE_DATA_DIR"] = str(real_root)
    environment["HERE_ENV_FILE"] = str(real_root / "absent.env")
    command = [
        interpreter,
        "-I",
        "-B",
        "-c",
        f"from here.entrypoints import {entry}; raise SystemExit({entry}() or 0)",
        "--smoke-check",
        str(report),
    ]
    result = subprocess.run(command, cwd=tmp_path, env=environment, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    data = json.loads(report.read_text(encoding="utf-8"))
    assert data["status"] == "ok"
    assert data["checks"]["desktop_visible"] is True
    assert data["checks"]["jobs_drained"] is True
    assert data["checks"]["helper_rejected_request"] is True
    assert data["checks"]["helper_timeout_reaped"] is True
    assert data["checks"]["provider_configured"] is False
    assert data["checks"]["hardware_opened"] is False
    assert data["checks"]["recording_started"] is False
    assert data["checks"]["editable_finder"] is False
    assert canary.read_text(encoding="utf-8") == "preserve-existing-data"
    assert sorted(path.name for path in real_root.iterdir()) == ["canary.txt"]


def test_installer_mutex_blocks_only_while_owned_process_is_alive():
    if os.name != "nt":
        pytest.skip("Windows installation lifetime mutex")
    import ctypes
    from ctypes import wintypes

    from here.distribution import ApplicationMutex

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenMutexW.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.LPCWSTR]
    kernel.OpenMutexW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    name = "Local\\here-test-" + os.urandom(12).hex()
    with ApplicationMutex(name):
        handle = kernel.OpenMutexW(0x00100000, False, name)
        assert handle
        kernel.CloseHandle(handle)
    assert not kernel.OpenMutexW(0x00100000, False, name)
