"""Exercise packaging behavior through a real noneditable wheel, outside the repo."""

import json
import os
import subprocess
from pathlib import Path

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


def test_windows_acceptance_isolates_and_restores_inherited_transcription_path(tmp_path):
    if os.name != "nt":
        pytest.skip("Windows acceptance script environment")
    root = Path(__file__).resolve().parents[1]
    trap = tmp_path / "outside transcript trap"
    trap.mkdir()
    canary = trap / "preserve.txt"
    canary.write_text("owned external data", encoding="utf-8")
    bundle = tmp_path / "fake bundle"
    bundle.mkdir()
    evidence = tmp_path / "evidence"
    harness = tmp_path / "spawn-boundary.ps1"
    harness.write_text(
        """param($SmokeScript, $Bundle, $Evidence, $Trap)
$ErrorActionPreference = 'Stop'
$env:TRANSCRIPTIONS_DIR = $Trap
function Start-Process {
    param($FilePath, $ArgumentList, $WorkingDirectory, [switch]$PassThru, $WindowStyle)
    if ($env:TRANSCRIPTIONS_DIR) { throw 'Inherited external transcripts reached child spawn' }
    # Deliberately stop at the real script's first process boundary: no app launched.
    throw 'expected-isolated-spawn-boundary'
}
try {
    & $SmokeScript -BundlePath $Bundle -EvidenceDirectory $Evidence
    throw 'Acceptance unexpectedly bypassed the spawn boundary'
} catch {
    if ($_.Exception.Message -ne 'expected-isolated-spawn-boundary') { throw }
}
if ($env:TRANSCRIPTIONS_DIR -ne $Trap) { throw 'Inherited transcripts were not restored' }
Write-Output 'isolated process environment and restored caller environment'
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        [
            "pwsh",
            "-NoProfile",
            "-File",
            str(harness),
            str(root / "scripts/smoke_windows.ps1"),
            str(bundle),
            str(evidence),
            str(trap),
        ],
        cwd=tmp_path,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert b"restored caller environment" in result.stdout
    assert canary.read_text(encoding="utf-8") == "owned external data"
    assert sorted(path.name for path in trap.iterdir()) == ["preserve.txt"]
