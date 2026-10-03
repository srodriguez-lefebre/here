import sys
import time
from pathlib import Path

import pytest


def test_helper_timeout_reaps_process_and_reader(tmp_path):
    from here.audio_helper import OwnedHelper

    script = tmp_path / "stalled.py"
    script.write_text("import time; time.sleep(60)")
    helper = OwnedHelper("devices", {}, command=[sys.executable, str(script)])
    started = time.monotonic()
    with pytest.raises(TimeoutError):
        helper.result(0.1)
    assert time.monotonic() - started < 3
    assert helper.process.poll() is not None
    assert not helper.reader.is_alive()


def test_helper_result_waits_for_actual_process_exit(tmp_path):
    from here.audio_helper import OwnedHelper

    script = tmp_path / "result_then_stall.py"
    script.write_text(
        'import time; print(\'{"kind":"result","value":[]}\', flush=True); time.sleep(60)'
    )
    helper = OwnedHelper("devices", {}, command=[sys.executable, str(script)])
    with pytest.raises(TimeoutError):
        helper.result(0.2)
    assert helper.process.poll() is not None


def test_frozen_launcher_uses_internal_dispatch(monkeypatch):
    from here.audio_helper import dispatch_internal_helper, helper_command

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    assert helper_command() == [
        str(Path(sys.executable).with_name("here-cli.exe")),
        "--here-audio-helper-v1",
    ]
    assert dispatch_internal_helper(["here", "--ordinary-flag"]) is None


def test_diagnostics_holds_reservation_until_cleanup(tmp_path):
    from here.application.diagnostics import AudioDiagnosticsService
    from here.recording.reservation import AudioReservation

    reservation = AudioReservation()
    service = AudioDiagnosticsService(reservation=reservation)
    reservation.acquire()
    try:
        with pytest.raises(RuntimeError, match="ocupados"):
            service.devices()
    finally:
        reservation.release()


def test_controller_cannot_start_during_diagnostic_cleanup(tmp_path):
    import threading

    from here.application import HereApplicationController, StartRequest
    from here.application.diagnostics import AudioDiagnosticsService

    cleanup, release = threading.Event(), threading.Event()

    class Helper:
        def result(self, timeout):
            cleanup.set()
            assert release.wait(3)
            return []

    service = AudioDiagnosticsService(helper_factory=lambda *args: Helper())
    diagnostic = threading.Thread(target=service.devices)
    diagnostic.start()
    assert cleanup.wait(2)
    try:

        def forbidden(*args):
            raise AssertionError("Diagnostic lease must reject capture before allocation")

        with pytest.raises(RuntimeError, match="ocupados"):
            HereApplicationController(capture_factory=forbidden, live_factory=forbidden).start(
                StartRequest(output_dir=tmp_path)
            )
    finally:
        release.set()
        diagnostic.join(3)
    assert not diagnostic.is_alive()


def test_frozen_launch_with_windowed_stdio_none(monkeypatch):
    from here.audio_helper import helper_command

    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "stdin", None)
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)
    assert Path(helper_command()[0]).name == "here-cli.exe"


@pytest.mark.skipif(sys.platform != "win32", reason="Windows process ownership contract")
def test_windows_helper_dies_when_owner_is_killed(tmp_path):
    import ctypes
    import subprocess

    child = tmp_path / "child.py"
    child.write_text("import time; time.sleep(60)")
    owner = tmp_path / "owner.py"
    owner.write_text(
        "import sys,time\nfrom pathlib import Path\n"
        "from here.audio_helper import OwnedHelper\n"
        "h=OwnedHelper('devices', {}, command=[sys.executable, sys.argv[1]])\n"
        "Path(sys.argv[2]).write_text(str(h.process.pid))\n"
        "time.sleep(60)\n"
    )
    marker = tmp_path / "pid.txt"
    process = subprocess.Popen(
        [sys.executable, str(owner), str(child), str(marker)],
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.restype = ctypes.c_void_p
    kernel.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
    kernel.CloseHandle.argtypes = [ctypes.c_void_p]
    handle = None
    try:
        deadline = time.monotonic() + 5
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists()
        handle = kernel.OpenProcess(0x00100000, False, int(marker.read_text()))
        assert handle
        process.kill()
        process.wait(3)
        assert kernel.WaitForSingleObject(handle, 3000) == 0
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if handle:
            # If the assertion failed, terminate this synthetic child explicitly.
            cleanup = kernel.OpenProcess(0x0001, False, int(marker.read_text()))
            if cleanup:
                kernel.TerminateProcess.argtypes = [ctypes.c_void_p, ctypes.c_uint]
                kernel.TerminateProcess(cleanup, 1)
                kernel.CloseHandle(cleanup)
            kernel.CloseHandle(handle)
