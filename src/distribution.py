"""Bounded local distribution checks and Windows installation lifetime guard."""

from __future__ import annotations

import contextlib
import ctypes
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
from pathlib import Path

APP_MUTEX = "here.application.94BE3378-6B95-48A1-BD13-C534746C7319"


class ApplicationMutex:
    """Keep the Inno application mutex alive without preventing multiple launches."""

    def __init__(self, name: str = APP_MUTEX):
        self.name = name
        self.handle = None

    def __enter__(self):
        if sys.platform == "win32":
            from ctypes import wintypes

            self.kernel = ctypes.WinDLL("kernel32", use_last_error=True)
            self.kernel.CreateMutexW.argtypes = [wintypes.LPVOID, wintypes.BOOL, wintypes.LPCWSTR]
            self.kernel.CreateMutexW.restype = wintypes.HANDLE
            self.kernel.CloseHandle.argtypes = [wintypes.HANDLE]
            self.handle = self.kernel.CreateMutexW(None, False, self.name)
            if not self.handle:
                raise ctypes.WinError(ctypes.get_last_error())
        return self

    def __exit__(self, *arguments):
        if self.handle:
            self.kernel.CloseHandle(self.handle)
            self.handle = None


def _helper_proof() -> dict[str, bool]:
    """Exercise the real fixed child dispatch, rejecting before any hardware import."""
    from here.audio_helper import _encode, helper_command
    from here.recording.process_owner import WindowsProcessOwner

    def launch():
        process = subprocess.Popen(
            helper_command(),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            owner = WindowsProcessOwner(process) if sys.platform == "win32" else None
        except BaseException:
            process.kill()
            process.wait()
            process.stdin.close()
            process.stdout.close()
            raise
        return process, owner

    process, owner = launch()
    try:
        output, _ = process.communicate(
            _encode({"action": "invalid-smoke-request", "parameters": {}}), timeout=10
        )
        message = json.loads(output)
        rejected = process.returncode == 1 and message == {
            "kind": "error",
            "message": "Unsupported internal audio action",
        }
        if not rejected:
            raise RuntimeError("Frozen helper failed the bounded rejection check")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if owner:
            owner.close()
        process.stdin.close()
        process.stdout.close()
    process, owner = launch()
    try:
        # Keep stdin open: a real child waits on its fixed initial frame. Reap it
        # on a bounded deadline, without supplying an action or opening devices.
        try:
            process.wait(timeout=0.25)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        else:
            raise RuntimeError("Helper did not wait for its initial request")
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
        if owner:
            owner.close()
        process.stdin.close()
        process.stdout.close()
    return {"helper_rejected_request": rejected, "helper_timeout_reaped": True}


@contextlib.contextmanager
def _isolated_environment(root: Path):
    keys = (
        "HERE_DATA_DIR",
        "HERE_ENV_FILE",
        "HERE_SETTINGS_FILE",
        "OPENAI_API_KEY",
        "TRANSCRIPTIONS_DIR",
    )
    previous = {key: os.environ.get(key) for key in keys}
    os.environ["HERE_DATA_DIR"] = str(root / "data")
    os.environ["HERE_ENV_FILE"] = str(root / "absent.env")
    os.environ["HERE_SETTINGS_FILE"] = str(root / "preferences.ini")
    os.environ.pop("OPENAI_API_KEY", None)
    os.environ.pop("TRANSCRIPTIONS_DIR", None)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def smoke_check(report_path: Path) -> int:
    """Initialize the production desktop on empty owned paths; never record/upload."""
    checks = {}
    report = {
        "schema": 1,
        "status": "failed",
        "version": importlib.metadata.version("here"),
        "python": platform.python_version(),
        "os": platform.platform(),
        "frozen": bool(getattr(sys, "frozen", False)),
        "checks": checks,
    }
    try:
        with tempfile.TemporaryDirectory(prefix="here-smoke-") as scratch:
            with _isolated_environment(Path(scratch)):
                # Imports are local-only. Constructing OpenAI clients and invoking
                # capture/diagnostic operations are deliberately outside this check.
                import here.application  # noqa: F401
                import here.recording.windows  # noqa: F401
                import here.transcription.client  # noqa: F401
                import soundfile
                from here.config.settings import get_settings
                from here.ui.gui import create_production_desktop
                from PySide6.QtCore import qVersion

                desktop = create_production_desktop()
                desktop.show()
                desktop.application.processEvents()
                checks["desktop_visible"] = desktop.main_window.isVisible()
                checks["provider_configured"] = get_settings().OPENAI_API_KEY is not None
                checks["hardware_opened"] = False
                checks["recording_started"] = desktop.controller.snapshot.has_active_work
                checks["editable_finder"] = any(
                    name.startswith("__editable__") for name in sys.modules
                )
                checks["windowed_stdio_absent"] = sys.stdout is None and sys.stderr is None
                report["qt"] = qVersion()
                report["libsndfile"] = soundfile.__libsndfile_version__
                # Exercise the actual packaged settings editor on the temporary
                # smoke-owned file. The synthetic key is never sent to a provider
                # and is removed before the fixed helper checks.
                from here.config.paths import get_env_file
                from here.ui.configuration import ConfigurationDialog
                from here.ui.preferences import VisualPreferences
                from PySide6.QtWidgets import QDialog, QLineEdit, QPushButton, QSlider, QSpinBox

                dialog = ConfigurationDialog(desktop.preferences, desktop.main_window)
                key_input = dialog.findChild(QLineEdit, "apiKeyInput")
                masked = key_input.echoMode() == QLineEdit.EchoMode.Password
                key_input.setText("synthetic-smoke-configuration")
                dialog.findChild(QSpinBox, "indicatorSizeInput").setValue(144)
                dialog.findChild(QSlider, "indicatorTransparencyInput").setValue(40)
                dialog.findChild(QPushButton, "saveConfigurationButton").click()
                configured = get_settings().OPENAI_API_KEY
                checks["configuration_editor_saved"] = (
                    masked
                    and dialog.result() == QDialog.DialogCode.Accepted
                    and configured is not None
                    and configured.get_secret_value() == "synthetic-smoke-configuration"
                )
                restored = VisualPreferences(desktop.preferences._settings)
                checks["indicator_preferences_saved"] = (
                    restored.size == 144
                    and abs(restored.opacity - 0.6) < 0.01
                    and desktop.overlay.width() == 144
                )
                get_env_file().unlink(missing_ok=True)
                dialog.close()
                if not all(
                    checks[name]
                    for name in ("configuration_editor_saved", "indicator_preferences_saved")
                ):
                    raise RuntimeError("Packaged configuration check failed")
                desktop.request_exit()
                deadline = time.monotonic() + 10
                while desktop.jobs.busy and time.monotonic() < deadline:
                    desktop.application.processEvents()
                    time.sleep(0.01)
                if desktop.jobs.busy:
                    raise TimeoutError("Desktop jobs did not drain")
                checks["jobs_drained"] = True
                desktop.main_window.close()
                desktop.overlay.close()
                desktop.application.processEvents()
                checks.update(_helper_proof())
                if not checks["desktop_visible"] or checks["recording_started"]:
                    raise RuntimeError("Desktop startup check failed")
                report["status"] = "ok"
    except Exception as error:
        report["error"] = f"{type(error).__name__}: {error}"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0 if report["status"] == "ok" else 1


def dispatch_smoke_check():
    if len(sys.argv) == 3 and sys.argv[1] == "--smoke-check":
        return smoke_check(Path(sys.argv[2]).resolve())
    return None
