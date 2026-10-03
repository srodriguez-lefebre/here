from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"


class _CallbackFlags:
    def __bool__(self) -> bool:
        return False


class _InputStream:
    def __init__(self, *args: object, **kwargs: object) -> None:
        del args, kwargs

    def __enter__(self) -> "_InputStream":
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: object,
    ) -> bool:
        del exc_type, exc, traceback
        return False


if "sounddevice" not in sys.modules:
    sounddevice_stub = types.ModuleType("sounddevice")
    sounddevice_stub.CallbackFlags = _CallbackFlags
    sounddevice_stub.InputStream = _InputStream
    sys.modules["sounddevice"] = sounddevice_stub

here_package = types.ModuleType("here")
here_package.__file__ = str(SRC_DIR / "__init__.py")
here_package.__path__ = [str(SRC_DIR)]
sys.modules["here"] = here_package


@pytest.fixture(autouse=True)
def _reset_settings_cache(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("HERE_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("HERE_ENV_FILE", str(tmp_path / "absent.env"))

    import here.config.settings as settings_module

    settings_module._settings_instance = None
    yield
    settings_module._settings_instance = None


@pytest.fixture
def owned_desktops(qtbot, monkeypatch):
    """Retain each composition root and drain it before Qtbot deletes its widgets."""
    from here.ui.app import HereDesktop

    original = HereDesktop.__init__

    def initialize(desktop, *args, **kwargs):
        original(desktop, *args, **kwargs)

        def before_close(widget):
            desktop.jobs.close()
            desktop.bridge.close()
            qtbot.waitUntil(lambda: not desktop.jobs.busy)

        # Pytest-qt closes widgets before fixture teardown. Its before-close hook
        # owns the whole desktop, not just weak references to the two widgets.
        qtbot.addWidget(desktop.main_window, before_close_func=before_close)
        qtbot.addWidget(desktop.overlay)

    monkeypatch.setattr(HereDesktop, "__init__", initialize)
