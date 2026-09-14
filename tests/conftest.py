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
def _reset_settings_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    import here.config.settings as settings_module

    settings_module._settings_instance = None
    yield
    settings_module._settings_instance = None
