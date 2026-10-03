"""Real child imports must not inherit conftest's optional audio stub."""

import subprocess
import sys

import pytest


@pytest.mark.parametrize(
    "module",
    [
        "here.recording.journal",
        "here.application.recovery",
        "here.recording.helper_worker",
        "here.entrypoints",
        "here.ui.gui",
    ],
)
def test_local_import_does_not_initialize_portaudio(tmp_path, module):
    script = tmp_path / "import_without_portaudio.py"
    script.write_text(
        "import importlib, importlib.abc, sys\n"
        "class NoPortAudio(importlib.abc.MetaPathFinder):\n"
        "    def find_spec(self, fullname, path=None, target=None):\n"
        "        if fullname == 'sounddevice':\n"
        "            raise OSError('PortAudio library not found')\n"
        "sys.meta_path.insert(0, NoPortAudio())\n"
        f"importlib.import_module({module!r})\n"
        "assert 'sounddevice' not in sys.modules\n"
    )
    result = subprocess.run(
        [sys.executable, str(script)],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
