"""Windows x64 onedir built against an installed noneditable here wheel."""

import os
from pathlib import Path

from PyInstaller.utils.hooks import collect_submodules, copy_metadata

root = Path(SPECPATH).parent
notices = Path(os.environ["HERE_BUILD_NOTICES"])
hidden = collect_submodules(
    "here", filter=lambda name: not any(
        part in name for part in ("recording.linux", "ui.preview", "ui.fake_controller")
    )
)
data = copy_metadata("here", recursive=True) + [(str(notices), "notices")]
options = dict(
    pathex=[], binaries=[], datas=data, hiddenimports=hidden,
    hookspath=[], hooksconfig={}, runtime_hooks=[],
    excludes=["tkinter", "sounddevice", "here.recording.linux", "PySide6.QtMultimedia"],
    noarchive=False,
)
gui = Analysis([str(root / "packaging/gui_entry.py")], **options)
cli = Analysis([str(root / "packaging/cli_entry.py")], **options)


def product_binaries(binaries):
    # M1 uses raster-painted Widgets without external images or OpenGL widgets.
    # Retain actual platform/raster resources; do not ship future playback/image
    # codecs, optional software OpenGL or PDF plugins only because Qt supplies them.
    unused_qt = {"qt6opengl.dll", "qt6pdf.dll", "qt6qml.dll", "qt6qmlmeta.dll",
                 "qt6qmlmodels.dll", "qt6qmlworkerscript.dll", "qt6quick.dll",
                 "qt6svg.dll", "qt6virtualkeyboard.dll"}
    return [item for item in binaries if Path(item[0]).name.lower() not in unused_qt and not any(
        marker in item[0].replace("\\", "/").lower()
        for marker in ("/imageformats/", "/iconengines/", "/multimedia/", "opengl32sw.dll",
                       "/generic/", "/platforminputcontexts/", "/networkinformation/")
    ) and ("/platforms/" not in item[0].replace("\\", "/").lower()
           or Path(item[0]).name.lower() in {"qwindows.dll", "qoffscreen.dll"})]


gui.binaries = product_binaries(gui.binaries)
cli.binaries = product_binaries(cli.binaries)
gui_exe = EXE(
    PYZ(gui.pure), gui.scripts, [], exclude_binaries=True, name="here",
    debug=False, bootloader_ignore_signals=False, strip=False, upx=False, console=False,
)
cli_exe = EXE(
    PYZ(cli.pure), cli.scripts, [], exclude_binaries=True, name="here-cli",
    debug=False, bootloader_ignore_signals=False, strip=False, upx=False, console=True,
)
COLLECT(gui_exe, cli_exe, gui.binaries, gui.datas, cli.binaries, cli.datas,
        strip=False, upx=False, name="here")
