"""Collect the Windows Media Foundation plugin for local WAV playback."""

from pathlib import Path

from PyInstaller.utils.hooks.qt import add_qt6_dependencies

hiddenimports, binaries, datas = add_qt6_dependencies(__file__)
# Filter before binary dependency analysis so the FFmpeg plugin cannot pull in
# its codec DLLs. The product uses QMediaPlayer and QAudioOutput without QtMultimediaWidgets.
binaries = [
    (source, destination)
    for source, destination in binaries
    if Path(destination).name.lower() != "multimedia"
    or Path(source).name.lower() == "windowsmediaplugin.dll"
]
