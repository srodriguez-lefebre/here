# Native notice sources

The build copies installed wheel license trees and the actual Python runtime
license into the bundle, alongside the selected full upstream texts in this
directory. `sources.json` preserves source provenance and hashes; the frozen
bundle inventory records actual native files, versions, hashes and source files.

Qt Core/Gui/Widgets/Network and PySide/Shiboken are dynamic libraries under
`_internal/PySide6` and `_internal/shiboken6`. Compatible replacement builds may
replace these libraries; the application does not enforce runtime binary hashes.
The full LGPL/GPL alternatives, exception texts, module library attributions and
upstream source links accompany them. See `upstream/*/LICENSES` and
`upstream/*/attributions.json`; `sources.json` links the matching source commits.

Qt Multimedia adds the dynamically linked module and native Windows Media
Foundation plugin for local WAV playback. The
[Qt Multimedia 6.11.2 source](https://github.com/qt/qtmultimedia/tree/6f162ccac1425edbd7b4d1582fabab5973b43d6c)
contains the module and plugin. Its license texts and the dr_libs and TLSF
attributions and full terms are under `upstream/qtmultimedia-6.11.2`. The native
backend uses system Media Foundation. Windows 11 N requires Microsoft's
[Media Feature Pack](https://support.microsoft.com/en-us/windows/experience/platform-variants/media-feature-pack-for-windows-n)
for this service.

SoundFile's actual native copyright/source notes accompany full libsndfile and
named-codec license texts. Internal codec versions are unknown; candidate source
versions identify the text provenance and are not asserted binary versions. The
same boundary applies to unexposed libffi/liblzma/mpdecimal versions. This
collection does not claim an independent legal or exact native source-build audit.

Frozen `qwindows` startup verifies the raster-painted Widgets. The distribution
uses the native playback plugin and excludes the FFmpeg backend and its codec
DLLs. Notices conservatively retain applicable module source attributions
without asserting each optional source component was compiled.
