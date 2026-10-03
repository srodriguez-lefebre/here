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

SoundFile's actual native copyright/source notes accompany full libsndfile and
named-codec license texts. Internal codec versions are unknown; candidate source
versions identify the text provenance and are not asserted binary versions. The
same boundary applies to unexposed libffi/liblzma/mpdecimal versions. This
collection does not claim an independent legal or exact native source-build audit.

No QtMultimedia, FFmpeg, PDF/SVG/image plugin, QML/virtual keyboard or optional
software OpenGL capability is promised by M1. Current raster-painted Widgets
are verified through actual frozen `qwindows` startup. Notices conservatively
retain applicable module source attributions without asserting each optional
source component was compiled.
