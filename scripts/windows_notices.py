"""Collect full namespaced notices and audit the actual Windows payload/provenance."""

from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUILD_ONLY = {
    "altgraph",
    "packaging",
    "pefile",
    "pyinstaller",
    "pyinstaller-hooks-contrib",
    "pywin32-ctypes",
    "setuptools",
}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def normalize(name):
    return name.lower().replace("_", "-")


def distributions():
    return sorted(metadata.distributions(), key=lambda dist: normalize(dist.metadata["Name"]))


def prepare(destination, wheel):
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copytree(ROOT / "packaging/notices", destination, dirs_exist_ok=True)
    entries = []
    for dist in distributions():
        name = normalize(dist.metadata["Name"])
        component = destination / "wheels" / f"{name}-{dist.version}"
        component.mkdir(parents=True, exist_ok=True)
        component.joinpath("METADATA").write_text(
            dist.read_text("METADATA") or "", encoding="utf-8"
        )
        copied = []
        for relative in dist.files or []:
            parts = Path(relative).parts
            basename = Path(relative).name.lower()
            if not (
                "licenses" in parts
                or basename.startswith(("license", "copying", "notice", "authors"))
            ):
                continue
            source = Path(dist.locate_file(relative))
            if not source.is_file() or source.suffix.lower() in {".py", ".pyc", ".dll", ".pyd"}:
                continue
            if ".." in parts:
                raise RuntimeError(f"Unsafe license path in {name}: {relative}")
            target = component / Path(relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            copied.append(target.relative_to(destination).as_posix())
        entries.append(
            {
                "name": name,
                "version": dist.version,
                "texts": copied,
                "role": "bootloader"
                if name == "pyinstaller"
                else (
                    "build environment; retained if analyzed code ships"
                    if name in BUILD_ONLY
                    else "locked runtime wheel"
                ),
            }
        )
    runtime = destination / "python-runtime"
    runtime.mkdir(exist_ok=True)
    shutil.copyfile(Path(sys.base_prefix) / "LICENSE.txt", runtime / "LICENSE.txt")
    # SoundFile's actual wheel contains codec copyright/source notes in a neutral
    # 'licensing' package and the native libsndfile license alongside its DLL.
    soundfile = metadata.distribution("soundfile")
    for relative in ("licensing/license_notes.md", "_soundfile_data/COPYING"):
        source = Path(soundfile.locate_file(relative))
        target = destination / "soundfile-native" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    sources = json.loads((destination / "sources.json").read_text(encoding="utf-8"))
    for item in sources:
        if digest(destination / item["path"]) != item["sha256"]:
            raise RuntimeError(f"Upstream notice digest mismatch: {item['path']}")
    write_json(destination / "wheel-notices.json", entries)
    lines = [
        "# Third-party notices",
        "",
        "here 0.2.0 Windows distribution.",
        "",
        "Full license/copyright texts are retained under component-specific paths.",
        "Wheel versions below describe the exact installed locked build environment.",
        "The adjacent bundle inventory maps actual shipped native files to their hashes",
        "and matching wheel/runtime files. Build-only hooks are not application payload.",
        "",
        "| Component | Version | Text directory |",
        "| --- | --- | --- |",
    ]
    lines += [
        f"| {item['name']} | {item['version']} | wheels/{item['name']}-{item['version']}/ |"
        for item in entries
    ]
    lines += [
        "",
        "Additional native and missing-wheel notices:",
        "",
        "- Qt Core/Gui/Widgets/Network and PySide/Shiboken 6.11.2: upstream Qt/PySide license",
        "  alternatives and selected library attribution records with their full referenced",
        "  texts are under upstream/qtbase-6.11.2 and upstream/pyside-shiboken-6.11.2.",
        "  Conservatively retained module source notices do not assert every optional",
        "  component was compiled. No QtMultimedia, FFmpeg, image/icon plugin or software",
        "  OpenGL fallback capability is included by this M1 configuration.",
        "- PyAudioWPatch/PortAudio: upstream/pyaudiowpatch-0.2.12.8 includes the full Apache",
        "  text and vendored PortAudio copyright/license; native wheel hashes are audited.",
        "- SoundFile/libsndfile: soundfile-native retains the actual wheel's LGPL text and",
        "  named codec copyright/source notes. libsndfile reports 1.2.2. Internal FLAC, Ogg,",
        "  Vorbis, Opus, LAME and mpg123 versions are unexposed/unknown. Full corresponding",
        "  license/copyright texts are under upstream/codecs. Source notice candidate",
        "  release versions in directory names are provenance, not asserted binary versions.",
        "- Python: python-runtime/LICENSE.txt is copied from the actual build runtime.",
        "  upstream/python-3.13.15 includes CPython incorporated-software acknowledgements,",
        "  full OpenSSL/Expat/zlib and MSVC runtime-family notices and conservative builder",
        "  notice texts for native components. libffi/liblzma/mpdecimal internal versions",
        "  remain unconfirmed. Actual reported runtime versions are in build-provenance.json.",
        "- Loguru's release wheel omits its MIT text; upstream/loguru-0.7.3 supplies it.",
        "- PyInstaller's installed COPYING.txt includes the bootloader distribution exception.",
        "- Inno Setup 6.7.0: upstream/inno-6.7.0/LICENSE.txt accompanies installer/runtime.",
        "",
        "sources.json records immutable upstream URLs and copied text SHA-256 values.",
        "This is a notice/provenance inventory, not an independent legal certificate.",
        "",
    ]
    (destination / "THIRD_PARTY_NOTICES.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Prepared {len(entries)} namespaced wheel/runtime notice entries")


def source_records():
    records = {}
    for dist in distributions():
        for relative in dist.files or []:
            if Path(relative).suffix.lower() not in {".dll", ".pyd", ".exe"}:
                continue
            source = Path(dist.locate_file(relative))
            if source.is_file():
                records.setdefault(digest(source), []).append(
                    {
                        "kind": "wheel",
                        "distribution": dist.metadata["Name"],
                        "version": dist.version,
                        "path": str(relative),
                    }
                )
    base = Path(sys.base_prefix)
    for pattern in ("*.dll", "*.pyd"):
        for source in base.rglob(pattern):
            records.setdefault(digest(source), []).append(
                {
                    "kind": "Python runtime",
                    "version": platform.python_version(),
                    "path": source.relative_to(base).as_posix(),
                }
            )

    # Analysis is also authoritative for redistributed Windows API/UCRT files
    # outside the Python/wheel trees. Parse its literal TOC as data, never execute it.
    def visit(value):
        if isinstance(value, (tuple, list)):
            if (
                len(value) == 3
                and isinstance(value[1], str)
                and value[2] in {"BINARY", "EXTENSION"}
            ):
                source = Path(value[1])
                if source.is_file():
                    file_hash = digest(source)
                    if file_hash not in records:
                        windows_root = Path(os.environ["SystemRoot"]).resolve()
                        if not source.resolve().is_relative_to(windows_root):
                            raise RuntimeError(
                                f"Foreign developer native DLL in analysis: {source}"
                            )
                        records[file_hash] = [
                            {
                                "kind": "PyInstaller collected native file",
                                "path": str(source),
                                "file_version": pe_version(source),
                            }
                        ]
            else:
                for item in value:
                    visit(item)

    for toc in (ROOT / "build/windows/pyinstaller").rglob("Analysis-*.toc"):
        visit(ast.literal_eval(toc.read_text(encoding="utf-8")))
    return records


def pe_version(path):
    import pefile

    image = pefile.PE(str(path), fast_load=True)
    try:
        image.parse_data_directories([pefile.DIRECTORY_ENTRY["IMAGE_DIRECTORY_ENTRY_RESOURCE"]])
        for group in getattr(image, "FileInfo", []):
            for item in group:
                for table in getattr(item, "StringTable", []):
                    value = table.entries.get(b"FileVersion")
                    if value:
                        return value.decode(errors="replace")
    finally:
        image.close()
    return None


def audit(bundle, wheel):
    import _ssl
    import pyexpat
    import zlib

    provenance = source_records()
    files = []
    for path in sorted(bundle.rglob("*")):
        if not path.is_file():
            continue
        relative = path.relative_to(bundle).as_posix()
        if path.name.lower() == ".env" or path.suffix.lower() in {".wav", ".mp3", ".flac"}:
            raise RuntimeError(f"Personal-data file category in bundle: {relative}")
        row = {"path": relative, "bytes": path.stat().st_size, "sha256": digest(path)}
        if path.suffix.lower() in {".dll", ".pyd", ".exe"}:
            row["file_version"] = pe_version(path)
            row["provenance"] = provenance.get(row["sha256"], [])
            if not row["provenance"] and path.name not in {"here.exe", "here-cli.exe"}:
                raise RuntimeError(f"Unattributed native payload: {relative}")
            if path.name in {"here.exe", "here-cli.exe"}:
                row["provenance"] = [
                    {
                        "kind": "PyInstaller bootloader/application archive",
                        "version": metadata.version("pyinstaller"),
                    }
                ]
        files.append(row)
    destination = bundle.parent
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=True
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    inputs = {
        path.relative_to(ROOT).as_posix(): digest(path)
        for folder in ("src", "packaging", "scripts")
        for path in (ROOT / folder).rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    write_json(
        destination / "build-provenance.json",
        {
            "schema": 1,
            "version": metadata.version("here"),
            "source_commit": result.stdout.strip(),
            "tracked_tree_dirty": bool(dirty),
            "source_inputs_sha256": inputs,
            "wheel": {"filename": wheel.name, "sha256": digest(wheel)},
            "lock_sha256": digest(ROOT / "uv.lock"),
            "python": sys.version,
            "python_runtime": str(Path(sys.base_prefix)),
            "python_toolchain_files": {
                filename: digest(Path(sys.base_prefix) / filename)
                for filename in ("python.exe", "python313.dll")
            },
            "os": platform.platform(),
            "runtime_reported_versions": {
                "OpenSSL": _ssl.OPENSSL_VERSION,
                "Expat": pyexpat.EXPAT_VERSION,
                "zlib": zlib.ZLIB_RUNTIME_VERSION,
            },
            "installed_distributions": {
                dist.metadata["Name"]: dist.version for dist in distributions()
            },
            "signing": "unsigned application; no release certificate configured",
        },
    )
    for filename in ("bundle-inventory.json", "build-provenance.json"):
        if filename == "build-provenance.json":
            shutil.copyfile(destination / filename, bundle / "_internal/notices" / filename)
    # The inventory excludes itself to avoid a circular hash, and includes the
    # final shipped provenance file. Its own digest is in the external checksums.
    files = [
        item
        for item in files
        if item["path"]
        not in {
            "_internal/notices/build-provenance.json",
            "_internal/notices/bundle-inventory.json",
        }
    ]
    shipped_provenance = bundle / "_internal/notices/build-provenance.json"
    files.append(
        {
            "path": "_internal/notices/build-provenance.json",
            "bytes": shipped_provenance.stat().st_size,
            "sha256": digest(shipped_provenance),
        }
    )
    write_json(
        destination / "bundle-inventory.json",
        {"schema": 1, "excludes": ["_internal/notices/bundle-inventory.json"], "files": files},
    )
    shutil.copyfile(
        destination / "bundle-inventory.json", bundle / "_internal/notices/bundle-inventory.json"
    )
    print(
        f"Audited {len(files)} files, {sum('provenance' in item for item in files)} native payloads"
    )


def checksums(destination, compiler):
    write_json(
        destination / "compiler-provenance.json", {"version": "6.7.0", "sha256": digest(compiler)}
    )
    bundle = destination / "here"
    archive = destination / "here-0.2.0-windows-x64.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as output:
        for path in sorted(bundle.rglob("*")):
            if path.is_file():
                output.write(path, path.relative_to(destination).as_posix())
    paths = sorted(
        path for path in destination.iterdir() if path.is_file() and path.name != "SHA256SUMS.txt"
    )
    (destination / "SHA256SUMS.txt").write_text(
        "".join(f"{digest(path)}  {path.name}\n" for path in paths), encoding="utf-8"
    )
    print(f"Hashed {len(paths)} artifacts; application signing not performed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("prepare", "audit", "checksums"))
    parser.add_argument("path", type=Path)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--compiler", type=Path)
    arguments = parser.parse_args()
    if arguments.action == "prepare":
        prepare(arguments.path, arguments.wheel)
    elif arguments.action == "audit":
        audit(arguments.path, arguments.wheel)
    else:
        checksums(arguments.path, arguments.compiler)
