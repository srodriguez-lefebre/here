"""Exercise the native search payload shipped in a real Windows x64 bundle."""

import hashlib
import json
import os
import sqlite3
import struct
from contextlib import closing
from pathlib import Path

import pytest

_NATIVE_DLL = "_internal/here/memory/_native_search_v2.dll"
_NOTICES = "_internal/notices"
_PROVENANCE = f"{_NOTICES}/build-provenance.json"
_ENTRYPOINT = "sqlite3_here_search_v2_init"


@pytest.fixture(scope="module")
def windows_bundle():
    configured = os.environ.get("HERE_WINDOWS_BUNDLE")
    if not configured:
        pytest.skip("Set HERE_WINDOWS_BUNDLE to the built Windows x64 here directory")
    assert os.name == "nt", "The native Windows bundle acceptance requires Windows"
    bundle = Path(configured).resolve(strict=True)
    assert (bundle / "here.exe").is_file(), f"Expected the here bundle directory: {bundle}"
    return bundle


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _audited_file(bundle, files, relative):
    path = bundle / relative
    assert path.is_file(), f"Missing shipped file: {relative}"
    assert path.resolve().is_relative_to(bundle), f"File resolves outside the bundle: {relative}"
    matches = [row for row in files if row["path"] == relative]
    assert len(matches) == 1, f"Expected one inventory entry for {relative}"
    row = matches[0]
    assert row["bytes"] == path.stat().st_size
    assert row["sha256"] == _sha256(path)
    return row


@pytest.fixture(scope="module")
def audited_search_dll(windows_bundle):
    inventory = _read_json(windows_bundle / _NOTICES / "bundle-inventory.json")
    dll_row = _audited_file(windows_bundle, inventory["files"], _NATIVE_DLL)
    return windows_bundle / _NATIVE_DLL, dll_row, inventory["files"]


def test_windows_bundle_loads_the_audited_x64_search_dll(audited_search_dll):
    dll, _, _ = audited_search_dll
    with dll.open("rb") as stream:
        assert stream.read(2) == b"MZ"
        stream.seek(0x3C)
        (pe_offset,) = struct.unpack("<I", stream.read(4))
        stream.seek(pe_offset)
        assert stream.read(6) == b"PE\0\0\x64\x86", "Expected the AMD64 PE machine type"

    with closing(sqlite3.connect(":memory:")) as connection:
        connection.enable_load_extension(True)
        try:
            connection.load_extension(str(dll), entrypoint=_ENTRYPOINT)
        finally:
            connection.enable_load_extension(False)


def test_windows_bundle_links_search_dll_to_build_inputs_and_shipped_notices(
    windows_bundle, audited_search_dll
):
    _, dll_row, files = audited_search_dll
    _audited_file(windows_bundle, files, _PROVENANCE)
    build_provenance = _read_json(windows_bundle / _PROVENANCE)
    native = build_provenance["native_search_v2"]

    assert native["path"] == _NATIVE_DLL
    assert native["sha256"] == dll_row["sha256"]
    assert native["architecture"] == "x86_64"
    assert native["entrypoint"] == _ENTRYPOINT
    assert {"kind": "here SQLite extension", **native} in dll_row["provenance"]

    source = native["source"]
    assert source["path"] == "native/search_v2.c"
    assert source["sha256"] == build_provenance["source_inputs_sha256"][source["path"]]
    compiler = native["compiler"]
    assert compiler["version"] == "0.15.2"
    headers = native["sqlite_headers"]
    assert headers["version"] == "3.53.1"
    assert set(headers["files"]) == {"sqlite3.h", "sqlite3ext.h"}
    for digest in (
        source["sha256"],
        compiler["sha256"],
        headers["sha256"],
        *headers["files"].values(),
    ):
        assert len(digest) == 64
        assert all(character in "0123456789abcdef" for character in digest)

    notices = native["notices"]
    assert notices, "Expected native search notice paths"
    assert len(notices) == len(set(notices))
    for relative in notices:
        assert relative.startswith(f"{_NOTICES}/")
        notice_row = _audited_file(windows_bundle, files, relative)
        assert notice_row["bytes"] > 0
