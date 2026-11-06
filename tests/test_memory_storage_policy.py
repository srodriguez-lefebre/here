"""Probe the Windows volume capability before selecting SQLite journaling."""

from __future__ import annotations

import ctypes
import importlib
import os

import pytest


@pytest.mark.skipif(os.name != "nt", reason="Windows volume capability policy")
def test_mapped_remote_volume_uses_rollback_journal_on_real_database(tmp_path, monkeypatch):
    database = importlib.import_module("here.memory.database")
    # Simulate the actual OS capability, retaining real path checks and SQLite I/O.
    # No network drive, remote folder, account or provider is created or accessed.
    monkeypatch.setattr(ctypes.windll.kernel32, "GetDriveTypeW", lambda path: 4)
    library = database.open_library(tmp_path / "synthetic-mapped-volume")
    try:
        assert library.connection.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
        assert library.connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert library.connection.execute("PRAGMA synchronous").fetchone()[0] == 2
    finally:
        library.close()
