"""Catalog storage contracts, using disposable real SQLite databases."""

from __future__ import annotations

import importlib
import sqlite3
from pathlib import Path

import pytest


def _database():
    return importlib.import_module("here.memory.database")


def _open(tmp_path):
    database = _database()
    library = database.open_library(tmp_path)
    database.migrate(library)
    return library


def _meeting(connection, meeting_id="meeting-a"):
    connection.execute(
        "INSERT INTO meetings (id, title, status, created_at, updated_at) "
        "VALUES (?, 'Synthetic', 'completed', '2026-11-05T12:00:00Z', '2026-11-05T12:00:00Z')",
        (meeting_id,),
    )


def _revision(connection, revision_id="revision-a", meeting_id="meeting-a", number=1):
    connection.execute(
        "INSERT INTO transcript_revisions "
        "(id, meeting_id, revision_number, raw_text, provider, model, created_at, origin, "
        "content_hash, status) VALUES (?, ?, ?, 'Evidence', 'synthetic', 'fixture', "
        "'2026-11-05T12:00:00Z', 'import', ?, 'complete')",
        (revision_id, meeting_id, number, "a" * 64),
    )


def test_fresh_migration_creates_all_twelve_tables_and_valid_catalog(tmp_path):
    database = _database()
    library = database.open_library(tmp_path)
    try:
        report = database.migrate(library)
        connection = library.connection
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
        assert {
            "schema_migrations",
            "meetings",
            "audio_assets",
            "transcript_revisions",
            "segments",
            "participants",
            "meeting_participants",
            "speaker_assignments",
            "lifecycle_events",
            "ingest_sources",
            "pending_operations",
            "deleted_origins",
        } <= tables
        required_columns = {
            "schema_migrations": {"version", "name", "checksum"},
            "meetings": {
                "id",
                "legacy_session_id",
                "title",
                "notes",
                "started_at",
                "completed_at",
                "duration_ms",
                "paused_ms",
                "status",
                "active_revision_id",
                "row_version",
                "created_at",
                "updated_at",
            },
            "audio_assets": {
                "id",
                "meeting_id",
                "relative_path",
                "size_bytes",
                "sha256",
                "codec",
                "sample_rate",
                "channels",
                "duration_ms",
                "status",
            },
            "transcript_revisions": {
                "id",
                "meeting_id",
                "revision_number",
                "raw_text",
                "presentation_text",
                "provider",
                "model",
                "created_at",
                "origin",
                "content_hash",
                "status",
            },
            "segments": {
                "id",
                "revision_id",
                "ordinal",
                "text",
                "start_ms",
                "end_ms",
                "timing_source",
                "speaker_key",
            },
            "participants": {"id", "display_name"},
            "meeting_participants": {"meeting_id", "participant_id", "role"},
            "speaker_assignments": {"meeting_id", "speaker_key", "participant_id"},
            "lifecycle_events": {
                "id",
                "meeting_id",
                "kind",
                "occurred_at",
                "recorded_duration_ms",
                "payload",
            },
            "ingest_sources": {
                "id",
                "origin_path",
                "origin_fingerprint",
                "meeting_id",
                "revision_id",
            },
            "pending_operations": {
                "id",
                "kind",
                "phase",
                "meeting_id",
                "staging_path",
                "destination_path",
            },
            "deleted_origins": {"origin_fingerprint", "deleted_at"},
        }
        for table, columns in required_columns.items():
            actual = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
            assert columns <= actual, table
        assert (tmp_path / "library.sqlite3").is_file()
        assert report.from_version == 0
        assert report.to_version >= 1
        assert tuple(report.applied_versions) == tuple(range(1, report.to_version + 1))
        rows = connection.execute(
            "SELECT version, name, checksum FROM schema_migrations"
        ).fetchall()
        assert len(rows) == report.to_version
        assert all(name and len(checksum) == 64 for _, name, checksum in rows)
        assert connection.execute("PRAGMA user_version").fetchone()[0] == report.to_version
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        again = database.migrate(library)
        assert tuple(again.applied_versions) == ()
        assert again.from_version == again.to_version == report.to_version
    finally:
        library.close()


def test_every_connection_enables_foreign_keys_full_sync_wal_and_bounded_timeout(tmp_path):
    first = _open(tmp_path)
    second = _database().open_library(tmp_path)
    try:
        assert first.connection is not second.connection
        for library in (first, second):
            connection = library.connection
            assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
            assert connection.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
            assert connection.execute("PRAGMA synchronous").fetchone()[0] == 2
            assert 1 <= connection.execute("PRAGMA busy_timeout").fetchone()[0] <= 5000
    finally:
        second.close()
        first.close()


def test_future_schema_is_typed_rejection_and_does_not_modify_data(tmp_path):
    database = _database()
    library = _open(tmp_path)
    try:
        connection = library.connection
        _meeting(connection)
        connection.execute("PRAGMA user_version=999999")
        connection.commit()
        with pytest.raises(database.UnsupportedSchema):
            database.migrate(library)
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 999999
        assert connection.execute("SELECT title FROM meetings").fetchone()[0] == "Synthetic"
    finally:
        library.close()


def test_open_future_database_rejects_before_changing_journal_or_creating_backups(tmp_path):
    database = _database()
    path = tmp_path / "library.sqlite3"
    with sqlite3.connect(path) as connection:
        connection.execute("CREATE TABLE future_data (value TEXT)")
        connection.execute("INSERT INTO future_data VALUES ('future-client-data')")
        connection.execute("PRAGMA user_version=999999")
    before = path.read_bytes()
    with pytest.raises(database.UnsupportedSchema):
        database.open_library(tmp_path)
    assert path.read_bytes() == before
    assert sorted(item.name for item in tmp_path.iterdir()) == ["library.sqlite3"]


def test_database_symlink_is_rejected_without_touching_external_target(tmp_path):
    database = _database()
    external = tmp_path / "external.sqlite3"
    with sqlite3.connect(external) as connection:
        connection.execute("CREATE TABLE keep_private (value TEXT)")
        connection.execute("INSERT INTO keep_private VALUES ('synthetic external target')")
    root = tmp_path / "managed"
    root.mkdir()
    try:
        (root / "library.sqlite3").symlink_to(external)
    except OSError:
        pytest.skip("Symlink creation privilege is unavailable")
    before = external.read_bytes()
    with pytest.raises((database.IntegrityFailure, ValueError)):
        database.open_library(root)
    assert external.read_bytes() == before
    assert not Path(str(external) + "-wal").exists()


def test_migration_backup_uses_sqlite_api_and_includes_committed_wal(tmp_path, monkeypatch):
    database = _database()
    connect = sqlite3.connect
    backup_calls = []

    class ObservedConnection(sqlite3.Connection):
        def backup(self, target, *args, **kwargs):
            backup_calls.append(True)
            return super().backup(target, *args, **kwargs)

    def observed_connect(*args, **kwargs):
        kwargs["factory"] = ObservedConnection
        return connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", observed_connect)
    library = database.open_library(tmp_path)
    try:
        connection = library.connection
        connection.execute("PRAGMA wal_autocheckpoint=0")
        connection.execute("CREATE TABLE synthetic_pre_migration (value TEXT)")
        connection.execute("INSERT INTO synthetic_pre_migration VALUES ('committed-in-wal')")
        connection.commit()
        assert Path(str(tmp_path / "library.sqlite3") + "-wal").stat().st_size > 0
        report = database.migrate(library)
        assert backup_calls, "Migrations must use Connection.backup, including with WAL enabled"
        backup_path = Path(report.backup_path)
        assert backup_path.is_file()
        assert backup_path.resolve().is_relative_to(tmp_path.resolve())
        assert backup_path.resolve() != (tmp_path / "library.sqlite3").resolve()
        assert backup_path in library.managed_backup_paths()
        with connect(backup_path) as backup:
            assert backup.execute("SELECT value FROM synthetic_pre_migration").fetchone() == (
                "committed-in-wal",
            )
            assert backup.execute("PRAGMA user_version").fetchone()[0] == 0
            assert backup.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        library.close()


def test_failed_migration_rolls_back_ddl_and_version_and_retains_backup(tmp_path):
    database = _database()
    library = database.open_library(tmp_path)
    try:
        connection = library.connection
        connection.execute("CREATE TABLE synthetic_existing (value TEXT)")
        connection.execute("INSERT INTO synthetic_existing VALUES ('preserve-me')")
        connection.commit()
        created_tables = []

        def reject_second_table(action, name, _column, _database_name, _trigger):
            if action == sqlite3.SQLITE_CREATE_TABLE and not name.startswith("sqlite_"):
                created_tables.append(name)
                if len(created_tables) == 2:
                    return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        connection.set_authorizer(reject_second_table)
        try:
            with pytest.raises(database.IntegrityFailure):
                database.migrate(library)
        finally:
            connection.set_authorizer(None)
        assert len(created_tables) == 2, "Fault must happen after the first migration DDL"
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 0
        assert connection.execute("SELECT value FROM synthetic_existing").fetchone()[0] == (
            "preserve-me"
        )
        remaining = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
        assert [row[0] for row in remaining] == ["synthetic_existing"]
        assert list(library.managed_backup_paths())
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        library.close()


def test_existing_v1_upgrade_backs_up_before_applying_numbered_migration(tmp_path, monkeypatch):
    database = _database()
    migrations = tuple(database.MIGRATIONS)
    synthetic_version = migrations[-1].version + 1
    with monkeypatch.context() as v1_only:
        v1_only.setattr(database, "MIGRATIONS", migrations[:1])
        library = _open(tmp_path)
    try:
        connection = library.connection
        _meeting(connection)
        connection.commit()
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        monkeypatch.setattr(
            database,
            "MIGRATIONS",
            (
                *migrations,
                database.Migration(
                    version=synthetic_version,
                    name="synthetic_upgrade",
                    sql="CREATE TABLE synthetic_upgrade (value TEXT NOT NULL);",
                ),
            ),
        )
        report = database.migrate(library)
        assert (report.from_version, report.to_version) == (1, synthetic_version)
        assert tuple(report.applied_versions) == tuple(range(2, synthetic_version + 1))
        assert connection.execute("SELECT COUNT(*) FROM synthetic_upgrade").fetchone()[0] == 0
        assert connection.execute("SELECT title FROM meetings").fetchone()[0] == "Synthetic"
        backup_path = Path(report.backup_path)
        assert backup_path in library.managed_backup_paths()
        with sqlite3.connect(backup_path) as backup:
            assert backup.execute("PRAGMA user_version").fetchone()[0] == 1
            assert backup.execute("SELECT title FROM meetings").fetchone()[0] == "Synthetic"
            assert (
                backup.execute(
                    "SELECT name FROM sqlite_master WHERE name='synthetic_upgrade'"
                ).fetchone()
                is None
            )
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        library.close()


def test_failed_upgrade_preserves_v1_schema_rows_and_migration_ledger(tmp_path, monkeypatch):
    database = _database()
    migrations = tuple(database.MIGRATIONS)
    synthetic_version = migrations[-1].version + 1
    with monkeypatch.context() as v1_only:
        v1_only.setattr(database, "MIGRATIONS", migrations[:1])
        library = _open(tmp_path)
    try:
        connection = library.connection
        _meeting(connection)
        connection.commit()
        before = [tuple(row) for row in connection.execute("SELECT * FROM schema_migrations")]
        before_schema = [
            tuple(row)
            for row in connection.execute("SELECT * FROM sqlite_master ORDER BY type, name")
        ]
        monkeypatch.setattr(
            database,
            "MIGRATIONS",
            (
                *migrations,
                database.Migration(
                    version=synthetic_version,
                    name="synthetic_failing_upgrade",
                    sql="CREATE TABLE should_rollback (value TEXT); INSERT INTO absent VALUES (1);",
                ),
            ),
        )
        with pytest.raises(database.IntegrityFailure):
            database.migrate(library)
        assert connection.execute("PRAGMA user_version").fetchone()[0] == 1
        assert [
            tuple(row) for row in connection.execute("SELECT * FROM schema_migrations")
        ] == before
        assert [
            tuple(row)
            for row in connection.execute("SELECT * FROM sqlite_master ORDER BY type, name")
        ] == before_schema
        assert connection.execute("SELECT title FROM meetings").fetchone()[0] == "Synthetic"
        assert (
            connection.execute(
                "SELECT name FROM sqlite_master WHERE name='should_rollback'"
            ).fetchone()
            is None
        )
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        library.close()


@pytest.mark.parametrize(
    "violation",
    [
        "orphan-revision",
        "duplicate-revision-number",
        "duplicate-segment-ordinal",
        "negative-segment-start",
        "reversed-segment-bounds",
        "invalid-timing-source",
        "invalid-meeting-status",
        "cross-meeting-active-revision",
    ],
)
def test_schema_rejects_broken_relations_and_durable_values(tmp_path, violation):
    library = _open(tmp_path)
    try:
        connection = library.connection
        _meeting(connection)
        _revision(connection)
        connection.commit()
        with pytest.raises(sqlite3.IntegrityError):
            if violation == "orphan-revision":
                _revision(connection, "orphan", "absent")
            elif violation == "duplicate-revision-number":
                _revision(connection, "duplicate")
            elif violation == "invalid-meeting-status":
                connection.execute("UPDATE meetings SET status='paused'")
            elif violation == "cross-meeting-active-revision":
                _meeting(connection, "meeting-b")
                connection.execute(
                    "UPDATE meetings SET active_revision_id='revision-a' WHERE id='meeting-b'"
                )
            else:
                values = {
                    "duplicate-segment-ordinal": (0, 10, "provider"),
                    "negative-segment-start": (-1, 10, "provider"),
                    "reversed-segment-bounds": (20, 10, "provider"),
                    "invalid-timing-source": (0, 10, "guessed"),
                }[violation]
                sql = (
                    "INSERT INTO segments (id, revision_id, ordinal, text, start_ms, end_ms, "
                    "timing_source) VALUES (?, 'revision-a', 0, 'Evidence', ?, ?, ?)"
                )
                if violation == "duplicate-segment-ordinal":
                    connection.execute(sql, ("segment-existing", *values))
                connection.execute(sql, ("segment-invalid", *values))
        connection.rollback()
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        library.close()


def test_untimed_segments_keep_null_bounds_and_source_order(tmp_path):
    library = _open(tmp_path)
    try:
        connection = library.connection
        _meeting(connection)
        _revision(connection)
        connection.executemany(
            "INSERT INTO segments (id, revision_id, ordinal, text, timing_source) "
            "VALUES (?, 'revision-a', ?, ?, 'absent')",
            [
                ("segment-b", 1, "Later source paragraph"),
                ("segment-a", 0, "First source paragraph"),
            ],
        )
        rows = connection.execute(
            "SELECT text, start_ms, end_ms FROM segments ORDER BY ordinal"
        ).fetchall()
        assert [tuple(row) for row in rows] == [
            ("First source paragraph", None, None),
            ("Later source paragraph", None, None),
        ]
    finally:
        library.close()
