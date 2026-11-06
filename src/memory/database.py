"""Owned SQLite connections, transactional numbered migrations and managed backups."""

from __future__ import annotations

import hashlib
import os
import sqlite3
from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from time import monotonic
from uuid import uuid4

from here.output.paths import session_artifact_path, validate_session_directory


class LibraryError(RuntimeError):
    """Recoverable local catalog failure; never implies loss of captured audio."""


class UnsupportedSchema(LibraryError):
    """The installed client cannot interpret a newer library schema."""


class IntegrityFailure(LibraryError):
    """A migration or catalog mutation violates the durable memory contract."""


class Busy(LibraryError):
    """Another worker holds the bounded catalog write lock."""


def storage_error(error: sqlite3.Error) -> LibraryError:
    if getattr(error, "sqlite_errorcode", 0) & 0xFF in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED):
        return Busy("Library is busy; retry the local operation")
    return IntegrityFailure(f"Library integrity operation failed ({type(error).__name__})")


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    name: str
    sql: str

    @property
    def checksum(self) -> str:
        return hashlib.sha256(self.sql.encode("utf-8")).hexdigest()


_INITIAL_SQL = """
CREATE TABLE schema_migrations (
    version INTEGER PRIMARY KEY, name TEXT NOT NULL, checksum TEXT NOT NULL
);
CREATE TABLE meetings (
    id TEXT PRIMARY KEY NOT NULL, legacy_session_id TEXT, title TEXT NOT NULL DEFAULT '',
    notes TEXT NOT NULL DEFAULT '', started_at TEXT, completed_at TEXT,
    duration_ms INTEGER CHECK (duration_ms >= 0), paused_ms INTEGER NOT NULL DEFAULT 0
        CHECK (paused_ms >= 0),
    status TEXT NOT NULL CHECK (status IN
        ('recording','processing','completed','failed','cancelled','interrupted','deleting')),
    active_revision_id TEXT, row_version INTEGER NOT NULL DEFAULT 1 CHECK (row_version > 0),
    created_at TEXT NOT NULL, updated_at TEXT NOT NULL,
    FOREIGN KEY (id, active_revision_id) REFERENCES transcript_revisions(meeting_id, id)
);
CREATE INDEX meetings_legacy_id ON meetings(legacy_session_id);
CREATE TABLE audio_assets (
    id TEXT PRIMARY KEY NOT NULL,
    meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    relative_path TEXT NOT NULL UNIQUE, size_bytes INTEGER NOT NULL CHECK (size_bytes >= 0),
    sha256 TEXT NOT NULL CHECK (length(sha256) = 64), codec TEXT NOT NULL,
    sample_rate INTEGER NOT NULL CHECK (sample_rate > 0),
    channels INTEGER NOT NULL CHECK (channels > 0),
    duration_ms INTEGER NOT NULL CHECK (duration_ms >= 0),
    status TEXT NOT NULL CHECK (status IN ('available','missing','corrupt'))
);
CREATE TABLE transcript_revisions (
    id TEXT PRIMARY KEY NOT NULL,
    meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    revision_number INTEGER NOT NULL CHECK (revision_number > 0), raw_text TEXT NOT NULL,
    presentation_text TEXT, provider TEXT NOT NULL, model TEXT NOT NULL, created_at TEXT NOT NULL,
    origin TEXT NOT NULL, content_hash TEXT NOT NULL CHECK (length(content_hash) = 64),
    status TEXT NOT NULL CHECK (status IN ('complete','partial')),
    UNIQUE (meeting_id, revision_number), UNIQUE (meeting_id, id)
);
CREATE TABLE segments (
    id TEXT PRIMARY KEY NOT NULL,
    revision_id TEXT NOT NULL REFERENCES transcript_revisions(id) ON DELETE CASCADE,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0), text TEXT NOT NULL,
    start_ms INTEGER CHECK (start_ms >= 0), end_ms INTEGER CHECK (end_ms >= 0),
    timing_source TEXT NOT NULL CHECK (timing_source IN ('provider','imported_explicit','absent')),
    speaker_key TEXT, UNIQUE (revision_id, ordinal), CHECK (end_ms >= start_ms),
    CHECK (timing_source != 'absent' OR (start_ms IS NULL AND end_ms IS NULL))
);
CREATE TABLE participants (id TEXT PRIMARY KEY NOT NULL, display_name TEXT NOT NULL);
CREATE TABLE meeting_participants (
    meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    participant_id TEXT NOT NULL REFERENCES participants(id), role TEXT,
    PRIMARY KEY (meeting_id, participant_id)
);
CREATE TABLE speaker_assignments (
    meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    revision_id TEXT NOT NULL, speaker_key TEXT NOT NULL,
    participant_id TEXT REFERENCES participants(id),
    PRIMARY KEY (meeting_id, revision_id, speaker_key),
    FOREIGN KEY (meeting_id, revision_id) REFERENCES transcript_revisions(meeting_id, id)
        ON DELETE CASCADE
);
CREATE TABLE lifecycle_events (
    id TEXT PRIMARY KEY NOT NULL,
    meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    kind TEXT NOT NULL, occurred_at TEXT NOT NULL,
    recorded_duration_ms INTEGER CHECK (recorded_duration_ms >= 0),
    payload TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE ingest_sources (
    id TEXT PRIMARY KEY NOT NULL, origin_path TEXT NOT NULL, origin_fingerprint TEXT NOT NULL,
    meeting_id TEXT NOT NULL REFERENCES meetings(id) ON DELETE CASCADE,
    revision_id TEXT, UNIQUE (origin_path, origin_fingerprint),
    FOREIGN KEY (meeting_id, revision_id) REFERENCES transcript_revisions(meeting_id, id)
);
CREATE TABLE pending_operations (
    id TEXT PRIMARY KEY NOT NULL, kind TEXT NOT NULL,
    phase TEXT NOT NULL CHECK (phase IN ('prepared','complete','reverted')),
    meeting_id TEXT, staging_path TEXT, destination_path TEXT,
    payload_json TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE deleted_origins (
    origin_fingerprint TEXT PRIMARY KEY NOT NULL, deleted_at TEXT NOT NULL
);
"""
_REVISION_AUDIO_SQL = """
CREATE UNIQUE INDEX audio_assets_meeting_id ON audio_assets(meeting_id, id);
CREATE TABLE revision_audio_assets (
    meeting_id TEXT NOT NULL, revision_id TEXT NOT NULL, asset_id TEXT NOT NULL,
    ordinal INTEGER NOT NULL CHECK (ordinal >= 0), source_file TEXT NOT NULL,
    PRIMARY KEY (revision_id, ordinal), UNIQUE (revision_id, source_file),
    FOREIGN KEY (meeting_id, revision_id) REFERENCES transcript_revisions(meeting_id, id)
        ON DELETE CASCADE,
    FOREIGN KEY (meeting_id, asset_id) REFERENCES audio_assets(meeting_id, id)
);
"""
MIGRATIONS = (
    Migration(1, "initial_catalog", _INITIAL_SQL),
    Migration(2, "revision_audio_assets", _REVISION_AUDIO_SQL),
)
BUSY_TIMEOUT_MS = 250


@dataclass(frozen=True, slots=True)
class MigrationReport:
    from_version: int
    to_version: int
    applied_versions: tuple[int, ...] = ()
    backup_path: Path | None = None


@dataclass(slots=True)
class LibraryConnection:
    connection: sqlite3.Connection
    data_dir: Path
    storage_warning: str | None = None

    def execute(self, sql: str, parameters: tuple[object, ...] = ()) -> sqlite3.Cursor:
        """Translate bounded read locks as well as transaction write locks."""
        try:
            return self.connection.execute(sql, parameters)
        except sqlite3.Error as error:
            raise storage_error(error) from error

    def close(self) -> None:
        self.connection.close()

    def managed_backup_paths(self) -> tuple[Path, ...]:
        directory = self.data_dir / "backups"
        validate_session_directory(directory)
        if not directory.exists():
            return ()
        return tuple(
            session_artifact_path(directory, p.name)
            for p in sorted(directory.glob("library-*.sqlite3"))
        )


def _version(connection: sqlite3.Connection) -> int:
    return int(connection.execute("PRAGMA user_version").fetchone()[0])


def _supported(connection: sqlite3.Connection) -> int:
    version = _version(connection)
    if version > MIGRATIONS[-1].version:
        raise UnsupportedSchema(f"Library schema {version} requires a newer here client")
    return version


def open_library(data_dir: Path) -> LibraryConnection:
    directory = Path(data_dir).absolute()
    validate_session_directory(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = session_artifact_path(directory, "library.sqlite3")
    # SQLite may access its sidecars before our first PRAGMA; validate them too.
    for suffix in ("-wal", "-shm", "-journal"):
        session_artifact_path(directory, path.name + suffix)
    connection = sqlite3.connect(path, timeout=BUSY_TIMEOUT_MS / 1000, isolation_level=None)
    try:
        _supported(connection)  # Refuse future schemas before changing persistent pragmas.
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        # WAL's shared memory is local-host only; UNC roots use rollback journaling.
        remote = str(directory).startswith(("\\\\", "//"))
        if os.name == "nt":
            import ctypes

            drive_type = ctypes.windll.kernel32.GetDriveTypeW(str(directory.anchor))
            # Only proven fixed/removable/RAM volumes can use local WAL semantics.
            remote = remote or drive_type not in (2, 3, 6)
        mode = "DELETE" if remote else "WAL"
        connection.execute(f"PRAGMA journal_mode={mode}")
        connection.execute("PRAGMA synchronous=FULL")
        connection.row_factory = sqlite3.Row
        warning = (
            (
                "Biblioteca en volumen remoto; se usa journal local de reversión, "
                "con menor concurrencia"
            )
            if remote
            else None
        )
        return LibraryConnection(connection, directory, warning)
    except sqlite3.Error as error:
        connection.close()
        raise storage_error(error) from error
    except BaseException:
        connection.close()
        raise


def _check(connection: sqlite3.Connection) -> None:
    if connection.execute("PRAGMA foreign_key_check").fetchall():
        raise IntegrityFailure("Library foreign key check failed")
    if [row[0] for row in connection.execute("PRAGMA integrity_check")] != ["ok"]:
        raise IntegrityFailure("Library integrity check failed")


def _backup(library: LibraryConnection, version: int) -> Path:
    directory = library.data_dir / "backups"
    validate_session_directory(directory)
    directory.mkdir(exist_ok=True)
    path: Path = session_artifact_path(directory, f"library-v{version}-{uuid4().hex}.sqlite3")
    source_path = session_artifact_path(library.data_dir, "library.sqlite3")
    # A separate read connection snapshots committed WAL under the migrator's write lock.
    # Backing up the connection holding BEGIN IMMEDIATE would wait on itself.
    deadline = monotonic() + BUSY_TIMEOUT_MS / 1000

    def progress(status: int, remaining: int, total: int) -> None:
        if status in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED) and monotonic() > deadline:
            raise Busy("Library backup is busy; retry migration")

    try:
        with closing(sqlite3.connect(source_path, isolation_level=None)) as source:
            with closing(sqlite3.connect(path, isolation_level=None)) as destination:
                source.backup(destination, pages=128, progress=progress, sleep=0.01)
                _check(destination)
    except BaseException:
        path.unlink(missing_ok=True)
        raise
    return path


def _statements(sql: str):
    statement = ""
    for character in sql:
        statement += character
        if character == ";" and sqlite3.complete_statement(statement):
            yield statement
            statement = ""
    if statement.strip():
        yield statement


def migrate(library: LibraryConnection) -> MigrationReport:
    connection = library.connection
    if connection.in_transaction:
        raise IntegrityFailure("Migration requires its own transaction")
    backup_path = None
    try:
        _supported(connection)
        connection.execute("BEGIN IMMEDIATE")
        version = _supported(connection)
        existing = connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
        ).fetchone()
        if existing:
            applied = {
                row[0]: (row[1], row[2])
                for row in connection.execute(
                    "SELECT version, name, checksum FROM schema_migrations"
                )
            }
            expected = {m.version: (m.name, m.checksum) for m in MIGRATIONS if m.version <= version}
            if applied != expected:
                raise IntegrityFailure("Migration ledger checksum or version mismatch")
        elif version:
            raise IntegrityFailure("Missing library migration ledger")
        pending = tuple(m for m in MIGRATIONS if m.version > version)
        if pending:
            has_tables = connection.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' LIMIT 1"
            ).fetchone()
            if has_tables:
                backup_path = _backup(library, version)
            for migration in pending:
                if migration.version != _version(connection) + 1:
                    raise IntegrityFailure("Library migrations must be contiguous")
                for statement in _statements(migration.sql):
                    connection.execute(statement)
                connection.execute(
                    "INSERT INTO schema_migrations VALUES (?, ?, ?)",
                    (migration.version, migration.name, migration.checksum),
                )
                connection.execute(f"PRAGMA user_version={migration.version}")
        _check(connection)
        connection.commit()
        return MigrationReport(
            version, _version(connection), tuple(m.version for m in pending), backup_path
        )
    except sqlite3.Error as error:
        connection.rollback()
        raise storage_error(error) from error
    except BaseException:
        connection.rollback()
        raise
