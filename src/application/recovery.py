"""Local, explicit and idempotent recovery. Discovery never invokes a provider."""

import sqlite3
from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from here.application.processing import SessionProcessor, _validated_recovery_source
from here.config.paths import get_data_dir
from here.memory.artifacts import managed_path, read_pending_intent
from here.memory.database import Busy, LibraryError
from here.memory.service import (
    Conflict,
    MemoryService,
    NotFound,
    OwnershipBusy,
    recovery_read_view,
)
from here.output.metadata import ErrorMetadataDocument, SessionMetadata
from here.output.ownership import FileLease, LeaseBusy
from here.output.paths import UnsafeSessionPath, session_artifact_path, validate_session_directory
from here.output.session_writer import ERRORS_FILE, read_session_metadata
from here.recording.journal import CaptureJournal
from here.recording.models import CaptureFailed


_INVALID_PENDING_MARKER = "Invalid pending catalog marker; local recovery required"
_PENDING_PUBLICATION = "Library publication pending local recovery"


def _pending_replay(directory: Path, metadata: SessionMetadata) -> tuple[bool, str | None]:
    """Check replay eligibility and any warning using the current pending marker."""
    marker = session_artifact_path(directory, "memory_pending.json")
    if not marker.exists():
        return False, None
    pending = read_pending_intent(directory, metadata.meeting_id)
    if metadata.status != "pending":
        return True, None
    # A final seal cannot make an uncommitted manifest final.
    if pending.hashes is not None:
        return False, "Final catalog evidence is uncommitted; pending intent retained"
    return False, None


def validate_recovery_audio(directory: Path, metadata: SessionMetadata) -> bool:
    available = False
    for source in metadata.capture_sources:
        if source.audio_file:
            actual = _validated_recovery_source(
                session_artifact_path(directory, source.audio_file), source
            )
            available |= actual.frames > 0
    if metadata.recoverable_audio:
        # Validate redirects/nonregular entries before distinguishing an absent derived WAV.
        # Only absence may use the validated raw sources above; corrupt existing audio fails.
        normalized = session_artifact_path(directory, metadata.recoverable_audio)
        try:
            normalized.stat()
        except FileNotFoundError:
            pass
        else:
            expected = metadata.sources[0] if len(metadata.sources) == 1 else None
            actual = _validated_recovery_source(normalized, expected)
            available |= actual.frames > 0
    return available


@dataclass(frozen=True, slots=True)
class RecoveryCandidate:
    session_dir: Path
    display_id: str
    recorded_duration_seconds: float
    status: str
    error_summary: str | None
    can_retry: bool
    capture_id: str | None = None


class _CatalogOwnershipUnavailable(LibraryError):
    """Catalog failure leaves legacy ownership unknown, never implicitly unowned."""


class RecoveryService:
    def __init__(self, root: Path, *, memory_service: MemoryService | None = None) -> None:
        self.root = root.absolute()
        self._memory = memory_service
        self._data_dir = memory_service.data_dir if memory_service is not None else get_data_dir()
        self._data_dir = self._data_dir.absolute()
        if self._memory is not None:
            self._memory._register_sessions_root(self.root)

    def _memory_service(self) -> MemoryService:
        """Initialize only for existing writer operations, outside read guards."""
        if self._memory is None:
            self._memory = MemoryService(self._data_dir, sessions_root=self.root)
        return self._memory

    @contextmanager
    def _read_view(self, directory: Path, source_id: str | None) -> Iterator[str | None]:
        """Translate only unresolved legacy setup failures, never caller errors."""
        with ExitStack() as view:
            try:
                context = (
                    self._memory.recovery_read_view(directory, source_id)
                    if self._memory is not None
                    else recovery_read_view(self._data_dir, directory, source_id)
                )
                identity = view.enter_context(context)
            except (NotFound, Busy, UnsafeSessionPath):
                raise
            except (LibraryError, ValueError) as exc:
                try:
                    canonical = source_id is not None and str(UUID(source_id)) == source_id
                except ValueError:
                    canonical = False
                if canonical:
                    raise
                raise _CatalogOwnershipUnavailable(
                    "Library ownership unavailable; local recovery required"
                ) from exc
            yield identity

    def _owner_identity(self, directory: Path, source_id: str | None) -> str | None:
        """Resolve the ownership target only; this closed view grants no audio access."""
        with self._read_view(directory, source_id) as identity:
            return identity

    @contextmanager
    def _inspection(
        self, directory: Path, source_id: str | None, owned_identity: str | None
    ) -> Iterator[None]:
        """Keep final suppression/alias exclusion through protected material inspection."""
        with self._read_view(directory, source_id) as identity:
            if identity != owned_identity:
                raise OwnershipBusy("Recovery ownership changed; rediscover the session")
            yield

    @contextmanager
    def _ownership(
        self, identity: str | None, *, directory: Path | None = None
    ) -> Iterator[tuple[str | None, Callable[[], None]]]:
        """Guard inspection/replay and recheck a stale selection before mutation."""
        if directory is not None:
            identity = self._owner_identity(directory, identity)
        try:
            canonical = identity is not None and str(UUID(identity)) == identity
        except ValueError:
            canonical = False
        if not canonical:
            yield identity, lambda: None
            return
        assert identity is not None
        with ExitStack() as owners:
            try:
                held: set[Path] = set()

                def acquire(*, create: bool) -> None:
                    for path in (
                        CaptureJournal.owner_path(self.root, identity),
                        managed_path(self._data_dir, f".owners/{identity}.lock"),
                    ):
                        if path in held:
                            continue
                        if create:
                            path.parent.mkdir(parents=True, exist_ok=True)
                        try:
                            lease = FileLease(path, timeout=0, create=create)
                        except FileNotFoundError:
                            continue
                        owners.callback(lease.close)
                        held.add(path)

                acquire(create=False)
                yield identity, lambda: acquire(create=True)
            except LeaseBusy as exc:
                raise OwnershipBusy(str(exc)) from exc

    def _replay_pending_publication(
        self, directory: Path, source_id: str | None, owner: str | None
    ) -> tuple[bool, str | None] | Busy:
        """Replay under the writer scope or defer SQLite contention to the final read."""
        memory = self._memory_service()
        with ExitStack() as writer:
            try:
                writer.enter_context(
                    memory.recovery_publication(directory, source_id, expected_identity=owner)
                )
            except UnsafeSessionPath:
                raise
            except ValueError:
                return True, _INVALID_PENDING_MARKER
            except Conflict:
                return True, _PENDING_PUBLICATION
            except Busy as exc:
                if type(exc) is not Busy or not isinstance(exc.__cause__, sqlite3.Error):
                    raise
                # The final read can revalidate a declared canonical identity directly.
                # An unmapped legacy origin may still require the writer's seal checks.
                if source_id is None or source_id != owner:
                    raise
                try:
                    canonical = str(UUID(source_id)) == source_id
                except ValueError:
                    canonical = False
                if not canonical:
                    raise
                return exc

            metadata = read_session_metadata(directory)
            if metadata is None or metadata.meeting_id != source_id:
                raise UnsafeSessionPath("Recovery identity changed")
            try:
                should_replay, warning = _pending_replay(directory, metadata)
            except UnsafeSessionPath:
                raise
            except ValueError:
                return True, _INVALID_PENDING_MARKER
            if not should_replay:
                return False, warning
            result = SessionProcessor(memory_service=memory).publish_memory(directory)

        if result.status != "rejected":
            return False, None
        details = "; ".join(
            str(getattr(warning, "message", warning)) for warning in result.warnings
        )
        return False, f"{_PENDING_PUBLICATION}: {details}" if details else _PENDING_PUBLICATION

    def discover(self) -> list[RecoveryCandidate]:
        validate_session_directory(self.root)
        try:
            if (self._data_dir / "library.sqlite3").exists() or (
                self._data_dir / "meetings"
            ).exists():
                self._memory_service().reconcile()
        except (OSError, ValueError, RuntimeError):
            pass
        if not self.root.exists():
            return []
        candidates: dict[tuple[str, Path], RecoveryCandidate] = {}
        completed: set[tuple[str, Path]] = set()
        for directory in sorted(self.root.iterdir()):
            if directory.name.startswith("."):
                continue
            try:
                metadata = read_session_metadata(directory)
                if metadata is None:
                    continue
                local_recovery_only = False
                publication_busy: Busy | None = None
                source_id = metadata.meeting_id
                with self._ownership(source_id, directory=directory) as (owner, claim):
                    with self._inspection(directory, source_id, owner):
                        metadata = read_session_metadata(directory)
                        if metadata is None or metadata.meeting_id != source_id:
                            raise UnsafeSessionPath("Recovery identity changed")
                        try:
                            should_replay, catalog_warning = _pending_replay(directory, metadata)
                        except UnsafeSessionPath:
                            raise
                        except ValueError:
                            should_replay = False
                            local_recovery_only = True
                            catalog_warning = _INVALID_PENDING_MARKER
                    if should_replay:
                        # Retain the upgraded leases while the writer revalidates authority.
                        claim()
                        replay = self._replay_pending_publication(
                            directory, source_id, owner
                        )
                        if isinstance(replay, Busy):
                            publication_busy = replay
                        else:
                            local_recovery_only, catalog_warning = replay
                    with self._inspection(directory, source_id, owner):
                        metadata = read_session_metadata(directory)
                        if metadata is None or metadata.meeting_id != source_id:
                            raise UnsafeSessionPath("Recovery identity changed")
                        if publication_busy is not None:
                            if metadata.status != "completed":
                                continue
                            catalog_warning = f"{_PENDING_PUBLICATION}: {publication_busy}"
                        # A copied manifest does not own another directory with the same ID.
                        key = (metadata.meeting_id or directory.name, directory)
                        if metadata.status == "completed" or local_recovery_only:
                            if catalog_warning:
                                candidates[key] = RecoveryCandidate(
                                    directory,
                                    metadata.session_id,
                                    metadata.duration_seconds,
                                    metadata.status,
                                    catalog_warning,
                                    False,
                                    metadata.meeting_id,
                                )
                            else:
                                completed.add(key)
                            continue
                        available = validate_recovery_audio(directory, metadata)
                        error_summary = metadata.failure_stage
                        errors_path = session_artifact_path(directory, ERRORS_FILE)
                        if errors_path.exists():
                            errors = ErrorMetadataDocument.model_validate_json(
                                errors_path.read_text(encoding="utf-8")
                            )
                            if errors.errors:
                                last_error = errors.errors[-1]
                                error_summary = last_error.cause_message or last_error.message
                        if catalog_warning:
                            error_summary = "; ".join(
                                message for message in (error_summary, catalog_warning) if message
                            )
                        candidates[key] = RecoveryCandidate(
                            directory,
                            metadata.session_id,
                            metadata.duration_seconds,
                            metadata.status,
                            error_summary,
                            available,
                            metadata.meeting_id,
                        )
            except _CatalogOwnershipUnavailable as exc:
                # Metadata passed path validation, but unknown ownership permits
                # no marker, error-document or audio inspection, nor mutation.
                assert metadata is not None
                key = (metadata.meeting_id or directory.name, directory)
                candidates[key] = RecoveryCandidate(
                    directory,
                    metadata.session_id,
                    metadata.duration_seconds,
                    metadata.status,
                    str(exc),
                    False,
                    metadata.meeting_id,
                )
            except (OSError, ValueError, RuntimeError):
                continue
        capture_root = self.root / ".captures"
        try:
            validate_session_directory(capture_root)
            journals = sorted(capture_root.iterdir()) if capture_root.exists() else []
        except (OSError, ValueError):
            journals = []
        for directory in journals:
            try:
                if str(UUID(directory.name)) != directory.name:
                    raise UnsafeSessionPath("Invalid capture identity")
                with self._ownership(directory.name, directory=directory) as (owner, _):
                    with self._inspection(directory, directory.name, owner):
                        journal = CaptureJournal.load(self.root, directory.name)
                        # Only metadata at this validated reservation supersedes the journal.
                        key = (journal.document.capture_id, journal.destination)
                        if key in completed or key in candidates:
                            continue
                        recording = journal.recording_session()
                        candidates[key] = RecoveryCandidate(
                            journal.destination,
                            journal.document.started_at.strftime("%Y%m%d_%H%M%S"),
                            recording.duration_seconds,
                            "interrupted",
                            "Capture interrupted before final persistence",
                            recording.duration_seconds > 0,
                            journal.document.capture_id,
                        )
            except (OSError, ValueError, RuntimeError):
                continue
        return [item for key, item in candidates.items() if key not in completed]

    def materialize(self, candidate: RecoveryCandidate) -> Path:
        if candidate.session_dir.absolute().parent != self.root:
            raise UnsafeSessionPath("Recovery destination is outside sessions root")
        with self._ownership(
            candidate.capture_id, directory=candidate.session_dir
        ) as (owner, claim):
            with self._inspection(candidate.session_dir, candidate.capture_id, owner):
                missing_metadata = read_session_metadata(candidate.session_dir) is None
                if missing_metadata:
                    if candidate.capture_id is None:
                        raise ValueError("Recovery metadata no longer exists")
                    if str(UUID(candidate.capture_id)) != candidate.capture_id:
                        raise UnsafeSessionPath("Invalid capture identity")
                    # Recognize the actual journal before creating writer leases;
                    # pure manifest inspection leaves no owner-file artifacts.
                    journal = CaptureJournal.load(self.root, candidate.capture_id)
                    if journal.destination != candidate.session_dir.absolute():
                        raise UnsafeSessionPath("Recovery destination changed")
            if missing_metadata:
                claim()
            return self._materialize_owned(candidate, owner, reconstruct=missing_metadata)

    def _materialize_owned(
        self,
        candidate: RecoveryCandidate,
        owned_identity: str | None,
        *,
        reconstruct: bool,
    ) -> Path:
        directory = candidate.session_dir.absolute()
        if directory.parent != self.root:
            raise UnsafeSessionPath("Recovery destination is outside sessions root")
        with self._inspection(directory, candidate.capture_id, owned_identity):
            metadata = read_session_metadata(directory)
            if metadata is not None:
                if metadata.meeting_id != candidate.capture_id:
                    raise UnsafeSessionPath("Recovery identity changed")
                validate_recovery_audio(directory, metadata)
                return directory
            if candidate.capture_id is None or not reconstruct:
                raise ValueError("Recovery metadata no longer exists")
            journal = CaptureJournal.load(self.root, candidate.capture_id)
            if journal.destination != directory:
                raise UnsafeSessionPath("Recovery destination changed")
            session = journal.recording_session()
            failure = CaptureFailed(
                session, RuntimeError("Capture interrupted before final persistence")
            )
        # The writer revalidates under its own publication lock. Existing real
        # native/canonical leases stay retained by materialize across this handoff.
        return (
            SessionProcessor(memory_service=self._memory_service())
            .persist_capture_failure(
                failure,
                self.root,
                started_at=journal.document.started_at,
            )
            .session_dir
        )
