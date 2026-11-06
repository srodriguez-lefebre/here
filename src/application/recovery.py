"""Local, explicit and idempotent recovery. Discovery never invokes a provider."""

from collections.abc import Callable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from here.application.processing import SessionProcessor, _validated_recovery_source
from here.config.paths import get_data_dir
from here.memory.artifacts import managed_path, read_pending_intent
from here.memory.database import LibraryError
from here.memory.service import MemoryService, NotFound, OwnershipBusy
from here.output.metadata import ErrorMetadataDocument, SessionMetadata
from here.output.ownership import FileLease, LeaseBusy
from here.output.paths import UnsafeSessionPath, session_artifact_path, validate_session_directory
from here.output.session_writer import ERRORS_FILE, read_session_metadata
from here.recording.journal import CaptureJournal
from here.recording.models import CaptureFailed


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
    def __init__(self, root: Path) -> None:
        self.root = root.absolute()

    def _owner_identity(self, directory: Path, source_id: str | None) -> str | None:
        """A legacy manifest keeps its ID; its normalized origin locates its owner."""
        data_dir = get_data_dir()
        if (data_dir / "library.sqlite3").exists():
            try:
                parent = MemoryService(data_dir).retry_parent(directory, source_id)
            except (NotFound, OwnershipBusy, UnsafeSessionPath):
                raise
            except (LibraryError, ValueError) as exc:
                raise _CatalogOwnershipUnavailable(
                    "Library ownership unavailable; local recovery required"
                ) from exc
            if parent is not None:
                return parent[0]
        return source_id

    @contextmanager
    def _ownership(
        self, identity: str | None, *, directory: Path | None = None
    ) -> Iterator[Callable[[], None]]:
        """Guard inspection/replay and recheck a stale selection before mutation."""
        if directory is not None:
            identity = self._owner_identity(directory, identity)
        try:
            canonical = identity is not None and str(UUID(identity)) == identity
        except ValueError:
            canonical = False
        if not canonical:
            yield lambda: None
            return
        assert identity is not None
        with ExitStack() as owners:
            try:
                held: set[Path] = set()

                def acquire(*, create: bool) -> None:
                    for path in (
                        CaptureJournal.owner_path(self.root, identity),
                        managed_path(get_data_dir(), f".owners/{identity}.lock"),
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
                yield lambda: acquire(create=True)
            except LeaseBusy as exc:
                raise OwnershipBusy(str(exc)) from exc

    def discover(self) -> list[RecoveryCandidate]:
        validate_session_directory(self.root)
        try:
            data_dir = get_data_dir()
            if (data_dir / "library.sqlite3").exists() or (data_dir / "meetings").exists():
                MemoryService(data_dir).reconcile()
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
                catalog_warning = None
                invalid_marker = False
                with self._ownership(metadata.meeting_id, directory=directory) as claim:
                    marker = session_artifact_path(directory, "memory_pending.json")
                    if marker.exists():
                        try:
                            pending = read_pending_intent(directory, metadata.meeting_id)
                        except UnsafeSessionPath:
                            raise
                        except ValueError:
                            # Invalid data grants no replay or provider authority.
                            # Do not echo untrusted contents or inspect its audio.
                            pending = None
                            invalid_marker = True
                            catalog_warning = (
                                "Invalid pending catalog marker; local recovery required"
                            )
                        if pending is not None and metadata.status != "pending":
                            claim()
                            result = SessionProcessor().publish_memory(
                                directory,
                                revision_of=pending.revision_of,
                                status_only=pending.intent == "status",
                            )
                            if result.status == "rejected":
                                details = "; ".join(
                                    str(getattr(warning, "message", warning))
                                    for warning in result.warnings
                                )
                                catalog_warning = "Library publication pending local recovery"
                                if details:
                                    catalog_warning += f": {details}"
                        elif pending is not None and pending.hashes is not None:
                            # A final seal cannot make an uncommitted manifest final.
                            # Preserve both pieces of evidence without acquiring a
                            # writer lease or adopting the previous transcript.
                            catalog_warning = (
                                "Final catalog evidence is uncommitted; pending intent retained"
                            )
                # A copied manifest does not own other directories with the same ID.
                key = (metadata.meeting_id or directory.name, directory)
                if metadata.status == "completed" or invalid_marker:
                    if catalog_warning:
                        # This warning needs local catalog recovery only.
                        # Do not inspect retry audio or offer another provider call.
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
                with self._ownership(metadata.meeting_id, directory=directory):
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
                with self._ownership(directory.name):
                    journal = CaptureJournal.load(self.root, directory.name)
                    # Only metadata at this journal's validated reservation supersedes it.
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
        with self._ownership(candidate.capture_id, directory=candidate.session_dir) as claim:
            if read_session_metadata(candidate.session_dir) is None:
                if candidate.capture_id is None:
                    raise ValueError("Recovery metadata no longer exists")
                if str(UUID(candidate.capture_id)) != candidate.capture_id:
                    raise UnsafeSessionPath("Invalid capture identity")
                # Recognize an actual owned journal before creating its writer
                # leases; pure manifest inspection leaves no protocol artifacts.
                journal_path = session_artifact_path(
                    self.root / ".captures" / candidate.capture_id, "journal.json"
                )
                if not journal_path.is_file():
                    raise FileNotFoundError(journal_path)
                claim()
            return self._materialize_owned(candidate)

    def _materialize_owned(self, candidate: RecoveryCandidate) -> Path:
        directory = candidate.session_dir.absolute()
        if directory.parent != self.root:
            raise UnsafeSessionPath("Recovery destination is outside sessions root")
        metadata = read_session_metadata(directory)
        if metadata is not None:
            if metadata.meeting_id != candidate.capture_id:
                raise UnsafeSessionPath("Recovery identity changed")
            validate_recovery_audio(directory, metadata)
            return directory
        if candidate.capture_id is None:
            raise ValueError("Recovery metadata no longer exists")
        journal = CaptureJournal.load(self.root, candidate.capture_id)
        if journal.destination != directory:
            raise UnsafeSessionPath("Recovery destination changed")
        session = journal.recording_session()
        failure = CaptureFailed(
            session, RuntimeError("Capture interrupted before final persistence")
        )
        return (
            SessionProcessor()
            .persist_capture_failure(
                failure,
                self.root,
                started_at=journal.document.started_at,
            )
            .session_dir
        )
