"""Local, explicit and idempotent recovery. Discovery never invokes a provider."""

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from pathlib import Path
from uuid import UUID

from here.application.processing import SessionProcessor, _validated_recovery_source
from here.config.paths import get_data_dir
from here.memory.artifacts import managed_path, read_pending_intent
from here.memory.service import MemoryService, OwnershipBusy
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


class RecoveryService:
    def __init__(self, root: Path) -> None:
        self.root = root.absolute()

    def _owner_identity(self, directory: Path, source_id: str | None) -> str | None:
        """A legacy manifest keeps its ID; its normalized origin locates its owner."""
        try:
            if source_id is not None and str(UUID(source_id)) == source_id:
                return source_id
        except ValueError:
            pass
        data_dir = get_data_dir()
        if (data_dir / "library.sqlite3").exists():
            parent = MemoryService(data_dir).retry_parent(directory, source_id)
            if parent is not None:
                return parent[0]
        return source_id

    @contextmanager
    def _ownership(
        self, identity: str | None, *, directory: Path | None = None, claim: bool = False
    ) -> Iterator[None]:
        """Guard inspection/replay and recheck a stale selection before mutation."""
        if directory is not None:
            identity = self._owner_identity(directory, identity)
        try:
            canonical = identity is not None and str(UUID(identity)) == identity
        except ValueError:
            canonical = False
        if not canonical:
            yield
            return
        assert identity is not None
        with ExitStack() as owners:
            try:
                for path in (
                    CaptureJournal.owner_path(self.root, identity),
                    managed_path(get_data_dir(), f".owners/{identity}.lock"),
                ):
                    if claim:
                        path.parent.mkdir(parents=True, exist_ok=True)
                    if claim or path.exists():
                        lease = FileLease(path, timeout=0)
                        owners.callback(lease.close)
                yield
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
                with self._ownership(metadata.meeting_id, directory=directory):
                    marker = session_artifact_path(directory, "memory_pending.json")
                    if marker.exists() and metadata.status != "pending":
                        expected, intent = read_pending_intent(directory, metadata.meeting_id)
                        SessionProcessor().publish_memory(
                            directory,
                            revision_of=expected,
                            status_only=intent == "status",
                        )
                # A copied manifest does not own other directories with the same ID.
                key = (metadata.meeting_id or directory.name, directory)
                if metadata.status == "completed":
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
                candidates[key] = RecoveryCandidate(
                    directory,
                    metadata.session_id,
                    metadata.duration_seconds,
                    metadata.status,
                    error_summary,
                    available,
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
        with self._ownership(candidate.capture_id, directory=candidate.session_dir, claim=True):
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
