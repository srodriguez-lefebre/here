"""Local, explicit and idempotent recovery. Discovery never invokes a provider."""

from dataclasses import dataclass
from pathlib import Path

from here.application.processing import SessionProcessor, _validated_recovery_source
from here.output.metadata import ErrorMetadataDocument, SessionMetadata
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

    def discover(self) -> list[RecoveryCandidate]:
        validate_session_directory(self.root)
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
                # A copied manifest does not own other directories with the same ID.
                key = (metadata.meeting_id or directory.name, directory)
                if metadata.status == "completed":
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
