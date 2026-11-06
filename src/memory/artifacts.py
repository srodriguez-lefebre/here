"""Read source artifacts without changing their origin or inferring evidence."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import Any
from uuid import UUID

import soundfile as sf  # type: ignore[import-untyped]
from here.output.metadata import SessionMetadata
from here.output.paths import (
    UnsafeSessionPath,
    session_artifact_path,
    staged_artifact_path,
    validate_session_directory,
)

SESSION_ARTIFACT_NAMES = frozenset(
    {
        "session.json",
        "transcript.txt",
        "transcript.md",
        "raw_transcript.txt",
        "segments.json",
        "chunks.json",
        "events.json",
        "errors.json",
        "audio.wav",
    }
)


def declared_artifact_roles(metadata: SessionMetadata) -> set[str]:
    audio = {item.audio_file for item in metadata.capture_sources if item.audio_file}
    if metadata.recoverable_audio:
        audio.add(metadata.recoverable_audio)
    if any(Path(name).suffix.lower() != ".wav" for name in audio):
        raise ValueError("Session audio references must name WAV artifacts")
    allowed = set(SESSION_ARTIFACT_NAMES) | audio
    if any(name not in allowed for name in metadata.output_files):
        raise ValueError("Session manifest references an unknown artifact role")
    return allowed


def validate_artifact_roles(
    directory: Path, names: Iterable[str], *, require_audio_declaration: bool = False
) -> None:
    """A persisted hash/cleanup plan cannot itself grant a file an artifact role."""
    remaining = []
    for name in names:
        if not isinstance(name, str):
            raise ValueError("Artifact names must be strings")
        session_artifact_path(directory, name)
        if require_audio_declaration and Path(name).suffix.lower() == ".wav":
            remaining.append(name)
            continue
        if name in SESSION_ARTIFACT_NAMES or re.fullmatch(
            r"(?:audio_[0-9a-f]{32}|source_[0-9]{2,})\.wav", name
        ):
            continue
        if Path(name).suffix.lower() != ".wav":
            raise ValueError("Persisted plan references an unknown artifact role")
        remaining.append(name)
    if remaining:
        # Nonstandard v1 WAV names need an explicit validated session/audio role.
        # Reading this fixed protocol manifest never dereferences those references.
        path = session_artifact_path(directory, "session.json")
        if not path.exists():
            raise ValueError("Unrecognized WAV has no explicit session audio role")
        metadata = SessionMetadata.model_validate_json(path.read_text(encoding="utf-8-sig"))
        allowed = declared_artifact_roles(metadata)
        if require_audio_declaration:
            allowed = {item.audio_file for item in metadata.capture_sources if item.audio_file}
            if metadata.recoverable_audio:
                allowed.add(metadata.recoverable_audio)
            if "audio.wav" in metadata.output_files:
                allowed.add("audio.wav")
        if metadata.schema_version != 1 or not set(remaining) <= allowed:
            raise ValueError("Unrecognized WAV is not declared session audio")


def read_meeting_state(directory: Path, name: str, meeting_id: str) -> dict[str, Any]:
    """Only generated state filenames authorize reading a state envelope."""
    filename = Path(name)
    version, separator, identity = filename.stem.partition("-")
    try:
        valid = (
            filename.suffix == ".json"
            and separator
            and int(version) > 0
            and str(UUID(identity)) == identity
        )
    except ValueError:
        valid = False
    if not valid:
        raise ValueError("Unrecognized meeting-state entry retained")
    path = session_artifact_path(directory, name)
    state = json.loads(path.read_text(encoding="utf-8"))
    meeting = state.get("meeting") if isinstance(state, dict) else None
    if (
        not isinstance(state, dict)
        or state.get("schema_version") != 1
        or not isinstance(meeting, dict)
        or meeting.get("id") != meeting_id
        or meeting.get("row_version") != int(version)
    ):
        raise ValueError("Invalid managed meeting-state envelope")
    return meeting


def write_durable_json(path: Path, value: object) -> None:
    """Flush a small owned journal/receipt before its same-volume atomic replace."""
    with staged_artifact_path(path) as staged:
        with staged.open("w", encoding="utf-8") as stream:
            json.dump(value, stream, ensure_ascii=False)
            stream.flush()
            os.fsync(stream.fileno())


def managed_path(root: Path, relative: str, *, directory: bool = False) -> Path:
    """Validate every nested component using the shared output path policy."""
    if not relative or "\\" in relative or PureWindowsPath(relative).drive:
        raise UnsafeSessionPath("Invalid managed relative path")
    parts = PurePosixPath(relative).parts
    if PurePosixPath(relative).is_absolute() or any(p in {".", ".."} for p in parts):
        raise UnsafeSessionPath("Invalid managed relative path")
    current = root
    validate_session_directory(current)
    for index, part in enumerate(parts):
        # Validate a directory name as a filename before creating or descending it.
        if index < len(parts) - 1 or directory:
            if not (current / part).exists():
                session_artifact_path(current, part)
            current = current / part
            validate_session_directory(current)
        else:
            current = session_artifact_path(current, part)
    return current


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized_origin(directory: Path) -> str:
    directory = directory.absolute()
    validate_session_directory(directory)
    return os.path.normcase(str(directory.resolve()))


@dataclass(frozen=True)
class PendingIntent:
    revision_of: str | None
    intent: str
    publication_id: str | None = None
    source_kind: str = "session"
    hashes: tuple[tuple[str, str], ...] | None = None


def read_pending_intent(directory: Path, meeting_id: str | None) -> PendingIntent:
    """Read only the typed application-owned replay envelope, never provider work."""
    pending = json.loads(
        session_artifact_path(directory, "memory_pending.json").read_text(encoding="utf-8")
    )
    if not isinstance(pending, dict):
        raise ValueError("Invalid pending catalog envelope")
    expected = pending.get("revision_of")
    intent = pending.get("intent", "ingest")
    version = pending.get("schema_version")
    publication_id = pending.get("publication_id")
    source_kind = pending.get("source_kind", "session")
    if (
        type(version) is not int
        or version not in {1, 2, 3}
        or pending.get("meeting_id") != meeting_id
        or not isinstance(intent, str)
        or intent not in {"ingest", "status"}
        or expected is not None
        and (not isinstance(expected, str) or str(UUID(expected)) != expected)
        or intent == "status"
        and expected is None
        or version in {2, 3}
        and (not isinstance(publication_id, str) or str(UUID(publication_id)) != publication_id)
        or not isinstance(source_kind, str)
        or source_kind not in {"session", "legacy_audio_only"}
    ):
        raise ValueError("Invalid pending catalog intent")
    hashes = pending.get("hashes") if version == 3 else None
    if version == 3:
        if (
            not isinstance(hashes, dict)
            or "session.json" not in hashes
            or any(
                not isinstance(name, str)
                or not isinstance(digest, str)
                or re.fullmatch(r"[0-9a-f]{64}", digest) is None
                for name, digest in hashes.items()
            )
        ):
            raise ValueError("Invalid pending catalog evidence seal")
        validate_artifact_roles(directory, hashes)
    return PendingIntent(
        expected,
        intent,
        publication_id if version in {2, 3} else None,
        source_kind,
        tuple(sorted(hashes.items())) if hashes is not None else None,
    )


def publication_hashes(
    directory: Path, metadata: SessionMetadata, *, staged_metadata: Path | None = None
) -> dict[str, str]:
    """Measure exactly the local evidence authorized for final catalog publication.

    The writer supplies its own staged metadata, so the seal measures actual bytes,
    including platform newline encoding, before replacing the final manifest.
    """
    declared_artifact_roles(metadata)
    names = {"session.json", *metadata.output_files}
    names.update(item.audio_file for item in metadata.capture_sources if item.audio_file)
    if metadata.recoverable_audio:
        names.add(metadata.recoverable_audio)
    paths = {name: session_artifact_path(directory, name) for name in names}
    for name in ("raw_transcript.txt", "transcript.txt", "segments.json"):
        path = session_artifact_path(directory, name)
        if path.exists():
            paths[name] = path
    if not metadata.recoverable_audio:
        audio = session_artifact_path(directory, "audio.wav")
        if audio.exists():
            paths["audio.wav"] = audio
    if staged_metadata is not None:
        paths["session.json"] = staged_metadata
    return {name: file_hash(path) for name, path in paths.items() if path.is_file()}


def validate_pending_evidence(
    directory: Path, metadata: SessionMetadata, pending: PendingIntent
) -> None:
    if pending.hashes is not None and (
        metadata.status == "pending"
        or publication_hashes(directory, metadata) != dict(pending.hashes)
    ):
        raise ValueError("Pending catalog evidence is uncommitted or changed; intent retained")


def _milliseconds(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        return None
    return round(value * 1000) if math.isfinite(value) and value >= 0 else None


@dataclass(frozen=True)
class SessionArtifacts:
    directory: Path
    metadata: SessionMetadata
    files: tuple[tuple[str, str], ...]
    audio: tuple[dict[str, Any], ...]
    raw_text: str
    presentation_text: str
    segments: tuple[dict[str, Any], ...]
    origin_path: str
    fingerprint: str
    content_hash: str


def read_artifacts(directory: Path) -> SessionArtifacts:
    return _read_artifacts(directory)


def read_retry_artifacts(directory: Path) -> SessionArtifacts:
    """Trusted explicit M1 reprocessing keeps the source's optional opaque ID."""
    return _read_artifacts(directory, trusted_legacy=True)


def _read_artifacts(directory: Path, *, trusted_legacy: bool = False) -> SessionArtifacts:
    directory = directory.absolute()
    validate_session_directory(directory)
    metadata = SessionMetadata.model_validate_json(
        session_artifact_path(directory, "session.json").read_text(encoding="utf-8-sig")
    )
    if metadata.schema_version != 1:
        raise ValueError("Unsupported session schema")
    if (
        not trusted_legacy
        and metadata.meeting_id is not None
        and str(UUID(metadata.meeting_id)) != metadata.meeting_id
    ):
        raise ValueError("Invalid meeting UUID")
    audio_references = {p.audio_file for p in metadata.capture_sources if p.audio_file}
    if metadata.recoverable_audio:
        audio_references.add(metadata.recoverable_audio)
    declared_artifact_roles(metadata)
    names = ["session.json", *metadata.output_files, *audio_references]
    # Validate ALL manifest references before any referenced file is read.
    paths = {name: session_artifact_path(directory, name) for name in names}
    for name in ("raw_transcript.txt", "transcript.txt", "segments.json"):
        path = session_artifact_path(directory, name)
        if path.exists():
            paths[name] = path
    audio_names = [metadata.recoverable_audio] if metadata.recoverable_audio else []
    if trusted_legacy and audio_names and not paths[audio_names[0]].exists():
        # An explicit M1 retry may reconstruct a missing derived mix from its
        # validated raw sources; preserve those as the original revision evidence.
        audio_names = []
    if not audio_names and session_artifact_path(directory, "audio.wav").exists():
        audio_names = ["audio.wav"]
        paths["audio.wav"] = session_artifact_path(directory, "audio.wav")
    if not audio_names:
        audio_names = [p.audio_file for p in metadata.capture_sources if p.audio_file]
    audio = []
    for name in dict.fromkeys(audio_names):
        path = paths[name]
        info = sf.info(path)
        audio.append(
            dict(
                name=name,
                sha256=file_hash(path),
                size_bytes=path.stat().st_size,
                codec=info.subtype or info.format,
                sample_rate=info.samplerate,
                channels=info.channels,
                duration_ms=round(info.duration * 1000),
            )
        )
    if not audio:
        raise ValueError("Session has no recoverable audio")
    presentation = paths.get("transcript.txt")
    raw = paths.get("raw_transcript.txt") or presentation
    raw_text = raw.read_text(encoding="utf-8-sig") if raw and raw.exists() else ""
    presentation_text = (
        presentation.read_text(encoding="utf-8-sig")
        if presentation and presentation.exists()
        else raw_text
    )
    evidence = []
    segment_path = paths.get("segments.json")
    if segment_path and segment_path.exists():
        document = json.loads(segment_path.read_text(encoding="utf-8-sig"))
        if document.get("schema_version", 1) != 1:
            raise ValueError("Unsupported segment schema")
        for ordinal, item in enumerate(document["segments"]):
            start, end = _milliseconds(item.get("start")), _milliseconds(item.get("end"))
            if start is not None and end is not None and end < start:
                start = end = None
            timing = item.get("timing_source") or "provider"
            if start is None and end is None or timing == "absent":
                start = end = None
                timing = "absent"
            if timing not in {"provider", "imported_explicit", "absent"}:
                raise ValueError("Unknown segment timing source")
            speaker = item.get("speaker")
            scope = item.get("speaker_scope") or f"chunk:{item.get('chunk_index', 'session')}"
            key = json.dumps([scope, speaker], ensure_ascii=False) if speaker else None
            evidence.append(
                dict(
                    ordinal=ordinal,
                    text=item["text"],
                    start_ms=start,
                    end_ms=end,
                    timing_source=timing,
                    speaker_key=key,
                )
            )
    else:
        evidence = [
            dict(
                ordinal=i,
                text=text.strip(),
                start_ms=None,
                end_ms=None,
                timing_source="absent",
                speaker_key=None,
            )
            for i, text in enumerate(raw_text.split("\n\n"))
            if text.strip()
        ]
    files = tuple(sorted((name, file_hash(path)) for name, path in paths.items() if path.is_file()))
    fingerprint = hashlib.sha256(json.dumps(files).encode()).hexdigest()
    content = [
        raw_text,
        presentation_text,
        evidence,
        [a["sha256"] for a in audio],
        metadata.status,
        metadata.transcription_model,
    ]
    content_hash = hashlib.sha256(json.dumps(content, ensure_ascii=False).encode()).hexdigest()
    return SessionArtifacts(
        directory,
        metadata,
        files,
        tuple(audio),
        raw_text,
        presentation_text,
        tuple(evidence),
        normalized_origin(directory),
        fingerprint,
        content_hash,
    )
