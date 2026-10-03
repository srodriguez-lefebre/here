from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Literal

import pytest
from here.output.metadata import ChunkMetadata
from here.output.session_writer import (
    AUDIO_FILE,
    CHUNKS_FILE,
    MARKDOWN_FILE,
    METADATA_FILE,
    SEGMENTS_FILE,
    TRANSCRIPT_ENCODING,
    TRANSCRIPT_FILE,
    SessionArtifactPaths,
    write_session_artifacts,
)
from here.recording.models import RecordedAudioSource, RecordingSession
from here.transcription.segments import TranscriptSegment


def _session(tmp_path: Path) -> RecordingSession:
    audio_path = tmp_path / "source.wav"
    audio_path.write_bytes(b"not real wav")
    return RecordingSession(
        sources=[
            RecordedAudioSource(
                path=audio_path,
                sample_rate=16000,
                channels=1,
                frames=48000,
                label="microphone",
                device_name="Asterisk Nova",
            )
        ]
    )


def test_write_session_artifacts_creates_folder_text_markdown_and_metadata(
    tmp_path: Path,
) -> None:
    artifacts = write_session_artifacts(
        session=_session(tmp_path),
        target_dir=tmp_path / "transcriptions",
        transcript_text="Speaker 1: hola",
        completed_at=datetime(2026, 5, 22, 10, 30, 0),
        transcription_model="gpt-4o-transcribe-diarize",
        cleanup_model="gpt-4.1-mini",
        cleanup_enabled=False,
        alt_model_used=False,
        live_pipeline_attempted=True,
        live_pipeline_used=True,
        fallback_used=False,
        chunks=[
            ChunkMetadata(
                index=1,
                mode="live",
                start_seconds=0.0,
                end_seconds=3.0,
                duration_seconds=3.0,
                source_count=1,
                transcription_started_at=datetime(2026, 5, 22, 10, 29, 0),
                transcription_finished_at=datetime(2026, 5, 22, 10, 29, 5),
                status="completed",
            )
        ],
    )

    assert artifacts.session_dir.name == "20260522_103000"
    assert artifacts.transcript_path.name == TRANSCRIPT_FILE
    assert artifacts.markdown_path.name == MARKDOWN_FILE
    assert artifacts.metadata_path.name == METADATA_FILE
    assert artifacts.chunks_path.name == CHUNKS_FILE
    assert artifacts.audio_path is None
    assert artifacts.transcript_path.read_text(encoding=TRANSCRIPT_ENCODING) == "Speaker 1: hola"

    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert metadata["schema_version"] == 1
    assert metadata["session_id"] == "20260522_103000"
    assert metadata["duration_seconds"] == 3.0
    assert metadata["sources"] == [
        {
            "label": "microphone",
            "device_name": "Asterisk Nova",
            "sample_rate": 16000,
            "channels": 1,
            "frames": 48000,
            "duration_seconds": 3.0,
        }
    ]
    assert metadata["output_files"] == [TRANSCRIPT_FILE, MARKDOWN_FILE, METADATA_FILE, CHUNKS_FILE]
    assert "source.wav" not in artifacts.metadata_path.read_text(encoding="utf-8")

    chunks = json.loads(artifacts.chunks_path.read_text(encoding="utf-8"))
    assert chunks["schema_version"] == 1
    assert chunks["chunks"][0]["mode"] == "live"
    assert chunks["chunks"][0]["source_count"] == 1
    assert chunks["chunks"][0]["status"] == "completed"
    assert "Speaker 1: hola" not in artifacts.chunks_path.read_text(encoding="utf-8")
    assert "gpt-4o-transcribe-diarize" not in artifacts.chunks_path.read_text(encoding="utf-8")

    markdown = artifacts.markdown_path.read_text(encoding="utf-8")
    assert "# Recording 2026-05-22 10:30" in markdown
    assert "- Session ID: `20260522_103000`" in markdown
    assert "- microphone (Asterisk Nova): 1 channel(s), 16000 Hz, 3s" in markdown
    assert "Speaker 1: hola" in markdown


def test_write_session_artifacts_uses_unique_session_folder(tmp_path: Path) -> None:
    target_dir = tmp_path / "transcriptions"
    (target_dir / "20260522_103000").mkdir(parents=True)

    artifacts = write_session_artifacts(
        session=_session(tmp_path),
        target_dir=target_dir,
        transcript_text="hola",
        completed_at=datetime(2026, 5, 22, 10, 30, 0),
        transcription_model="gpt-4o-transcribe-diarize",
        cleanup_model="gpt-4.1-mini",
        cleanup_enabled=False,
        alt_model_used=False,
        live_pipeline_attempted=False,
        live_pipeline_used=False,
        fallback_used=False,
    )

    assert artifacts.session_dir.name == "20260522_103000_2"
    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert metadata["session_id"] == "20260522_103000_2"
    chunks = json.loads(artifacts.chunks_path.read_text(encoding="utf-8"))
    assert chunks == {"schema_version": 1, "chunks": []}


def test_write_session_artifacts_derives_audio_path_from_recoverable_audio(
    tmp_path: Path,
) -> None:
    artifacts = write_session_artifacts(
        session=_session(tmp_path),
        target_dir=tmp_path / "transcriptions",
        completed_at=datetime(2026, 5, 22, 10, 30, 0),
        transcription_model="gpt-4o-transcribe-diarize",
        cleanup_model="gpt-4.1-mini",
        cleanup_enabled=False,
        alt_model_used=False,
        live_pipeline_attempted=False,
        live_pipeline_used=False,
        fallback_used=False,
        recoverable_audio=AUDIO_FILE,
    )

    assert artifacts.audio_path == artifacts.session_dir / AUDIO_FILE
    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert metadata["recoverable_audio"] == AUDIO_FILE
    assert AUDIO_FILE in metadata["output_files"]


def _write_segment_artifacts(
    tmp_path: Path,
    *,
    segments: list[TranscriptSegment] | None,
    session_dir: Path | None = None,
    status: Literal["pending", "completed", "failed", "cancelled"] = "completed",
) -> SessionArtifactPaths:
    return write_session_artifacts(
        session=_session(tmp_path),
        target_dir=tmp_path / "transcriptions",
        completed_at=datetime(2026, 5, 22, 10, 30, 0),
        transcription_model="gpt-4o-transcribe-diarize",
        cleanup_model="gpt-4.1-mini",
        cleanup_enabled=False,
        alt_model_used=False,
        live_pipeline_attempted=False,
        live_pipeline_used=False,
        fallback_used=False,
        segments=segments,
        session_dir=session_dir,
        status=status,
    )


@pytest.mark.parametrize("status", ["failed", "cancelled", "pending", "completed"])
@pytest.mark.parametrize("rewrite", [False, True], ids=["fresh", "rewrite"])
def test_session_without_segment_evidence_has_no_segment_artifact(
    tmp_path: Path,
    status: Literal["pending", "completed", "failed", "cancelled"],
    rewrite: bool,
) -> None:
    session_dir = None
    if rewrite:
        previous = _write_segment_artifacts(
            tmp_path, segments=[TranscriptSegment("old provider evidence", 0.0, 1.0, "A")]
        )
        assert previous.segments_path is not None
        previous_document = json.loads(previous.segments_path.read_text(encoding="utf-8"))
        assert previous_document["segments"][0]["text"] == "old provider evidence"
        assert previous.metadata.status == "completed"
        session_dir = previous.session_dir

    artifacts = _write_segment_artifacts(
        tmp_path, segments=None, session_dir=session_dir, status=status
    )

    if rewrite:
        assert artifacts.session_dir == session_dir
    assert artifacts.segments_path is None
    assert SEGMENTS_FILE not in artifacts.metadata.output_files
    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert metadata["status"] == status
    assert SEGMENTS_FILE not in metadata["output_files"]
    assert not (artifacts.session_dir / SEGMENTS_FILE).exists()


@pytest.mark.parametrize("rewrite", [False, True], ids=["fresh", "rewrite"])
def test_explicit_empty_segment_evidence_writes_current_document(
    tmp_path: Path, rewrite: bool
) -> None:
    session_dir = None
    if rewrite:
        previous = _write_segment_artifacts(
            tmp_path, segments=[TranscriptSegment("old provider evidence", 0.0, 1.0, "A")]
        )
        assert previous.segments_path is not None
        assert previous.segments_path.exists()
        session_dir = previous.session_dir

    artifacts = _write_segment_artifacts(tmp_path, segments=[], session_dir=session_dir)

    if rewrite:
        assert artifacts.session_dir == session_dir
    assert artifacts.segments_path == artifacts.session_dir / SEGMENTS_FILE
    assert SEGMENTS_FILE in artifacts.metadata.output_files
    metadata = json.loads(artifacts.metadata_path.read_text(encoding="utf-8"))
    assert SEGMENTS_FILE in metadata["output_files"]
    document = json.loads(artifacts.segments_path.read_text(encoding="utf-8"))
    assert document["schema_version"] == 1
    assert document["semantics"] == "provider_evidence"
    assert document["segments"] == []


@pytest.mark.parametrize("segments", [None, [], [TranscriptSegment("new evidence")]])
def test_failed_metadata_publication_preserves_previous_segment_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, segments
) -> None:
    previous = _write_segment_artifacts(
        tmp_path, segments=[TranscriptSegment("old provider evidence", 0.0, 1.0, "A")]
    )
    assert previous.segments_path is not None
    previous_metadata = previous.metadata_path.read_bytes()
    previous_segments = previous.segments_path.read_bytes()
    assert SEGMENTS_FILE in json.loads(previous_metadata)["output_files"]
    write_text = Path.write_text

    def fail_metadata_publication(path: Path, *args, **kwargs):
        if path.name == METADATA_FILE or path.name.startswith(f".{METADATA_FILE}."):
            raise OSError("synthetic metadata publication failure")
        return write_text(path, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_metadata_publication)

    with pytest.raises(OSError, match="synthetic metadata publication failure"):
        _write_segment_artifacts(
            tmp_path, segments=segments, session_dir=previous.session_dir, status="failed"
        )

    assert previous.metadata_path.read_bytes() == previous_metadata
    assert previous.segments_path.read_bytes() == previous_segments


@pytest.mark.parametrize("segments", [[], [TranscriptSegment("new evidence")]])
@pytest.mark.parametrize("failed_file", [SEGMENTS_FILE, METADATA_FILE])
@pytest.mark.parametrize("partial", [False, True], ids=["before-write", "partial-write"])
@pytest.mark.parametrize("rewrite", [False, True], ids=["fresh", "rewrite"])
def test_pair_staging_failure_preserves_previous_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, segments, failed_file, partial, rewrite
) -> None:
    session_dir = tmp_path / "transcriptions" / "session"
    session_dir.mkdir(parents=True)
    metadata_path = session_dir / METADATA_FILE
    segments_path = session_dir / SEGMENTS_FILE
    if rewrite:
        _write_segment_artifacts(
            tmp_path, segments=[TranscriptSegment("old evidence")], session_dir=session_dir
        )
    old_metadata = metadata_path.read_bytes() if rewrite else None
    old_segments = segments_path.read_bytes() if rewrite else None
    write_text = Path.write_text

    def fail_staging(path: Path, text, *args, **kwargs):
        if path.name == failed_file or path.name.startswith(f".{failed_file}."):
            if partial:
                write_text(path, text[:8], *args, **kwargs)
            raise OSError("synthetic staging failure")
        return write_text(path, text, *args, **kwargs)

    monkeypatch.setattr(Path, "write_text", fail_staging)
    with pytest.raises(OSError, match="synthetic staging failure"):
        _write_segment_artifacts(tmp_path, segments=segments, session_dir=session_dir)

    if rewrite:
        assert metadata_path.read_bytes() == old_metadata
        assert segments_path.read_bytes() == old_segments
        assert SEGMENTS_FILE in json.loads(old_metadata)["output_files"]
    else:
        assert not metadata_path.exists()
        assert not segments_path.exists()
    assert not list(session_dir.glob(".*.stage"))
    assert not list(session_dir.glob(".*.backup"))


@pytest.mark.parametrize("segments", [None, [], [TranscriptSegment("new evidence")]])
@pytest.mark.parametrize("rewrite", [False, True], ids=["fresh", "rewrite"])
def test_metadata_replace_failure_rolls_back_segment_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, segments, rewrite
) -> None:
    session_dir = tmp_path / "transcriptions" / "session"
    session_dir.mkdir(parents=True)
    metadata_path = session_dir / METADATA_FILE
    segments_path = session_dir / SEGMENTS_FILE
    if rewrite:
        _write_segment_artifacts(
            tmp_path, segments=[TranscriptSegment("old evidence")], session_dir=session_dir
        )
    old_metadata = metadata_path.read_bytes() if rewrite else None
    old_segments = segments_path.read_bytes() if rewrite else None
    replace = Path.replace
    reached_publication = []

    def fail_metadata_replace(path: Path, target):
        if Path(target) == metadata_path:
            reached_publication.append(True)
            if segments is None:
                assert not segments_path.exists()
            else:
                current = json.loads(segments_path.read_text(encoding="utf-8"))
                assert [item["text"] for item in current["segments"]] == [
                    segment.text for segment in segments
                ]
            # Cause a real filesystem replace failure after segment publication.
            path.unlink()
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_metadata_replace)
    with pytest.raises(OSError):
        _write_segment_artifacts(tmp_path, segments=segments, session_dir=session_dir)

    assert reached_publication == [True]
    if rewrite:
        assert metadata_path.read_bytes() == old_metadata
        assert segments_path.read_bytes() == old_segments
        assert SEGMENTS_FILE in json.loads(old_metadata)["output_files"]
    else:
        assert not metadata_path.exists()
        assert not segments_path.exists()
    assert not list(session_dir.glob(".*.stage"))
    assert not list(session_dir.glob(".*.backup"))


def test_failed_segment_rollback_retains_prior_evidence_backup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    previous = _write_segment_artifacts(tmp_path, segments=[TranscriptSegment("old evidence")])
    old_metadata = previous.metadata_path.read_bytes()
    old_segments = previous.segments_path.read_bytes()
    replace = Path.replace

    def fail_publication_and_rollback(path: Path, target):
        if Path(target) == previous.metadata_path:
            path.unlink()
        if path.suffix == ".backup":
            raise OSError("synthetic rollback failure")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_publication_and_rollback)
    with pytest.raises(OSError, match="synthetic rollback failure") as failure:
        _write_segment_artifacts(
            tmp_path, segments=[TranscriptSegment("new evidence")], session_dir=previous.session_dir
        )

    assert previous.metadata_path.read_bytes() == old_metadata
    backups = list(previous.session_dir.glob(f".{SEGMENTS_FILE}.*.backup"))
    assert len(backups) == 1
    assert backups[0].read_bytes() == old_segments
    assert str(backups[0]) in " ".join(failure.value.__notes__)
    assert not list(previous.session_dir.glob(".*.stage"))


@pytest.mark.parametrize("segments", [None, [], [TranscriptSegment("new evidence")]])
def test_successful_pair_rewrite_publishes_current_files_and_keeps_unowned_staging(
    tmp_path: Path, segments
) -> None:
    previous = _write_segment_artifacts(tmp_path, segments=[TranscriptSegment("old evidence")])
    unrelated = previous.session_dir / ".segments.json.unrelated.stage"
    unrelated.write_bytes(b"another operation owns this file")

    current = _write_segment_artifacts(
        tmp_path, segments=segments, session_dir=previous.session_dir, status="pending"
    )

    metadata = json.loads(current.metadata_path.read_text(encoding="utf-8"))
    assert metadata["status"] == "pending"
    if segments is None:
        assert not (current.session_dir / SEGMENTS_FILE).exists()
        assert SEGMENTS_FILE not in metadata["output_files"]
    else:
        assert SEGMENTS_FILE in metadata["output_files"]
        document = json.loads(current.segments_path.read_text(encoding="utf-8"))
        assert [item["text"] for item in document["segments"]] == [item.text for item in segments]
    assert unrelated.read_bytes() == b"another operation owns this file"
    assert list(current.session_dir.glob(".*.stage")) == [unrelated]
    assert not list(current.session_dir.glob(".*.backup"))


@pytest.mark.parametrize("failed_step", ["backup", "new-segments"])
def test_segment_replace_failure_preserves_previous_pair(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failed_step
) -> None:
    previous = _write_segment_artifacts(tmp_path, segments=[TranscriptSegment("old evidence")])
    old_metadata = previous.metadata_path.read_bytes()
    old_segments = previous.segments_path.read_bytes()
    replace = Path.replace

    def fail_segment_replace(path: Path, target):
        if failed_step == "backup" and path == previous.segments_path:
            raise OSError("synthetic segment publication failure")
        if (
            failed_step == "new-segments"
            and path.suffix == ".stage"
            and (Path(target) == previous.segments_path)
        ):
            raise OSError("synthetic segment publication failure")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_segment_replace)
    with pytest.raises(OSError, match="synthetic segment publication failure"):
        _write_segment_artifacts(
            tmp_path, segments=[TranscriptSegment("new evidence")], session_dir=previous.session_dir
        )

    assert previous.metadata_path.read_bytes() == old_metadata
    assert previous.segments_path.read_bytes() == old_segments
    assert not list(previous.session_dir.glob(".*.stage"))
    assert not list(previous.session_dir.glob(".*.backup"))


@pytest.mark.parametrize("link_kind", ["hardlink", "symlink"])
def test_linked_segment_entry_is_rejected_without_modifying_external_target(
    tmp_path: Path, link_kind
) -> None:
    previous = _write_segment_artifacts(tmp_path, segments=[TranscriptSegment("old evidence")])
    old_metadata = previous.metadata_path.read_bytes()
    old_segments = previous.segments_path.read_bytes()
    external = tmp_path / "external-evidence.json"
    previous.segments_path.replace(external)
    if link_kind == "hardlink":
        os.link(external, previous.segments_path)
    else:
        try:
            previous.segments_path.symlink_to(external)
        except OSError as error:
            pytest.skip(f"Host cannot create symlinks: {error}")
    with pytest.raises(ValueError, match="session artifact"):
        _write_segment_artifacts(
            tmp_path,
            segments=[TranscriptSegment("new evidence")],
            session_dir=previous.session_dir,
        )
    assert previous.metadata_path.read_bytes() == old_metadata
    assert previous.segments_path.read_bytes() == old_segments
    assert external.read_bytes() == old_segments
    if link_kind == "symlink":
        assert previous.segments_path.is_symlink()
    else:
        assert previous.segments_path.samefile(external)
