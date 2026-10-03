from __future__ import annotations

import json
import os
import wave
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
from here.transcription.segments import TranscriptSegment, scoped_segments, shift_segments


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


def test_direct_typed_reversed_evidence_is_persisted_without_contradictory_times(tmp_path):
    artifacts = _write_segment_artifacts(
        tmp_path,
        segments=[TranscriptSegment("original evidence", 2.0, 1.0, "A", 3, "chunk:3")],
    )
    document = json.loads(artifacts.segments_path.read_text(encoding="utf-8"))
    assert document["segments"] == [
        {
            "text": "original evidence",
            "start": None,
            "end": None,
            "speaker": "A",
            "chunk_index": 3,
            "speaker_scope": "chunk:3",
        }
    ]


@pytest.mark.parametrize(
    "invalid",
    [
        pytest.param(-1, id="negative"),
        pytest.param(float("nan"), id="nan"),
        pytest.param(float("inf"), id="infinity"),
        pytest.param(float("-inf"), id="negative-infinity"),
        pytest.param(True, id="true"),
        pytest.param(False, id="false"),
        pytest.param(10**400, id="huge-positive"),
        pytest.param(-(10**400), id="huge-negative"),
        pytest.param("invalid", id="invalid-string"),
        pytest.param("-1", id="negative-string"),
        pytest.param("NaN", id="nan-string"),
        pytest.param("inf", id="infinity-string"),
        pytest.param("1e400", id="overflow-string"),
        pytest.param("", id="empty-string"),
        pytest.param("  ", id="whitespace-string"),
    ],
)
@pytest.mark.parametrize("bounds", ["start", "end", "both"])
def test_direct_invalid_bounds_persist_as_missing_and_survive_shift_and_scope(
    tmp_path, invalid, bounds
):
    segment = TranscriptSegment(
        "original evidence",
        invalid if bounds in {"start", "both"} else 0.0,
        invalid if bounds in {"end", "both"} else 0.0,
        "A",
        4,
        "chunk:4",
    )
    artifacts = _write_segment_artifacts(tmp_path, segments=[segment])
    document = json.loads(artifacts.segments_path.read_text(encoding="utf-8"))
    expected_start = None if bounds in {"start", "both"} else 0.0
    expected_end = None if bounds in {"end", "both"} else 0.0
    assert document["segments"] == [
        {
            "text": "original evidence",
            "start": expected_start,
            "end": expected_end,
            "speaker": "A",
            "chunk_index": 4,
            "speaker_scope": "chunk:4",
        }
    ]
    expected_shifted = (
        None if bounds in {"start", "both"} else 10.0,
        None if bounds in {"end", "both"} else 10.0,
    )
    shifted = shift_segments([segment], 10)[0]
    scoped = scoped_segments([segment], chunk_index=7, offset_seconds=10)[0]
    for updated in (shifted, scoped):
        assert (updated.start, updated.end) == expected_shifted
        assert updated.text == "original evidence"
        assert updated.speaker == "A"
    assert shifted.speaker_scope == "chunk:4"
    assert scoped.chunk_index == 7
    assert scoped.speaker_scope == "chunk:7"


@pytest.mark.parametrize(
    "start,end,expected",
    [
        (" 0 ", "0", (0.0, 0.0)),
        ("2", "10", (2.0, 10.0)),
        ("10", "2", (None, None)),
        ("invalid", "1.5", (None, 1.5)),
        ("1.5", "invalid", (1.5, None)),
    ],
)
def test_direct_numeric_strings_use_numeric_validation_before_persistence(
    tmp_path, start, end, expected
):
    artifacts = _write_segment_artifacts(
        tmp_path, segments=[TranscriptSegment("evidence", start, end, "A")]
    )
    segment = json.loads(artifacts.segments_path.read_text(encoding="utf-8"))["segments"][0]
    assert (segment["start"], segment["end"]) == expected
    assert segment["text"] == "evidence"
    assert segment["speaker"] == "A"


@pytest.fixture
def prior_failed_session(tmp_path):
    """Author an existing recovery session independently of the writer under test."""
    directory = tmp_path / "sessions" / "saved"
    directory.mkdir(parents=True)
    audio = directory / AUDIO_FILE
    with wave.open(str(audio), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(8000)
        wav.writeframes(b"\x01\x00" * 80)
    meeting_id = "4800458c-c1f5-44e5-83d4-31823519a135"
    documents = {
        METADATA_FILE: {
            "schema_version": 1,
            "session_id": "saved",
            "meeting_id": meeting_id,
            "started_at": "2026-10-03T10:00:00",
            "completed_at": "2026-10-03T10:00:01",
            "duration_seconds": 0.01,
            "status": "failed",
            "failure_stage": "offline_transcription",
            "recoverable_audio": AUDIO_FILE,
            "sources": [
                {
                    "label": "mic",
                    "sample_rate": 8000,
                    "channels": 1,
                    "frames": 80,
                    "duration_seconds": 0.01,
                }
            ],
            "transcription_model": "authored",
            "cleanup_model": "authored",
            "cleanup_enabled": False,
            "alt_model_used": False,
            "live_pipeline_attempted": False,
            "live_pipeline_used": False,
            "fallback_used": False,
            "output_files": [
                METADATA_FILE,
                AUDIO_FILE,
                CHUNKS_FILE,
                SEGMENTS_FILE,
                "errors.json",
                "events.json",
            ],
        },
        CHUNKS_FILE: {"schema_version": 1, "chunks": []},
        SEGMENTS_FILE: {"schema_version": 1, "segments": [{"text": "old evidence"}]},
        "errors.json": {
            "schema_version": 1,
            "errors": [
                {
                    "stage": "offline_transcription",
                    "type": "OSError",
                    "message": "authored failure",
                    "retryable": True,
                    "occurred_at": "2026-10-03T10:00:01",
                }
            ],
        },
        "events.json": {
            "schema_version": 1,
            "events": [
                {
                    "kind": "stopped",
                    "occurred_at": "2026-10-03T10:00:01",
                    "recorded_duration_seconds": 0.01,
                }
            ],
        },
    }
    for name, document in documents.items():
        (directory / name).write_text(json.dumps(document, indent=2), encoding="utf-8")
    session = RecordingSession(
        [RecordedAudioSource(audio, 8000, 1, 80, "mic")], meeting_id=meeting_id
    )
    return directory, session, {path.name: path.read_bytes() for path in directory.iterdir()}


def _complete_failed_session(directory, session, *, segments):
    return write_session_artifacts(
        session=session,
        session_dir=directory,
        target_dir=directory.parent,
        completed_at=datetime(2026, 10, 3, 10, 0, 2),
        transcription_model="authored",
        cleanup_model="authored",
        cleanup_enabled=False,
        alt_model_used=False,
        live_pipeline_attempted=False,
        live_pipeline_used=False,
        fallback_used=False,
        transcript_text="current transcript",
        recoverable_audio=AUDIO_FILE,
        segments=segments,
    )


@pytest.mark.parametrize("failed_file", ["errors.json", "events.json"])
def test_obsolete_move_failure_preserves_prior_recovery(
    prior_failed_session, monkeypatch, failed_file
):
    from here.application.recovery import RecoveryService

    directory, session, before = prior_failed_session
    replace = Path.replace

    def fail_obsolete_move(path, target):
        if path == directory / failed_file:
            if failed_file == "events.json":
                assert not (directory / "errors.json").exists()
            raise PermissionError("synthetic obsolete move failure")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_obsolete_move)
    with pytest.raises(PermissionError, match="synthetic obsolete move failure"):
        _complete_failed_session(directory, session, segments=[TranscriptSegment("new evidence")])

    for name, content in before.items():
        assert (directory / name).read_bytes() == content
    (candidate,) = RecoveryService(directory.parent).discover()
    assert candidate.status == "failed"
    assert candidate.can_retry
    assert candidate.error_summary == "authored failure"
    assert not list(directory.glob(".*.backup"))
    assert not list(directory.glob(".*.stage"))


@pytest.mark.parametrize("segments", [None, [], [TranscriptSegment("new evidence")]])
def test_metadata_replace_failure_restores_obsolete_entries(
    prior_failed_session, monkeypatch, segments
):
    directory, session, before = prior_failed_session
    replace = Path.replace

    def fail_metadata_replace(path, target):
        if Path(target) == directory / METADATA_FILE:
            assert not (directory / "errors.json").exists()
            assert not (directory / "events.json").exists()
            assert (directory / METADATA_FILE).read_bytes() == before[METADATA_FILE]
            raise PermissionError("synthetic metadata replace failure")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_metadata_replace)
    with pytest.raises(PermissionError, match="synthetic metadata replace failure"):
        _complete_failed_session(directory, session, segments=segments)

    for name, content in before.items():
        assert (directory / name).read_bytes() == content
    assert not list(directory.glob(".*.backup"))
    assert not list(directory.glob(".*.stage"))


def test_obsolete_rollback_failure_retains_backup_and_restores_other_entries(
    prior_failed_session, monkeypatch
):
    directory, session, before = prior_failed_session
    replace = Path.replace

    def fail_publication_and_one_rollback(path, target):
        if Path(target) == directory / METADATA_FILE:
            raise OSError("synthetic metadata replace failure")
        if path.suffix == ".backup" and Path(target) == directory / "errors.json":
            raise OSError("synthetic obsolete rollback failure")
        return replace(path, target)

    monkeypatch.setattr(Path, "replace", fail_publication_and_one_rollback)
    with pytest.raises(OSError, match="synthetic obsolete rollback failure") as failure:
        _complete_failed_session(directory, session, segments=[TranscriptSegment("new evidence")])

    for name in (METADATA_FILE, SEGMENTS_FILE, "events.json", AUDIO_FILE):
        assert (directory / name).read_bytes() == before[name]
    (backup,) = directory.glob(".*.backup")
    assert backup.read_bytes() == before["errors.json"]
    assert str(backup) in " ".join(failure.value.__notes__)
    assert not list(directory.glob(".*.stage"))


@pytest.mark.parametrize(
    "failed_file,suffix",
    [
        ("errors.json", ".backup"),
        ("events.json", ".backup"),
        (SEGMENTS_FILE, ".backup"),
        (SEGMENTS_FILE, ".stage"),
        (METADATA_FILE, ".stage"),
    ],
)
def test_postcommit_cleanup_failure_returns_completed_session(
    prior_failed_session, monkeypatch, failed_file, suffix
):
    from here.application.recovery import RecoveryService

    directory, session, before = prior_failed_session
    unknown = directory / ".errors.json.unowned.backup"
    unknown.write_bytes(b"unrelated operation")
    unlink = Path.unlink

    def fail_owned_cleanup(path, *args, **kwargs):
        if path != unknown and (
            path.name == failed_file
            or (path.name.startswith(f".{failed_file}.") and path.suffix == suffix)
        ):
            metadata = json.loads((directory / METADATA_FILE).read_bytes())
            if metadata["status"] == "completed":
                raise PermissionError("synthetic owned cleanup failure")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_owned_cleanup)
    artifacts = _complete_failed_session(
        directory, session, segments=[TranscriptSegment("new evidence")]
    )

    assert artifacts.metadata.status == "completed"
    metadata = json.loads(artifacts.metadata_path.read_bytes())
    assert metadata["status"] == "completed"
    assert metadata["meeting_id"] == "4800458c-c1f5-44e5-83d4-31823519a135"
    assert "errors.json" not in metadata["output_files"]
    assert "events.json" not in metadata["output_files"]
    assert not artifacts.errors_path.exists()
    assert not artifacts.events_path.exists()
    assert artifacts.transcript_path.read_text(encoding=TRANSCRIPT_ENCODING) == "current transcript"
    assert json.loads(artifacts.segments_path.read_bytes())["segments"][0]["text"] == "new evidence"
    assert artifacts.audio_path.read_bytes() == before[AUDIO_FILE]
    assert RecoveryService(directory.parent).discover() == []
    assert unknown.read_bytes() == b"unrelated operation"
    retained = [path for path in directory.glob(".*.backup") if path != unknown]
    if suffix == ".backup":
        (backup,) = retained
        assert backup.read_bytes() == before[failed_file]
    else:
        assert retained == []
