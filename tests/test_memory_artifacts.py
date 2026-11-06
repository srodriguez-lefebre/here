"""Read untrusted session manifests through the public ingestion boundary."""

from __future__ import annotations

import importlib
import json
import os
import sqlite3
import subprocess

import pytest
from memory_fixtures import (
    HOSTILE_ARTIFACT_PATHS,
    MEETING_ID,
    make_hostile_session,
    make_session,
    write_json,
)


def _service(root):
    return importlib.import_module("here.memory.service").MemoryService(root)


@pytest.mark.parametrize("hostile", [*HOSTILE_ARTIFACT_PATHS, "nested/../../outside.wav"])
def test_manifest_paths_are_rejected_before_external_reads_or_publication(
    tmp_path, monkeypatch, hostile
):
    service = _service(tmp_path / "data")
    source = make_hostile_session(tmp_path / "source", hostile)
    sentinel = tmp_path / "outside.wav"
    sentinel.write_bytes(b"synthetic external material must not be opened")
    metadata_before = source.metadata.read_bytes()
    original_open = type(sentinel).open

    def guarded_open(path, *args, **kwargs):
        if path.absolute() == sentinel.absolute():
            pytest.fail("An untrusted reference was dereferenced before validation")
        return original_open(path, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(type(sentinel), "open", guarded_open)
        result = service.ingest_session(source.directory)
    assert result.status == "rejected"
    assert result.warnings
    assert source.metadata.read_bytes() == metadata_before
    assert source.audio.is_file()
    assert sentinel.read_bytes() == b"synthetic external material must not be opened"
    with sqlite3.connect(tmp_path / "data" / "library.sqlite3") as connection:
        assert connection.execute("SELECT COUNT(*) FROM meetings").fetchone()[0] == 0


def test_segment_evidence_preserves_source_order_nullable_time_and_scoped_speakers(tmp_path):
    service = _service(tmp_path / "data")
    source = make_session(
        tmp_path / "source",
        meeting_id=MEETING_ID,
        segments=[
            {"text": "First untimed", "speaker": "speaker_0", "speaker_scope": "request-a"},
            {
                "text": "Later timed",
                "start": 1.25,
                "end": 1.75,
                "speaker": "speaker_0",
                "speaker_scope": "request-b",
            },
            {"text": "Contradictory", "start": 2.0, "end": 1.0, "speaker": "speaker_1"},
            {"text": "Non-finite", "start": float("nan"), "end": float("inf")},
        ],
    )
    result = service.ingest_session(source.directory)
    assert result.status == "inserted"
    with sqlite3.connect(tmp_path / "data" / "library.sqlite3") as connection:
        rows = connection.execute(
            "SELECT ordinal, text, start_ms, end_ms, timing_source, speaker_key "
            "FROM segments ORDER BY ordinal"
        ).fetchall()
        assert [row[:5] for row in rows] == [
            (0, "First untimed", None, None, "absent"),
            (1, "Later timed", 1250, 1750, "provider"),
            (2, "Contradictory", None, None, "absent"),
            (3, "Non-finite", None, None, "absent"),
        ]
        assert rows[0][5] and rows[1][5] and rows[0][5] != rows[1][5]
        assert rows[2][5]
        assert rows[3][5] is None
        assert connection.execute("SELECT COUNT(*) FROM participants").fetchone()[0] == 0


def test_legacy_text_without_segments_does_not_invent_timestamps_or_speakers(tmp_path):
    service = _service(tmp_path / "data")
    source = make_session(
        tmp_path / "source", transcript_text="At 00:12 we agreed.\n\nNo timestamp."
    )
    (source.directory / "segments.json").unlink()
    metadata = json.loads(source.metadata.read_text(encoding="utf-8"))
    metadata["output_files"].remove("segments.json")
    write_json(source.metadata, metadata)
    assert service.ingest_session(source.directory).status == "inserted"
    with sqlite3.connect(tmp_path / "data" / "library.sqlite3") as connection:
        assert connection.execute(
            "SELECT ordinal, text, start_ms, end_ms, timing_source, speaker_key "
            "FROM segments ORDER BY ordinal"
        ).fetchall() == [
            (0, "At 00:12 we agreed.", None, None, "absent", None),
            (1, "No timestamp.", None, None, "absent", None),
        ]


def test_hostile_pending_paths_do_not_touch_external_files_or_block_valid_sibling(tmp_path):
    root = tmp_path / "data"
    service = _service(root)
    source = make_session(tmp_path / "source", meeting_id=MEETING_ID)
    assert service.ingest_session(source.directory).status == "inserted"
    external = tmp_path / "outside"
    external.mkdir()
    sentinel = external / "keep.txt"
    sentinel.write_text("synthetic external data", encoding="utf-8")
    with sqlite3.connect(root / "library.sqlite3") as connection:
        connection.execute("UPDATE pending_operations SET phase='prepared'")
        for identity, path in (("traversal", "../outside"), ("absolute", str(external))):
            connection.execute(
                "INSERT INTO pending_operations "
                "(id, kind, phase, meeting_id, staging_path, destination_path, payload_json) "
                "VALUES (?, 'publish_revision', 'prepared', ?, ?, ?, '{}')",
                (identity, MEETING_ID, path, path),
            )
    report = service.reconcile()
    assert report.warnings
    assert sentinel.read_text(encoding="utf-8") == "synthetic external data"
    assert service.ingest_session(source.directory).status == "unchanged"
    with sqlite3.connect(root / "library.sqlite3") as connection:
        assert (
            connection.execute(
                "SELECT COUNT(*) FROM pending_operations WHERE phase='prepared' "
                "AND id NOT IN ('traversal','absolute')"
            ).fetchone()[0]
            == 0
        )
        assert connection.execute("SELECT COUNT(*) FROM meetings").fetchone()[0] == 1
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []


@pytest.mark.skipif(os.name != "nt", reason="Windows junction contract")
def test_nested_junction_in_managed_destination_is_rejected_and_valid_sibling_survives(tmp_path):
    root = tmp_path / "data"
    service = _service(root)
    external = tmp_path / "external"
    external.mkdir()
    sentinel = external / "keep.txt"
    sentinel.write_text("synthetic junction target", encoding="utf-8")
    meetings = root / "meetings"
    meetings.mkdir(exist_ok=True)
    junction = meetings / MEETING_ID
    # cmd mklink is a creation builtin; all paths are literal argv inside temporary roots.
    process = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(external)],
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert process.returncode == 0, process.stdout + process.stderr
    try:
        source = make_session(tmp_path / "hostile", meeting_id=MEETING_ID)
        assert service.ingest_session(source.directory).status == "rejected"
        valid = make_session(tmp_path / "valid")
        assert service.ingest_session(valid.directory).status == "inserted"
        assert sentinel.read_text(encoding="utf-8") == "synthetic junction target"
        assert set(path.name for path in external.iterdir()) == {"keep.txt"}
    finally:
        # Remove only the verified reparse point, never traverse the external directory.
        assert junction.absolute().is_relative_to(root.absolute())
        junction.rmdir()
