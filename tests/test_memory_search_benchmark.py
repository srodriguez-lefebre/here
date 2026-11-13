"""Opt-in real 1k/100k query cost, including generation and the installed guard.

Run only with an exclusive host lease and HERE_RUN_SEARCH_BENCHMARK=1. This file
measures the nonempty real B guard; Qt heartbeat remains a separate task-2 gate.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import statistics
import sys
from pathlib import Path
from time import perf_counter

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("HERE_RUN_SEARCH_BENCHMARK") != "1",
        reason="Opt-in 1,000/100,000 benchmark requires exclusive host runtime lease",
    ),
]


def test_warm_search_p95_with_actual_corpus_generation_and_visibility_guard(
    tmp_path, monkeypatch, record_property
):
    from search_performance_fixtures import build_guarded_scale_catalog
    from test_memory_delete import _view

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    # Builder validates physical counts, actual B snapshot/alias and ALL44 public
    # results before warmups. It never removes suppressed SQL/FTS documents.
    bundle = build_guarded_scale_catalog(tmp_path / "guarded-scale-library")
    record_property("machine", platform.platform())
    record_property("machine_architecture", platform.machine())
    record_property("processor", platform.processor())
    record_property("logical_cpus", os.cpu_count())
    record_property("python", platform.python_version())
    record_property("python_full", sys.version)
    record_property("meetings", bundle.physical_meetings)
    record_property("segments", bundle.physical_segments)
    record_property("persistent_fts_documents", 98901)
    record_property("registry_records", len(bundle.registry_record_ids))
    record_property("pending_records", 110)
    record_property("complete_records", 110)
    record_property("resolved_suppressed_ids", len(bundle.suppressed_meeting_ids))
    record_property("alias_meeting_id", bundle.alias_meeting_id)
    record_property("alias_origin", bundle.alias_origin)
    record_property("alias_fingerprint", bundle.alias_fingerprint)
    record_property("suppression_fingerprint", _view(bundle.data_dir).fingerprint)
    record_property("expected_total_hits", len(bundle.expected_segment_ids))
    record_property("query", bundle.query.text)
    record_property("include_partial", bundle.query.include_partial)
    record_property("limit", bundle.query.limit)
    record_property("setup_and_validation_seconds", bundle.setup_seconds)
    record_property("warmup_samples", 5)
    record_property("required_measured_samples", 40)
    record_property("budget_ms", 500)
    record_property("percentile_method", "statistics.quantiles(n=100, method='inclusive')[94]")
    source_files = {
        "search": Path(sys.modules["here.memory.search"].__file__),
        "service": Path(sys.modules["here.memory.service"].__file__),
        "models": Path(sys.modules["here.memory.models"].__file__),
        "corpus": Path(sys.modules["search_fixtures"].__file__),
        "performance_helper": Path(sys.modules["search_performance_fixtures"].__file__),
        "benchmark": Path(__file__),
        "config": Path(__file__).resolve().parents[1] / "pyproject.toml",
    }
    for name, path in source_files.items():
        record_property(name + "_raw_sha256", hashlib.sha256(path.read_bytes()).hexdigest())
    warmups = []
    timings = []
    try:
        # Every interval includes the ENTIRE public call: real lock, validation,
        # alias resolution, connection membership, full generation, fetch/close.
        # Result verification is outside the interval, with no patched fast path.
        for _ in range(5):
            started = perf_counter()
            try:
                page = bundle.search.search(bundle.query)
            finally:
                warmups.append((perf_counter() - started) * 1000)
            assert tuple(hit.segment_id for hit in page.items) == bundle.expected_first_segment_ids
            assert page.degraded is False
        for _ in range(40):
            started = perf_counter()
            try:
                page = bundle.search.search(bundle.query)
            finally:
                timings.append((perf_counter() - started) * 1000)
            assert tuple(hit.segment_id for hit in page.items) == bundle.expected_first_segment_ids
            assert page.degraded is False
    finally:
        # Preserve all40 raw values and computed p95 BEFORE any budget assertion.
        # A request/correctness error instead retains truthful partial samples.
        record_property("warmup_times_ms", json.dumps(warmups))
        record_property("raw_samples_ms", json.dumps(timings))
        record_property("actual_measured_samples", len(timings))
        p95 = (
            statistics.quantiles(timings, n=100, method="inclusive")[94]
            if len(timings) == 40
            else None
        )
        record_property("p95_ms", p95)
    assert len(warmups) == 5 and len(timings) == 40
    assert p95 is not None and p95 < 500, f"Warm public search p95 {p95} ms exceeds 500 ms"
