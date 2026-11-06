"""Opt-in real 1k/100k query cost, including generation and the installed guard.

Run only with an exclusive host lease and HERE_RUN_SEARCH_BENCHMARK=1. This file
does not certify the Qt heartbeat (task 2) or deferred B5 suppression integration.
"""

from __future__ import annotations

import os
import platform
import statistics
from time import perf_counter

import pytest
from test_memory_search import _api, _open_catalog, _service

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        os.environ.get("HERE_RUN_SEARCH_BENCHMARK") != "1",
        reason="Opt-in 1,000/100,000 benchmark requires exclusive host runtime lease",
    ),
]


def test_warm_search_p95_with_actual_corpus_generation_and_visibility_guard(
    tmp_path, record_property
):
    from search_fixtures import COHORT_TOKEN, scale_corpus

    _, models = _api()
    corpus = scale_corpus()
    assert (corpus.meeting_count, corpus.segment_count) == (1000, 100000)
    library, _ = _open_catalog(tmp_path, corpus)
    try:
        assert library.execute("SELECT count(*) FROM meetings").fetchone()[0] == 1000
        assert library.execute("SELECT count(*) FROM segments").fetchone()[0] == 100000
        service = _service(tmp_path)
        query = models.SearchQuery(text=COHORT_TOKEN, include_partial=True, limit=10)
        # Public search is timed in full: no patched digest, fake guard, cached
        # generation, direct SQL fast path or precomputed result page.
        for _ in range(5):
            assert len(service.search(query).items) == 10
        timings = []
        for _ in range(40):
            started = perf_counter()
            page = service.search(query)
            timings.append((perf_counter() - started) * 1000)
            assert len(page.items) == 10
        p95 = statistics.quantiles(timings, n=100, method="inclusive")[94]
        record_property("machine", platform.platform())
        record_property("processor", platform.processor())
        record_property("python", platform.python_version())
        record_property("meetings", 1000)
        record_property("segments", 100000)
        record_property("warm_samples", 40)
        record_property("p95_ms", p95)
        assert p95 < 500, f"Warm public search p95 {p95:.2f} ms exceeds 500 ms"
    finally:
        library.close()
