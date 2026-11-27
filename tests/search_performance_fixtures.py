"""Shared opt-in query/UI performance setup; one immutable corpus generator.

No Qt, provider, media, native ownership or runtime work occurs at module import.
The builder performs real TEMP-root setup and validation OUTSIDE measured windows.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from here.memory.models import SearchQuery
    from here.memory.search import SearchService
    from here.memory.service import MemoryService


@dataclass(frozen=True, slots=True)
class GuardedScaleCatalog:
    data_dir: Path
    memory: MemoryService
    search: SearchService
    query: SearchQuery
    expected_segment_ids: tuple[str, ...]
    suppressed_meeting_ids: tuple[str, ...]
    registry_record_ids: tuple[str, ...]
    alias_meeting_id: str
    alias_origin: str
    alias_fingerprint: str
    setup_seconds: float
    physical_meetings: int = 1000
    physical_segments: int = 100000

    @property
    def expected_first_segment_ids(self) -> tuple[str, ...]:
        return self.expected_segment_ids[: self.query.limit]


def build_guarded_scale_catalog(data_dir: Path) -> GuardedScaleCatalog:
    """Build the same 1k/100k corpus with 220 records and 221 resolved IDs.

    Suppression leaves every SQL/FTS document intact, so actual rank statistics,
    registry validation, alias resolution and fresh-connection membership all cost
    real work. Both query and UI consumers retain the SAME memory instance here.
    """
    from here.memory.database import open_library
    from here.memory.models import IngestSource, SearchQuery
    from here.memory.search import SearchService
    from here.memory.service import MemoryService
    from search_fixtures import COHORT_TOKEN, scale_corpus
    from test_memory_delete import _record, _view
    from test_memory_search import _identity, _open_catalog

    started = perf_counter()
    data_dir = Path(data_dir).absolute()
    corpus = scale_corpus()  # The original immutable generator, never copied.
    assert (corpus.meeting_count, corpus.segment_count) == (1000, 100000)
    registry_indices = (*range(5), *range(100, 315))
    registry_ids = tuple(corpus.meeting_id(index) for index in registry_indices)
    alias_id = corpus.meeting_id(5)
    alias_origin = os.path.normcase(os.path.normpath(str(data_dir.parent / "offline-alias")))
    alias_fingerprint = "d" * 64
    library, repo = _open_catalog(data_dir, corpus)
    try:
        with repo.transaction():
            repo.insert_ingest_source(
                IngestSource(
                    id=_identity(9400),
                    origin_path=alias_origin,
                    origin_fingerprint=alias_fingerprint,
                    meeting_id=alias_id,
                    revision_id=corpus.revision_id(5),
                )
            )
        pair = {"origin_path": alias_origin, "origin_fingerprint": alias_fingerprint}
        for ordinal, identity in enumerate(registry_ids):
            _record(
                data_dir,
                identity=identity,
                phase="pending" if ordinal % 2 else "complete",
                origins=(pair,) if identity == corpus.meeting_id(100) else (),
            )
        suppressed = tuple(sorted((*registry_ids, alias_id)))
        assert _view(data_dir).suppressed_meeting_ids == suppressed
    finally:
        library.close()
    memory = MemoryService(data_dir)
    service = SearchService(lambda: open_library(data_dir), reindexer=memory.reindex)
    query = SearchQuery(text=COHORT_TOKEN, include_partial=True, limit=10)
    # Cohort rows6..49 have equal BM25 inputs; fixture dates tie in groups of4.
    # Expected order is derived from that authored fixture contract, not search.
    expected_indices = sorted(range(6, 50), key=lambda index: (-(index // 4), index))
    bundle = GuardedScaleCatalog(
        data_dir=data_dir,
        memory=memory,
        search=service,
        query=query,
        expected_segment_ids=tuple(corpus.segment_id(index, 0) for index in expected_indices),
        suppressed_meeting_ids=suppressed,
        registry_record_ids=registry_ids,
        alias_meeting_id=alias_id,
        alias_origin=alias_origin,
        alias_fingerprint=alias_fingerprint,
        setup_seconds=0.0,
    )
    assert_guarded_scale_catalog(bundle)
    return replace(bundle, setup_seconds=perf_counter() - started)


def assert_guarded_scale_catalog(bundle: GuardedScaleCatalog) -> None:
    """Revalidate actual counts/suppression/public results, OUTSIDE measurement."""
    from here.memory.database import open_library
    from test_memory_delete import _view
    from test_memory_search import _search_all

    assert len(bundle.registry_record_ids) == 220
    assert len(bundle.suppressed_meeting_ids) == 221
    snapshot = _view(bundle.data_dir)
    assert snapshot.suppressed_meeting_ids == bundle.suppressed_meeting_ids
    assert bundle.alias_meeting_id in snapshot.suppressed_meeting_ids
    assert bundle.alias_meeting_id not in bundle.registry_record_ids
    assert snapshot.fingerprint
    assert len(tuple((bundle.data_dir / ".deletions").glob("*.json"))) == 220
    library = open_library(bundle.data_dir)
    try:
        assert library.execute("SELECT count(*) FROM meetings").fetchone()[0] == 1000
        assert library.execute("SELECT count(*) FROM segments").fetchone()[0] == 100000
        # 999 complete active revisions *99 active segments; the one partial
        # revision stays out of persistent FTS. Historical segments stay in SQL.
        assert library.execute("SELECT count(*) FROM segments_fts").fetchone()[0] == 98901
        for identity in (bundle.registry_record_ids[0], bundle.alias_meeting_id):
            assert (
                library.execute(
                    "SELECT count(*) FROM segments_fts WHERE meeting_id=?", (identity,)
                ).fetchone()[0]
                == 99
            )
            assert (
                library.execute("SELECT status FROM meetings WHERE id=?", (identity,)).fetchone()[0]
                == "completed"
            )
        alias = library.execute(
            "SELECT origin_path,origin_fingerprint FROM ingest_sources WHERE meeting_id=?",
            (bundle.alias_meeting_id,),
        ).fetchone()
        assert tuple(alias) == (bundle.alias_origin, bundle.alias_fingerprint)
    finally:
        library.close()
    hits = _search_all(bundle.search, bundle.query)
    assert tuple(hit.segment_id for hit in hits) == bundle.expected_segment_ids
    assert len(hits) == 44
    assert all(hit.meeting_id not in bundle.suppressed_meeting_ids for hit in hits)
    assert not any(hit.partial for hit in hits)
