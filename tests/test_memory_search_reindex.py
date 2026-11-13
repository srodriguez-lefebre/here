"""R53 facade delegation and composition; actual index equivalence stays guarded.

These controls do not replace B's real reindex/SQL/R45 tests with callback doubles.
No provider, capture, native deletion or new public UI maintenance action is used.
"""

from __future__ import annotations

import importlib
import sqlite3

import pytest
from test_memory_search import _Factory, _api, _open_catalog, _publish

TOKEN = "facadeneedle"


def _forbidden_resources(monkeypatch, search):
    """A facade callback owns maintenance; the reader must allocate nothing."""
    database = importlib.import_module("here.memory.database")
    backend = importlib.import_module("here.memory.service")
    observed = []

    def forbidden(*args, **kwargs):
        observed.append((args, kwargs))
        pytest.fail("R53 facade touched reader/writer resources instead of delegating")

    original_bindings = (
        (database, "open_library"), (database, "migrate"),
        (backend, "fts_available"), (backend, "_FileLock"),
        (backend, "MemoryService"), (sqlite3, "connect"),
    )
    for module, name in original_bindings:
        original = getattr(module, name)
        for alias, binding in tuple(vars(search).items()):
            if binding is original:
                monkeypatch.setattr(search, alias, forbidden)
        monkeypatch.setattr(module, name, forbidden)
    monkeypatch.setattr(search.SearchService, "_snapshot", forbidden)
    return forbidden, observed


def test_constructor_stores_maintenance_callback_without_invoking_or_opening_resources(monkeypatch):
    search, models = _api()
    factory, resources = _forbidden_resources(monkeypatch, search)
    calls = []

    def maintenance():
        calls.append(True)
        return models.ReindexReport()

    search.SearchService(factory, reindexer=maintenance)
    assert calls == []
    assert resources == []


def test_reindex_delegates_once_and_returns_the_same_report_and_warning_objects(monkeypatch):
    search, models = _api()
    backend = importlib.import_module("here.memory.service")
    warning = backend.FtsUnavailable()
    warnings = (warning,)
    report = models.ReindexReport(indexed_meetings=3, indexed_segments=7, warnings=warnings)
    factory, resources = _forbidden_resources(monkeypatch, search)
    calls = []

    def maintenance():
        calls.append(True)
        return report

    facade = search.SearchService(factory, reindexer=maintenance)
    result = facade.reindex()
    assert calls == [True]
    assert result is report
    assert result.warnings is warnings
    assert result.warnings[0] is warning
    assert (result.indexed_meetings, result.indexed_segments) == (3, 7)
    assert resources == []


@pytest.mark.parametrize("error_name", ["LibraryError", "Busy", "InvalidInput"])
def test_reindex_propagates_the_exact_canonical_error_without_retry_or_reader_resources(
    monkeypatch, error_name
):
    search, _ = _api()
    database = importlib.import_module("here.memory.database")
    backend = importlib.import_module("here.memory.service")
    error_type = backend.InvalidInput if error_name == "InvalidInput" else getattr(
        database, error_name
    )
    expected = (
        error_type("Synthetic maintenance input", reason="corpus_changed")
        if error_name == "InvalidInput" else error_type("Synthetic maintenance failure")
    )
    factory, resources = _forbidden_resources(monkeypatch, search)
    calls = []

    def maintenance():
        calls.append(True)
        raise expected

    facade = search.SearchService(factory, reindexer=maintenance)
    with pytest.raises(error_type) as caught:
        facade.reindex()
    assert caught.value is expected
    assert calls == [True]
    assert resources == []


def test_without_callback_only_reindex_fails_with_canonical_error_and_zero_resources(monkeypatch):
    search, _ = _api()
    database = importlib.import_module("here.memory.database")
    factory, resources = _forbidden_resources(monkeypatch, search)
    facade = search.SearchService(factory)
    with pytest.raises(database.LibraryError):
        facade.reindex()
    assert resources == []


@pytest.fixture
def reader_catalog(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    root = tmp_path / "custom-memory"
    library, repo = _open_catalog(root)
    try:
        entities = _publish(repo, 31, title="Synthetic R53", texts=(TOKEN, TOKEN + " evidence"))
        yield root, entities
    finally:
        library.close()


def test_absent_maintenance_callback_preserves_actual_capabilities_list_and_search(reader_catalog):
    search, models = _api()
    database = importlib.import_module("here.memory.database")
    root, entities = reader_catalog
    factory = _Factory(root)
    facade = search.SearchService(factory)
    with pytest.raises(database.LibraryError):
        facade.reindex()
    assert factory.connections == []
    assert facade.capabilities().fts_available is True
    listed = facade.list_meetings(models.MeetingQuery())
    assert [item.meeting.id for item in listed.items] == [entities[0]]
    hits = facade.search(models.SearchQuery(text=TOKEN))
    assert {hit.segment_id for hit in hits.items} == set(entities[2])
    assert all(connection.closed for connection in factory.connections)


def _real_composition(reader_catalog, tmp_path, monkeypatch):
    backend = importlib.import_module("here.memory.service")
    application = importlib.import_module("here.application")
    contract = importlib.import_module("here.ui.contract")
    root, entities = reader_catalog
    default_root = tmp_path / "unused-default-memory"
    monkeypatch.setenv("HERE_DATA_DIR", str(default_root))
    memory = backend.MemoryService(root)
    controller = application.HereApplicationController(memory_service=memory)
    assert controller.memory_service is memory

    def forbidden_second_memory(*args, **kwargs):
        pytest.fail("R53 composition constructed a second/default-root MemoryService")

    monkeypatch.setattr(backend.MemoryService, "__init__", forbidden_second_memory)
    return backend, contract, memory, controller, entities, default_root


@pytest.mark.parametrize("injection", ["controller", "adapter"])
def test_real_composition_binds_same_injected_memory_reindex_and_custom_root_once(
    reader_catalog, tmp_path, monkeypatch, injection
):
    _, models = _api()
    backend, contract, memory, controller, entities, default_root = _real_composition(
        reader_catalog, tmp_path, monkeypatch
    )
    actual_reindex = backend.MemoryService.reindex
    calls = []
    reports = []

    def observed_reindex(instance):
        assert instance is memory
        calls.append(instance)
        report = actual_reindex(instance)
        reports.append(report)
        return report

    original_open = contract.open_library
    reader_opens = []

    def observed_reader_open(root):
        assert root == memory.data_dir
        reader_opens.append(root)
        return original_open(root)

    monkeypatch.setattr(backend.MemoryService, "reindex", observed_reindex)
    monkeypatch.setattr(contract, "open_library", observed_reader_open)
    options = {"memory_service": memory} if injection == "adapter" else {}
    adapter = contract.ApplicationUiAdapter(controller, tmp_path / "sessions", **options)
    # R53 explicitly permits this composition seam; no protocol name is pinned.
    query = adapter._query_backend()
    assert calls == [] and reader_opens == []
    report = query.reindex()
    assert calls == [memory]
    assert report is reports[0]
    assert report.warnings is reports[0].warnings
    assert (report.indexed_meetings, report.indexed_segments) == (1, 2)
    assert reader_opens == [], "Facade maintenance must not open its query factory"
    assert [item.meeting.id for item in query.list_meetings(models.MeetingQuery()).items] == [
        entities[0]
    ]
    assert {hit.segment_id for hit in query.search(models.SearchQuery(text=TOKEN)).items} == set(
        entities[2]
    )
    assert reader_opens and all(root == memory.data_dir for root in reader_opens)
    assert not (default_root / "library.sqlite3").exists()


def test_memory_backend_without_maintenance_keeps_actual_adapter_reads_available(
    reader_catalog, tmp_path, monkeypatch
):
    _, models = _api()
    database = importlib.import_module("here.memory.database")
    _, contract, memory, controller, entities, default_root = _real_composition(
        reader_catalog, tmp_path, monkeypatch
    )

    class ReadOnlyMemory:
        data_dir = memory.data_dir

        def get_meeting(self, identity):
            return memory.get_meeting(identity)

        def list_participants(self, meeting_id=None):
            return memory.list_participants(meeting_id)

        def update_meeting(self, meeting_id, patch, expected_version):
            raise database.LibraryError("This test backend is read only")

    adapter = contract.ApplicationUiAdapter(
        controller, tmp_path / "readonly-sessions", memory_service=ReadOnlyMemory()
    )
    query = adapter._query_backend()
    with pytest.raises(database.LibraryError):
        query.reindex()
    assert [item.meeting.id for item in adapter.list_meetings(models.MeetingQuery()).items] == [
        entities[0]
    ]
    assert {hit.segment_id for hit in adapter.search(models.SearchQuery(text=TOKEN)).items} == set(
        entities[2]
    )
    assert adapter.get_meeting(entities[0]).meeting.id == entities[0]
    assert not (default_root / "library.sqlite3").exists()


def test_explicit_query_backend_is_returned_unchanged_without_resolving_memory(
    tmp_path, monkeypatch
):
    _, models = _api()
    contract = importlib.import_module("here.ui.contract")
    listing = models.MeetingPage()
    results = models.SearchPage()

    class ExplicitQueries:
        def list_meetings(self, query):
            return listing

        def search(self, query):
            return results

    explicit = ExplicitQueries()
    adapter = contract.ApplicationUiAdapter(
        object(), tmp_path / "override-sessions", search_service=explicit
    )

    def forbidden_memory_resolution():
        pytest.fail("Explicit query override incorrectly resolved a memory writer")

    monkeypatch.setattr(adapter, "_memory_backend", forbidden_memory_resolution)
    assert adapter._query_backend() is explicit
    assert adapter.list_meetings(models.MeetingQuery()) is listing
    assert adapter.search(models.SearchQuery(text=TOKEN)) is results
