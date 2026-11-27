"""Exercise navigation regressions with explicitly delivered worker results."""

from dataclasses import replace

import pytest
import test_ui_playback_navigation as navigation_tests
from here.memory.database import Busy
from PySide6.QtWidgets import QLabel
from test_ui_library import _detail, _text
from test_ui_playback_navigation import _buttons, _passive, _transcript

dto = navigation_tests.dto
navigation_library = navigation_tests.navigation_library
evidence = navigation_tests.evidence
fake_media = navigation_tests.fake_media

pytestmark = pytest.mark.usefixtures("owned_desktops", "fake_media")


@pytest.fixture
def complete_job(navigation_library, monkeypatch):
    desktop, _, _ = navigation_library
    jobs = desktop.jobs
    pending = {}
    next_request = 10_000

    def submit(kind, operation):
        nonlocal next_request
        assert kind not in pending
        next_request += 1
        pending[kind] = next_request
        return next_request

    def complete(kind, value=None, *, error=None):
        request = pending.pop(kind)
        jobs.finished.emit(request, kind, value, error)
        jobs.settled.emit(request)

    monkeypatch.setattr(jobs, "submit", submit)
    return complete


def test_same_meeting_navigation_preserves_pending_title_edit(
    navigation_library, evidence, complete_job
):
    _, _, view = navigation_library
    detail, target, _ = evidence
    view.open_target(_passive(target))
    complete_job("library-detail", detail)
    complete_job("library-citation", target)
    (save,) = _buttons(view, r"Guardar título")
    save.click()
    assert not save.isEnabled()

    view.open_target(_passive(target))
    enabled_while_pending = save.isEnabled()
    complete_job("library-citation", target)
    failure = "Synthetic title edit failed"
    complete_job("library-title", error=Busy(failure))

    assert (enabled_while_pending, failure in _text(view), save.isEnabled()) == (False, True, True)


@pytest.fixture
def revisions(evidence):
    _, target, _ = evidence
    detail = _detail(text="Current synthetic evidence")
    historical_revision = replace(detail.revisions[0], id="00000000-0000-0000-0000-000000000031")
    historical_segment = replace(
        detail.segments[0],
        id="00000000-0000-0000-0000-000000000032",
        revision_id=historical_revision.id,
        text="Earlier synthetic evidence",
    )
    detail = replace(
        detail,
        revisions=(historical_revision, replace(detail.revisions[0], revision_number=2)),
        segments=(historical_segment, *detail.segments),
    )
    return detail, historical_segment, target


def test_opening_current_citation_retains_earlier_evidence(
    navigation_library, revisions, complete_job
):
    _, _, view = navigation_library
    detail, historical_segment, target = revisions
    target = replace(target, text=detail.segments[-1].text)

    view.open_target(_passive(target))
    complete_job("library-detail", detail)
    assert historical_segment.text in _text(view)
    complete_job("library-citation", target)

    selected = _transcript(view).currentItem()
    assert selected is not None and target.text in selected.text()
    assert historical_segment.text in _text(view)


def test_historical_citation_with_missing_audio_still_links_to_current_revision(
    navigation_library, revisions, complete_job
):
    _, _, view = navigation_library
    detail, historical_segment, target = revisions
    historical = replace(
        target,
        revision_id=historical_segment.revision_id,
        segment_id=historical_segment.id,
        text=historical_segment.text,
        status="missing",
        audio_path=None,
    )
    view.open_target(_passive(historical))
    complete_job("library-detail", detail)
    complete_job("library-citation", historical)

    selected = _transcript(view).currentItem()
    assert selected is not None and historical.text in selected.text()
    assert (
        any(
            label.isVisibleTo(view) and label.text() == "Revisión histórica"
            for label in view.findChildren(QLabel)
        ),
        bool(_buttons(view, r"revisión actual")),
    ) == (True, True)
    (current_revision,) = _buttons(view, r"revisión actual")
    current_revision.click()
    complete_job("library-citation", replace(target, text=detail.segments[-1].text))
    assert detail.segments[-1].text in _transcript(view).currentItem().text()


def test_busy_detail_does_not_offer_playback_without_authoritative_evidence(
    navigation_library, evidence, complete_job
):
    _, _, view = navigation_library
    _, target, _ = evidence
    view.open_target(_passive(target))
    complete_job("library-detail", error=Busy("Synthetic busy meeting detail"))
    complete_job("library-citation", target)

    playback = _buttons(view, r"Reproducir desde")
    assert not playback or target.text in _text(view)


def test_successful_same_meeting_navigation_clears_previous_resolver_error(
    navigation_library, evidence, complete_job
):
    _, _, view = navigation_library
    detail, target, _ = evidence
    failure = "Synthetic busy citation resolver"
    view.open_target(_passive(target))
    complete_job("library-detail", detail)
    complete_job("library-citation", error=Busy(failure))
    assert failure in _text(view)

    view.open_target(_passive(target))
    complete_job("library-citation", target)

    selected = _transcript(view).currentItem()
    assert selected is not None and target.text in selected.text()
    assert failure not in _text(view)
