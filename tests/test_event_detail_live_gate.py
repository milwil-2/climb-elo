"""HTML tests for the live-view gate on /events/{id} and /live/{id} (#134).

The "Live view ->" link on the event detail page is now gated on the
canonical ``event_status()`` predicate (#136). The ``/live/{event_id}``
route soft-404s when the event is not in the LIVE state — defense in
depth so stale bookmarks don't surface an empty live page.

Mirrors the per-test-DB fixture pattern in ``tests/test_html_forecast_routes.py``.
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

import climbing_elo.api.routes as _routes
import climbing_elo.database as _db
from climbing_elo.api.app import create_app
from climbing_elo.models import (
    Athlete,
    Base,
    Discipline,
    Event,
    EventTier,
    Gender,
    Rating,
    Result,
    Round,
    RoundType,
)


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "live_gate_test.db"


@pytest.fixture
def factory(db_path: Path):
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


@pytest.fixture
def client(db_path: Path, factory) -> TestClient:
    original_session = _routes._session
    original_get_engine = _db.get_engine

    def patched_session():
        return factory()

    def patched_get_engine(_=None):
        return create_engine(f"sqlite:///{db_path}")

    _routes._session = patched_session  # type: ignore[assignment]
    _db.get_engine = patched_get_engine  # type: ignore[assignment]

    app = create_app()
    tc = TestClient(app)
    try:
        yield tc
    finally:
        _routes._session = original_session
        _db.get_engine = original_get_engine


def _seed_event(
    session: Session,
    *,
    name: str,
    start_date: date,
    with_final_results: bool = False,
) -> Event:
    event = Event(
        name=name,
        tier=EventTier.WORLD_CUP,
        country="AUT",
        season=start_date.year,
        start_date=start_date,
        discipline=Discipline.LEAD,
    )
    session.add(event)
    session.flush()

    if with_final_results:
        athlete = Athlete(name=f"{name}-winner", gender=Gender.M, nationality="USA")
        session.add(athlete)
        session.flush()
        session.add(
            Rating(
                athlete_id=athlete.id,
                discipline=Discipline.LEAD,
                mu=1700.0,
                sigma=100.0,
                n_events=5,
                provisional=False,
            )
        )
        rnd = Round(
            event_id=event.id,
            round_type=RoundType.FINAL,
            gender=Gender.M,
        )
        session.add(rnd)
        session.flush()
        session.add(
            Result(
                round_id=rnd.id,
                athlete_id=athlete.id,
                rank=1,
                raw_score="TOP",
                dns=False,
            )
        )
    session.commit()
    return event


# ---------------------------------------------------------------------------
# /events/{id} — "Live view" link gate
# ---------------------------------------------------------------------------


def test_event_detail_hides_live_link_for_upcoming(client: TestClient, factory) -> None:
    """An upcoming event (>30 days out) should NOT render the Live view link."""
    with factory() as s:
        event = _seed_event(
            s, name="Future WC", start_date=date.today() + timedelta(days=60)
        )
        eid = event.id

    r = client.get(f"/events/{eid}")
    assert r.status_code == 200
    assert "Live view" not in r.text


def test_event_detail_hides_live_link_for_finished(client: TestClient, factory) -> None:
    """A past event with a final result is FINISHED — no Live view link."""
    with factory() as s:
        event = _seed_event(
            s,
            name="Past WC",
            start_date=date.today() - timedelta(days=180),
            with_final_results=True,
        )
        eid = event.id

    r = client.get(f"/events/{eid}")
    assert r.status_code == 200
    assert "Live view" not in r.text


def test_event_detail_shows_live_link_when_live(client: TestClient, factory) -> None:
    """An event whose start_date is today and has no final results is LIVE."""
    with factory() as s:
        event = _seed_event(
            s,
            name="Live WC",
            start_date=date.today(),
            with_final_results=False,
        )
        eid = event.id

    r = client.get(f"/events/{eid}")
    assert r.status_code == 200
    assert "Live view" in r.text
    assert f'href="/live/{eid}"' in r.text


# ---------------------------------------------------------------------------
# /live/{id} — soft-404 for non-live events
# ---------------------------------------------------------------------------


def test_live_route_returns_404_for_finished_event(client: TestClient, factory) -> None:
    with factory() as s:
        event = _seed_event(
            s,
            name="Old WC",
            start_date=date.today() - timedelta(days=365),
            with_final_results=True,
        )
        eid = event.id

    r = client.get(f"/live/{eid}")
    assert r.status_code == 404
    assert "isn't currently live" in r.text
    assert f'href="/events/{eid}"' in r.text


def test_live_route_returns_404_for_upcoming_event(client: TestClient, factory) -> None:
    with factory() as s:
        event = _seed_event(
            s,
            name="Future WC",
            start_date=date.today() + timedelta(days=21),
        )
        eid = event.id

    r = client.get(f"/live/{eid}")
    assert r.status_code == 404
    assert "isn't currently live" in r.text


def test_live_route_returns_200_when_live(client: TestClient, factory) -> None:
    """An event in the live window with no final results renders the live page."""
    with factory() as s:
        event = _seed_event(
            s,
            name="Live WC",
            start_date=date.today() - timedelta(days=1),
            with_final_results=False,
        )
        eid = event.id

    r = client.get(f"/live/{eid}")
    assert r.status_code == 200
