"""HTML route tests for the persistent-forecast UI work (Lane F).

Covers:
- ``GET /model-performance`` renders 200 + headline copy.
- ``GET /events/{event_id}`` renders the "Predicted vs Actual" recap panel
  when an :class:`EventForecast` set + final results exist.
- ``GET /events/{event_id}`` still works (no recap section) when no forecast
  is stored.
- ``GET /predictions/{event_id}`` renders **recap mode** for a finished event
  with stored forecasts.
- ``GET /predictions/{event_id}`` renders **forward mode** for an upcoming
  event (no final results yet).

Mirrors the per-test-DB fixture pattern in ``tests/test_v1_forecast_routes.py``
so the module-scoped client in ``tests/test_api.py`` isn't disturbed.
"""

from __future__ import annotations

from datetime import date, datetime, timezone
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
    EventForecast,
    EventForecastScore,
    EventTier,
    Gender,
    Rating,
    Result,
    Round,
    RoundType,
)


# ---------------------------------------------------------------------------
# Fixtures — per-test SQLite DB + patched session/engine getters.
# ---------------------------------------------------------------------------


@pytest.fixture
def db_path(tmp_path: Path) -> Path:
    return tmp_path / "html_forecast_test.db"


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


# ---------------------------------------------------------------------------
# Seeders
# ---------------------------------------------------------------------------


def _seed_event_with_final(
    session: Session,
    *,
    name: str = "Test World Cup",
    season: int = 2026,
    start_date: date | None = None,
    discipline: Discipline = Discipline.LEAD,
    n_athletes: int = 6,
) -> tuple[Event, list[Athlete], Round]:
    """Seed an event with a FINAL round + n_athletes ranked results.

    Returns (event, athletes, final_round).
    """
    start_date = start_date or date(2026, 6, 1)
    event = Event(
        name=name,
        tier=EventTier.WORLD_CUP,
        country="AUT",
        season=season,
        start_date=start_date,
        discipline=discipline,
    )
    session.add(event)
    session.flush()

    athletes: list[Athlete] = []
    for i in range(n_athletes):
        a = Athlete(name=f"Athlete-{name}-{i}", gender=Gender.M, nationality="USA")
        session.add(a)
        athletes.append(a)
    session.flush()

    final_round = Round(
        event_id=event.id,
        round_type=RoundType.FINAL,
        gender=Gender.M,
    )
    session.add(final_round)
    session.flush()

    for rank, athlete in enumerate(athletes, start=1):
        session.add(
            Result(
                round_id=final_round.id,
                athlete_id=athlete.id,
                rank=rank,
                raw_score=str(50 - rank),
                dns=False,
            )
        )

    # Each athlete needs a Rating row so the projections-inputs builder in
    # forward mode has something to read (not used by the recap tests but
    # cheap to keep both paths exercising the same fixture).
    for athlete in athletes:
        session.add(
            Rating(
                athlete_id=athlete.id,
                discipline=discipline,
                mu=1600.0,
                sigma=100.0,
                n_events=10,
                provisional=False,
            )
        )

    session.flush()
    return event, athletes, final_round


def _seed_forecast_set(
    session: Session,
    *,
    event_id: int,
    athletes: list[Athlete],
    gender: Gender = Gender.M,
    is_backfill: bool = False,
    engine_version: str = "abc123def456-cafebab",
) -> None:
    """Seed one EventForecast row per athlete + one EventForecastScore row."""
    n = len(athletes)
    for i, athlete in enumerate(athletes):
        # Probabilities monotone non-increasing with i. Top finisher gets the
        # highest win/podium prob.
        decay = max(0.0, 1.0 - i / max(n - 1, 1))
        session.add(
            EventForecast(
                event_id=event_id,
                gender=gender,
                athlete_id=athlete.id,
                prob_qualify=1.0,
                prob_reach_semi=0.9 * decay + 0.1,
                prob_reach_final=0.7 * decay + 0.05,
                prob_podium=0.4 * decay + 0.02,
                prob_win=0.3 * decay + 0.01,
                expected_rank=float(i + 1),
                mu_at_forecast=1700.0 - i * 20,
                sigma_at_forecast=110.0,
                n_simulations=10_000,
                roster_source="confirmed",
                is_backfill=is_backfill,
                engine_version=engine_version,
                generated_at=datetime.now(timezone.utc),
            )
        )

    session.add(
        EventForecastScore(
            event_id=event_id,
            gender=gender,
            is_backfill=is_backfill,
            engine_version=engine_version,
            n_athletes=n,
            n_simulations=10_000,
            brier_semi=0.10,
            brier_final=0.12,
            brier_podium=0.15,
            brier_win=0.05,
            logloss_semi=0.30,
            logloss_final=0.34,
            logloss_podium=0.40,
            logloss_win=0.15,
            top3_intersection=2,
            top8_intersection=5,
            spearman_rank=0.72,
            computed_at=datetime.now(timezone.utc),
        )
    )


# ---------------------------------------------------------------------------
# /model-performance
# ---------------------------------------------------------------------------


def test_model_performance_page_renders(client: TestClient) -> None:
    r = client.get("/model-performance")
    assert r.status_code == 200, r.text
    body = r.text
    # Headline + filter labels are present even with zero scored events.
    assert "Model" in body
    assert "performance" in body
    assert "Season" in body
    assert "Discipline" in body
    # Include-backfill toggle is rendered.
    assert "include_backfill" in body


def test_model_performance_page_includes_seeded_event(
    client: TestClient, factory
) -> None:
    """When a scored event exists for the default season, it appears in the
    per-event table and the aggregate-card count > 0."""
    today_year = date.today().year
    with factory() as session:
        event, athletes, _ = _seed_event_with_final(
            session,
            name="Bern Boulder",
            season=today_year,
            start_date=date(today_year, 6, 1),
            discipline=Discipline.BOULDER,
        )
        _seed_forecast_set(session, event_id=event.id, athletes=athletes)
        session.commit()

    r = client.get("/model-performance")
    assert r.status_code == 200, r.text
    body = r.text
    assert "Bern Boulder" in body
    # "Events scored" card shows at least 1.
    assert "Events scored" in body


# ---------------------------------------------------------------------------
# /events/{id}  recap panel
# ---------------------------------------------------------------------------


def test_event_detail_recap_panel_renders(client: TestClient, factory) -> None:
    with factory() as session:
        event, athletes, _ = _seed_event_with_final(session, name="Innsbruck Lead")
        _seed_forecast_set(session, event_id=event.id, athletes=athletes)
        session.commit()
        event_id = event.id

    r = client.get(f"/events/{event_id}")
    assert r.status_code == 200, r.text
    body = r.text
    assert "Predicted vs Actual" in body
    # Score-row summary line is present (Brier value rendered).
    assert "Podium Brier" in body
    assert "Top-3 intersection" in body


def test_event_detail_no_panel_when_no_forecast(client: TestClient, factory) -> None:
    with factory() as session:
        _seed_event_with_final(session, name="Ungforecasted Cup")
        session.commit()

    # Look up the seeded event id via API (single event in the DB).
    with factory() as session:
        event = session.query(Event).one()
        event_id = event.id

    r = client.get(f"/events/{event_id}")
    assert r.status_code == 200, r.text
    body = r.text
    # The recap header is the unique marker for the panel — must NOT appear.
    assert "Predicted vs Actual" not in body
    # But the event still renders normally.
    assert "Ungforecasted Cup" in body


# ---------------------------------------------------------------------------
# /predictions/{id}  recap + forward modes
# ---------------------------------------------------------------------------


def test_predictions_recap_mode(client: TestClient, factory) -> None:
    with factory() as session:
        event, athletes, _ = _seed_event_with_final(
            session, name="Salt Lake Speed", discipline=Discipline.SPEED
        )
        _seed_forecast_set(session, event_id=event.id, athletes=athletes)
        session.commit()
        event_id = event.id

    r = client.get(f"/predictions/{event_id}")
    assert r.status_code == 200, r.text
    body = r.text
    # Recap-mode-specific copy: "recap" headline + score summary line.
    assert "recap" in body.lower()
    assert "Top-3 intersection" in body
    assert "Podium Brier" in body


def test_predictions_forward_mode(client: TestClient, factory) -> None:
    """Upcoming event with no FINAL results yet renders the forward
    projection — no recap panel."""
    with factory() as session:
        # Seed an upcoming event WITHOUT FINAL results — just an empty
        # qualification round so likely_competitors can still produce rows.
        future = date.today().replace(year=date.today().year + 1)
        event = Event(
            name="Future Lead Cup",
            tier=EventTier.WORLD_CUP,
            country="USA",
            season=future.year,
            start_date=future,
            discipline=Discipline.LEAD,
        )
        session.add(event)
        session.flush()
        rnd = Round(
            event_id=event.id,
            round_type=RoundType.QUALIFICATION,
            gender=Gender.M,
        )
        session.add(rnd)
        session.flush()

        # Add 4 athletes with results in the qualification round (so the
        # forward route picks them up directly without falling back to
        # likely_competitors, which needs season-history context).
        for i in range(4):
            a = Athlete(name=f"Forward-{i}", gender=Gender.M, nationality="USA")
            session.add(a)
            session.flush()
            session.add(
                Rating(
                    athlete_id=a.id,
                    discipline=Discipline.LEAD,
                    mu=1600.0 + i * 20,
                    sigma=100.0,
                    n_events=10,
                    provisional=False,
                )
            )
            session.add(
                Result(
                    round_id=rnd.id,
                    athlete_id=a.id,
                    rank=i + 1,
                    raw_score=str(40 - i),
                    dns=False,
                )
            )
        session.commit()
        event_id = event.id

    r = client.get(f"/predictions/{event_id}")
    assert r.status_code == 200, r.text
    body = r.text
    # Forward-mode-specific copy: "forward projection" headline.
    assert "forward projection" in body.lower()
    # And no recap-only labels.
    assert "Top-3 intersection" not in body
