"""Tests for the Supabase-egress optimizations (2026-08).

Covers the three query-shape guarantees the egress work relies on:

1. ``RatingHistory.contributing_pairs`` is deferred - the ~1.1KB JSON blob
   (93% of the row's bytes) must not appear in a default ORM load.
2. Sites that DO read the blob (breakdown, profile opponents preview) opt back
   in via ``undefer`` and still see the data.
3. ``_ticker_context`` is cached - repeat page renders must not re-run the
   ticker queries while ratings are unchanged.
"""

from __future__ import annotations

from datetime import date

from sqlalchemy import select
from sqlalchemy.orm import undefer

from climbing_elo.models import (
    Athlete,
    Discipline,
    Event,
    EventTier,
    Gender,
    RatingHistory,
    Round,
    RoundType,
)


def _seed_history(db_session) -> tuple[Athlete, Event]:
    ath = Athlete(name="Deferred Tester", gender=Gender.M)
    ev = Event(
        name="Test WC",
        season=2024,
        discipline=Discipline.LEAD,
        tier=EventTier.WORLD_CUP,
        start_date=date(2024, 6, 1),
    )
    db_session.add_all([ath, ev])
    db_session.flush()
    rnd = Round(event_id=ev.id, round_type=RoundType.FINAL, gender=Gender.M)
    db_session.add(rnd)
    db_session.flush()
    db_session.add(
        RatingHistory(
            athlete_id=ath.id,
            event_id=ev.id,
            round_id=rnd.id,
            mu_before=1500.0,
            mu_after=1512.0,
            sigma_before=200.0,
            sigma_after=195.0,
            contributing_pairs=[{"opponent_id": 999, "result": "W", "delta": 12.0}],
            kind="pair",
        )
    )
    db_session.flush()
    return ath, ev


def test_contributing_pairs_not_in_default_select():
    """The deferred JSON column must be absent from a plain ORM SELECT."""
    sql = str(select(RatingHistory))
    assert "contributing_pairs" not in sql, sql


def test_contributing_pairs_loads_via_undefer(db_session):
    ath, ev = _seed_history(db_session)
    row = db_session.execute(
        select(RatingHistory)
        .options(undefer(RatingHistory.contributing_pairs))
        .where(RatingHistory.athlete_id == ath.id)
    ).scalar_one()
    assert row.contributing_pairs == [
        {"opponent_id": 999, "result": "W", "delta": 12.0}
    ]


def test_contributing_pairs_lazy_loads_when_accessed(db_session):
    """Without undefer, access still works (lazy load) - nothing breaks."""
    ath, _ = _seed_history(db_session)
    row = db_session.execute(
        select(RatingHistory).where(RatingHistory.athlete_id == ath.id)
    ).scalar_one()
    assert row.contributing_pairs[0]["opponent_id"] == 999


def test_ticker_context_is_cached(db_session, monkeypatch):
    from climbing_elo.api import routes

    calls = {"n": 0}
    real = routes._ticker_context_uncached

    def counting(session):
        calls["n"] += 1
        return real(session)

    monkeypatch.setattr(routes, "_ticker_context_uncached", counting)
    first = routes._ticker_context(db_session)
    second = routes._ticker_context(db_session)
    assert calls["n"] == 1
    assert second == first
