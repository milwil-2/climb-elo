"""Tests for the HTML-page response cache and batched 90d-delta (Issue #97).

Covers:
- ``ratings_fingerprint`` changes when ratings mutate.
- ``html_page_cache`` hit returns the identical object; a fingerprint shift
  (ratings mutate) misses, so stale data is never served.
- The batched ``_get_90d_deltas`` matches the legacy per-athlete
  ``_get_90d_delta`` on the eight_athletes fixture.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from sqlalchemy.orm import Session

from climbing_elo.api.routes import _get_90d_delta, _get_90d_deltas
from climbing_elo.cache import TTLCache, ratings_fingerprint
from climbing_elo.models import (
    Athlete,
    Discipline,
    Event,
    EventTier,
    Rating,
    RatingHistory,
    Round,
    RoundType,
)


# ---------------------------------------------------------------------------
# ratings_fingerprint
# ---------------------------------------------------------------------------


def _add_rating(session: Session, athlete_id: int, mu: float) -> Rating:
    r = Rating(
        athlete_id=athlete_id,
        discipline=Discipline.LEAD,
        mu=mu,
        sigma=100.0,
        n_events=5,
        provisional=False,
    )
    session.add(r)
    session.flush()
    return r


def test_fingerprint_stable_until_mutation(
    db_session: Session, eight_athletes: list[Athlete]
):
    _add_rating(db_session, eight_athletes[0].id, 1700.0)
    fp1 = ratings_fingerprint(db_session)
    # Recomputing without changes yields the same fingerprint.
    assert ratings_fingerprint(db_session) == fp1


def test_fingerprint_changes_when_mu_mutates(
    db_session: Session, eight_athletes: list[Athlete]
):
    r = _add_rating(db_session, eight_athletes[0].id, 1700.0)
    fp1 = ratings_fingerprint(db_session)

    r.mu = 1850.0
    db_session.flush()
    fp2 = ratings_fingerprint(db_session)
    assert fp2 != fp1


def test_fingerprint_changes_when_rating_added(
    db_session: Session, eight_athletes: list[Athlete]
):
    _add_rating(db_session, eight_athletes[0].id, 1700.0)
    fp1 = ratings_fingerprint(db_session)

    _add_rating(db_session, eight_athletes[1].id, 1600.0)
    fp2 = ratings_fingerprint(db_session)
    assert fp2 != fp1


def test_empty_table_fingerprint(db_session: Session):
    # No ratings — must not raise (coalesce guards the NULL aggregates).
    assert ratings_fingerprint(db_session) == "0:0.0:0.0"


# ---------------------------------------------------------------------------
# html_page_cache behaviour (using the same TTLCache primitive)
# ---------------------------------------------------------------------------


def test_cache_hit_returns_identical_object(
    db_session: Session, eight_athletes: list[Athlete]
):
    cache = TTLCache(ttl_seconds=600)
    _add_rating(db_session, eight_athletes[0].id, 1700.0)

    key = f"html:leaderboard:fp:{ratings_fingerprint(db_session)}"
    payload = {"rows": [{"id": 1, "mu": 1700.0}], "total_count": 1}
    cache.set(key, payload)

    assert cache.get(key) is payload  # identical object, no recompute


def test_fingerprint_change_invalidates_cache(
    db_session: Session, eight_athletes: list[Athlete]
):
    cache = TTLCache(ttl_seconds=600)
    r = _add_rating(db_session, eight_athletes[0].id, 1700.0)

    key1 = f"html:leaderboard:fp:{ratings_fingerprint(db_session)}"
    cache.set(key1, {"stale": True})

    # Ratings mutate -> fingerprint shifts -> different key -> cache miss.
    r.mu = 1900.0
    db_session.flush()
    key2 = f"html:leaderboard:fp:{ratings_fingerprint(db_session)}"

    assert key2 != key1
    assert cache.get(key2) is None  # new key not populated -> would recompute


# ---------------------------------------------------------------------------
# batched 90d delta == legacy per-athlete delta
# ---------------------------------------------------------------------------


def _seed_history(
    session: Session,
    athletes: list[Athlete],
) -> None:
    """Give each athlete a handful of LEAD history rows across distinct events."""
    base = date(2024, 1, 1)
    for ai, athlete in enumerate(athletes):
        # 4 events per athlete so the "3 most recent" window actually clips one.
        for ei in range(4):
            event = Event(
                name=f"WC {ai}-{ei}",
                tier=EventTier.WORLD_CUP,
                season=2024,
                start_date=base + timedelta(days=ei * 30),
                discipline=Discipline.LEAD,
            )
            session.add(event)
            session.flush()
            rnd = Round(
                event_id=event.id,
                round_type=RoundType.FINAL,
                gender=athlete.gender,
            )
            session.add(rnd)
            session.flush()
            mu_before = 1500.0 + ai * 10 + ei * 7
            mu_after = mu_before + (ei + 1) * 3 - ai
            session.add(
                RatingHistory(
                    athlete_id=athlete.id,
                    event_id=event.id,
                    round_id=rnd.id,
                    mu_before=mu_before,
                    mu_after=mu_after,
                    sigma_before=100.0,
                    sigma_after=98.0,
                    kind="pair",
                )
            )
    session.flush()


def test_batched_delta_matches_legacy(
    db_session: Session, eight_athletes: list[Athlete]
):
    _seed_history(db_session, eight_athletes)
    ids = [a.id for a in eight_athletes]

    batched = _get_90d_deltas(db_session, ids, Discipline.LEAD)
    for aid in ids:
        legacy = _get_90d_delta(db_session, aid, Discipline.LEAD)
        assert batched.get(aid, 0.0) == legacy, f"mismatch for athlete {aid}"


def test_batched_delta_empty_inputs(db_session: Session):
    assert _get_90d_deltas(db_session, [], Discipline.LEAD) == {}


def test_batched_delta_no_history_returns_zero_via_helper(
    db_session: Session, eight_athletes: list[Athlete]
):
    # No RatingHistory at all -> batched omits, legacy helper defaults to 0.0.
    aid = eight_athletes[0].id
    assert _get_90d_deltas(db_session, [aid], Discipline.LEAD) == {}
    assert _get_90d_delta(db_session, aid, Discipline.LEAD) == 0.0


def test_batched_delta_respects_discipline_filter(
    db_session: Session, eight_athletes: list[Athlete]
):
    _seed_history(db_session, eight_athletes[:1])
    aid = eight_athletes[0].id
    # History is all LEAD; querying BOULDER must yield nothing.
    assert _get_90d_deltas(db_session, [aid], Discipline.BOULDER) == {}
    assert _get_90d_deltas(db_session, [aid], Discipline.LEAD).get(aid) is not None


@pytest.mark.parametrize("n_events", [1, 2, 3, 5])
def test_batched_delta_window_size(
    db_session: Session, eight_athletes: list[Athlete], n_events: int
):
    """Delta must use exactly the 3 most-recent rows regardless of total count."""
    athlete = eight_athletes[0]
    base = date(2024, 1, 1)
    for ei in range(n_events):
        event = Event(
            name=f"WC {ei}",
            tier=EventTier.WORLD_CUP,
            season=2024,
            start_date=base + timedelta(days=ei * 30),
            discipline=Discipline.LEAD,
        )
        db_session.add(event)
        db_session.flush()
        rnd = Round(
            event_id=event.id, round_type=RoundType.FINAL, gender=athlete.gender
        )
        db_session.add(rnd)
        db_session.flush()
        db_session.add(
            RatingHistory(
                athlete_id=athlete.id,
                event_id=event.id,
                round_id=rnd.id,
                mu_before=1500.0 + ei,
                mu_after=1510.0 + ei,
                sigma_before=100.0,
                sigma_after=98.0,
                kind="pair",
            )
        )
    db_session.flush()

    batched = _get_90d_deltas(db_session, [athlete.id], Discipline.LEAD)
    legacy = _get_90d_delta(db_session, athlete.id, Discipline.LEAD)
    assert batched.get(athlete.id, 0.0) == legacy
