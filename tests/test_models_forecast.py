"""Persistent forecast model schema + engine version stamp.

Covers the additive plan from ``.claude/plans/joyful-swinging-map.md`` plus
the #124 follow-up that folds ``engine_version`` into the EventForecast
unique key:

* ``EventForecast`` and ``EventForecastScore`` rows persist and round-trip
  through SQLAlchemy.
* The ``EventForecast`` unique constraint
  (``uq_event_forecast_event_gender_athlete_backfill_version``) blocks
  duplicates only when every key column matches — including
  ``engine_version`` — so prior-engine snapshots survive a version bump.
* ``EventForecastScore`` 's
  ``uq_event_forecast_score_event_gender_backfill_version`` blocks duplicates
  only when every key column matches — including ``engine_version`` (#131) —
  so prior-engine score rows survive a version bump.
* ``engine_version_tag()`` returns a deterministic ``<12hex>-<sha>`` string
  whose shape is stable across calls within a process.
"""

from __future__ import annotations

import re

import pytest
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from climbing_elo.engine.elo import EloConfig, engine_version_tag
from climbing_elo.models import (
    Athlete,
    Event,
    EventForecast,
    EventForecastScore,
    Gender,
)


_VERSION_RE = re.compile(r"^[0-9a-f]{12}-(?:[0-9a-f]{7,12}|unknown)$")


def _make_athlete(db_session: Session, name: str = "Test Climber") -> Athlete:
    athlete = Athlete(name=name, gender=Gender.M)
    db_session.add(athlete)
    db_session.flush()
    return athlete


def test_event_forecast_persists(db_session: Session, sample_event: Event) -> None:
    athlete = _make_athlete(db_session)
    forecast = EventForecast(
        event_id=sample_event.id,
        gender=Gender.M,
        athlete_id=athlete.id,
        prob_qualify=1.0,
        prob_reach_semi=0.8,
        prob_reach_final=0.5,
        prob_podium=0.2,
        prob_win=0.1,
        expected_rank=5.5,
        mu_at_forecast=1750.0,
        sigma_at_forecast=120.0,
        n_simulations=10_000,
        roster_source="confirmed",
        is_backfill=False,
        engine_version=engine_version_tag(),
    )
    db_session.add(forecast)
    db_session.flush()

    fetched = db_session.query(EventForecast).filter_by(id=forecast.id).one()
    assert fetched.prob_qualify == 1.0
    assert fetched.prob_win == 0.1
    assert fetched.roster_source == "confirmed"
    assert fetched.is_backfill is False
    assert fetched.generated_at is not None


def test_event_forecast_unique_constraint_blocks_duplicates(
    db_session: Session, sample_event: Event
) -> None:
    athlete = _make_athlete(db_session)
    base_kwargs = dict(
        event_id=sample_event.id,
        gender=Gender.M,
        athlete_id=athlete.id,
        prob_qualify=1.0,
        prob_reach_semi=0.5,
        prob_reach_final=0.3,
        prob_podium=0.1,
        prob_win=0.05,
        expected_rank=8.0,
        mu_at_forecast=1700.0,
        sigma_at_forecast=150.0,
        n_simulations=10_000,
        roster_source="likely",
        is_backfill=False,
        engine_version=engine_version_tag(),
    )
    db_session.add(EventForecast(**base_kwargs))
    db_session.flush()

    # All five key columns match (including engine_version) -> blocked.
    db_session.add(EventForecast(**base_kwargs))
    with pytest.raises(IntegrityError):
        db_session.flush()
    db_session.rollback()

    # A backfill row with the SAME (event, gender, athlete, engine_version)
    # is allowed -- ``is_backfill`` is part of the key so live + retro
    # coexist.
    backfill_kwargs = {**base_kwargs, "is_backfill": True}
    db_session.add(EventForecast(**backfill_kwargs))
    db_session.flush()


def test_event_forecast_allows_different_engine_versions(
    db_session: Session, sample_event: Event
) -> None:
    """#124: prior-engine snapshots must survive a version bump.

    Two rows that match on (event, gender, athlete, is_backfill) but differ
    only in engine_version should both insert successfully.
    """
    athlete = _make_athlete(db_session)
    base_kwargs = dict(
        event_id=sample_event.id,
        gender=Gender.M,
        athlete_id=athlete.id,
        prob_qualify=1.0,
        prob_reach_semi=0.5,
        prob_reach_final=0.3,
        prob_podium=0.1,
        prob_win=0.05,
        expected_rank=8.0,
        mu_at_forecast=1700.0,
        sigma_at_forecast=150.0,
        n_simulations=10_000,
        roster_source="likely",
        is_backfill=False,
    )

    db_session.add(EventForecast(**base_kwargs, engine_version="aaaaaaaaaaaa-1234567"))
    db_session.flush()
    db_session.add(EventForecast(**base_kwargs, engine_version="bbbbbbbbbbbb-1234567"))
    db_session.flush()  # must not raise

    persisted = (
        db_session.query(EventForecast)
        .filter_by(
            event_id=sample_event.id,
            gender=Gender.M,
            athlete_id=athlete.id,
            is_backfill=False,
        )
        .all()
    )
    assert len(persisted) == 2
    assert {row.engine_version for row in persisted} == {
        "aaaaaaaaaaaa-1234567",
        "bbbbbbbbbbbb-1234567",
    }


def test_event_forecast_blocks_full_key_duplicates(
    db_session: Session, sample_event: Event
) -> None:
    """#124: rows that match on all five key columns (including
    engine_version) still raise IntegrityError."""
    athlete = _make_athlete(db_session)
    kwargs = dict(
        event_id=sample_event.id,
        gender=Gender.M,
        athlete_id=athlete.id,
        prob_qualify=1.0,
        prob_reach_semi=0.5,
        prob_reach_final=0.3,
        prob_podium=0.1,
        prob_win=0.05,
        expected_rank=8.0,
        mu_at_forecast=1700.0,
        sigma_at_forecast=150.0,
        n_simulations=10_000,
        roster_source="likely",
        is_backfill=False,
        engine_version="aaaaaaaaaaaa-1234567",
    )
    db_session.add(EventForecast(**kwargs))
    db_session.flush()

    db_session.add(EventForecast(**kwargs))
    with pytest.raises(IntegrityError):
        db_session.flush()
    db_session.rollback()


def test_event_forecast_score_persists_and_blocks_duplicates(
    db_session: Session, sample_event: Event
) -> None:
    score_kwargs = dict(
        event_id=sample_event.id,
        gender=Gender.M,
        is_backfill=False,
        engine_version=engine_version_tag(),
        n_athletes=8,
        n_simulations=10_000,
        brier_semi=0.15,
        brier_final=0.18,
        brier_podium=0.10,
        brier_win=0.05,
        logloss_semi=0.40,
        logloss_final=0.45,
        logloss_podium=0.30,
        logloss_win=0.20,
        top3_intersection=2,
        top8_intersection=7,
        spearman_rank=0.78,
    )
    db_session.add(EventForecastScore(**score_kwargs))
    db_session.flush()

    fetched = db_session.query(EventForecastScore).one()
    assert fetched.n_athletes == 8
    assert fetched.spearman_rank == pytest.approx(0.78)
    assert fetched.computed_at is not None

    # All four key columns match (including engine_version) -> blocked.
    db_session.add(EventForecastScore(**score_kwargs))
    with pytest.raises(IntegrityError):
        db_session.flush()
    db_session.rollback()


def test_event_forecast_score_allows_different_engine_versions(
    db_session: Session, sample_event: Event
) -> None:
    """#131: prior-engine score rows must survive a version bump.

    Two rows that match on (event, gender, is_backfill) but differ only in
    engine_version should both insert successfully.
    """
    base_kwargs = dict(
        event_id=sample_event.id,
        gender=Gender.M,
        is_backfill=False,
        n_athletes=8,
        n_simulations=10_000,
        brier_semi=0.15,
        brier_final=0.18,
        brier_podium=0.10,
        brier_win=0.05,
        logloss_semi=0.40,
        logloss_final=0.45,
        logloss_podium=0.30,
        logloss_win=0.20,
        top3_intersection=2,
        top8_intersection=7,
        spearman_rank=0.78,
    )

    db_session.add(
        EventForecastScore(**base_kwargs, engine_version="aaaaaaaaaaaa-1234567")
    )
    db_session.flush()
    db_session.add(
        EventForecastScore(**base_kwargs, engine_version="bbbbbbbbbbbb-1234567")
    )
    db_session.flush()  # must not raise

    persisted = (
        db_session.query(EventForecastScore)
        .filter_by(
            event_id=sample_event.id,
            gender=Gender.M,
            is_backfill=False,
        )
        .all()
    )
    assert len(persisted) == 2
    assert {row.engine_version for row in persisted} == {
        "aaaaaaaaaaaa-1234567",
        "bbbbbbbbbbbb-1234567",
    }


def test_event_forecast_score_blocks_full_key_duplicates(
    db_session: Session, sample_event: Event
) -> None:
    """#131: rows that match on all four key columns (including
    engine_version) still raise IntegrityError."""
    kwargs = dict(
        event_id=sample_event.id,
        gender=Gender.M,
        is_backfill=False,
        engine_version="aaaaaaaaaaaa-1234567",
        n_athletes=8,
        n_simulations=10_000,
        brier_semi=0.15,
        brier_final=0.18,
        brier_podium=0.10,
        brier_win=0.05,
        logloss_semi=0.40,
        logloss_final=0.45,
        logloss_podium=0.30,
        logloss_win=0.20,
        top3_intersection=2,
        top8_intersection=7,
        spearman_rank=0.78,
    )
    db_session.add(EventForecastScore(**kwargs))
    db_session.flush()

    db_session.add(EventForecastScore(**kwargs))
    with pytest.raises(IntegrityError):
        db_session.flush()
    db_session.rollback()


def test_event_forecast_score_allows_null_spearman(
    db_session: Session, sample_event: Event
) -> None:
    """Spearman is undefined for 1-athlete fields / all-tied predictions."""
    score = EventForecastScore(
        event_id=sample_event.id,
        gender=Gender.F,
        is_backfill=False,
        engine_version=engine_version_tag(),
        n_athletes=1,
        n_simulations=10_000,
        brier_semi=0.0,
        brier_final=0.0,
        brier_podium=0.0,
        brier_win=0.0,
        logloss_semi=0.0,
        logloss_final=0.0,
        logloss_podium=0.0,
        logloss_win=0.0,
        top3_intersection=1,
        top8_intersection=1,
        spearman_rank=None,
    )
    db_session.add(score)
    db_session.flush()
    assert db_session.query(EventForecastScore).one().spearman_rank is None


def test_engine_version_tag_shape_and_stability() -> None:
    tag = engine_version_tag()
    assert isinstance(tag, str)
    assert tag, "engine_version_tag returned empty string"
    assert _VERSION_RE.match(tag) is not None, f"unexpected shape: {tag!r}"

    # Stable across calls within a session.
    assert engine_version_tag() == tag

    # Knob change → different config hash, same git SHA suffix.
    custom = engine_version_tag(EloConfig(margin_cap=2.0))
    assert custom != tag
    assert custom.rsplit("-", 1)[1] == tag.rsplit("-", 1)[1]
