"""Tests for SQLAlchemy models."""
from datetime import date

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
import pytest

from climbing_elo.models import (
    Athlete,
    Discipline,
    Event,
    EventTier,
    Gender,
    Rating,
    RatingHistory,
    Result,
    Round,
    RoundType,
)


def test_create_athlete(db_session):
    athlete = Athlete(name="Adam Ondra", gender=Gender.M, nationality="CZE")
    db_session.add(athlete)
    db_session.flush()
    assert athlete.id is not None


def test_athlete_unique_constraint(db_session):
    db_session.add(Athlete(name="Test", gender=Gender.M))
    db_session.add(Athlete(name="Test", gender=Gender.M))
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_athlete_same_name_different_gender(db_session):
    db_session.add(Athlete(name="Test", gender=Gender.M))
    db_session.add(Athlete(name="Test", gender=Gender.F))
    db_session.flush()


def test_create_event(db_session):
    event = Event(
        name="Innsbruck World Cup",
        tier=EventTier.WORLD_CUP,
        season=2024,
        start_date=date(2024, 6, 1),
        discipline=Discipline.LEAD,
    )
    db_session.add(event)
    db_session.flush()
    assert event.id is not None


def test_round_result_relationship(db_session, sample_event, eight_athletes):
    rnd = Round(
        event_id=sample_event.id,
        round_type=RoundType.FINAL,
        gender=Gender.M,
        athlete_count=8,
    )
    db_session.add(rnd)
    db_session.flush()

    for i, athlete in enumerate(eight_athletes):
        db_session.add(Result(
            round_id=rnd.id,
            athlete_id=athlete.id,
            rank=i + 1,
        ))
    db_session.flush()

    results = db_session.execute(
        select(Result).where(Result.round_id == rnd.id)
    ).scalars().all()
    assert len(results) == 8


def test_result_unique_constraint(db_session, sample_event, eight_athletes):
    rnd = Round(
        event_id=sample_event.id,
        round_type=RoundType.FINAL,
        gender=Gender.M,
    )
    db_session.add(rnd)
    db_session.flush()

    db_session.add(Result(round_id=rnd.id, athlete_id=eight_athletes[0].id, rank=1))
    db_session.add(Result(round_id=rnd.id, athlete_id=eight_athletes[0].id, rank=2))
    with pytest.raises(IntegrityError):
        db_session.flush()


def test_rating_crud(db_session, eight_athletes):
    rating = Rating(
        athlete_id=eight_athletes[0].id,
        discipline=Discipline.LEAD,
        mu=1600.0,
        sigma=120.0,
        n_events=5,
        provisional=False,
    )
    db_session.add(rating)
    db_session.flush()

    loaded = db_session.execute(
        select(Rating).where(Rating.athlete_id == eight_athletes[0].id)
    ).scalar_one()
    assert loaded.mu == 1600.0
    assert loaded.provisional is False


def test_rating_history_json(db_session, sample_event, eight_athletes):
    rnd = Round(
        event_id=sample_event.id,
        round_type=RoundType.FINAL,
        gender=Gender.M,
    )
    db_session.add(rnd)
    db_session.flush()

    pairs = [{"opponent_id": 2, "result": "won", "delta": 5.3}]
    rh = RatingHistory(
        athlete_id=eight_athletes[0].id,
        event_id=sample_event.id,
        round_id=rnd.id,
        mu_before=1500.0,
        mu_after=1510.0,
        sigma_before=350.0,
        sigma_after=343.0,
        contributing_pairs=pairs,
    )
    db_session.add(rh)
    db_session.flush()

    loaded = db_session.execute(
        select(RatingHistory).where(RatingHistory.id == rh.id)
    ).scalar_one()
    assert loaded.contributing_pairs[0]["delta"] == 5.3
