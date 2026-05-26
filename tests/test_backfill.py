"""Integration test for the backfill pipeline."""

from datetime import date

from sqlalchemy import select

from climbing_elo.engine.backfill import run_backfill
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


def _seed_event(session, name, event_date, athletes, final_order):
    """Create an event with a final round and seed results."""
    event = Event(
        name=name,
        tier=EventTier.WORLD_CUP,
        season=event_date.year,
        start_date=event_date,
        discipline=Discipline.LEAD,
    )
    session.add(event)
    session.flush()

    rnd = Round(
        event_id=event.id,
        round_type=RoundType.FINAL,
        gender=Gender.M,
        athlete_count=len(final_order),
    )
    session.add(rnd)
    session.flush()

    for rank, athlete_idx in enumerate(final_order, 1):
        session.add(
            Result(
                round_id=rnd.id,
                athlete_id=athletes[athlete_idx].id,
                rank=rank,
            )
        )
    session.flush()
    return event


def test_backfill_three_events(db_session):
    """Run backfill over 3 events and verify ratings converge sensibly."""
    athletes = []
    for name in ["Alpha", "Beta", "Gamma", "Delta"]:
        a = Athlete(name=name, gender=Gender.M)
        db_session.add(a)
        athletes.append(a)
    db_session.flush()

    # Event 1: Alpha > Beta > Gamma > Delta
    _seed_event(db_session, "WC Innsbruck", date(2024, 3, 1), athletes, [0, 1, 2, 3])
    # Event 2: Beta > Alpha > Delta > Gamma
    _seed_event(db_session, "WC Chamonix", date(2024, 5, 1), athletes, [1, 0, 3, 2])
    # Event 3: Alpha > Alpha again > Gamma > Beta — wait, can't repeat.
    # Event 3: Alpha > Gamma > Beta > Delta
    _seed_event(db_session, "WC Briançon", date(2024, 7, 1), athletes, [0, 2, 1, 3])
    db_session.commit()

    report = run_backfill(db_session, Discipline.LEAD)

    assert report.events_processed == 3
    assert report.rounds_processed == 3
    assert len(report.athletes_rated) == 4
    assert len(report.errors) == 0

    ratings = {
        r.athlete_id: r
        for r in db_session.execute(
            select(Rating).where(Rating.discipline == Discipline.LEAD)
        ).scalars()
    }

    # Alpha won 2 of 3 events and placed 2nd in the other — should be top
    alpha_id = athletes[0].id
    beta_id = athletes[1].id
    delta_id = athletes[3].id

    assert ratings[alpha_id].mu > ratings[beta_id].mu
    assert ratings[beta_id].mu > ratings[delta_id].mu

    # All should have n_events = 3
    for r in ratings.values():
        assert r.n_events == 3
        assert r.provisional is False  # >= PROVISIONAL_THRESHOLD

    # Rating history should have entries
    history_count = db_session.execute(select(RatingHistory)).all()
    assert len(history_count) >= 12  # 4 athletes × 3 events


def test_backfill_reproducibility(db_session):
    """Running backfill twice on the same data should produce identical results."""
    athletes = []
    for name in ["X", "Y", "Z"]:
        a = Athlete(name=name, gender=Gender.M)
        db_session.add(a)
        athletes.append(a)
    db_session.flush()

    _seed_event(db_session, "Test WC 1", date(2024, 1, 1), athletes, [0, 1, 2])
    _seed_event(db_session, "Test WC 2", date(2024, 6, 1), athletes, [2, 0, 1])
    db_session.commit()

    run_backfill(db_session, Discipline.LEAD)
    first_run = {
        r.athlete_id: r.mu
        for r in db_session.execute(
            select(Rating).where(Rating.discipline == Discipline.LEAD)
        ).scalars()
    }

    # Reset ratings and history
    for r in db_session.execute(select(Rating)).scalars():
        db_session.delete(r)
    for rh in db_session.execute(select(RatingHistory)).scalars():
        db_session.delete(rh)
    db_session.commit()

    run_backfill(db_session, Discipline.LEAD)
    second_run = {
        r.athlete_id: r.mu
        for r in db_session.execute(
            select(Rating).where(Rating.discipline == Discipline.LEAD)
        ).scalars()
    }

    for aid in first_run:
        assert abs(first_run[aid] - second_run[aid]) < 0.0001, (
            f"Athlete {aid}: first={first_run[aid]}, second={second_run[aid]}"
        )
