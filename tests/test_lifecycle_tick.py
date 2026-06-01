"""Tests for the event-lifecycle state machine (#142 PR 1)."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from sqlalchemy.orm import Session

from climbing_elo.engine.event_status import (
    IMMINENT_WINDOW_DAYS,
    LIVE_WINDOW_DAYS,
    LOCKED_WINDOW_DAYS,
)
from climbing_elo.engine.lifecycle import tick
from climbing_elo.models import (
    Athlete,
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


TODAY = date(2026, 5, 30)


# ---------------------------------------------------------------------------
# Seeders
# ---------------------------------------------------------------------------


def _seed_event(
    session: Session,
    *,
    name: str,
    start_date: date,
    discipline: Discipline = Discipline.LEAD,
) -> Event:
    event = Event(
        name=name,
        tier=EventTier.WORLD_CUP,
        season=start_date.year,
        start_date=start_date,
        discipline=discipline,
    )
    session.add(event)
    session.flush()
    return event


def _seed_roster_with_ratings(
    session: Session, n: int = 6, gender: Gender = Gender.M
) -> list[Athlete]:
    """Seed athletes with Ratings so snapshot_forecast can find a roster."""
    athletes = []
    for i in range(n):
        a = Athlete(
            name=f"Athlete-{gender.value}-{i}", gender=gender, nationality="USA"
        )
        session.add(a)
        athletes.append(a)
    session.flush()
    for i, a in enumerate(athletes):
        session.add(
            Rating(
                athlete_id=a.id,
                discipline=Discipline.LEAD,
                mu=1700.0 - i * 20,
                sigma=100.0,
                n_events=10,
                provisional=False,
            )
        )
    session.flush()
    return athletes


def _seed_final_results(
    session: Session, event: Event, athletes: list[Athlete], gender: Gender = Gender.M
) -> Round:
    """Seed a FINAL round + ranked Result rows so the event scores as FINISHED."""
    rnd = Round(
        event_id=event.id,
        round_type=RoundType.FINAL,
        gender=gender,
        athlete_count=len(athletes),
    )
    session.add(rnd)
    session.flush()
    for rank, a in enumerate(athletes, start=1):
        session.add(
            Result(
                round_id=rnd.id,
                athlete_id=a.id,
                rank=rank,
                raw_score=str(50 - rank),
                dns=False,
            )
        )
    session.flush()
    return rnd


def _seed_qual_results(
    session: Session, event: Event, athletes: list[Athlete], gender: Gender = Gender.M
) -> Round:
    """Seed a QUALIFICATION round so snapshot_forecast sees a confirmed roster."""
    rnd = Round(
        event_id=event.id,
        round_type=RoundType.QUALIFICATION,
        gender=gender,
        athlete_count=len(athletes),
    )
    session.add(rnd)
    session.flush()
    for rank, a in enumerate(athletes, start=1):
        session.add(
            Result(
                round_id=rnd.id,
                athlete_id=a.id,
                rank=rank,
                raw_score=str(50 - rank),
                dns=False,
            )
        )
    session.flush()
    return rnd


# ---------------------------------------------------------------------------
# Status branches — what fires for each lifecycle state
# ---------------------------------------------------------------------------


def test_upcoming_event_no_action(db_session: Session) -> None:
    """An event > 30 days out (UPCOMING) is outside the candidate window."""
    _seed_event(db_session, name="Far Future", start_date=TODAY + timedelta(days=60))
    result = tick(db_session, today=TODAY)
    assert result.total_actions() == 0
    assert result.skipped == []  # event isn't even in the candidate set


def test_imminent_event_no_action(db_session: Session) -> None:
    """An event in IMMINENT (7-30 days out) is in the window but no transition fires."""
    _seed_event(
        db_session,
        name="Imminent",
        start_date=TODAY + timedelta(days=IMMINENT_WINDOW_DAYS - 1),
    )
    result = tick(db_session, today=TODAY)
    assert result.snapshots_created == []
    assert result.scores_created == []


def test_locked_event_triggers_snapshot(db_session: Session) -> None:
    """A LOCKED event (1-7 days out) with a roster gets a forecast snapshot."""
    event = _seed_event(
        db_session,
        name="Locked WC",
        start_date=TODAY + timedelta(days=LOCKED_WINDOW_DAYS - 2),
    )
    _seed_roster_with_ratings(db_session, n=6, gender=Gender.M)
    _seed_roster_with_ratings(db_session, n=6, gender=Gender.F)
    # Confirmed roster comes from a qualification-round Result.
    quals_m = _seed_qual_results(
        db_session,
        event,
        list(db_session.query(Athlete).filter(Athlete.gender == Gender.M).all()),
    )
    quals_f = _seed_qual_results(
        db_session,
        event,
        list(db_session.query(Athlete).filter(Athlete.gender == Gender.F).all()),
        gender=Gender.F,
    )
    _ = quals_m, quals_f
    db_session.commit()

    result = tick(db_session, today=TODAY)
    db_session.commit()

    snapshot_genders = {g for _, g, _ in result.snapshots_created}
    assert snapshot_genders == {"M", "F"}
    forecasts = db_session.query(EventForecast).filter_by(event_id=event.id).all()
    assert len(forecasts) > 0


def test_locked_event_snapshot_is_idempotent(db_session: Session) -> None:
    """Calling tick twice on the same LOCKED event only snapshots once."""
    event = _seed_event(
        db_session, name="Locked Twice", start_date=TODAY + timedelta(days=3)
    )
    athletes = _seed_roster_with_ratings(db_session, n=6, gender=Gender.M)
    _seed_qual_results(db_session, event, athletes)
    db_session.commit()

    first = tick(db_session, today=TODAY)
    db_session.commit()
    second = tick(db_session, today=TODAY)
    db_session.commit()

    assert any(g == "M" for _, g, _ in first.snapshots_created)
    # Second tick skips because rows now exist.
    assert all(
        reason == "forecast-already-exists"
        for _, _, reason in second.skipped
        if _ == event.id
    )  # noqa: E501
    assert not any(g == "M" for _, g, _ in second.snapshots_created)


def test_locked_event_with_no_roster_skipped(db_session: Session) -> None:
    """LOCKED event with no Ratings and no roster source skips gracefully."""
    _seed_event(
        db_session, name="No Roster Locked", start_date=TODAY + timedelta(days=3)
    )
    # No athletes / ratings / quals — likely_competitors will return empty.
    result = tick(db_session, today=TODAY)
    db_session.commit()

    reasons = {r for _, _, r in result.skipped}
    assert "no-roster" in reasons
    assert result.snapshots_created == []


# ---------------------------------------------------------------------------
# FINISHED branch — scoring
# ---------------------------------------------------------------------------


def test_finished_event_with_forecast_triggers_score(db_session: Session) -> None:
    """A FINISHED event with a forecast and final results gets scored."""
    event = _seed_event(
        db_session,
        name="Finished WC",
        start_date=TODAY - timedelta(days=LIVE_WINDOW_DAYS + 1),
    )
    athletes = _seed_roster_with_ratings(db_session, n=6, gender=Gender.M)
    _seed_final_results(db_session, event, athletes)
    # Snapshot a forecast first (as the LOCKED tick would have done).
    from climbing_elo.engine.forecasting import snapshot_forecast

    snapshot_forecast(db_session, event_id=event.id, gender=Gender.M, is_backfill=False)
    db_session.commit()

    result = tick(db_session, today=TODAY)
    db_session.commit()

    assert (event.id, "M") in result.scores_created
    scores = db_session.query(EventForecastScore).filter_by(event_id=event.id).all()
    assert len(scores) == 1


def test_finished_event_without_forecast_skipped(db_session: Session) -> None:
    """A FINISHED event with no prior forecast row is surfaced as skipped,
    not silently snapshotted post-hoc (which would be misleading)."""
    event = _seed_event(
        db_session,
        name="Missed WC",
        start_date=TODAY - timedelta(days=LIVE_WINDOW_DAYS + 1),
    )
    athletes = _seed_roster_with_ratings(db_session, n=6, gender=Gender.M)
    _seed_final_results(db_session, event, athletes)
    db_session.commit()

    result = tick(db_session, today=TODAY)
    db_session.commit()

    reasons = {(e, g, r) for e, g, r in result.skipped}
    assert (event.id, "M", "no-forecast-to-score") in reasons
    assert result.scores_created == []


def test_finished_event_score_is_idempotent(db_session: Session) -> None:
    """Two ticks on a scored event don't write the score row twice."""
    event = _seed_event(
        db_session,
        name="Already Scored",
        start_date=TODAY - timedelta(days=LIVE_WINDOW_DAYS + 1),
    )
    athletes = _seed_roster_with_ratings(db_session, n=6, gender=Gender.M)
    _seed_final_results(db_session, event, athletes)
    from climbing_elo.engine.forecasting import snapshot_forecast

    snapshot_forecast(db_session, event_id=event.id, gender=Gender.M, is_backfill=False)
    db_session.commit()

    tick(db_session, today=TODAY)
    db_session.commit()
    second = tick(db_session, today=TODAY)
    db_session.commit()

    assert all(g != "M" for e, g in second.scores_created if e == event.id)
    scores = db_session.query(EventForecastScore).filter_by(event_id=event.id).all()
    assert len(scores) == 1


def test_finished_event_with_no_scrape_yet_skipped(db_session: Session) -> None:
    """If the event is past the LIVE window but final-round Results haven't
    been scraped yet (rare race), tick surfaces a clean skip reason."""
    event = _seed_event(
        db_session,
        name="Finished but unscraped",
        start_date=TODAY - timedelta(days=LIVE_WINDOW_DAYS + 1),
    )
    athletes = _seed_roster_with_ratings(db_session, n=6, gender=Gender.M)
    # Forecast row exists (snapshot happened pre-event) but no final Results.
    from climbing_elo.engine.forecasting import snapshot_forecast

    # Need at least a qualification roster for snapshot_forecast to fire.
    _seed_qual_results(db_session, event, athletes)
    snapshot_forecast(db_session, event_id=event.id, gender=Gender.M, is_backfill=False)
    db_session.commit()

    result = tick(db_session, today=TODAY)
    db_session.commit()

    reasons = {(e, g, r) for e, g, r in result.skipped}
    assert (event.id, "M", "no-final-results-loaded") in reasons


# ---------------------------------------------------------------------------
# Candidate window
# ---------------------------------------------------------------------------


def test_event_outside_candidate_window_is_ignored(db_session: Session) -> None:
    """Events well outside the lifecycle boundary aren't even queried."""
    # Far past — already scored or out of scope.
    _seed_event(db_session, name="Ancient", start_date=date(2014, 5, 1))
    # Far future — UPCOMING, nothing to do.
    _seed_event(db_session, name="Decade Out", start_date=date(2036, 5, 1))
    result = tick(db_session, today=TODAY)
    assert result.total_actions() == 0
    assert result.skipped == []


def test_tick_returns_empty_for_empty_db(db_session: Session) -> None:
    """Smoke test — no events means no work."""
    result = tick(db_session, today=TODAY)
    assert result.total_actions() == 0
    assert result.skipped == []


# ---------------------------------------------------------------------------
# Smoke: tick walks both genders independently
# ---------------------------------------------------------------------------


def test_tick_handles_genders_independently(db_session: Session) -> None:
    """One gender's transitions don't bleed into the other."""
    event = _seed_event(db_session, name="Split", start_date=TODAY + timedelta(days=3))
    # Only seed men's roster; women has no athletes.
    athletes_m = _seed_roster_with_ratings(db_session, n=6, gender=Gender.M)
    _seed_qual_results(db_session, event, athletes_m)
    db_session.commit()

    result = tick(db_session, today=TODAY)
    db_session.commit()

    m_actions = [g for _, g, _ in result.snapshots_created]
    assert "M" in m_actions
    # Women's branch should have skipped (no roster), not crashed.
    f_skips = [r for _, g, r in result.skipped if g == "F"]
    assert "no-roster" in f_skips


# ---------------------------------------------------------------------------
# Parametrize across status boundaries
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "days_to_start,expect_snapshot",
    [
        (-2, False),  # LIVE — not snapshotted by this tick
        (0, False),  # LIVE
        (1, True),  # LOCKED (lower bound)
        (LOCKED_WINDOW_DAYS, True),  # LOCKED (upper bound)
        (LOCKED_WINDOW_DAYS + 1, False),  # IMMINENT
        (IMMINENT_WINDOW_DAYS + 1, False),  # UPCOMING
    ],
)
def test_snapshot_only_fires_on_locked(
    db_session: Session, days_to_start: int, expect_snapshot: bool
) -> None:
    event = _seed_event(
        db_session, name="Param", start_date=TODAY + timedelta(days=days_to_start)
    )
    athletes = _seed_roster_with_ratings(db_session, n=6, gender=Gender.M)
    _seed_qual_results(db_session, event, athletes)
    db_session.commit()

    result = tick(db_session, today=TODAY)
    db_session.commit()

    snapshotted = any(g == "M" for _, g, _ in result.snapshots_created)
    assert snapshotted is expect_snapshot
