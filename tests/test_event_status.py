"""Tests for the event-lifecycle predicate (#136)."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from sqlalchemy.orm import Session

from climbing_elo.engine.event_status import (
    IMMINENT_WINDOW_DAYS,
    LIVE_WINDOW_DAYS,
    LOCKED_WINDOW_DAYS,
    EventStatus,
    bulk_event_status,
    event_status,
)
from climbing_elo.models import (
    Discipline,
    Event,
    EventTier,
    Gender,
    Result,
    Round,
    RoundType,
)


TODAY = date(2026, 5, 30)


def _make_event(db_session: Session, start_date: date, *, name: str = "Test") -> Event:
    event = Event(
        name=name,
        tier=EventTier.WORLD_CUP,
        season=start_date.year,
        start_date=start_date,
        discipline=Discipline.LEAD,
    )
    db_session.add(event)
    db_session.commit()
    return event


def _seed_final_with_result(db_session: Session, event: Event, athlete_id: int) -> None:
    rnd = Round(
        event_id=event.id,
        round_type=RoundType.FINAL,
        gender=Gender.M,
        athlete_count=1,
    )
    db_session.add(rnd)
    db_session.flush()
    db_session.add(
        Result(
            round_id=rnd.id,
            athlete_id=athlete_id,
            rank=1,
            dnf=False,
            dns=False,
        )
    )
    db_session.commit()


# ---------------------------------------------------------------------------
# event_status — date-based transitions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "days_to_start,expected",
    [
        (IMMINENT_WINDOW_DAYS + 1, EventStatus.UPCOMING),
        (IMMINENT_WINDOW_DAYS, EventStatus.IMMINENT),
        (LOCKED_WINDOW_DAYS + 1, EventStatus.IMMINENT),
        (LOCKED_WINDOW_DAYS, EventStatus.LOCKED),
        (1, EventStatus.LOCKED),
    ],
)
def test_pre_start_states(
    db_session: Session, days_to_start: int, expected: EventStatus
) -> None:
    event = _make_event(db_session, TODAY + timedelta(days=days_to_start))
    assert event_status(event, today=TODAY) == expected


def test_first_day_of_event_is_live_when_no_final_results(db_session: Session) -> None:
    event = _make_event(db_session, TODAY)
    assert event_status(event, today=TODAY, has_final_results=False) == EventStatus.LIVE


def test_last_day_of_live_window_is_live(db_session: Session) -> None:
    event = _make_event(db_session, TODAY - timedelta(days=LIVE_WINDOW_DAYS))
    assert event_status(event, today=TODAY, has_final_results=False) == EventStatus.LIVE


def test_past_live_window_is_finished(db_session: Session) -> None:
    event = _make_event(db_session, TODAY - timedelta(days=LIVE_WINDOW_DAYS + 1))
    assert (
        event_status(event, today=TODAY, has_final_results=False)
        == EventStatus.FINISHED
    )


def test_final_result_collapses_to_finished_inside_window(
    db_session: Session,
) -> None:
    """An event mid-weekend with results already in is FINISHED, not LIVE."""
    event = _make_event(db_session, TODAY - timedelta(days=1))
    assert (
        event_status(event, today=TODAY, has_final_results=True) == EventStatus.FINISHED
    )


# ---------------------------------------------------------------------------
# event_status — session-driven FINISHED detection
# ---------------------------------------------------------------------------


def test_event_status_queries_session_for_finished(
    db_session: Session, eight_athletes
) -> None:
    event = _make_event(db_session, TODAY - timedelta(days=1))
    # With no results in the DB, session-mode should report LIVE.
    assert event_status(event, today=TODAY, session=db_session) == EventStatus.LIVE
    _seed_final_with_result(db_session, event, athlete_id=eight_athletes[0].id)
    # Now FINISHED.
    assert event_status(event, today=TODAY, session=db_session) == EventStatus.FINISHED


def test_dns_only_final_does_not_count_as_finished(
    db_session: Session, eight_athletes
) -> None:
    event = _make_event(db_session, TODAY - timedelta(days=1))
    rnd = Round(
        event_id=event.id,
        round_type=RoundType.FINAL,
        gender=Gender.M,
        athlete_count=1,
    )
    db_session.add(rnd)
    db_session.flush()
    db_session.add(
        Result(
            round_id=rnd.id,
            athlete_id=eight_athletes[0].id,
            rank=None,
            dnf=False,
            dns=True,
        )
    )
    db_session.commit()
    assert event_status(event, today=TODAY, session=db_session) == EventStatus.LIVE


# ---------------------------------------------------------------------------
# bulk_event_status — batch path
# ---------------------------------------------------------------------------


def test_bulk_event_status_returns_one_per_event(
    db_session: Session, eight_athletes
) -> None:
    upcoming = _make_event(db_session, TODAY + timedelta(days=60), name="Future")
    locked = _make_event(db_session, TODAY + timedelta(days=3), name="Locked")
    live = _make_event(db_session, TODAY - timedelta(days=1), name="Live")
    finished = _make_event(db_session, TODAY - timedelta(days=2), name="Finished")
    _seed_final_with_result(db_session, finished, athlete_id=eight_athletes[0].id)

    result = bulk_event_status(
        [upcoming, locked, live, finished], today=TODAY, session=db_session
    )

    assert result == {
        upcoming.id: EventStatus.UPCOMING,
        locked.id: EventStatus.LOCKED,
        live.id: EventStatus.LIVE,
        finished.id: EventStatus.FINISHED,
    }


def test_bulk_event_status_empty_input(db_session: Session) -> None:
    assert bulk_event_status([], today=TODAY, session=db_session) == {}
