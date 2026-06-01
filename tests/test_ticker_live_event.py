"""Tests for the nav/ticker 'live event' badge population (#134 follow-up).

``_ticker_context()`` now consults the canonical ``event_status()`` predicate
to surface a live event in the sticky nav badge + the ticker scroll. Before
this change it was a hard-coded ``None`` TODO.
"""

from __future__ import annotations

from datetime import date, timedelta

from sqlalchemy.orm import Session

from climbing_elo.api.routes import _ticker_context
from climbing_elo.models import (
    Athlete,
    Discipline,
    Event,
    EventTier,
    Gender,
    Result,
    Round,
    RoundType,
)


def _make_event(
    db_session: Session,
    *,
    name: str,
    start_date: date,
    with_final_result: bool = False,
    athlete: Athlete | None = None,
) -> Event:
    event = Event(
        name=name,
        tier=EventTier.WORLD_CUP,
        season=start_date.year,
        start_date=start_date,
        discipline=Discipline.LEAD,
    )
    db_session.add(event)
    db_session.flush()
    if with_final_result:
        assert athlete is not None
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
                athlete_id=athlete.id,
                rank=1,
                dnf=False,
                dns=False,
            )
        )
    db_session.commit()
    return event


def test_no_live_event_returns_none(db_session: Session) -> None:
    """Empty DB → live_event is None and ticker doesn't crash."""
    ctx = _ticker_context(db_session)
    assert ctx["live_event"] is None
    assert "ticker_items" in ctx


def test_finished_event_does_not_surface_as_live(
    db_session: Session, eight_athletes
) -> None:
    """A past event with final results is FINISHED — not surfaced as live."""
    _make_event(
        db_session,
        name="Past WC",
        start_date=date.today() - timedelta(days=2),
        with_final_result=True,
        athlete=eight_athletes[0],
    )
    assert _ticker_context(db_session)["live_event"] is None


def test_upcoming_event_does_not_surface_as_live(db_session: Session) -> None:
    """A future event is UPCOMING — not surfaced as live."""
    _make_event(
        db_session, name="Future WC", start_date=date.today() + timedelta(days=14)
    )
    assert _ticker_context(db_session)["live_event"] is None


def test_live_event_surfaced_with_id_and_name(db_session: Session) -> None:
    """An event in the live window with no final results is surfaced."""
    event = _make_event(db_session, name="Madrid Boulder", start_date=date.today())
    live = _ticker_context(db_session)["live_event"]
    assert live is not None
    assert live["id"] == event.id
    assert live["name"] == "Madrid Boulder"


def test_most_recent_live_event_wins(db_session: Session) -> None:
    """Two LIVE events → the most-recently-started one is surfaced."""
    _make_event(db_session, name="Older", start_date=date.today() - timedelta(days=2))
    newer = _make_event(db_session, name="Newer", start_date=date.today())
    live = _ticker_context(db_session)["live_event"]
    assert live is not None
    assert live["id"] == newer.id
