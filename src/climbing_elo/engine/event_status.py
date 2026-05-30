"""Canonical event-lifecycle predicate.

Five states an event can be in:

    UPCOMING    start_date > today + 30 days
    IMMINENT    today + 7  < start_date <= today + 30
    LOCKED      today + 1 <= start_date <= today + 7
    LIVE        start_date <= today <= start_date + LIVE_WINDOW_DAYS
                AND no non-DNS final-round Result stored
    FINISHED    at least one non-DNS Result in a final round

States are computed (not stored). Adding a column is deferred until query-time
cost on list views becomes a measurable issue. See issue #136 for the design
discussion + rationale on each threshold.

Consumers (all separately ticketed):
    - #134 — "Live view ->" link gates on status == LIVE
    - #135 — auto-start LivePoller when an event transitions into LIVE
    - _ticker_context() nav badge — populates from status == LIVE

Cancelled events are out of scope; CLAUDE.md's existing approach (leave
start_date in the past with no results) keeps them in FINISHED-or-stale
states until that ticket lands.
"""

from __future__ import annotations

import enum
from datetime import date
from typing import Iterable, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from climbing_elo.models import Event, Result, Round, RoundType


LIVE_WINDOW_DAYS = 4
LOCKED_WINDOW_DAYS = 7
IMMINENT_WINDOW_DAYS = 30


class EventStatus(str, enum.Enum):
    UPCOMING = "upcoming"
    IMMINENT = "imminent"
    LOCKED = "locked"
    LIVE = "live"
    FINISHED = "finished"


def event_status(
    event: Event,
    *,
    today: Optional[date] = None,
    has_final_results: Optional[bool] = None,
    session: Optional[Session] = None,
) -> EventStatus:
    """Derive an event's lifecycle state.

    ``has_final_results`` short-circuits the FINISHED check — pass it when
    batching over many events to avoid N+1 EXISTS subqueries. If omitted and
    ``session`` is given, the predicate runs the query itself. If both are
    omitted the FINISHED check is skipped — the predicate may incorrectly
    return LIVE for an event that has actually concluded; callers in that
    state should treat it as a hint rather than authoritative.
    """
    today = today or date.today()

    if has_final_results is None and session is not None:
        has_final_results = _has_final_results(session, event.id)

    days_to_start = (event.start_date - today).days

    if days_to_start > IMMINENT_WINDOW_DAYS:
        return EventStatus.UPCOMING
    if days_to_start > LOCKED_WINDOW_DAYS:
        return EventStatus.IMMINENT
    if days_to_start >= 1:
        return EventStatus.LOCKED

    days_since_start = (today - event.start_date).days
    if days_since_start <= LIVE_WINDOW_DAYS and not has_final_results:
        return EventStatus.LIVE
    return EventStatus.FINISHED


def bulk_event_status(
    events: Iterable[Event],
    *,
    today: Optional[date] = None,
    session: Optional[Session] = None,
) -> dict[int, EventStatus]:
    """Batch variant — one EXISTS query for all events, no N+1.

    Returns ``{event_id: EventStatus}``. Pass a session to enable the
    FINISHED check; without it, recently-started events that have actually
    finished may be tagged LIVE.
    """
    today = today or date.today()
    events = list(events)

    finished_ids: set[int] = set()
    if session is not None and events:
        finished_ids = set(
            session.execute(
                select(Round.event_id)
                .join(Result, Result.round_id == Round.id)
                .where(
                    Round.event_id.in_([e.id for e in events]),
                    Round.round_type == RoundType.FINAL,
                    Result.dns.is_(False),
                )
                .distinct()
            ).scalars()
        )

    return {
        event.id: event_status(
            event, today=today, has_final_results=(event.id in finished_ids)
        )
        for event in events
    }


def _has_final_results(session: Session, event_id: int) -> bool:
    """True when at least one non-DNS Result exists in a final round."""
    return (
        session.execute(
            select(Result.id)
            .join(Round, Result.round_id == Round.id)
            .where(
                Round.event_id == event_id,
                Round.round_type == RoundType.FINAL,
                Result.dns.is_(False),
            )
            .limit(1)
        ).first()
        is not None
    )
