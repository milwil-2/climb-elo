"""One pass of the event-lifecycle state machine (#142 PR 1).

Replaces the date-window-driven cron logic (``snapshot_forecasts.py
--within-days 7`` + ``score_forecasts.py --all-unscored``) with a
transition-driven model. Each call walks every event in the active
lifecycle window and, for each gender, fires the appropriate side effect:

    status              action
    ──────────────────  ─────────────────────────────────────────────────
    LOCKED              snapshot_forecast() if no row exists
    FINISHED            score_forecast()    if no score row exists
    everything else     no-op

All actions are idempotent: the underlying ``snapshot_forecast`` /
``score_forecast`` calls use UPSERT and skip when nothing changed. Safe
to run every minute.

The lifecycle window — ``[today - LIVE_WINDOW_DAYS, today + LOCKED_WINDOW_DAYS]``
— is the only set of events for which any state transition could fire.
Anything outside is UPCOMING (too far in the future) or older-than-recently-
finished (already scored or out of scope).

Host-agnostic: callable from a CLI script (GH Actions cron), a Vercel cron,
or the Fly.io daemon loop planned in PR 4.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from climbing_elo.engine.event_status import (
    LIVE_WINDOW_DAYS,
    LOCKED_WINDOW_DAYS,
    EventStatus,
    bulk_event_status,
)
from climbing_elo.engine.forecast_scoring import score_forecast
from climbing_elo.engine.forecasting import snapshot_forecast
from climbing_elo.models import Event, EventForecast, EventForecastScore, Gender

log = logging.getLogger(__name__)


@dataclass
class TickResult:
    """Structured summary of one lifecycle pass."""

    snapshots_created: list[tuple[int, str, int]] = field(default_factory=list)
    """(event_id, gender_value, n_athletes) for each snapshot fired."""

    scores_created: list[tuple[int, str]] = field(default_factory=list)
    """(event_id, gender_value) for each score row written."""

    skipped: list[tuple[int, str, str]] = field(default_factory=list)
    """(event_id, gender_value, reason) for diagnosable no-ops."""

    def total_actions(self) -> int:
        return len(self.snapshots_created) + len(self.scores_created)


def _candidate_events(session: Session, today: date) -> list[Event]:
    """Events whose ``start_date`` is in the active lifecycle window.

    Any event outside this window can only be UPCOMING / IMMINENT (too early
    to lock) or past the FINISHED-write deadline (anything that wanted to
    score has already had a chance). The window deliberately includes a
    couple of extra LIVE-window days on either side as a safety buffer
    against clock skew.
    """
    lo = today - timedelta(days=LIVE_WINDOW_DAYS + 1)
    hi = today + timedelta(days=LOCKED_WINDOW_DAYS + 1)
    return list(
        session.execute(
            select(Event)
            .where(Event.start_date >= lo, Event.start_date <= hi)
            .order_by(Event.start_date.asc(), Event.id.asc())
        ).scalars()
    )


def _has_forecast(session: Session, event_id: int, gender: Gender) -> bool:
    """True when at least one live forecast row exists for the (event, gender)."""
    return (
        session.execute(
            select(EventForecast.id)
            .where(
                EventForecast.event_id == event_id,
                EventForecast.gender == gender,
                EventForecast.is_backfill.is_(False),
            )
            .limit(1)
        ).first()
        is not None
    )


def _has_score(session: Session, event_id: int, gender: Gender) -> bool:
    """True when a live score row exists for the (event, gender)."""
    return (
        session.execute(
            select(EventForecastScore.id)
            .where(
                EventForecastScore.event_id == event_id,
                EventForecastScore.gender == gender,
                EventForecastScore.is_backfill.is_(False),
            )
            .limit(1)
        ).first()
        is not None
    )


def _handle_locked(
    session: Session, event: Event, gender: Gender, result: TickResult
) -> None:
    """Snapshot a forecast for a LOCKED event if one doesn't exist yet."""
    if _has_forecast(session, event.id, gender):
        result.skipped.append((event.id, gender.value, "forecast-already-exists"))
        return
    rows = snapshot_forecast(
        session, event_id=event.id, gender=gender, is_backfill=False
    )
    if not rows:
        result.skipped.append((event.id, gender.value, "no-roster"))
        return
    result.snapshots_created.append((event.id, gender.value, len(rows)))
    log.info(
        "lifecycle: snapshot event=%s gender=%s n=%d",
        event.id,
        gender.value,
        len(rows),
    )


def _handle_finished(
    session: Session, event: Event, gender: Gender, result: TickResult
) -> None:
    """Score a FINISHED event if a forecast exists and no score row does yet."""
    if _has_score(session, event.id, gender):
        result.skipped.append((event.id, gender.value, "score-already-exists"))
        return
    if not _has_forecast(session, event.id, gender):
        # No pre-event forecast was ever taken — this is the "we missed it"
        # branch. Recovery is via scripts/backfill_forecasts.py (retro-replay
        # lane), not the live lane, so we surface it and move on.
        result.skipped.append((event.id, gender.value, "no-forecast-to-score"))
        return
    score_row = score_forecast(
        session, event_id=event.id, gender=gender, is_backfill=False
    )
    if score_row is None:
        # score_forecast returns None when finals aren't loaded yet — the
        # event is FINISHED by status (calendar window passed) but we haven't
        # actually scraped the final-round results into Result rows. Scrape
        # cron is the canonical backstop; the next tick after the scrape
        # catches up will resolve this.
        result.skipped.append((event.id, gender.value, "no-final-results-loaded"))
        return
    result.scores_created.append((event.id, gender.value))
    log.info("lifecycle: score event=%s gender=%s", event.id, gender.value)


def tick(session: Session, *, today: Optional[date] = None) -> TickResult:
    """Run one pass of the lifecycle state machine.

    Picks events near the lifecycle boundary, computes status via the
    canonical :func:`event_status` predicate, and fires snapshot or score
    side effects for status transitions. All actions are idempotent — calling
    ``tick`` repeatedly is safe and side-effect-free once everything is up
    to date.

    The caller controls transaction boundaries: this function only flushes
    the session via ``snapshot_forecast`` / ``score_forecast`` (which
    themselves only flush). Commit when done — atomic per-tick commit
    keeps a half-finished tick from leaving partial state.

    Returns:
        :class:`TickResult` summarizing actions taken and reasons skipped.
    """
    today = today or date.today()
    result = TickResult()

    events = _candidate_events(session, today)
    if not events:
        return result

    statuses = bulk_event_status(events, today=today, session=session)

    for event in events:
        status = statuses.get(event.id)
        for gender in (Gender.M, Gender.F):
            if status == EventStatus.LOCKED:
                _handle_locked(session, event, gender, result)
            elif status == EventStatus.FINISHED:
                _handle_finished(session, event, gender, result)
            # UPCOMING / IMMINENT / LIVE → no-op. LIVE events without a prior
            # forecast aren't auto-recovered here (see _handle_finished's
            # no-forecast-to-score branch — same reasoning).

    return result
