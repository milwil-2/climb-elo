#!/usr/bin/env python3
"""Snapshot live Monte Carlo forecasts for upcoming events.

Walks the ``events`` table for entries starting within the next ``--within-days``
days (default 7) and freezes one row per (event, gender) via
:func:`climbing_elo.engine.forecasting.snapshot_forecast` with
``is_backfill=False``. Designed for daily cron use alongside the scrape job.

Idempotency is enforced by the
``uq_event_forecast_event_gender_athlete_backfill_version`` unique constraint
on ``event_forecasts``: re-running at the same ``engine_version_tag()``
overwrites the prior row via the upsert path, and a snapshot at a new engine
version inserts a fresh row alongside the prior-version one rather than
replacing it.

Usage
-----
    uv run python scripts/snapshot_forecasts.py
    uv run python scripts/snapshot_forecasts.py --within-days 14
    uv run python scripts/snapshot_forecasts.py --event-id 1234

Exit codes
----------
* ``0`` on success (even when no events qualify).
* ``1`` when ``DATABASE_URL`` is unset.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, timedelta

from sqlalchemy import select

from climbing_elo.database import init_db
from climbing_elo.engine.elo import engine_version_tag
from climbing_elo.engine.forecasting import snapshot_forecast
from climbing_elo.models import Event, Gender

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _events_in_window(session, *, today: date, within_days: int) -> list[Event]:
    horizon = today + timedelta(days=within_days)
    return list(
        session.execute(
            select(Event)
            .where(Event.start_date >= today, Event.start_date <= horizon)
            .order_by(Event.start_date.asc(), Event.id.asc())
        )
        .scalars()
        .all()
    )


def _snapshot_event(session, *, event: Event) -> tuple[int, int, str]:
    """Snapshot both genders for one event. Returns (m_rows, f_rows, source).

    Idempotency is delegated to the
    ``uq_event_forecast_event_gender_athlete_backfill_version`` unique
    constraint: same engine version → upsert overwrites in place; new engine
    version → insert alongside the prior row. No application-level skip
    needed.
    """
    counts = {"M": 0, "F": 0}
    sources: list[str] = []
    for gender in (Gender.M, Gender.F):
        rows = snapshot_forecast(
            session,
            event_id=event.id,
            gender=gender,
            is_backfill=False,
        )
        if len(rows) < 2:
            log.warning(
                "skip: event=%s gender=%s — roster has %d athletes (<2)",
                event.id,
                gender.value,
                len(rows),
            )
            # Roll back the partial forecast rows for this gender so we don't
            # leave dangling <2-athlete snapshots in the DB.
            for r in rows:
                session.delete(r)
            session.flush()
            continue
        counts[gender.value] = len(rows)
        # Every row in a single snapshot shares the same roster_source.
        sources.append(rows[0].roster_source)
    source = sources[0] if sources else "none"
    return counts["M"], counts["F"], source


def main() -> int:
    if not os.environ.get("DATABASE_URL"):
        log.error("DATABASE_URL is required")
        return 1

    parser = argparse.ArgumentParser(
        description="Freeze live Monte Carlo forecasts for upcoming events.",
    )
    parser.add_argument(
        "--within-days",
        type=int,
        default=7,
        metavar="N",
        help="Snapshot events starting in [today, today+N days]. Default: 7.",
    )
    parser.add_argument(
        "--event-id",
        type=int,
        default=None,
        metavar="ID",
        help=(
            "Snapshot exactly this event (both genders) regardless of date. "
            "Overwrites any existing live snapshot at the current engine "
            "version; inserts a new row alongside prior-version snapshots."
        ),
    )
    args = parser.parse_args()

    if args.within_days < 0:
        log.error("--within-days must be >= 0 (got %d)", args.within_days)
        return 1

    SessionFactory = init_db()
    engine_version = engine_version_tag()

    with SessionFactory() as session:
        if args.event_id is not None:
            event = session.get(Event, args.event_id)
            if event is None:
                log.error("event_id=%s not found", args.event_id)
                return 1
            events: list[Event] = [event]
        else:
            today = date.today()
            events = _events_in_window(
                session, today=today, within_days=args.within_days
            )

        if not events:
            log.info(
                "no upcoming events in window=[today, today+%d days]",
                args.within_days,
            )
            return 0

        log.info(
            "snapshotting %d event(s); engine_version=%s",
            len(events),
            engine_version,
        )
        for event in events:
            m_rows, f_rows, source = _snapshot_event(
                session,
                event=event,
            )
            session.commit()
            log.info(
                "snapshot: event=%s name=%s M=%d F=%d source=%s",
                event.id,
                event.name,
                m_rows,
                f_rows,
                source,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
