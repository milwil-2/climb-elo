#!/usr/bin/env python3
"""Replay historical forecasts and score them against actual results.

For each finished event in the given season range, snapshots a forecast
"as-of" the event's ``start_date`` (so ``RatingHistory`` rows from earlier
events drive the sim, never the event itself), then immediately scores it.
Both rows land in the ``is_backfill=True`` lane so the live snapshot flow is
untouched.

Usage
-----
    uv run python scripts/backfill_forecasts.py --from-season 2024 --to-season 2025
    uv run python scripts/backfill_forecasts.py --from-season 2023 --to-season 2025 \\
        --discipline lead --gender F

Exit codes
----------
* ``0`` on success (even when no events qualify).
* ``1`` when ``DATABASE_URL`` is unset or args are invalid.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from sqlalchemy import select

from climbing_elo.database import init_db
from climbing_elo.engine.forecast_scoring import score_forecast
from climbing_elo.engine.forecasting import snapshot_forecast
from climbing_elo.models import (
    Discipline,
    Event,
    Gender,
    Result,
    Round,
    RoundType,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


DISCIPLINE_MAP = {
    "lead": Discipline.LEAD,
    "boulder": Discipline.BOULDER,
    "speed": Discipline.SPEED,
}

GENDER_MAP = {
    "M": Gender.M,
    "F": Gender.F,
}

_MIN_ROSTER_SIZE = 4
_PROGRESS_EVERY = 10


def _finished_events(
    session,
    *,
    from_season: int,
    to_season: int,
    discipline: Discipline | None,
) -> list[Event]:
    """Events in the season range with at least one final-round Result."""
    base = (
        select(Event)
        .join(Round, Round.event_id == Event.id)
        .join(Result, Result.round_id == Round.id)
        .where(
            Event.season >= from_season,
            Event.season <= to_season,
            Round.round_type == RoundType.FINAL,
            Result.dns.is_(False),
        )
    )
    if discipline is not None:
        base = base.where(Event.discipline == discipline)
    base = base.distinct().order_by(Event.start_date.asc(), Event.id.asc())
    return list(session.execute(base).scalars().all())


def _backfill_roster_size(session, *, event_id: int, gender: Gender) -> int:
    """Count of distinct non-DNS athletes for this event+gender."""
    rows = session.execute(
        select(Result.athlete_id)
        .join(Round, Result.round_id == Round.id)
        .where(
            Round.event_id == event_id,
            Round.gender == gender,
            Result.dns.is_(False),
        )
        .distinct()
    ).all()
    return len(rows)


def main() -> int:
    if not os.environ.get("DATABASE_URL"):
        log.error("DATABASE_URL is required")
        return 1

    parser = argparse.ArgumentParser(
        description=(
            "Retro-replay forecasts for finished events in a season range "
            "and score them. Writes to the is_backfill=True lane only."
        ),
    )
    parser.add_argument(
        "--from-season",
        type=int,
        required=True,
        metavar="YEAR",
        help="First season to include (inclusive).",
    )
    parser.add_argument(
        "--to-season",
        type=int,
        required=True,
        metavar="YEAR",
        help="Last season to include (inclusive).",
    )
    parser.add_argument(
        "--discipline",
        choices=list(DISCIPLINE_MAP.keys()),
        default=None,
        help="Only replay this discipline (default: all 3).",
    )
    parser.add_argument(
        "--gender",
        choices=list(GENDER_MAP.keys()),
        default=None,
        help="Only replay this gender (default: both).",
    )
    args = parser.parse_args()

    if args.from_season > args.to_season:
        log.error(
            "--from-season (%d) must be <= --to-season (%d)",
            args.from_season,
            args.to_season,
        )
        return 1

    discipline = DISCIPLINE_MAP[args.discipline] if args.discipline else None
    genders: tuple[Gender, ...] = (
        (GENDER_MAP[args.gender],) if args.gender else (Gender.M, Gender.F)
    )

    SessionFactory = init_db()

    with SessionFactory() as session:
        events = _finished_events(
            session,
            from_season=args.from_season,
            to_season=args.to_season,
            discipline=discipline,
        )
        total = len(events)
        if total == 0:
            log.info(
                "no finished events in seasons [%d, %d]%s",
                args.from_season,
                args.to_season,
                f" for discipline={args.discipline}" if discipline else "",
            )
            return 0

        log.info(
            "backfilling %d event(s); seasons=[%d, %d]; discipline=%s; gender=%s",
            total,
            args.from_season,
            args.to_season,
            args.discipline or "all",
            args.gender or "both",
        )

        for i, event in enumerate(events, start=1):
            for gender in genders:
                roster_size = _backfill_roster_size(
                    session, event_id=event.id, gender=gender
                )
                if roster_size < _MIN_ROSTER_SIZE:
                    log.warning(
                        "skip: event=%s gender=%s — roster size %d < %d",
                        event.id,
                        gender.value,
                        roster_size,
                        _MIN_ROSTER_SIZE,
                    )
                    continue
                snapshot_forecast(
                    session,
                    event_id=event.id,
                    gender=gender,
                    is_backfill=True,
                    as_of_date=event.start_date,
                )
                score_forecast(
                    session,
                    event_id=event.id,
                    gender=gender,
                    is_backfill=True,
                )
                session.commit()
            if i % _PROGRESS_EVERY == 0 or i == total:
                log.info(
                    "[%d/%d] season=%d event=%s name=%s",
                    i,
                    total,
                    event.season,
                    event.id,
                    event.name,
                )

    return 0


if __name__ == "__main__":
    sys.exit(main())
