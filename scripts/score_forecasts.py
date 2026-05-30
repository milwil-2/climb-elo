#!/usr/bin/env python3
"""Score frozen Monte Carlo forecasts against actual results.

Default behaviour (``--all-unscored``): finds every ``(event, gender,
is_backfill=False)`` triple that already has rows in ``event_forecasts`` but no
matching row in ``event_forecast_scores`` AND a finished final round on file,
then calls :func:`climbing_elo.engine.forecast_scoring.score_forecast` for each.

Pass ``--event-id`` to (re-)score one event for both genders, overwriting any
existing score row via the engine's upsert. Pass ``--include-backfill`` to also
process ``is_backfill=True`` rows (used for the historical replay sweep).

Usage
-----
    uv run python scripts/score_forecasts.py
    uv run python scripts/score_forecasts.py --event-id 1234
    uv run python scripts/score_forecasts.py --include-backfill

Exit codes
----------
* ``0`` on success (even when nothing was scoreable).
* ``1`` when ``DATABASE_URL`` is unset.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

from sqlalchemy import select

from climbing_elo.database import init_db
from climbing_elo.engine.forecast_scoring import score_forecast
from climbing_elo.models import (
    Event,
    EventForecast,
    EventForecastScore,
    Gender,
    Result,
    Round,
    RoundType,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _has_final_results(session, *, event_id: int, gender: Gender) -> bool:
    row = session.execute(
        select(Result.id)
        .join(Round, Result.round_id == Round.id)
        .where(
            Round.event_id == event_id,
            Round.gender == gender,
            Round.round_type == RoundType.FINAL,
            Result.dns.is_(False),
        )
        .limit(1)
    ).first()
    return row is not None


def _unscored_triples(
    session, *, include_backfill: bool
) -> list[tuple[int, Gender, bool]]:
    """Return every (event_id, gender, is_backfill) that has forecast rows
    but no matching score row.
    """
    is_backfill_values = (False, True) if include_backfill else (False,)
    forecast_keys = session.execute(
        select(
            EventForecast.event_id,
            EventForecast.gender,
            EventForecast.is_backfill,
        )
        .where(EventForecast.is_backfill.in_(is_backfill_values))
        .distinct()
    ).all()
    score_keys = {
        (row.event_id, row.gender, row.is_backfill)
        for row in session.execute(
            select(
                EventForecastScore.event_id,
                EventForecastScore.gender,
                EventForecastScore.is_backfill,
            ).where(EventForecastScore.is_backfill.in_(is_backfill_values))
        ).all()
    }
    triples: list[tuple[int, Gender, bool]] = []
    for event_id, gender, is_backfill in forecast_keys:
        if (event_id, gender, is_backfill) in score_keys:
            continue
        triples.append((event_id, gender, is_backfill))
    return triples


def _score_event(
    session, *, event_id: int, is_backfill_modes: tuple[bool, ...]
) -> dict[str, dict[Gender, EventForecastScore | None]]:
    """Score one event across both genders for the requested modes."""
    out: dict[str, dict[Gender, EventForecastScore | None]] = {
        "live": {Gender.M: None, Gender.F: None},
        "backfill": {Gender.M: None, Gender.F: None},
    }
    for is_backfill in is_backfill_modes:
        bucket = "backfill" if is_backfill else "live"
        for gender in (Gender.M, Gender.F):
            score_row = score_forecast(
                session,
                event_id=event_id,
                gender=gender,
                is_backfill=is_backfill,
            )
            out[bucket][gender] = score_row
    return out


def _format_score_line(
    event: Event, scores: dict[Gender, EventForecastScore | None], suffix: str
) -> str:
    parts: list[str] = []
    for gender_label, gender in (("M", Gender.M), ("F", Gender.F)):
        row = scores.get(gender)
        if row is None:
            parts.append(f"{gender_label}=-")
            continue
        parts.append(
            f"{gender_label}={row.top3_intersection}/3 podium_brier={row.brier_podium:.4f}"
        )
    label = f"scored{suffix}"
    return f"{label}: event={event.id} {'; '.join(parts)}"


def main() -> int:
    if not os.environ.get("DATABASE_URL"):
        log.error("DATABASE_URL is required")
        return 1

    parser = argparse.ArgumentParser(
        description=(
            "Score frozen forecasts against actuals. Default = every "
            "unscored (event, gender, is_backfill=False) with finished finals."
        ),
    )
    parser.add_argument(
        "--all-unscored",
        action="store_true",
        default=False,
        help=(
            "Score every (event, gender, is_backfill=False) that has "
            "forecast rows, no score row, and a finished final. This is the "
            "default behaviour when --event-id is omitted."
        ),
    )
    parser.add_argument(
        "--event-id",
        type=int,
        default=None,
        metavar="ID",
        help=(
            "Score (and overwrite) this event for both genders. Honours "
            "--include-backfill."
        ),
    )
    parser.add_argument(
        "--include-backfill",
        action="store_true",
        help="Also score is_backfill=True rows.",
    )
    args = parser.parse_args()

    SessionFactory = init_db()

    modes: tuple[bool, ...] = (False, True) if args.include_backfill else (False,)

    with SessionFactory() as session:
        if args.event_id is not None:
            event = session.get(Event, args.event_id)
            if event is None:
                log.error("event_id=%s not found", args.event_id)
                return 1
            scored = _score_event(session, event_id=event.id, is_backfill_modes=modes)
            session.commit()
            log.info(_format_score_line(event, scored["live"], suffix=""))
            if args.include_backfill:
                log.info(
                    _format_score_line(event, scored["backfill"], suffix="(backfill)")
                )
            return 0

        # Default: scan everything unscored. ``--all-unscored`` is the
        # opt-in flag for clarity but the default loop runs anyway.
        triples = _unscored_triples(session, include_backfill=args.include_backfill)
        if not triples:
            log.info("nothing to score")
            return 0

        # Group by event_id so we can emit one log line per event.
        per_event: dict[int, list[tuple[Gender, bool]]] = {}
        for event_id, gender, is_backfill in triples:
            per_event.setdefault(event_id, []).append((gender, is_backfill))

        n_scored = 0
        for event_id, items in per_event.items():
            event = session.get(Event, event_id)
            if event is None:
                continue
            scored_live: dict[Gender, EventForecastScore | None] = {
                Gender.M: None,
                Gender.F: None,
            }
            scored_backfill: dict[Gender, EventForecastScore | None] = {
                Gender.M: None,
                Gender.F: None,
            }
            any_scored = False
            for gender, is_backfill in items:
                if not _has_final_results(session, event_id=event_id, gender=gender):
                    continue
                row = score_forecast(
                    session,
                    event_id=event_id,
                    gender=gender,
                    is_backfill=is_backfill,
                )
                if row is None:
                    continue
                any_scored = True
                if is_backfill:
                    scored_backfill[gender] = row
                else:
                    scored_live[gender] = row
            session.commit()
            if not any_scored:
                continue
            n_scored += 1
            if any(v is not None for v in scored_live.values()):
                log.info(_format_score_line(event, scored_live, suffix=""))
            if any(v is not None for v in scored_backfill.values()):
                log.info(
                    _format_score_line(event, scored_backfill, suffix="(backfill)")
                )

        log.info("scored %d event(s)", n_scored)

    return 0


if __name__ == "__main__":
    sys.exit(main())
