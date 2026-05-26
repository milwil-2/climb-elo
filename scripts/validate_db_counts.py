#!/usr/bin/env python3
"""Print event and result counts by season + discipline.

Outputs a human-readable table that makes data-coverage gaps immediately
visible — useful for comparing the local SQLite DB against a Supabase instance
after a historical backfill.

Usage
-----
::

    # Against the default local SQLite DB
    uv run python scripts/validate_db_counts.py

    # Against a specific SQLite file
    uv run python scripts/validate_db_counts.py --db path/to/climbing_elo.db

    # Against Supabase (reads DATABASE_URL from environment)
    DATABASE_URL=postgresql://... uv run python scripts/validate_db_counts.py

Flags
-----
--db          Path to a SQLite file (ignored when DATABASE_URL is set).
--min-year    Only show seasons >= this year (default: show all).
--max-year    Only show seasons <= this year (default: show all).
--csv         Output comma-separated values instead of a human-readable table.
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

# Make src/ importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sqlalchemy import func, select

from climbing_elo.database import get_engine
from climbing_elo.models import Base, Discipline, Event, Result, Round


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print event/result counts by season+discipline."
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=None,
        help="Path to SQLite DB file (ignored when DATABASE_URL is set).",
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=None,
        help="Only show seasons >= this year.",
    )
    parser.add_argument(
        "--max-year",
        type=int,
        default=None,
        help="Only show seasons <= this year.",
    )
    parser.add_argument(
        "--csv",
        action="store_true",
        default=False,
        help="Output CSV instead of a human-readable table.",
    )
    return parser.parse_args()


# Display label for each Discipline value.
_DISC_LABEL: dict[str, str] = {
    Discipline.LEAD.value: "Lead",
    Discipline.BOULDER.value: "Boulder",
    Discipline.SPEED.value: "Speed",
    Discipline.BOULDER_LEAD.value: "BoulderLead",
}


def get_counts(
    engine,
    min_year: int | None = None,
    max_year: int | None = None,
) -> list[dict]:
    """Return a list of dicts with keys: season, discipline, events, results.

    Each row represents one (season, discipline) combination that has at least
    one event.  Rows are sorted by season ascending, then discipline.
    """
    with engine.connect() as conn:
        # Subquery: results per event via round joins
        results_per_event = (
            select(
                Event.id.label("event_id"),
                func.count(Result.id).label("result_count"),
            )
            .join(Round, Round.event_id == Event.id)
            .join(Result, Result.round_id == Round.id)
            .group_by(Event.id)
            .subquery()
        )

        stmt = (
            select(
                Event.season,
                Event.discipline,
                func.count(Event.id).label("event_count"),
                func.coalesce(func.sum(results_per_event.c.result_count), 0).label(
                    "result_count"
                ),
            )
            .outerjoin(results_per_event, results_per_event.c.event_id == Event.id)
            .group_by(Event.season, Event.discipline)
            .order_by(Event.season, Event.discipline)
        )

        if min_year is not None:
            stmt = stmt.where(Event.season >= min_year)
        if max_year is not None:
            stmt = stmt.where(Event.season <= max_year)

        rows = conn.execute(stmt).fetchall()

    return [
        {
            "season": row.season,
            "discipline": row.discipline,
            "events": row.event_count,
            "results": int(row.result_count),
        }
        for row in rows
    ]


def _print_table(rows: list[dict]) -> None:
    if not rows:
        print("No data found.")
        return

    # Column widths
    col_season = max(6, *(len(str(r["season"])) for r in rows))
    col_disc = max(
        10, *(len(_DISC_LABEL.get(r["discipline"], r["discipline"])) for r in rows)
    )
    col_events = max(6, *(len(str(r["events"])) for r in rows))
    col_results = max(7, *(len(str(r["results"])) for r in rows))

    header = (
        f"{'Season':>{col_season}}  "
        f"{'Discipline':<{col_disc}}  "
        f"{'Events':>{col_events}}  "
        f"{'Results':>{col_results}}"
    )
    sep = "-" * len(header)

    print(header)
    print(sep)

    prev_season = None
    for r in rows:
        season = r["season"]
        disc_label = _DISC_LABEL.get(r["discipline"], r["discipline"])
        if prev_season is not None and season != prev_season:
            print()  # blank line between seasons
        print(
            f"{season:>{col_season}}  "
            f"{disc_label:<{col_disc}}  "
            f"{r['events']:>{col_events}}  "
            f"{r['results']:>{col_results}}"
        )
        prev_season = season

    print(sep)
    total_events = sum(r["events"] for r in rows)
    total_results = sum(r["results"] for r in rows)
    print(
        f"{'TOTAL':>{col_season}}  "
        f"{'':>{col_disc}}  "
        f"{total_events:>{col_events}}  "
        f"{total_results:>{col_results}}"
    )


def _print_csv(rows: list[dict]) -> None:
    writer = csv.DictWriter(
        sys.stdout,
        fieldnames=["season", "discipline", "events", "results"],
        lineterminator="\n",
    )
    writer.writeheader()
    for r in rows:
        writer.writerow(
            {
                "season": r["season"],
                "discipline": _DISC_LABEL.get(r["discipline"], r["discipline"]),
                "events": r["events"],
                "results": r["results"],
            }
        )


def main() -> None:
    args = _parse_args()

    kwargs = {}
    if args.db is not None:
        kwargs["db_path"] = args.db

    engine = get_engine(**kwargs)
    # Ensure tables exist (no-op if already present).
    Base.metadata.create_all(engine)

    rows = get_counts(engine, min_year=args.min_year, max_year=args.max_year)

    if args.csv:
        _print_csv(rows)
    else:
        _print_table(rows)


if __name__ == "__main__":
    main()
