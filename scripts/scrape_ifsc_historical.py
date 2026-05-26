#!/usr/bin/env python3
"""Historical IFSC results scraper — walks all seasons from 2012 to present.

This script is a dedicated historical backfill tool that iterates every season
in the ifsc.results.info API for a given year range and populates the database
with results for all three disciplines (lead, boulder, speed).

Key behaviours
--------------
- **Idempotent**: uses the existing ``(name, season, discipline)`` unique
  constraint on the ``events`` table and the per-athlete-round duplicate
  check in ``scrape_season`` to skip rows that already exist.
- **Configurable delay**: ``--delay-ms`` (default 200 ms) inserts a sleep
  between API requests to avoid hammering the IFSC server.
- **All disciplines**: scrapes lead, boulder, and speed in a single pass
  unless ``--discipline`` limits the run.
- **DATABASE_URL**: if set in the environment the script targets Postgres
  (Supabase); otherwise falls back to the local SQLite file at
  ``data/climbing_elo.db``.

Usage
-----
::

    uv run python scripts/scrape_ifsc_historical.py
    uv run python scripts/scrape_ifsc_historical.py --min-year 2018 --max-year 2022
    uv run python scripts/scrape_ifsc_historical.py --discipline lead --delay-ms 300
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Make src/ importable when the script is run directly (not installed).
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import climbing_elo.scraper.ifsc_api as ifsc_api
from climbing_elo.database import init_db

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

_ALL_DISCIPLINES = ("lead", "boulder", "speed")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Historical IFSC backfill: walks every season from --min-year to"
            " --max-year and scrapes finished events into the database."
        )
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=2012,
        help="Earliest season to scrape (default: 2012)",
    )
    parser.add_argument(
        "--max-year",
        type=int,
        default=2026,
        help="Latest season to scrape (default: 2026)",
    )
    parser.add_argument(
        "--discipline",
        choices=list(_ALL_DISCIPLINES) + ["all"],
        default="all",
        help=(
            "Discipline to scrape: lead, boulder, speed, or all (default: all)."
        ),
    )
    parser.add_argument(
        "--delay-ms",
        type=int,
        default=200,
        metavar="MS",
        help="Milliseconds to sleep between API requests (default: 200)",
    )
    return parser.parse_args()


def _run_scrape(
    min_year: int,
    max_year: int,
    disciplines: tuple[str, ...],
    delay_seconds: float,
) -> None:
    """Execute the historical scrape for the requested year range and disciplines.

    Temporarily overrides ``ifsc_api.REQUEST_DELAY`` with *delay_seconds* so
    that ``scrape_season`` (which calls ``time.sleep(REQUEST_DELAY)``) respects
    the user-supplied delay without requiring an API change.
    """
    original_delay = ifsc_api.REQUEST_DELAY
    ifsc_api.REQUEST_DELAY = delay_seconds

    try:
        SessionFactory = init_db()

        totals: dict[str, ifsc_api.ScrapeReport] = {}
        for discipline in disciplines:
            log.info(
                "=== Starting historical scrape: %s, seasons %d–%d, delay %.3fs ===",
                discipline.upper(),
                min_year,
                max_year,
                delay_seconds,
            )

            with SessionFactory() as session:
                report = ifsc_api.scrape_all_seasons(
                    session,
                    min_year=min_year,
                    max_year=max_year,
                    discipline=discipline,
                )
            totals[discipline] = report

            log.info(
                "=== %s done: %d seasons, %d events, %d results, %d athletes ===",
                discipline.upper(),
                report.seasons_scraped,
                report.events_scraped,
                report.results_created,
                report.athletes_created,
            )
            if report.errors:
                log.warning("%d errors for %s:", len(report.errors), discipline)
                for err in report.errors[:20]:
                    log.warning("  %s", err)

            # Brief extra pause between disciplines (not strictly needed but polite).
            if len(disciplines) > 1:
                time.sleep(delay_seconds)

        # Summary table
        print("\n=== Historical scrape summary ===")
        print(f"{'Discipline':<12} {'Seasons':>8} {'Events':>8} {'Results':>10} {'Athletes':>10}")
        print("-" * 52)
        for disc, r in totals.items():
            print(
                f"{disc.capitalize():<12} {r.seasons_scraped:>8} {r.events_scraped:>8}"
                f" {r.results_created:>10} {r.athletes_created:>10}"
            )
        total_events = sum(r.events_scraped for r in totals.values())
        total_results = sum(r.results_created for r in totals.values())
        print("-" * 52)
        print(
            f"{'TOTAL':<12} {'':>8} {total_events:>8} {total_results:>10}"
        )

    finally:
        ifsc_api.REQUEST_DELAY = original_delay


def main() -> None:
    args = _parse_args()

    if args.min_year > args.max_year:
        log.error(
            "--min-year (%d) must be ≤ --max-year (%d)", args.min_year, args.max_year
        )
        sys.exit(1)

    disciplines: tuple[str, ...] = (
        _ALL_DISCIPLINES if args.discipline == "all" else (args.discipline,)
    )
    delay_seconds = args.delay_ms / 1000.0

    print(
        f"Historical IFSC scrape: {args.min_year}–{args.max_year}, "
        f"disciplines={', '.join(disciplines)}, delay={args.delay_ms}ms"
    )
    print("This may take several minutes. Progress is logged to stderr.\n")

    _run_scrape(
        min_year=args.min_year,
        max_year=args.max_year,
        disciplines=disciplines,
        delay_seconds=delay_seconds,
    )


if __name__ == "__main__":
    main()
