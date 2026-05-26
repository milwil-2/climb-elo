#!/usr/bin/env python3
"""Scrape upcoming (scheduled / live) IFSC events into the database.

Checks the current season and the next ``--seasons-ahead`` seasons for events
whose d_cat status is one of: scheduled, registration, live.  Inserts bare
Event rows so the /predictions route can list them; no athlete results are
stored (the predictions page falls back to the manual form in that case).

Usage
-----
    uv run python scripts/scrape_upcoming.py
    uv run python scripts/scrape_upcoming.py --discipline boulder
    uv run python scripts/scrape_upcoming.py --discipline speed --seasons-ahead 1
    uv run python scripts/scrape_upcoming.py --all-disciplines
"""
from __future__ import annotations

import argparse
import logging

import httpx

from climbing_elo.database import init_db
from climbing_elo.scraper.ifsc_api import scrape_upcoming_events

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape upcoming IFSC events for predictions"
    )
    parser.add_argument(
        "--discipline",
        choices=["lead", "boulder", "speed"],
        default="lead",
        help="Discipline to check (default: lead)",
    )
    parser.add_argument(
        "--all-disciplines",
        action="store_true",
        help="Scrape all three disciplines (lead, boulder, speed)",
    )
    def _bounded_seasons_ahead(value: str) -> int:
        n = int(value)
        if n < 0 or n > 5:
            raise argparse.ArgumentTypeError("--seasons-ahead must be 0..5")
        return n

    parser.add_argument(
        "--seasons-ahead",
        type=_bounded_seasons_ahead,
        default=2,
        metavar="N",
        help="How many seasons beyond the current year to check (0..5, default: 2)",
    )
    args = parser.parse_args()

    disciplines = ["lead", "boulder", "speed"] if args.all_disciplines else [args.discipline]

    SessionFactory = init_db()

    with httpx.Client() as client:
        for disc in disciplines:
            print(f"\nScraping upcoming {disc.capitalize()} events...")
            with SessionFactory() as session:
                report = scrape_upcoming_events(
                    client,
                    session,
                    discipline=disc,
                    seasons_ahead=args.seasons_ahead,
                )
            print(f"  Events stored:  {report.events_stored}")
            print(f"  Events skipped: {report.events_skipped} (already in DB)")
            if report.errors:
                print(f"  Errors: {len(report.errors)}")
                for err in report.errors[:10]:
                    print(f"    - {err}")


if __name__ == "__main__":
    main()
