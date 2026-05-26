#!/usr/bin/env python3
"""Run historical backfill: compute ELO ratings from all results in the DB."""
import argparse
import logging
import sys

from climbing_elo.database import init_db
from climbing_elo.engine.backfill import run_backfill
from climbing_elo.models import Discipline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ELO backfill for a discipline")
    parser.add_argument(
        "--discipline",
        type=str,
        default="lead",
        choices=["lead", "boulder"],
        help="Discipline to backfill (default: lead)",
    )
    args = parser.parse_args()

    discipline = Discipline.BOULDER if args.discipline == "boulder" else Discipline.LEAD

    SessionFactory = init_db()

    print(f"Running backfill for {args.discipline.capitalize()}...")

    with SessionFactory() as session:
        report = run_backfill(session, discipline=discipline)

    print(f"\nBackfill complete:")
    print(f"  Discipline:       {args.discipline}")
    print(f"  Events processed: {report.events_processed}")
    print(f"  Rounds processed: {report.rounds_processed}")
    print(f"  Athletes rated:   {len(report.athletes_rated)}")
    if report.errors:
        print(f"  Errors:           {len(report.errors)}")
        for err in report.errors:
            print(f"    - {err}")
        sys.exit(1)


if __name__ == "__main__":
    main()
