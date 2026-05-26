#!/usr/bin/env python3
"""Run historical backfill: compute Lead ELO ratings from all results in the DB."""
import logging
import sys

from climbing_elo.database import init_db
from climbing_elo.engine.backfill import run_backfill
from climbing_elo.models import Discipline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main() -> None:
    SessionFactory = init_db()

    with SessionFactory() as session:
        report = run_backfill(session, discipline=Discipline.LEAD)

    print(f"\nBackfill complete:")
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
