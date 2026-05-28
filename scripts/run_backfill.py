#!/usr/bin/env python3
"""Run historical backfill: compute ELO ratings from all results in the DB.

Default invocation is **idempotent** — rounds with an existing pair-kind
``RatingHistory`` row are silently skipped. After an engine change that
needs *existing* rows to be recomputed (K-factor regrid, σ-formula bumps,
TPB activation, etc.), pass ``--force-reset`` to wipe the existing
ratings + history for the target discipline before recomputing.
"""

import argparse
import logging
import sys

from climbing_elo.database import init_db
from climbing_elo.engine.backfill import force_reset_for_discipline, run_backfill
from climbing_elo.models import Discipline

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

DISCIPLINE_MAP = {
    "lead": Discipline.LEAD,
    "boulder": Discipline.BOULDER,
    "speed": Discipline.SPEED,
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute ELO ratings from scraped results"
    )
    parser.add_argument(
        "--discipline",
        choices=list(DISCIPLINE_MAP.keys()),
        default="lead",
        help="Discipline to backfill (default: lead)",
    )
    parser.add_argument(
        "--force-reset",
        action="store_true",
        help=(
            "Before backfilling, DELETE every RatingHistory row attached to "
            "an event of this discipline and reset every Rating row of this "
            "discipline to engine defaults. Use after an engine change "
            "(K-regrid, σ-formula bump, TPB activation, etc.) where the "
            "existing rows were computed by the old engine — the normal "
            "backfill is idempotent and would otherwise skip them. "
            "Per-discipline scope: a reset of LEAD does not touch BOULDER "
            "or SPEED. The raw Athlete/Event/Round/Result data is untouched. "
            "RUN LOCALLY ONLY: a full reset+rebuild exceeds the GitHub "
            "Actions 60-min job timeout and will leave prod half-rebuilt. "
            "Point DATABASE_URL at the session pooler (port 5432) and run "
            "from a laptop."
        ),
    )
    args = parser.parse_args()

    discipline = DISCIPLINE_MAP[args.discipline]
    SessionFactory = init_db()

    with SessionFactory() as session:
        if args.force_reset:
            log.warning(
                "--force-reset: wiping rating_history + ratings for discipline=%s",
                args.discipline,
            )
            rows_deleted = force_reset_for_discipline(session, discipline)
            session.commit()
            log.warning(
                "--force-reset: deleted %d rating_history rows; rebuilding from scratch",
                rows_deleted,
            )

        print(f"Running backfill for {args.discipline.capitalize()} discipline...")
        report = run_backfill(session, discipline=discipline)

    print("\nBackfill complete:")
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
