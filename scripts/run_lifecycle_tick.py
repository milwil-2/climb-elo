#!/usr/bin/env python3
"""Run one pass of the event-lifecycle state machine (#142 PR 1).

Walks events near the lifecycle boundary, computes status via the canonical
:func:`climbing_elo.engine.event_status.event_status` predicate, and fires
snapshot-on-LOCKED / score-on-FINISHED side effects. All actions are
idempotent — safe to run every minute.

Usage
-----
    uv run python scripts/run_lifecycle_tick.py
    uv run python scripts/run_lifecycle_tick.py --dry-run
    uv run python scripts/run_lifecycle_tick.py --today 2026-05-29

Exit codes
----------
* ``0`` on success (regardless of how many actions fired).
* ``1`` when ``DATABASE_URL`` is unset.
* ``2`` on unhandled exception (the tick is wrapped in a try/except so we
  always log the failure cleanly).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from datetime import date, datetime

from climbing_elo.database import init_db
from climbing_elo.engine.lifecycle import tick

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


def _parse_today(value: str) -> date:
    """Parse a ``--today`` value as ISO date (``YYYY-MM-DD``)."""
    return datetime.strptime(value, "%Y-%m-%d").date()


def main() -> int:
    if not os.environ.get("DATABASE_URL"):
        log.error("DATABASE_URL is required")
        return 1

    parser = argparse.ArgumentParser(
        description=(
            "Run one pass of the event-lifecycle state machine — snapshot "
            "forecasts for newly-LOCKED events and score newly-FINISHED ones."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Roll back the transaction at the end. Use to preview what would change.",
    )
    parser.add_argument(
        "--today",
        type=_parse_today,
        default=None,
        metavar="YYYY-MM-DD",
        help="Override the lifecycle 'today' for replay / test scenarios.",
    )
    args = parser.parse_args()

    SessionFactory = init_db()

    try:
        with SessionFactory() as session:
            result = tick(session, today=args.today)
            if args.dry_run:
                session.rollback()
                log.info("dry-run: rolled back, no rows persisted")
            else:
                session.commit()
    except Exception:
        log.exception("lifecycle tick failed")
        return 2

    log.info(
        "tick done — %d snapshot(s), %d score(s), %d skipped",
        len(result.snapshots_created),
        len(result.scores_created),
        len(result.skipped),
    )
    for event_id, gender, n_rows in result.snapshots_created:
        log.info("  snapshot event=%s gender=%s n=%d", event_id, gender, n_rows)
    for event_id, gender in result.scores_created:
        log.info("  score    event=%s gender=%s", event_id, gender)
    for event_id, gender, reason in result.skipped:
        log.debug("  skip     event=%s gender=%s reason=%s", event_id, gender, reason)
    return 0


if __name__ == "__main__":
    sys.exit(main())
