#!/usr/bin/env python3
"""One-shot: re-normalize boulder Result rows with NULL ``score_normalized``.

Fixes #168 — after the scraper regex divergence was patched, the ~2,610 pre-2018
lowercase boulder rows already in the DB still have ``score_normalized = NULL``
because the scraper skips existing Result rows on re-scrape. Walk those rows,
recompute via :func:`climbing_elo.engine.elo.normalize_boulder_score` (which has
handled the omitted-attempts case since #115), and UPDATE.

Idempotent: rows that already have ``score_normalized`` set are ignored; rows
whose raw is unparseable (rare — genuine feed garbage) stay NULL and get
reported at the end.

Run against the session pooler (port 5432) — reads a modest number of rows
and does one UPDATE per fixable row, batched.

::

    DATABASE_URL=postgresql://.../postgres uv run \\
        python scripts/renormalize_boulder_null_scores.py --dry-run
    DATABASE_URL=postgresql://.../postgres uv run \\
        python scripts/renormalize_boulder_null_scores.py
"""

from __future__ import annotations

import argparse
import logging
import sys

from sqlalchemy import select

from climbing_elo.database import SessionLocal, init_db
from climbing_elo.engine.elo import normalize_boulder_score
from climbing_elo.models import Discipline, Event, Result, Round

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

BATCH_COMMIT = 500


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report counts and sample fixes without writing to the DB.",
    )
    args = parser.parse_args()

    init_db()

    session = SessionLocal()
    try:
        rows = (
            session.execute(
                select(Result)
                .join(Round, Result.round_id == Round.id)
                .join(Event, Round.event_id == Event.id)
                .where(
                    Event.discipline == Discipline.BOULDER,
                    Result.score_normalized.is_(None),
                    Result.dns.is_(False),
                    Result.dnf.is_(False),
                    Result.raw_score.isnot(None),
                    Result.raw_score != "",
                )
            )
            .scalars()
            .all()
        )

        log.info(
            "Found %d candidate boulder rows with NULL score_normalized", len(rows)
        )

        fixed = 0
        unfixable: list[str] = []
        pending = 0

        for res in rows:
            recomputed = normalize_boulder_score(res.raw_score)
            if recomputed is None:
                unfixable.append(res.raw_score)
                continue

            if args.dry_run:
                fixed += 1
                if fixed <= 10:
                    log.info(
                        "  dry-run would fix result.id=%d raw=%r → %s",
                        res.id,
                        res.raw_score,
                        recomputed,
                    )
                continue

            res.score_normalized = recomputed
            fixed += 1
            pending += 1
            if pending >= BATCH_COMMIT:
                session.commit()
                log.info("  committed batch of %d (running total: %d)", pending, fixed)
                pending = 0

        if not args.dry_run and pending:
            session.commit()
            log.info("  committed final batch of %d", pending)

        log.info("Fixed: %d", fixed)
        log.info("Unfixable (raw stayed NULL): %d", len(unfixable))
        if unfixable:
            sample = ", ".join(repr(x) for x in unfixable[:10])
            log.info("  unfixable sample: %s", sample)
            log.info(
                "  (these are genuine feed garbage the engine parser also "
                "cannot recover; leave as NULL)"
            )

        return 0
    finally:
        session.close()


if __name__ == "__main__":
    sys.exit(main())
