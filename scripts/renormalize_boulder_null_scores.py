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

from sqlalchemy import bindparam, select, update

from climbing_elo.database import init_db
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

    # The Supabase session pooler idle-times out fast — don't hold a session
    # open across the compute loop. Phase 1: fetch (id, raw) into a plain
    # tuple list, close. Phase 2: normalize in memory. Phase 3: open a fresh
    # short-lived session per batch and issue a bulk UPDATE via `executemany`.
    Session = init_db()

    # ─── Phase 1: fetch ────────────────────────────────────────────────────
    with Session() as session:
        candidates: list[tuple[int, str]] = list(
            session.execute(
                select(Result.id, Result.raw_score)
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
        )

    log.info(
        "Found %d candidate boulder rows with NULL score_normalized",
        len(candidates),
    )

    # ─── Phase 2: normalize in memory (no DB) ─────────────────────────────
    fixes: list[dict] = []
    unfixable: list[str] = []
    for row_id, raw in candidates:
        recomputed = normalize_boulder_score(raw)
        if recomputed is None:
            unfixable.append(raw)
        else:
            fixes.append({"id": row_id, "value": recomputed})

    log.info("Fixable: %d · unfixable: %d", len(fixes), len(unfixable))
    for i, f in enumerate(fixes[:10]):
        raw = next(r for rid, r in candidates if rid == f["id"])
        log.info("  sample fix result.id=%d raw=%r → %s", f["id"], raw, f["value"])

    if args.dry_run:
        log.info("(dry-run — no writes)")
        return 0

    # ─── Phase 3: bulk UPDATE in short-lived sessions ─────────────────────
    # Use the Core-level table directly (Result.__table__) rather than the ORM
    # mapping — the ORM bulk-update path insists the parameter dict include
    # the PK column literally, and clashes with our `.where()` clause. Core
    # bypasses that machinery cleanly.
    tbl = Result.__table__
    stmt = (
        update(tbl)
        .where(tbl.c.id == bindparam("row_id"))
        .values(score_normalized=bindparam("value"))
    )
    total = 0
    for start in range(0, len(fixes), BATCH_COMMIT):
        batch = [
            {"row_id": f["id"], "value": f["value"]}
            for f in fixes[start : start + BATCH_COMMIT]
        ]
        with Session() as session:
            session.execute(stmt, batch)
            session.commit()
        total += len(batch)
        log.info("  committed batch of %d (running total: %d)", len(batch), total)

    log.info("Fixed: %d", total)
    if unfixable:
        sample = ", ".join(repr(x) for x in unfixable[:10])
        log.info("Unfixable sample: %s", sample)
        log.info(
            "  (these are genuine feed garbage the engine parser also cannot "
            "recover; leave as NULL)"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
