#!/usr/bin/env python3
"""Offline boulder rating rebuild → gated push to prod.

Incident 2026-08: a ``--force-reset`` boulder backfill against the Supabase
session pooler crashed after 2 events, leaving ~94% of boulder Rating rows at
defaults and only 435 of ~26k RatingHistory rows. Two restart attempts died
the same way (pooler kills long-held connections), and chunked prod runs are
NOT equivalent to a single run — Glicko-2 volatility lives only in memory, so
each chunk re-seeds it and the ratings silently diverge.

Strategy (mirrors the documented #85/#81 offline-rebuild + swap pattern):

1. ``export``  — copy raw boulder data (athletes / boulder events / their
   rounds / their results) from prod into a local SQLite file, in small
   batched short-lived sessions (the only access pattern the pooler has
   proven to tolerate).
2. ``rebuild`` — run the full boulder backfill against the SQLite copy in
   ONE process/session: no network, volatility continuity preserved.
3. ``push``    — after sanity gates pass, replace prod's boulder
   rating_history and overwrite boulder Rating rows with the computed
   values, again in short batched sessions.
4. ``verify``  — print prod-side counts, default-μ share, and μ-p95.

Run (session pooler URL, NullPool recommended)::

    CLIMBING_ELO_DB_NULLPOOL=1 uv run --env-file .env \\
        python scripts/offline_rebuild_boulder.py --phase all
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from sqlalchemy import delete, func, insert, select, update
from sqlalchemy.exc import DBAPIError
from sqlalchemy.orm import Session as OrmSession
from sqlalchemy.orm import sessionmaker

from climbing_elo.database import get_engine, init_db
from climbing_elo.engine.backfill import run_backfill
from climbing_elo.models import (
    Athlete,
    Base,
    Discipline,
    Event,
    Rating,
    RatingHistory,
    Result,
    Round,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

LOCAL_DB = Path("data/offline_rebuild_boulder.sqlite")
BATCH_READ = 2000
BATCH_WRITE = 1000
IN_CHUNK = 300

# Sanity gates the local rebuild must pass before any prod write happens.
GATE_MIN_EVENTS = 100
GATE_MIN_HISTORY_ROWS = 15_000
GATE_P95_BAND = (1850.0, 2250.0)


def _with_retry(fn, attempts: int = 4, base_sleep: float = 3.0):
    """Run ``fn`` with retries on connection-level DB errors (fresh attempt each time)."""
    for i in range(attempts):
        try:
            return fn()
        except DBAPIError as exc:
            if i == attempts - 1:
                raise
            log.warning("DB error (attempt %d/%d): %s — retrying", i + 1, attempts, exc)
            time.sleep(base_sleep * (i + 1))


def _row_dict(obj, model, exclude: tuple[str, ...] = ()) -> dict:
    return {
        c.name: getattr(obj, c.name)
        for c in model.__table__.columns
        if c.name not in exclude
    }


def _chunks(seq: list, n: int):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def _local_factory() -> sessionmaker[OrmSession]:
    engine = get_engine(LOCAL_DB)
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)


# ─── preflight ──────────────────────────────────────────────────────────────


def preflight(ProdS: sessionmaker) -> int:
    """Print prod rating health per discipline; sanity-check scope assumptions."""
    with ProdS() as s:
        for disc in (Discipline.LEAD, Discipline.BOULDER, Discipline.SPEED):
            total = s.execute(
                select(func.count(Rating.id)).where(Rating.discipline == disc)
            ).scalar_one()
            defaults = s.execute(
                select(func.count(Rating.id)).where(
                    Rating.discipline == disc,
                    Rating.mu == 1500.0,
                    Rating.n_events == 0,
                )
            ).scalar_one()
            hist = s.execute(
                select(func.count(RatingHistory.id))
                .join(Event, RatingHistory.event_id == Event.id)
                .where(Event.discipline == disc)
            ).scalar_one()
            share = (defaults / total * 100) if total else 0.0
            log.info(
                "prod %-7s ratings=%d defaults=%d (%.1f%%) history=%d",
                disc.name,
                total,
                defaults,
                share,
                hist,
            )
            if disc != Discipline.BOULDER and total and share > 20.0:
                log.error(
                    "%s also looks broken (%.1f%% defaults) — this script only "
                    "rebuilds BOULDER; investigate before proceeding",
                    disc.name,
                    share,
                )
                return 1
    return 0


# ─── phase 1: export ────────────────────────────────────────────────────────


def export(ProdS: sessionmaker, LocalS: sessionmaker) -> int:
    if LOCAL_DB.exists():
        log.info("Removing stale local DB %s", LOCAL_DB)
        LOCAL_DB.unlink()
        # get_engine created tables on the old file; rebuild factory fresh.
    local = _local_factory()

    # Athletes — all of them (cheap, keeps FKs simple).
    def fetch_athletes():
        with ProdS() as s:
            return [
                _row_dict(o, Athlete)
                for o in s.execute(select(Athlete)).scalars().all()
            ]

    athletes = _with_retry(fetch_athletes)
    with local() as ls:
        for chunk in _chunks(athletes, BATCH_WRITE):
            ls.execute(insert(Athlete), chunk)
        ls.commit()
    log.info("Copied %d athletes", len(athletes))

    # Boulder events.
    def fetch_events():
        with ProdS() as s:
            return [
                _row_dict(o, Event)
                for o in s.execute(
                    select(Event).where(Event.discipline == Discipline.BOULDER)
                )
                .scalars()
                .all()
            ]

    events = _with_retry(fetch_events)
    event_ids = [e["id"] for e in events]
    with local() as ls:
        for chunk in _chunks(events, BATCH_WRITE):
            ls.execute(insert(Event), chunk)
        ls.commit()
    log.info("Copied %d boulder events", len(events))
    if not event_ids:
        log.error("No boulder events found in prod — aborting")
        return 1

    # Rounds of those events.
    rounds: list[dict] = []
    for id_chunk in _chunks(event_ids, IN_CHUNK):

        def fetch_rounds(ids=id_chunk):
            with ProdS() as s:
                return [
                    _row_dict(o, Round)
                    for o in s.execute(select(Round).where(Round.event_id.in_(ids)))
                    .scalars()
                    .all()
                ]

        rounds.extend(_with_retry(fetch_rounds))
    round_ids = [r["id"] for r in rounds]
    with local() as ls:
        for chunk in _chunks(rounds, BATCH_WRITE):
            ls.execute(insert(Round), chunk)
        ls.commit()
    log.info("Copied %d rounds", len(rounds))

    # Results of those rounds.
    n_results = 0
    for id_chunk in _chunks(round_ids, IN_CHUNK):

        def fetch_results(ids=id_chunk):
            with ProdS() as s:
                return [
                    _row_dict(o, Result)
                    for o in s.execute(select(Result).where(Result.round_id.in_(ids)))
                    .scalars()
                    .all()
                ]

        batch = _with_retry(fetch_results)
        with local() as ls:
            for chunk in _chunks(batch, BATCH_WRITE):
                ls.execute(insert(Result), chunk)
            ls.commit()
        n_results += len(batch)
        log.info("  results copied so far: %d", n_results)
    log.info("Export complete: %d results", n_results)
    return 0


# ─── phase 2: rebuild (local, single session) ───────────────────────────────


def rebuild(LocalS: sessionmaker) -> int:
    with LocalS() as ls:
        report = run_backfill(ls, discipline=Discipline.BOULDER)
    log.info(
        "Local rebuild: events=%d rounds=%d athletes=%d errors=%d",
        report.events_processed,
        report.rounds_processed,
        len(report.athletes_rated),
        len(report.errors),
    )
    for err in report.errors:
        log.error("  backfill error: %s", err)
    return 1 if report.errors else 0


def gates(LocalS: sessionmaker) -> int:
    """Sanity gates on the local rebuild — all must pass before push."""
    with LocalS() as ls:
        n_hist = ls.execute(select(func.count(RatingHistory.id))).scalar_one()
        n_events = ls.execute(
            select(func.count(func.distinct(RatingHistory.event_id)))
        ).scalar_one()
        mus = sorted(
            ls.execute(
                select(Rating.mu).where(
                    Rating.discipline == Discipline.BOULDER, Rating.n_events >= 5
                )
            )
            .scalars()
            .all()
        )
    if n_events < GATE_MIN_EVENTS:
        log.error(
            "GATE FAIL: only %d events with history (< %d)", n_events, GATE_MIN_EVENTS
        )
        return 1
    if n_hist < GATE_MIN_HISTORY_ROWS:
        log.error(
            "GATE FAIL: only %d history rows (< %d)", n_hist, GATE_MIN_HISTORY_ROWS
        )
        return 1
    if not mus:
        log.error("GATE FAIL: no rated athletes with n_events >= 5")
        return 1
    p95 = mus[max(0, int(0.95 * len(mus)) - 1)]
    log.info(
        "Gates: events=%d history=%d rated(n>=5)=%d mu-p95=%.1f",
        n_events,
        n_hist,
        len(mus),
        p95,
    )
    lo, hi = GATE_P95_BAND
    if not (lo <= p95 <= hi):
        log.error("GATE FAIL: mu-p95 %.1f outside [%.0f, %.0f]", p95, lo, hi)
        return 1
    log.info("All gates passed")
    return 0


# ─── phase 3: push ──────────────────────────────────────────────────────────


def push(ProdS: sessionmaker, LocalS: sessionmaker) -> int:
    # Read everything to push from the local DB first (no prod session held).
    with LocalS() as ls:
        hist_rows = [
            _row_dict(o, RatingHistory, exclude=("id",))
            for o in ls.execute(select(RatingHistory)).scalars().all()
        ]
        rating_rows = [
            _row_dict(o, Rating, exclude=("id",))
            for o in ls.execute(
                select(Rating).where(Rating.discipline == Discipline.BOULDER)
            )
            .scalars()
            .all()
        ]
        event_ids = list(ls.execute(select(Event.id)).scalars().all())
    log.info(
        "Pushing %d history rows + %d ratings across %d events",
        len(hist_rows),
        len(rating_rows),
        len(event_ids),
    )

    # 3a. Wipe existing prod boulder history (stale partial rows included).
    deleted = 0
    for id_chunk in _chunks(event_ids, IN_CHUNK):

        def do_delete(ids=id_chunk):
            with ProdS() as s:
                res = s.execute(
                    delete(RatingHistory.__table__).where(
                        RatingHistory.__table__.c.event_id.in_(ids)
                    )
                )
                s.commit()
                return res.rowcount or 0

        deleted += _with_retry(do_delete)
    log.info("Deleted %d stale prod boulder history rows", deleted)

    # 3b. Insert computed history rows.
    inserted = 0
    for chunk in _chunks(hist_rows, BATCH_WRITE):

        def do_insert(rows=chunk):
            with ProdS() as s:
                s.execute(insert(RatingHistory.__table__), rows)
                s.commit()

        _with_retry(do_insert)
        inserted += len(chunk)
        log.info("  history inserted: %d / %d", inserted, len(hist_rows))

    # 3c. Upsert boulder Rating rows.
    def fetch_existing():
        with ProdS() as s:
            return set(
                s.execute(
                    select(Rating.athlete_id).where(
                        Rating.discipline == Discipline.BOULDER
                    )
                )
                .scalars()
                .all()
            )

    existing = _with_retry(fetch_existing)
    updates = [r for r in rating_rows if r["athlete_id"] in existing]
    inserts = [r for r in rating_rows if r["athlete_id"] not in existing]

    from sqlalchemy import bindparam

    tbl = Rating.__table__
    upd = (
        update(tbl)
        .where(
            tbl.c.athlete_id == bindparam("b_aid"),
            tbl.c.discipline == Discipline.BOULDER,
        )
        .values(
            mu=bindparam("b_mu"),
            sigma=bindparam("b_sigma"),
            n_events=bindparam("b_n"),
            last_event_at=bindparam("b_last"),
            provisional=bindparam("b_prov"),
        )
    )
    done_upd = 0
    for chunk in _chunks(updates, BATCH_WRITE):
        params = [
            {
                "b_aid": r["athlete_id"],
                "b_mu": r["mu"],
                "b_sigma": r["sigma"],
                "b_n": r["n_events"],
                "b_last": r["last_event_at"],
                "b_prov": r["provisional"],
            }
            for r in chunk
        ]

        def do_update(p=params):
            with ProdS() as s:
                s.execute(upd, p)
                s.commit()

        _with_retry(do_update)
        done_upd += len(chunk)
        log.info("  ratings updated: %d / %d", done_upd, len(updates))

    for chunk in _chunks(inserts, BATCH_WRITE):

        def do_ins(rows=chunk):
            with ProdS() as s:
                s.execute(insert(Rating), rows)
                s.commit()

        _with_retry(do_ins)
    if inserts:
        log.info("  ratings inserted (no prior row): %d", len(inserts))
    log.info("Push complete")
    return 0


# ─── phase 4: verify (prod) ─────────────────────────────────────────────────


def verify(ProdS: sessionmaker) -> int:
    with ProdS() as s:
        total = s.execute(
            select(func.count(Rating.id)).where(Rating.discipline == Discipline.BOULDER)
        ).scalar_one()
        defaults = s.execute(
            select(func.count(Rating.id)).where(
                Rating.discipline == Discipline.BOULDER,
                Rating.mu == 1500.0,
                Rating.n_events == 0,
            )
        ).scalar_one()
        hist = s.execute(
            select(func.count(RatingHistory.id))
            .join(Event, RatingHistory.event_id == Event.id)
            .where(Event.discipline == Discipline.BOULDER)
        ).scalar_one()
        mus = sorted(
            s.execute(
                select(Rating.mu).where(
                    Rating.discipline == Discipline.BOULDER, Rating.n_events >= 5
                )
            )
            .scalars()
            .all()
        )
    share = (defaults / total * 100) if total else 0.0
    p95 = mus[max(0, int(0.95 * len(mus)) - 1)] if mus else float("nan")
    log.info(
        "PROD VERIFY: ratings=%d defaults=%d (%.1f%%) history=%d mu-p95(n>=5)=%.1f",
        total,
        defaults,
        share,
        hist,
        p95,
    )
    if share > 30.0 or hist < GATE_MIN_HISTORY_ROWS:
        log.error("VERIFY FAIL — prod still looks broken")
        return 1
    log.info("VERIFY OK")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--phase",
        choices=["preflight", "export", "rebuild", "push", "verify", "all"],
        required=True,
    )
    args = parser.parse_args()

    ProdS = init_db()
    LocalS = _local_factory()

    if args.phase == "preflight":
        return preflight(ProdS)
    if args.phase == "export":
        return export(ProdS, LocalS)
    if args.phase == "rebuild":
        rc = rebuild(LocalS)
        return rc or gates(LocalS)
    if args.phase == "push":
        rc = gates(LocalS)
        if rc:
            log.error("Refusing to push: gates failed")
            return rc
        return push(ProdS, LocalS)
    if args.phase == "verify":
        return verify(ProdS)

    # all: preflight → export → rebuild → gates → push → verify.
    for name, fn in (
        ("preflight", lambda: preflight(ProdS)),
        ("export", lambda: export(ProdS, LocalS)),
        ("rebuild", lambda: rebuild(LocalS)),
        ("gates", lambda: gates(LocalS)),
        ("push", lambda: push(ProdS, LocalS)),
        ("verify", lambda: verify(ProdS)),
    ):
        log.info("═══ phase: %s ═══", name)
        rc = fn()
        if rc:
            log.error(
                "Phase %s failed (rc=%d) — stopping before any further writes", name, rc
            )
            return rc
    return 0


if __name__ == "__main__":
    sys.exit(main())
