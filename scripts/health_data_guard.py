#!/usr/bin/env python3
"""Data + rating health guard (Issue #118).

Wired into ``scrape-supabase.yml`` after the backfill + combined steps so the
daily job FAILS loudly on the class of silent data bugs that previously hid for
months (#115 boulder parse failures, #117 scale drift, #95 σ-collapse). The
older ``validate_db_counts.py`` only *prints* counts; this is the real gate.

Three checks. Exits ``1`` if any **FAIL**; ``WARN`` findings are reported but do
not fail the build (used for known, deliberately-deferred states).

1. **Recoverable parse failures (FAIL)** — a non-DNS/DNF result whose stored
   ``score_normalized`` is NULL *but the current normalizer can parse its
   ``raw_score``*. This is the #115 class: the code can parse it, yet the stored
   value is NULL (stale data / a normalizer that improved without a re-backfill).
   Legitimately-unparseable rows (e.g. Lead-qualification two-route strings, for
   which the parser *also* returns None) are NOT flagged — only code-vs-data
   divergence is.

2. **Stored-vs-recomputed mismatch (WARN)** — a non-null stored score that differs
   from re-normalizing ``raw_score``. Runs **weekly** (Mondays, or ``--full``):
   it must re-fetch every stored score because a normalizer change invalidates
   old rows retroactively, and paying that ~6 MB pull daily was wasteful
   (Supabase egress). Post-#117 (decimal-only boulder raws now
   normalize to ``None`` rather than silently returning the decimal), this check
   is silent on 2025+ boulder because ``recomputed`` is ``None`` and the mismatch
   arm requires both sides non-null. Any residual mismatch surfacing in Lead/Speed
   would be a genuine drift signal — stays a WARN for now.

3. **Rating health (FAIL)** — any rating at the σ-floor (≤ ``SIGMA_FLOOR``, the
   #95 collapse signal); Lead/Boulder μ-p95 outside its calibrated elite band
   in ``MU_P95_BANDS``; or any μ_max ≥ ``MU_MAX_CEILING``. Speed + Combined
   μ-p95 are reported but not hard-failed (Speed is sparse; Combined is an
   aggregate that sits slightly higher by design).

Usage::

    DATABASE_URL=postgresql://... uv run python scripts/health_data_guard.py
    uv run python scripts/health_data_guard.py --db path/to/local.db   # SQLite
"""

from __future__ import annotations

import argparse
import sys
from datetime import date

from sqlalchemy import func, select

from climbing_elo.database import get_engine
from climbing_elo.engine.elo import normalize_boulder_score
from climbing_elo.models import Discipline, Event, Rating, Result, Round
from climbing_elo.scraper.ifsc_api import _parse_lead_score, _parse_speed_score

# --- Calibrated thresholds (see Issue #118 calibration against prod) ----------
SIGMA_FLOOR = 50.01  # ratings at/below this are σ-collapsed (#95)
# Per-discipline elite bands for the *developed* disciplines (Speed sparse,
# BL aggregate - both informational only). Boulder was recalibrated after the
# #117 ordinal-only normalization + re-backfill settled its μ-p95 at ~1887.5
# (down from 1929 at the 2026-05-28 calibration); backtest quality was
# unaffected, so the lower bound moved to 1850 rather than retuning the engine.
MU_P95_BANDS = {
    Discipline.LEAD: (1900.0, 2200.0),
    Discipline.BOULDER: (1850.0, 2200.0),
}
# No single μ should exceed this. 3100 leaves headroom over the genuine
# outlier (Garnbret, Lead μ≈2990 as of 2026-08) while still catching runaway
# inflation of the #95 class.
MU_MAX_CEILING = 3100.0
MISMATCH_TOL = 1.0  # |stored − recomputed| above this counts as a mismatch
SAMPLE = 5  # examples to print per finding


def _recompute(discipline: Discipline, raw: str | None) -> float | None:
    """Re-derive score_normalized from a stored raw_score, per discipline.

    Mirrors the scraper's ingest normalization so we can detect divergence
    between what the current code parses and what is stored.
    """
    if raw is None or not raw.strip():
        return None
    if discipline == Discipline.BOULDER:
        return normalize_boulder_score(raw)
    if discipline == Discipline.LEAD:
        return _parse_lead_score(raw)[1]
    if discipline == Discipline.SPEED:
        return _parse_speed_score(raw)[1]
    return None


def run_checks(session, full_scan: bool = True) -> tuple[list[str], list[str]]:
    """Return (failures, warnings) as human-readable lines.

    ``full_scan`` controls check 2 (stored-vs-recomputed mismatch), which must
    re-fetch every non-NULL stored score (~50k rows / ~6 MB from prod) because
    a normalizer code change can invalidate old rows retroactively. Check 1
    (recoverable parse failures) only needs the NULL-score rows - a handful -
    so it always runs. The daily scrape passes ``full_scan`` only on Mondays
    (or via ``--full``) to keep routine egress near zero.
    """
    failures: list[str] = []
    warnings: list[str] = []

    # --- Checks 1 & 2: fetch candidates server-side, recompute, classify ---
    # Check 1 candidates: non-DNS/DNF rows whose stored score is NULL.
    base_query = (
        select(
            Result.score_normalized,
            Result.raw_score,
            Event.discipline,
        )
        .join(Round, Result.round_id == Round.id)
        .join(Event, Round.event_id == Event.id)
        .where(Result.dns.is_(False), Result.dnf.is_(False))
    )
    rows = list(session.execute(base_query.where(Result.score_normalized.is_(None))))
    # Check 2 candidates: every non-NULL stored score (weekly / --full only).
    if full_scan:
        rows += list(
            session.execute(base_query.where(Result.score_normalized.is_not(None)))
        )

    recoverable: dict[Discipline, list[str]] = {}
    mismatched: dict[Discipline, list[str]] = {}
    for stored, raw, disc in rows:
        recomputed = _recompute(disc, raw)
        if stored is None and recomputed is not None:
            recoverable.setdefault(disc, []).append(f"{raw!r}→{recomputed}")
        elif (
            stored is not None
            and recomputed is not None
            and abs(stored - recomputed) > MISMATCH_TOL
        ):
            mismatched.setdefault(disc, []).append(f"{raw!r}: {stored}≠{recomputed}")

    for disc, items in recoverable.items():
        ex = ", ".join(items[:SAMPLE])
        failures.append(
            f"[parse-failure] {disc.value}: {len(items)} non-DNS/DNF result(s) "
            f"have NULL score_normalized but raw_score IS parseable "
            f"(code-vs-data divergence — re-normalize + re-backfill). e.g. {ex}"
        )
    for disc, items in mismatched.items():
        ex = ", ".join(items[:SAMPLE])
        warnings.append(
            f"[scale-drift] {disc.value}: {len(items)} stored score(s) differ from "
            f"re-normalizing raw_score (cf. #117). e.g. {ex}"
        )

    # --- Check 3: rating health -------------------------------------------
    # p95 needs a portable computation (SQLite lacks percentile_cont) — do it
    # in Python per discipline.
    disciplines = [
        d for (d,) in session.execute(select(Rating.discipline).distinct()).all()
    ]
    for disc in disciplines:
        mus = sorted(
            m
            for (m,) in session.execute(
                select(Rating.mu).where(Rating.discipline == disc)
            ).all()
        )
        n = len(mus)
        if n == 0:
            continue
        p95 = mus[min(n - 1, int(round(0.95 * (n - 1))))]
        mu_max = mus[-1]
        n_floor = session.execute(
            select(func.count(Rating.id)).where(
                Rating.discipline == disc, Rating.sigma <= SIGMA_FLOOR
            )
        ).scalar_one()
        tag = disc.value
        if n_floor > 0:
            failures.append(
                f"[rating-health] {tag}: {n_floor} rating(s) at the σ-floor "
                f"(≤{SIGMA_FLOOR}) — σ collapse (cf. #95)."
            )
        if mu_max >= MU_MAX_CEILING:
            failures.append(
                f"[rating-health] {tag}: μ_max={mu_max:.1f} ≥ {MU_MAX_CEILING}."
            )
        band = MU_P95_BANDS.get(disc)
        if band is not None and not (band[0] <= p95 <= band[1]):
            failures.append(
                f"[rating-health] {tag}: μ-p95={p95:.1f} outside band {band}."
            )
        else:
            note = "" if band is not None else " (informational)"
            print(
                f"  rating health {tag}: μ-p95={p95:.1f} μ_max={mu_max:.1f} "
                f"σ-floor={n_floor} n={n}{note}"
            )

    return failures, warnings


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Data + rating health guard (#118).")
    p.add_argument("--db", default=None, help="SQLite path (else uses DATABASE_URL).")
    p.add_argument(
        "--full",
        action="store_true",
        help="Force the full stored-vs-recomputed mismatch scan (check 2). "
        "Without it the scan runs only on Mondays; checks 1 and 3 always run.",
    )
    args = p.parse_args(argv)

    full_scan = args.full or date.today().isoweekday() == 1
    if not full_scan:
        print("  (mismatch scan skipped - weekly on Mondays; pass --full to force)")

    engine = get_engine(args.db) if args.db else get_engine()
    from sqlalchemy.orm import Session

    with Session(engine) as session:
        failures, warnings = run_checks(session, full_scan=full_scan)

    for w in warnings:
        print(f"WARN  {w}")
    for f in failures:
        print(f"FAIL  {f}")

    if failures:
        print(
            f"\n❌ health guard: {len(failures)} failure(s), {len(warnings)} warning(s)."
        )
        return 1
    print(f"\n✅ health guard passed ({len(warnings)} warning(s)).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
