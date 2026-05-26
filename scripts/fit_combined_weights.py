#!/usr/bin/env python3
"""Fit learned Boulder+Lead composite weights (Issue #54).

Problem
-------
``scripts/compute_combined_ratings.py`` aggregates per-discipline ratings into a
single ``BOULDER_LEAD`` rating using the geometric mean
``sqrt(mu_boulder * mu_lead)``. This is a sensible default but has never been
validated against the actual prediction target — combined-format event
outcomes (Olympics + World Championships, the format LA 2028 will use).

Approach
--------
Generalise the geometric mean to a weighted form::

    mu_combined = mu_lead ** w_lead * mu_boulder ** w_boulder
    w_lead + w_boulder = 1

then optimise ``(w_lead, w_boulder)`` on held-out combined-format events.

Combined-format ground truth
----------------------------
The DB does not store an explicit "combined" event — the IFSC publishes
Boulder and Lead separately, even at the World Championships. We synthesise
the combined outcome the same way the Olympics do (Tokyo 2020, Paris 2024):
**multiply each athlete's per-discipline rank** to produce a combined score.
Lower score wins. Only athletes who competed in BOTH disciplines at the same
WCh year+gender are included.

Each (year, gender) becomes one "virtual combined event":

- Predicted podium probabilities come from the Monte Carlo projector applied
  to that event's roster using ``mu_combined`` (with current candidate
  weights) and the RMS-pooled ``sigma_combined``.
- Actual podium = the three athletes with the smallest rank product.

Held-out construction (no data leakage)
---------------------------------------
For each WCh year ``Y`` used as a fold:

1. Copy the production DB to a temp file.
2. Run backfill on **both** disciplines with ``end_date=date(Y, 1, 1)`` —
   ratings reflect only events strictly before that year.
3. Score every (year, gender) tuple from the years in scope.

Step 2 is the expensive one. We do it once per cutoff year and cache the
per-discipline (mu, sigma) snapshots; the grid sweep over (w_L, w_B) reuses
those snapshots without re-running backfill, so adding grid points is free.

Scoring
-------
- **Primary**: mean log-loss of *podium* prediction across all (year, gender)
  rounds. Lower is better.
- **Tie-breaker / regression check**: mean Spearman rank correlation between
  predicted ``mu_combined`` and *actual* combined rank product. Higher is
  better.

A candidate (w_L, w_B) is shipped only if:

- ``log_loss(candidate) < log_loss(baseline)`` (strict improvement), AND
- ``rank_corr(candidate) >= 0.95 * rank_corr(baseline)`` (≤ 5% regression).

If no candidate satisfies both, the geometric-mean baseline wins and no JSON
is written — ``compute_combined_ratings.py`` then falls back to its default.

CLI usage
---------

    uv run python scripts/fit_combined_weights.py
    uv run python scripts/fit_combined_weights.py --method grid --step 0.05
    uv run python scripts/fit_combined_weights.py --output data/learned_combined_weights.json

Default grid: ``w_lead`` ∈ {0.0, 0.1, …, 1.0} (constraint ``w_B = 1 - w_L``).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import statistics
import sys
import tempfile
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

# Ensure src/ is importable when run directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from climbing_elo.database import (  # noqa: E402
    DEFAULT_DB_PATH,
    get_session_factory,
    init_db,
)
from climbing_elo.engine.backfill import run_backfill  # noqa: E402
from climbing_elo.engine.elo import DEFAULT_MU, DEFAULT_SIGMA  # noqa: E402
from climbing_elo.engine.evaluation import _log_loss, _spearman  # noqa: E402
from climbing_elo.engine.projections import (  # noqa: E402
    AthleteProjectionInput,
    compute_podium_probabilities,
)
from climbing_elo.models import (  # noqa: E402
    Discipline,
    Event,
    EventTier,
    Gender,
    Result,
    Round,
    RoundType,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


# Minimum athletes a (year, gender) combined event must contain to be scored.
# Too few and Monte Carlo podium probabilities are uninformative.
MIN_ATHLETES_PER_FOLD = 6

# How many Monte Carlo sims per (year, gender, candidate). 5_000 is plenty
# given the small field sizes (≤ ~30 athletes) and the deterministic seed
# below.
N_SIMULATIONS = 5_000
RNG_SEED = 42

# Geometric-mean baseline (w_lead = w_boulder = 0.5).
BASELINE_W_LEAD = 0.5
BASELINE_W_BOULDER = 0.5

# Threshold: ship learned weights only when rank-correlation regression is
# ≤ 5% of the baseline.
RANK_CORR_REGRESSION_TOLERANCE = 0.95


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CombinedEntry:
    """One athlete's per-discipline standing at a virtual combined event."""

    athlete_id: int
    rank_boulder: int
    rank_lead: int

    @property
    def combined_score(self) -> int:
        """Rank product — Tokyo 2020 / Paris 2024 combined format."""
        return self.rank_boulder * self.rank_lead


@dataclass(frozen=True)
class CombinedFold:
    """One (year, gender) virtual combined event used as held-out data."""

    year: int
    gender: Gender
    entries: tuple[CombinedEntry, ...]


@dataclass(frozen=True)
class RatingSnapshot:
    """Per-discipline (mu, sigma) for one athlete at a given cutoff."""

    mu: float
    sigma: float


@dataclass(frozen=True)
class FitMetrics:
    """Metrics computed for one weight candidate across all folds."""

    log_loss: float
    rank_corr: float


# ---------------------------------------------------------------------------
# Fold discovery
# ---------------------------------------------------------------------------


def _best_rank_for_athlete(
    session: Session,
    event_ids: list[int],
    athlete_id: int,
    gender: Gender,
) -> int | None:
    """Return the athlete's best finishing rank across the given events.

    "Best" prefers FINAL → SEMI → QUALIFICATION (the further the athlete
    advances, the more informative their rank). Within the most-advanced
    round, we take the minimum rank value. DNS results are ignored.

    Returns ``None`` if the athlete has no scoring result across the events.
    """
    if not event_ids:
        return None
    # Pull all (round_type, rank) tuples for this athlete across the events.
    rows = session.execute(
        select(Round.round_type, Result.rank)
        .join(Result, Result.round_id == Round.id)
        .where(
            Round.event_id.in_(event_ids),
            Round.gender == gender,
            Result.athlete_id == athlete_id,
            ~Result.dns,
            Result.rank.is_not(None),
        )
    ).all()
    if not rows:
        return None

    # Most-advanced round = max ROUND_ORDER value present.
    round_priority = {RoundType.FINAL: 2, RoundType.SEMI: 1, RoundType.QUALIFICATION: 0}
    best_priority = max(round_priority[rt] for rt, _ in rows)
    candidate_ranks = [rank for rt, rank in rows if round_priority[rt] == best_priority]
    return min(candidate_ranks)


def discover_combined_folds(
    session: Session,
    tiers: tuple[EventTier, ...] = (
        EventTier.OLYMPICS,
        EventTier.WORLD_CHAMPIONSHIP,
    ),
    min_athletes: int = MIN_ATHLETES_PER_FOLD,
) -> list[CombinedFold]:
    """List every (year, gender) for which we can synthesise a combined event.

    An (year, gender) tuple is included when at least ``min_athletes`` athletes
    competed in BOTH Boulder and Lead at events of the given tiers in that
    year, with at least one non-DNS finishing result in each discipline.
    """
    # All events of the target tiers, grouped by season then discipline.
    events = list(
        session.execute(
            select(Event.id, Event.season, Event.discipline)
            .where(Event.tier.in_(tiers))
            .where(Event.discipline.in_([Discipline.BOULDER, Discipline.LEAD]))
            .order_by(Event.season.asc())
        ).all()
    )
    # season → discipline → list[event_id]
    by_season: dict[int, dict[Discipline, list[int]]] = {}
    for eid, season, discipline in events:
        by_season.setdefault(season, {}).setdefault(discipline, []).append(eid)

    folds: list[CombinedFold] = []
    for season in sorted(by_season):
        per_disc = by_season[season]
        if Discipline.BOULDER not in per_disc or Discipline.LEAD not in per_disc:
            continue
        b_events = per_disc[Discipline.BOULDER]
        l_events = per_disc[Discipline.LEAD]

        # For each gender independently, find athletes with results in BOTH.
        for gender in (Gender.M, Gender.F):
            # Athletes with at least one non-DNS rank in Boulder this year.
            b_athletes = set(
                session.execute(
                    select(Result.athlete_id)
                    .join(Round, Round.id == Result.round_id)
                    .where(
                        Round.event_id.in_(b_events),
                        Round.gender == gender,
                        ~Result.dns,
                        Result.rank.is_not(None),
                    )
                    .distinct()
                ).scalars()
            )
            l_athletes = set(
                session.execute(
                    select(Result.athlete_id)
                    .join(Round, Round.id == Result.round_id)
                    .where(
                        Round.event_id.in_(l_events),
                        Round.gender == gender,
                        ~Result.dns,
                        Result.rank.is_not(None),
                    )
                    .distinct()
                ).scalars()
            )
            shared = sorted(b_athletes & l_athletes)
            if len(shared) < min_athletes:
                continue

            entries: list[CombinedEntry] = []
            for aid in shared:
                rb = _best_rank_for_athlete(session, b_events, aid, gender)
                rl = _best_rank_for_athlete(session, l_events, aid, gender)
                if rb is None or rl is None:
                    continue
                entries.append(
                    CombinedEntry(athlete_id=aid, rank_boulder=rb, rank_lead=rl)
                )
            if len(entries) < min_athletes:
                continue
            folds.append(
                CombinedFold(year=season, gender=gender, entries=tuple(entries))
            )
    return folds


# ---------------------------------------------------------------------------
# Rating snapshots per cutoff year
# ---------------------------------------------------------------------------


def _snapshot_for_discipline(
    session: Session,
    discipline: Discipline,
    athlete_ids: set[int],
) -> dict[int, RatingSnapshot]:
    """Read post-backfill (mu, sigma) for the requested athletes.

    Missing athletes get the default (1500, 350) — same fallback the
    BacktestRunner uses.
    """
    from climbing_elo.models import Rating

    out: dict[int, RatingSnapshot] = {}
    for r in session.execute(
        select(Rating).where(
            Rating.discipline == discipline,
            Rating.athlete_id.in_(list(athlete_ids)) if athlete_ids else False,
        )
    ).scalars():
        out[r.athlete_id] = RatingSnapshot(mu=r.mu, sigma=r.sigma)
    for aid in athlete_ids:
        out.setdefault(aid, RatingSnapshot(mu=DEFAULT_MU, sigma=DEFAULT_SIGMA))
    return out


def build_snapshots(
    source_db_path: Path,
    folds: list[CombinedFold],
) -> dict[int, dict[Discipline, dict[int, RatingSnapshot]]]:
    """Build per-cutoff (year, discipline, athlete_id) → (mu, sigma) snapshots.

    For each unique fold year, we copy the source DB into a temp file, run
    backfill on Boulder + Lead with ``end_date=date(year, 1, 1)``, then read
    the resulting Rating rows for the athletes who participate in any fold.

    The temp DB is removed when the function returns.
    """
    if not folds:
        return {}

    needed_years = sorted({fold.year for fold in folds})
    athlete_ids = {entry.athlete_id for fold in folds for entry in fold.entries}

    snapshots: dict[int, dict[Discipline, dict[int, RatingSnapshot]]] = {}

    tmpdir = Path(tempfile.mkdtemp(prefix="climbing_elo_fit_combined_"))
    try:
        pristine = tmpdir / "pristine.db"
        if source_db_path.exists():
            shutil.copy(source_db_path, pristine)
        else:
            init_db(pristine)

        for year in needed_years:
            working = tmpdir / f"working_{year}.db"
            shutil.copy(pristine, working)
            factory = get_session_factory(working)
            with factory() as session:
                cutoff = date(year, 1, 1)
                log.info(
                    "Backfilling Boulder + Lead with end_date=%s for fold year %d",
                    cutoff,
                    year,
                )
                run_backfill(session, Discipline.BOULDER, end_date=cutoff)
                run_backfill(session, Discipline.LEAD, end_date=cutoff)
                session.commit()
                snapshots[year] = {
                    Discipline.BOULDER: _snapshot_for_discipline(
                        session, Discipline.BOULDER, athlete_ids
                    ),
                    Discipline.LEAD: _snapshot_for_discipline(
                        session, Discipline.LEAD, athlete_ids
                    ),
                }
            # Working DB no longer needed.
            try:
                working.unlink()
            except OSError:
                pass
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return snapshots


# ---------------------------------------------------------------------------
# Candidate scoring
# ---------------------------------------------------------------------------


def combined_mu(
    mu_lead: float,
    mu_boulder: float,
    w_lead: float,
    w_boulder: float,
) -> float:
    """Generalised geometric mean used throughout the fitter.

    Mirrors the production formula in
    :func:`scripts.compute_combined_ratings.compute_combined_mu` so the fitter
    and the production write-out always agree.
    """
    if mu_lead <= 0 or mu_boulder <= 0:
        raise ValueError(
            f"combined_mu requires positive ratings; got mu_lead={mu_lead}, mu_boulder={mu_boulder}"
        )
    return (mu_lead**w_lead) * (mu_boulder**w_boulder)


def combined_sigma(sigma_lead: float, sigma_boulder: float) -> float:
    """RMS — matches production. Independent of (w_L, w_B)."""
    return math.sqrt((sigma_lead**2 + sigma_boulder**2) / 2.0)


def score_candidate(
    folds: list[CombinedFold],
    snapshots: dict[int, dict[Discipline, dict[int, RatingSnapshot]]],
    w_lead: float,
    w_boulder: float,
    n_simulations: int = N_SIMULATIONS,
    rng_seed: int = RNG_SEED,
) -> FitMetrics:
    """Score one (w_L, w_B) candidate across all folds.

    Returns NaN metrics if no fold could be scored (e.g. all athletes missing
    snapshots — shouldn't happen in production but guards tests).
    """
    podium_true: list[int] = []
    podium_prob: list[float] = []
    spearmans: list[float] = []

    for fold in folds:
        per_disc = snapshots.get(fold.year)
        if per_disc is None:
            continue
        boulder_snap = per_disc.get(Discipline.BOULDER, {})
        lead_snap = per_disc.get(Discipline.LEAD, {})

        # Build the projection inputs (drop athletes with no snapshot at all).
        inputs: list[AthleteProjectionInput] = []
        entry_by_aid: dict[int, CombinedEntry] = {}
        for entry in fold.entries:
            b = boulder_snap.get(entry.athlete_id)
            lead = lead_snap.get(entry.athlete_id)
            if b is None or lead is None:
                continue
            mu_c = combined_mu(lead.mu, b.mu, w_lead, w_boulder)
            sigma_c = combined_sigma(lead.sigma, b.sigma)
            inputs.append(
                AthleteProjectionInput(
                    athlete_id=entry.athlete_id, mu=mu_c, sigma=sigma_c
                )
            )
            entry_by_aid[entry.athlete_id] = entry
        if len(inputs) < MIN_ATHLETES_PER_FOLD:
            continue

        probs = compute_podium_probabilities(
            inputs, n_simulations=n_simulations, rng_seed=rng_seed
        )

        # Actual combined: lowest rank-product wins.
        ranked_entries = sorted(entry_by_aid.values(), key=lambda e: e.combined_score)
        actual_podium = {e.athlete_id for e in ranked_entries[:3]}

        for inp in inputs:
            podium_true.append(int(inp.athlete_id in actual_podium))
            podium_prob.append(probs[inp.athlete_id]["podium"])

        predicted_mu = [inp.mu for inp in inputs]
        # Spearman expects "higher = better" on both sides; combined_score is
        # "lower = better" so negate it.
        actual_neg_score = [
            -entry_by_aid[inp.athlete_id].combined_score for inp in inputs
        ]
        rho = _spearman(predicted_mu, actual_neg_score)
        if not math.isnan(rho):
            spearmans.append(rho)

    if not podium_true:
        return FitMetrics(log_loss=float("nan"), rank_corr=float("nan"))

    ll = _log_loss(podium_true, podium_prob)
    rc = statistics.fmean(spearmans) if spearmans else float("nan")
    return FitMetrics(log_loss=ll, rank_corr=rc)


# ---------------------------------------------------------------------------
# Grid search
# ---------------------------------------------------------------------------


def grid_search(
    folds: list[CombinedFold],
    snapshots: dict[int, dict[Discipline, dict[int, RatingSnapshot]]],
    step: float = 0.1,
    n_simulations: int = N_SIMULATIONS,
    rng_seed: int = RNG_SEED,
) -> tuple[tuple[float, float], FitMetrics, list[tuple[float, float, FitMetrics]]]:
    """Sweep (w_lead, w_boulder=1-w_lead) on a step-spaced grid.

    Returns:
        (best_weights, best_metrics, all_evaluations)
    """
    if step <= 0 or step > 1.0:
        raise ValueError(f"step must be in (0, 1]; got {step}")

    grid: list[float] = []
    w = 0.0
    while w <= 1.0 + 1e-9:
        grid.append(round(w, 6))
        w += step

    evaluations: list[tuple[float, float, FitMetrics]] = []
    best: tuple[tuple[float, float], FitMetrics] | None = None
    for w_lead in grid:
        w_boulder = round(1.0 - w_lead, 6)
        # Skip degenerate boundary points (mu**0 = 1, collapses the signal
        # from one discipline). We allow them but tie-break on rank-corr.
        try:
            metrics = score_candidate(
                folds,
                snapshots,
                w_lead,
                w_boulder,
                n_simulations=n_simulations,
                rng_seed=rng_seed,
            )
        except ValueError:
            continue
        evaluations.append((w_lead, w_boulder, metrics))
        if math.isnan(metrics.log_loss):
            continue
        if best is None:
            best = ((w_lead, w_boulder), metrics)
            continue
        # Primary: minimise log-loss. Tie-break (within 1e-9) by max rank-corr.
        cur_ll = best[1].log_loss
        if metrics.log_loss < cur_ll - 1e-9:
            best = ((w_lead, w_boulder), metrics)
        elif abs(metrics.log_loss - cur_ll) <= 1e-9:
            if (
                not math.isnan(metrics.rank_corr)
                and not math.isnan(best[1].rank_corr)
                and metrics.rank_corr > best[1].rank_corr
            ):
                best = ((w_lead, w_boulder), metrics)

    if best is None:
        return (
            (BASELINE_W_LEAD, BASELINE_W_BOULDER),
            FitMetrics(float("nan"), float("nan")),
            evaluations,
        )
    return best[0], best[1], evaluations


# ---------------------------------------------------------------------------
# Decision rule + JSON output
# ---------------------------------------------------------------------------


def decide_ship(
    best_metrics: FitMetrics,
    baseline_metrics: FitMetrics,
    best_weights: tuple[float, float],
    tolerance: float = RANK_CORR_REGRESSION_TOLERANCE,
) -> tuple[bool, str]:
    """Decide whether to ship the learned weights.

    Returns ``(ship?, reason)``. The reason string is printed at the end of
    the CLI run so a human reviewer can see *why* the decision went the way it
    did.
    """
    # Boundary: identical weights → don't bother shipping.
    if (
        abs(best_weights[0] - BASELINE_W_LEAD) < 1e-9
        and abs(best_weights[1] - BASELINE_W_BOULDER) < 1e-9
    ):
        return False, "Best weights match the baseline (0.5, 0.5) — no shipping needed."

    if math.isnan(best_metrics.log_loss) or math.isnan(baseline_metrics.log_loss):
        return False, "Could not compute log-loss for one or both candidates."

    if best_metrics.log_loss >= baseline_metrics.log_loss:
        return (
            False,
            f"Learned log-loss ({best_metrics.log_loss:.4f}) did not improve over "
            f"baseline ({baseline_metrics.log_loss:.4f}). Keeping geometric mean.",
        )

    # Rank-correlation regression check.
    if math.isnan(best_metrics.rank_corr) or math.isnan(baseline_metrics.rank_corr):
        # Can't reason about regression — be conservative.
        return (
            False,
            "Rank-correlation NaN; cannot verify regression bound. Keeping baseline.",
        )

    if best_metrics.rank_corr < baseline_metrics.rank_corr * tolerance:
        return (
            False,
            f"Learned rank-correlation ({best_metrics.rank_corr:.4f}) regressed "
            f">{int((1 - tolerance) * 100)}% vs baseline "
            f"({baseline_metrics.rank_corr:.4f}). Keeping baseline.",
        )

    return (
        True,
        f"Learned weights beat baseline (Δlog-loss = "
        f"{baseline_metrics.log_loss - best_metrics.log_loss:.4f}, "
        f"rank-corr {best_metrics.rank_corr:.4f} vs {baseline_metrics.rank_corr:.4f}). "
        "Shipping.",
    )


def write_weights_json(
    output_path: Path,
    weights: tuple[float, float],
    metrics: FitMetrics,
    baseline_metrics: FitMetrics,
    n_folds: int,
) -> None:
    """Atomically write the learned-weights JSON consumed by the production script."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "w_lead": weights[0],
        "w_boulder": weights[1],
        "log_loss": metrics.log_loss,
        "rank_corr": metrics.rank_corr,
        "baseline_log_loss": baseline_metrics.log_loss,
        "baseline_rank_corr": baseline_metrics.rank_corr,
        "n_folds": n_folds,
        "fit_date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    # Atomic write: temp file then rename.
    tmp = output_path.with_suffix(output_path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(output_path)
    log.info("Wrote learned weights to %s", output_path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument(
        "--method",
        choices=("grid",),
        default="grid",
        help="Optimisation method. Only 'grid' is implemented today (scipy would "
        "be smoother but pulls a dep — see follow-up issue if grid proves too coarse).",
    )
    p.add_argument(
        "--step",
        type=float,
        default=0.1,
        help="Grid step for w_lead (and 1-w_lead for w_boulder). Default 0.1.",
    )
    p.add_argument(
        "--source-db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Path to the source SQLite DB (default: data/climbing_elo.db).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "data"
        / "learned_combined_weights.json",
        help="Path to write the learned weights JSON (only written if learned beats baseline).",
    )
    p.add_argument(
        "--n-simulations",
        type=int,
        default=N_SIMULATIONS,
        help=f"Monte Carlo simulations per fold per candidate (default {N_SIMULATIONS}).",
    )
    p.add_argument(
        "--min-athletes",
        type=int,
        default=MIN_ATHLETES_PER_FOLD,
        help="Minimum shared B+L athletes a (year, gender) fold must contain.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    SessionFactory = init_db(args.source_db)
    with SessionFactory() as session:
        folds = discover_combined_folds(session, min_athletes=args.min_athletes)

    log.info("Discovered %d combined-format (year, gender) folds:", len(folds))
    for fold in folds:
        log.info(
            "  %d %s — %d athletes (both B & L)",
            fold.year,
            fold.gender.value,
            len(fold.entries),
        )
    if not folds:
        log.error(
            "No combined-format folds discovered. Cannot fit weights — keeping baseline."
        )
        return 1

    log.info(
        "Building per-year rating snapshots (this runs backfill once per unique year)…"
    )
    snapshots = build_snapshots(args.source_db, folds)

    log.info(
        "Scoring baseline (geometric mean) on %d folds with n_sim=%d…",
        len(folds),
        args.n_simulations,
    )
    baseline_metrics = score_candidate(
        folds,
        snapshots,
        BASELINE_W_LEAD,
        BASELINE_W_BOULDER,
        n_simulations=args.n_simulations,
    )
    log.info(
        "Baseline: log-loss=%.4f, rank-corr=%.4f",
        baseline_metrics.log_loss,
        baseline_metrics.rank_corr,
    )

    log.info("Running grid search with step=%.2f…", args.step)
    best_weights, best_metrics, evaluations = grid_search(
        folds, snapshots, step=args.step, n_simulations=args.n_simulations
    )

    log.info("Grid sweep results (sorted by log-loss):")
    sorted_evals = sorted(
        evaluations, key=lambda e: (math.isnan(e[2].log_loss), e[2].log_loss)
    )
    for w_l, w_b, m in sorted_evals[:10]:
        log.info(
            "  w_lead=%.2f w_boulder=%.2f → log-loss=%.4f, rank-corr=%.4f",
            w_l,
            w_b,
            m.log_loss,
            m.rank_corr,
        )

    print()
    print(
        f"Best learned weights: w_lead={best_weights[0]:.4f}, w_boulder={best_weights[1]:.4f}"
    )
    print(
        f"  log-loss   = {best_metrics.log_loss:.4f}  (baseline {baseline_metrics.log_loss:.4f})"
    )
    print(
        f"  rank-corr  = {best_metrics.rank_corr:.4f}  (baseline {baseline_metrics.rank_corr:.4f})"
    )

    ship, reason = decide_ship(best_metrics, baseline_metrics, best_weights)
    print()
    print(reason)

    if ship:
        write_weights_json(
            args.output,
            best_weights,
            best_metrics,
            baseline_metrics,
            n_folds=len(folds),
        )
    else:
        # Be explicit: ensure no stale JSON contaminates production.
        if args.output.exists():
            log.warning(
                "Learned weights did not improve. NOT overwriting existing %s "
                "— delete it manually if you want to revert to the baseline.",
                args.output,
            )
        print("Learned weights did not improve over geometric mean — keeping baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
