#!/usr/bin/env python3
"""K_FACTOR_TABLE regrid sweep for the post-Glicko-2 (#51) effective-K regime.

Background
----------

Issue #51 wired Glicko-2's RD (φ) into the per-pair ELO update. The effective K
used by each pairwise contribution is now ::

    K_eff = K_base(tier, round) · g(φ_opp) · margin_mult

— variable per opponent rather than constant. The legacy ``K_FACTOR_TABLE``
values (grid-tuned under constant K + ``PROVISIONAL_K_MULTIPLIER=2``) were
halved as a conservative starting point but never properly retuned. Symptom:
elite μ values inflated to ~3000-3400 (vs the historical 1500-1900 regime).

Strategy
--------

**Coordinate descent** over the K table. A full Cartesian product over
~16 cells × 7 grid points is intractable (~33B evals). Instead, hold all
other cells at their current value, sweep ONE cell, pick the winner, then
move to the next cell. Repeat for ``--passes`` passes until convergence.

The sweep grid is multiplicative around the current value: ::

    {0.5x, 0.75x, 1.0x, 1.5x, 2.0x}  (default)

Per grid point, the script:

1. Builds an :class:`EloConfig` with the candidate K table (no monkey-patching).
2. Restores a pristine copy of the source DB to a temp working copy.
3. Wipes ratings / rating-history and re-runs backfill for each discipline
   with ``config=trial_config``.
4. Scores every holdout round via :func:`engine.evaluation._score_split_events`
   reusing :func:`compute_podium_probabilities` for probabilistic metrics.
5. Computes the **μ-range stats** (min / p50 / p95 / p99 / max) from the
   trained Rating rows — the secondary acceptance gate alongside top-3 hit
   rate.

The winner per cell maximises ``top-3 hit-rate`` subject to ``μ-p95`` lying in
the target band ``[--mu-p95-min, --mu-p95-max]`` (default 1900-2200). When no
in-band candidate beats the current cell's baseline by ``>= --tolerance pp``,
the cell is held at its current value.

Output
------

* ``docs/K_REGRID_REPORT.md`` — per-cell sweep tables + the recommended K
  table as a paste-ready Python dict literal.

Does **not** modify ``src/climbing_elo/engine/elo.py``. Applying the
recommended values is a separate, reviewed manual step.

Usage
-----

::

    # Default 5-point grid × 2 passes (~40 min on the local DB).
    uv run python scripts/regrid_k_factors.py --db data/climbing_elo.db

    # Faster smoke run — 3 points × 1 pass.
    uv run python scripts/regrid_k_factors.py --db data/climbing_elo.db \
        --grid 0.5,1.0,2.0 --passes 1 --n-sims 1000

    # Custom output path:
    uv run python scripts/regrid_k_factors.py --db data/climbing_elo.db \
        --output docs/K_REGRID_REPORT.md
"""

from __future__ import annotations

import argparse
import copy
import logging
import shutil
import statistics
import sys
import tempfile
import time
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from climbing_elo.database import get_session_factory
from climbing_elo.engine.backfill import run_backfill
from climbing_elo.engine.elo import DEFAULT_CONFIG, EloConfig
from climbing_elo.engine.evaluation import (
    BacktestDataset,
    BacktestRunner,
    HoldoutMode,
    _aggregate_metrics,
)
from climbing_elo.models import (
    Discipline,
    Event,
    Rating,
    RatingHistory,
    Round,
    RoundType,
    EventTier,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_GRID: tuple[float, ...] = (0.5, 0.75, 1.0, 1.5, 2.0)
DEFAULT_PASSES = 2
DEFAULT_HOLDOUT_SEASONS = 2
DEFAULT_N_SIMS = 2_000
DEFAULT_RNG_SEED = 2026
DEFAULT_MU_P95_MIN = 1900.0
DEFAULT_MU_P95_MAX = 2200.0
DEFAULT_TOLERANCE_PP = 0.0

# Order cells from most to least populated — gives coordinate descent the
# biggest signal-first and lets us bail early on near-empty cells.
ALL_TIERS: tuple[EventTier, ...] = (
    EventTier.WORLD_CUP,
    EventTier.WORLD_CHAMPIONSHIP,
    EventTier.CONTINENTAL,
    EventTier.OLYMPICS,
)
ALL_ROUND_TYPES: tuple[RoundType, ...] = (
    RoundType.FINAL,
    RoundType.QUALIFICATION,
    RoundType.SEMI,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SweepResult:
    """One (tier, round_type, multiplier) trial's metric outcome."""

    tier: EventTier
    round_type: RoundType
    multiplier: float
    k_value: float
    top3_hit_rate: float
    log_loss_podium: float
    log_loss_win: float
    brier_podium: float
    mu_min: float
    mu_p50: float
    mu_p95: float
    mu_p99: float
    mu_max: float


@dataclass
class CellRecommendation:
    """Per-cell sweep summary + winning multiplier."""

    tier: EventTier
    round_type: RoundType
    baseline_k: float
    recommended_k: float
    baseline_top3: float
    recommended_top3: float
    baseline_mu_p95: float
    recommended_mu_p95: float
    in_band: bool
    held_unchanged: bool
    trials: list[SweepResult] = field(default_factory=list)


@dataclass
class RegridReport:
    """End-to-end regrid output."""

    started_at: str
    finished_at: str
    grid: tuple[float, ...]
    passes: int
    n_sims: int
    rng_seed: int
    holdout_seasons: int
    mu_p95_min: float
    mu_p95_max: float
    tolerance_pp: float
    disciplines: tuple[Discipline, ...]
    initial_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    final_metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    cells: list[CellRecommendation] = field(default_factory=list)
    recommended_k_table: dict[EventTier, dict[RoundType, float]] = field(
        default_factory=dict
    )
    baseline_k_table: dict[EventTier, dict[RoundType, float]] = field(
        default_factory=dict
    )


# ---------------------------------------------------------------------------
# DB harness — copy + isolated session for one trial
# ---------------------------------------------------------------------------


class _WorkingDB:
    """Maintain a pristine DB copy + a working copy that can be reset cheaply.

    Lifecycle:

    * ``__enter__`` copies the source DB to ``pristine.db`` in a temp dir.
    * ``reset()`` (called once per trial) copies pristine → working.
    * ``session()`` opens a SQLAlchemy session bound to the working copy.
    * ``__exit__`` removes the temp directory.
    """

    def __init__(self, source: Path):
        self.source = source
        self._tmpdir: Path | None = None
        self._pristine: Path | None = None
        self._working: Path | None = None

    def __enter__(self) -> "_WorkingDB":
        self._tmpdir = Path(tempfile.mkdtemp(prefix="climbing_elo_regrid_"))
        self._pristine = self._tmpdir / "pristine.db"
        self._working = self._tmpdir / "working.db"
        shutil.copy(self.source, self._pristine)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._tmpdir is not None:
            shutil.rmtree(self._tmpdir, ignore_errors=True)

    def reset(self) -> None:
        assert self._pristine is not None and self._working is not None
        shutil.copy(self._pristine, self._working)

    def session(self) -> Session:
        assert self._working is not None
        factory = get_session_factory(self._working)
        return factory()


# ---------------------------------------------------------------------------
# Trial runner — one EloConfig → metrics + μ stats
# ---------------------------------------------------------------------------


def _wipe_ratings(session: Session) -> None:
    """Clear Rating + RatingHistory before re-running backfill with a new config."""
    session.execute(delete(RatingHistory))
    session.execute(delete(Rating))
    session.commit()


def _populated_cells(
    source_db: Path, disciplines: Iterable[Discipline]
) -> set[tuple[EventTier, RoundType]]:
    """Inspect the DB and return the set of (tier, round_type) cells with rounds.

    Used by the regrid driver to skip sweep trials over cells whose K value
    can't possibly influence the metrics — those cells stay at their current
    value (and are emitted as ``UNCHANGED — no data`` in the report).
    """
    factory = get_session_factory(source_db)
    with factory() as s:
        rows = s.execute(
            select(Event.tier, Round.round_type, func.count(Round.id))
            .join(Event, Event.id == Round.event_id)
            .where(Event.discipline.in_(list(disciplines)))
            .group_by(Event.tier, Round.round_type)
        ).all()
    return {(t, rt) for t, rt, n in rows if n > 0}


def _percentiles(values: list[float], qs: Iterable[float]) -> list[float]:
    """Inclusive percentiles via :func:`statistics.quantiles` (n=100)."""
    if not values:
        return [float("nan")] * len(list(qs))
    if len(values) == 1:
        return [values[0]] * len(list(qs))
    qs = list(qs)
    quants = statistics.quantiles(values, n=100, method="inclusive")
    # quants is length 99 corresponding to percentiles 1..99 inclusive.
    out = []
    for q in qs:
        if q <= 0:
            out.append(min(values))
        elif q >= 100:
            out.append(max(values))
        else:
            # quants index i corresponds to percentile (i+1)
            idx = max(0, min(98, int(round(q)) - 1))
            out.append(quants[idx])
    return out


def _collect_mu_stats(
    session: Session, disciplines: Iterable[Discipline]
) -> dict[str, float]:
    """Return min / p50 / p95 / p99 / max of post-backfill μ across disciplines.

    Only athletes meeting ``n_events >= 3`` (the established-athlete bar used by
    the leaderboard) are included — provisional athletes carry cold-start noise
    that doesn't reflect the elite-inflation we care about.
    """
    mus: list[float] = []
    for d in disciplines:
        rows = session.execute(
            select(Rating.mu, Rating.n_events).where(Rating.discipline == d)
        ).all()
        mus.extend(r[0] for r in rows if (r[1] or 0) >= 3)
    if not mus:
        return {
            "mu_min": float("nan"),
            "mu_p50": float("nan"),
            "mu_p95": float("nan"),
            "mu_p99": float("nan"),
            "mu_max": float("nan"),
        }
    p50, p95, p99 = _percentiles(mus, (50, 95, 99))
    return {
        "mu_min": min(mus),
        "mu_p50": p50,
        "mu_p95": p95,
        "mu_p99": p99,
        "mu_max": max(mus),
    }


def _run_trial(
    db: _WorkingDB,
    trial_config: EloConfig,
    disciplines: tuple[Discipline, ...],
    holdout_seasons: int,
    n_sims: int,
    rng_seed: int,
) -> dict[str, Any]:
    """Re-backfill + score every holdout round under ``trial_config``.

    Returns a dict with aggregate metrics + μ stats. Side-effect-free w.r.t.
    the source DB — only the temp working copy is mutated.

    Strategy: pre-run backfill with the custom config (the harness's own
    backfill call would otherwise drop our config), then hand the session
    to :class:`BacktestRunner` with ``in_memory_session=session``. The
    runner's idempotency guard in :func:`run_backfill` will skip every
    already-rated round, so the scoring stage sees our trained state.
    """
    db.reset()

    all_predictions = []
    with db.session() as session:
        for discipline in disciplines:
            mode = HoldoutMode(n_seasons=holdout_seasons)
            splits = mode.splits(session, discipline)
            if not splits:
                continue
            split = splits[0]

            _wipe_ratings(session)
            run_backfill(
                session,
                discipline,
                end_date=split.train_end_date,
                config=trial_config,
            )
            session.commit()

            dataset = BacktestDataset(
                disciplines=(discipline,),
                n_simulations=n_sims,
                rng_seed=rng_seed,
            )
            runner = BacktestRunner(
                dataset=dataset,
                variant="current",
                oos_mode=mode,
                in_memory_session=session,
            )
            # ``run()`` writes nothing because output_dir is None. It calls
            # run_backfill again (no-op due to idempotency) then scores the
            # eval rounds and returns a BacktestReport with split-level
            # predictions wrapped up. We re-collect at the per-prediction
            # level by re-invoking the scoring method to keep the metrics
            # aggregation centralised.
            engine = runner.engine_factory(session)
            preds = runner._score_split_events(  # type: ignore[attr-defined]
                session, engine, split, discipline
            )
            all_predictions.extend(preds)

        mu_stats = _collect_mu_stats(session, disciplines)

    metrics = _aggregate_metrics(all_predictions)
    metrics.update(mu_stats)
    return metrics


# ---------------------------------------------------------------------------
# Coordinate descent over the K table
# ---------------------------------------------------------------------------


def _trial_table(
    base_table: dict[EventTier, dict[RoundType, float]],
    tier: EventTier,
    round_type: RoundType,
    multiplier: float,
) -> dict[EventTier, dict[RoundType, float]]:
    """Return a deep copy of ``base_table`` with one cell scaled by ``multiplier``."""
    new = copy.deepcopy(base_table)
    new[tier][round_type] = base_table[tier][round_type] * multiplier
    return new


def _cell_score(metrics: dict[str, Any]) -> float:
    """Selection metric — higher is better. Top-3 hit rate is the v1 single objective."""
    v = metrics.get("hit_rate_top3")
    if v is None or (isinstance(v, float) and v != v):  # NaN check
        return float("-inf")
    return float(v)


def _in_band(p95: float, lo: float, hi: float) -> bool:
    if p95 != p95:  # NaN
        return False
    return lo <= p95 <= hi


def _pick_winner(
    cell_baseline_score: float,
    cell_baseline_p95: float,
    trials: list[SweepResult],
    mu_p95_min: float,
    mu_p95_max: float,
    tolerance_pp: float,
) -> tuple[SweepResult, bool]:
    """Pick the best trial.

    Prefers in-band p95 trials. Among in-band candidates, picks max top-3 hit.
    Falls back to picking the trial whose p95 is closest to the band's centre
    if no candidate lands in the band. Returns ``(winner, held_unchanged)``
    where ``held_unchanged`` is True when no candidate clears the existing
    cell's baseline by ``tolerance_pp``.

    Empty-evidence cells (every trial has identical metrics — the K value
    has no measurable effect because the (tier, round) doesn't appear in the
    eval data) always hold unchanged. Sweeping zero-impact cells would just
    flip them around randomly between identical scores.
    """
    # Detect "no measurable effect" — if every trial agrees on top-3 and
    # μ-p95 to within tight tolerances, the cell carries no signal and we
    # should hold at the identity multiplier (1.0).
    top3s = [t.top3_hit_rate for t in trials]
    p95s = [t.mu_p95 for t in trials]
    no_signal = (
        len(trials) >= 2
        and (max(top3s) - min(top3s)) < 1e-9
        and (max(p95s) - min(p95s)) < 0.5
    )
    if no_signal:
        identity = next(
            (t for t in trials if abs(t.multiplier - 1.0) < 1e-9), trials[0]
        )
        return identity, True

    in_band = [t for t in trials if _in_band(t.mu_p95, mu_p95_min, mu_p95_max)]

    if in_band:
        winner = max(in_band, key=lambda t: t.top3_hit_rate)
    else:
        centre = (mu_p95_min + mu_p95_max) / 2.0
        winner = min(trials, key=lambda t: abs(t.mu_p95 - centre))

    # Tolerance gate — only adopt the change if it improves on baseline.
    # Special case: if the baseline is itself OUT of band, we always adopt
    # the winning candidate (the regrid's primary job is to fix elite μ
    # inflation, even at the cost of a tiny top-3 dip).
    baseline_in_band = _in_band(cell_baseline_p95, mu_p95_min, mu_p95_max)
    if baseline_in_band:
        improvement_pp = (winner.top3_hit_rate - cell_baseline_score) * 100.0
        if improvement_pp < tolerance_pp and not (
            _in_band(winner.mu_p95, mu_p95_min, mu_p95_max)
            and winner.top3_hit_rate > cell_baseline_score
        ):
            return winner, True
    return winner, False


# ---------------------------------------------------------------------------
# Main sweep driver
# ---------------------------------------------------------------------------


def run_regrid(
    source_db: Path,
    output_path: Path,
    grid: tuple[float, ...] = DEFAULT_GRID,
    passes: int = DEFAULT_PASSES,
    n_sims: int = DEFAULT_N_SIMS,
    rng_seed: int = DEFAULT_RNG_SEED,
    holdout_seasons: int = DEFAULT_HOLDOUT_SEASONS,
    mu_p95_min: float = DEFAULT_MU_P95_MIN,
    mu_p95_max: float = DEFAULT_MU_P95_MAX,
    tolerance_pp: float = DEFAULT_TOLERANCE_PP,
    disciplines: tuple[Discipline, ...] = (Discipline.LEAD, Discipline.BOULDER),
    cells: tuple[tuple[EventTier, RoundType], ...] | None = None,
    progress: bool = True,
) -> RegridReport:
    """Coordinate-descent regrid the K_FACTOR_TABLE against the backtest.

    See module docstring for design overview.

    Args:
        source_db: Path to the source SQLite DB (copied; never mutated).
        output_path: Where the markdown report is written.
        grid: Multiplicative sweep points around the current cell value.
        passes: Number of coordinate-descent passes over the cell list.
        n_sims: Monte Carlo simulations per round (passes through to
            :func:`compute_podium_probabilities`).
        rng_seed: Seed for reproducible MC draws.
        holdout_seasons: Trailing seasons to hold out.
        mu_p95_min, mu_p95_max: Target μ-p95 band — winners must land here.
        tolerance_pp: Minimum top-3 improvement (in percentage points) to
            accept a change. Set to 0.0 to always adopt the best in-band
            candidate.
        disciplines: Which disciplines to score (defaults to LEAD + BOULDER).
        cells: Optional explicit cell list. When ``None``, the script sweeps
            every ``(tier, round_type)`` pair in :data:`ALL_TIERS` ×
            :data:`ALL_ROUND_TYPES` whose baseline K is defined.
        progress: Print one line per trial when True.

    Returns:
        :class:`RegridReport` with per-cell results + the recommended K table.
    """
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    if cells is None:
        cells = tuple(
            (tier, rt)
            for tier in ALL_TIERS
            for rt in ALL_ROUND_TYPES
            if rt in DEFAULT_CONFIG.k_factor_table.get(tier, {})
        )

    populated = _populated_cells(source_db, disciplines)
    empty_cells = [c for c in cells if c not in populated]
    active_cells = [c for c in cells if c in populated]
    if progress and empty_cells:
        print(
            f"[regrid] skipping {len(empty_cells)} empty cells "
            f"(no rounds in source DB): "
            + ", ".join(f"{t.value}/{rt.value}" for t, rt in empty_cells),
            flush=True,
        )

    baseline_table = copy.deepcopy(DEFAULT_CONFIG.k_factor_table)
    current_table = copy.deepcopy(baseline_table)
    cell_lookup: dict[tuple[EventTier, RoundType], CellRecommendation] = {}

    # Record empty cells up-front as held-unchanged with empty trial lists.
    for tier, rt in empty_cells:
        cell_lookup[(tier, rt)] = CellRecommendation(
            tier=tier,
            round_type=rt,
            baseline_k=baseline_table[tier][rt],
            recommended_k=baseline_table[tier][rt],
            baseline_top3=float("nan"),
            recommended_top3=float("nan"),
            baseline_mu_p95=float("nan"),
            recommended_mu_p95=float("nan"),
            in_band=False,
            held_unchanged=True,
            trials=[],
        )

    with _WorkingDB(source_db) as db:
        # --- Initial metrics under the current config ------------------------
        if progress:
            print(
                f"[regrid] disciplines={[d.value for d in disciplines]} "
                f"holdout_seasons={holdout_seasons} n_sims={n_sims} "
                f"grid={grid} passes={passes}",
                flush=True,
            )
            print("[regrid] computing initial baseline metrics...", flush=True)

        initial = _run_trial(
            db,
            EloConfig(k_factor_table=copy.deepcopy(current_table)),
            disciplines,
            holdout_seasons,
            n_sims,
            rng_seed,
        )
        if progress:
            print(
                f"[regrid] initial: top-3={initial['hit_rate_top3']:.4f} "
                f"μ p50={initial['mu_p50']:.0f} p95={initial['mu_p95']:.0f} "
                f"p99={initial['mu_p99']:.0f}",
                flush=True,
            )

        baseline_top3 = float(initial["hit_rate_top3"])
        baseline_p95 = float(initial["mu_p95"])

        total_trials = passes * len(active_cells) * len(grid)
        trial_num = 0

        # --- Coordinate descent ---------------------------------------------
        for pass_idx in range(passes):
            for tier, rt in active_cells:
                cell_baseline_k = current_table[tier][rt]
                trials: list[SweepResult] = []

                for mult in grid:
                    trial_num += 1
                    candidate_table = _trial_table(current_table, tier, rt, mult)
                    cfg = EloConfig(k_factor_table=candidate_table)
                    metrics = _run_trial(
                        db,
                        cfg,
                        disciplines,
                        holdout_seasons,
                        n_sims,
                        rng_seed,
                    )
                    sr = SweepResult(
                        tier=tier,
                        round_type=rt,
                        multiplier=mult,
                        k_value=cell_baseline_k * mult,
                        top3_hit_rate=float(metrics["hit_rate_top3"]),
                        log_loss_podium=float(metrics["log_loss_podium"]),
                        log_loss_win=float(metrics["log_loss_win"]),
                        brier_podium=float(metrics["brier_podium"]),
                        mu_min=float(metrics["mu_min"]),
                        mu_p50=float(metrics["mu_p50"]),
                        mu_p95=float(metrics["mu_p95"]),
                        mu_p99=float(metrics["mu_p99"]),
                        mu_max=float(metrics["mu_max"]),
                    )
                    trials.append(sr)
                    if progress:
                        print(
                            f"  [{trial_num:3d}/{total_trials}] pass={pass_idx + 1} "
                            f"{tier.value}/{rt.value} mult={mult:.2f} k={sr.k_value:6.2f} "
                            f"→ top3={sr.top3_hit_rate:.4f} μp95={sr.mu_p95:.0f}",
                            flush=True,
                        )

                # Cell baseline (the *current* value's metric) — find the
                # mult=1.0 trial if present, else use the prior cell summary,
                # else use the global baseline.
                identity = next(
                    (t for t in trials if abs(t.multiplier - 1.0) < 1e-9), None
                )
                if identity is not None:
                    cell_baseline_top3 = identity.top3_hit_rate
                    cell_baseline_p95 = identity.mu_p95
                else:
                    cell_baseline_top3 = baseline_top3
                    cell_baseline_p95 = baseline_p95

                winner, held = _pick_winner(
                    cell_baseline_top3,
                    cell_baseline_p95,
                    trials,
                    mu_p95_min,
                    mu_p95_max,
                    tolerance_pp,
                )

                # Keep the most-recent pass's trials (the last pass sees the
                # globally most-converged neighbour cells, so its trial table
                # is the canonical per-cell sensitivity readout). The original
                # baseline_k is preserved across passes — it's the cell's value
                # before the very first sweep, not before the latest pass.
                if held:
                    current_value = current_table[tier][rt]
                    rec = CellRecommendation(
                        tier=tier,
                        round_type=rt,
                        baseline_k=baseline_table[tier][rt],
                        recommended_k=current_value,
                        baseline_top3=cell_baseline_top3,
                        recommended_top3=cell_baseline_top3,
                        baseline_mu_p95=cell_baseline_p95,
                        recommended_mu_p95=cell_baseline_p95,
                        in_band=_in_band(cell_baseline_p95, mu_p95_min, mu_p95_max),
                        held_unchanged=True,
                        trials=trials,
                    )
                    cell_lookup[(tier, rt)] = rec
                else:
                    current_table[tier][rt] = winner.k_value
                    rec = CellRecommendation(
                        tier=tier,
                        round_type=rt,
                        baseline_k=baseline_table[tier][rt],
                        recommended_k=winner.k_value,
                        baseline_top3=cell_baseline_top3,
                        recommended_top3=winner.top3_hit_rate,
                        baseline_mu_p95=cell_baseline_p95,
                        recommended_mu_p95=winner.mu_p95,
                        in_band=_in_band(winner.mu_p95, mu_p95_min, mu_p95_max),
                        held_unchanged=False,
                        trials=trials,
                    )
                    cell_lookup[(tier, rt)] = rec
                    if progress:
                        print(
                            f"  -> WINNER {tier.value}/{rt.value}: "
                            f"k {cell_baseline_k:.2f} → {winner.k_value:.2f} "
                            f"(mult {winner.multiplier:.2f}), "
                            f"top3 {cell_baseline_top3:.4f} → "
                            f"{winner.top3_hit_rate:.4f}, "
                            f"μp95 {cell_baseline_p95:.0f} → {winner.mu_p95:.0f}",
                            flush=True,
                        )

        # --- Final pass metrics under the recommended table ------------------
        if progress:
            print(
                "[regrid] computing final metrics under recommended K table...",
                flush=True,
            )
        final = _run_trial(
            db,
            EloConfig(k_factor_table=copy.deepcopy(current_table)),
            disciplines,
            holdout_seasons,
            n_sims,
            rng_seed,
        )
        if progress:
            print(
                f"[regrid] final: top-3={final['hit_rate_top3']:.4f} "
                f"μ p50={final['mu_p50']:.0f} p95={final['mu_p95']:.0f} "
                f"p99={final['mu_p99']:.0f}",
                flush=True,
            )

    finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    report = RegridReport(
        started_at=started_at,
        finished_at=finished_at,
        grid=tuple(grid),
        passes=passes,
        n_sims=n_sims,
        rng_seed=rng_seed,
        holdout_seasons=holdout_seasons,
        mu_p95_min=mu_p95_min,
        mu_p95_max=mu_p95_max,
        tolerance_pp=tolerance_pp,
        disciplines=disciplines,
        initial_metrics=initial,
        final_metrics=final,
        cells=[cell_lookup[c] for c in cells if c in cell_lookup],
        recommended_k_table=current_table,
        baseline_k_table=baseline_table,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_render_markdown(report))
    if progress:
        print(f"[regrid] wrote {output_path}", flush=True)

    return report


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


def _fmt_float(x: float, digits: int = 4) -> str:
    if x is None:
        return "—"
    if isinstance(x, float) and (x != x):  # NaN
        return "—"
    return f"{x:.{digits}f}"


def _render_markdown(report: RegridReport) -> str:
    lines: list[str] = []
    lines.append("# K_FACTOR_TABLE regrid report (Issue #80)")
    lines.append("")
    lines.append(f"- Started: `{report.started_at}`")
    lines.append(f"- Finished: `{report.finished_at}`")
    lines.append(f"- Disciplines: {', '.join(d.value for d in report.disciplines)}")
    lines.append(f"- Holdout seasons: {report.holdout_seasons}")
    lines.append(f"- MC simulations per round: {report.n_sims}")
    lines.append(f"- RNG seed: {report.rng_seed}")
    lines.append(f"- Grid: {list(report.grid)}")
    lines.append(f"- Passes: {report.passes}")
    lines.append(
        f"- μ-p95 target band: [{report.mu_p95_min:.0f}, {report.mu_p95_max:.0f}]"
    )
    lines.append(f"- Tolerance: {report.tolerance_pp:.2f}pp")
    lines.append("")

    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Current (baseline) | Recommended |")
    lines.append("|---|---|---|")
    init = report.initial_metrics
    final = report.final_metrics
    lines.append(
        f"| Top-3 hit rate | {_fmt_float(init['hit_rate_top3'])} | "
        f"{_fmt_float(final['hit_rate_top3'])} |"
    )
    lines.append(
        f"| Top-1 hit rate | {_fmt_float(init['hit_rate_top1'])} | "
        f"{_fmt_float(final['hit_rate_top1'])} |"
    )
    lines.append(
        f"| Log-loss (win) | {_fmt_float(init['log_loss_win'])} | "
        f"{_fmt_float(final['log_loss_win'])} |"
    )
    lines.append(
        f"| Log-loss (podium) | {_fmt_float(init['log_loss_podium'])} | "
        f"{_fmt_float(final['log_loss_podium'])} |"
    )
    lines.append(
        f"| Brier (podium) | {_fmt_float(init['brier_podium'])} | "
        f"{_fmt_float(final['brier_podium'])} |"
    )
    lines.append(
        f"| μ min  | {_fmt_float(init['mu_min'], 0)} | "
        f"{_fmt_float(final['mu_min'], 0)} |"
    )
    lines.append(
        f"| μ p50  | {_fmt_float(init['mu_p50'], 0)} | "
        f"{_fmt_float(final['mu_p50'], 0)} |"
    )
    lines.append(
        f"| μ p95  | {_fmt_float(init['mu_p95'], 0)} | "
        f"{_fmt_float(final['mu_p95'], 0)} |"
    )
    lines.append(
        f"| μ p99  | {_fmt_float(init['mu_p99'], 0)} | "
        f"{_fmt_float(final['mu_p99'], 0)} |"
    )
    lines.append(
        f"| μ max  | {_fmt_float(init['mu_max'], 0)} | "
        f"{_fmt_float(final['mu_max'], 0)} |"
    )
    lines.append("")

    lines.append("## Per-cell sweep results")
    lines.append("")
    for rec in report.cells:
        # Status reflects the FINAL outcome relative to the original baseline,
        # not the last-pass micro-decision (which could be "held" simply
        # because pass 2 saw no further improvement on the value pass 1 picked).
        original_baseline = rec.baseline_k
        if not rec.trials:
            status = "UNCHANGED — no data"
        elif abs(rec.recommended_k - original_baseline) < 1e-9:
            status = "HELD UNCHANGED"
        else:
            status = "UPDATED"
        in_band_str = "in band" if rec.in_band else "out of band"
        lines.append(
            f"### {rec.tier.value} / {rec.round_type.value} — **{status}** ({in_band_str})"
        )
        lines.append("")
        if not rec.trials:
            lines.append(
                f"- No rounds with this (tier, round_type) appear in the source DB; "
                f"K held at the current default of **{rec.baseline_k:.2f}**."
            )
            lines.append("")
            continue
        lines.append(
            f"- Baseline K: **{rec.baseline_k:.2f}** "
            f"(top-3={_fmt_float(rec.baseline_top3)}, "
            f"μ-p95={_fmt_float(rec.baseline_mu_p95, 0)})"
        )
        lines.append(
            f"- Recommended K: **{rec.recommended_k:.2f}** "
            f"(top-3={_fmt_float(rec.recommended_top3)}, "
            f"μ-p95={_fmt_float(rec.recommended_mu_p95, 0)})"
        )
        lines.append("")
        lines.append("| Mult | K | Top-3 | LL podium | μ p50 | μ p95 | μ p99 |")
        lines.append("|---|---|---|---|---|---|---|")
        for t in rec.trials:
            mark = " ←" if abs(t.k_value - rec.recommended_k) < 1e-9 else ""
            lines.append(
                f"| {t.multiplier:.2f} | {t.k_value:.2f} | "
                f"{_fmt_float(t.top3_hit_rate)} | "
                f"{_fmt_float(t.log_loss_podium)} | "
                f"{_fmt_float(t.mu_p50, 0)} | "
                f"{_fmt_float(t.mu_p95, 0)} | "
                f"{_fmt_float(t.mu_p99, 0)} |{mark}"
            )
        lines.append("")

    lines.append("## Recommended K_FACTOR_TABLE (paste into `_DEFAULT_K_FACTORS`)")
    lines.append("")
    lines.append("```python")
    lines.append(
        f"# Recommended K_FACTOR_TABLE (regrid {report.finished_at[:10]} against backtest):"
    )
    lines.append("_DEFAULT_K_FACTORS: dict[EventTier, dict[RoundType, float]] = {")
    for tier in ALL_TIERS:
        if tier not in report.recommended_k_table:
            continue
        lines.append(f"    EventTier.{tier.name}: {{")
        for rt in (RoundType.FINAL, RoundType.SEMI, RoundType.QUALIFICATION):
            if rt not in report.recommended_k_table[tier]:
                continue
            new_k = report.recommended_k_table[tier][rt]
            old_k = report.baseline_k_table[tier][rt]
            comment = (
                f"  # was {old_k:.1f}" if abs(new_k - old_k) > 1e-9 else "  # unchanged"
            )
            lines.append(f"        RoundType.{rt.name}: {new_k:.2f},{comment}")
        lines.append("    },")
    lines.append("}")
    lines.append("```")
    lines.append("")
    lines.append("## Next steps")
    lines.append("")
    lines.append("1. Review the per-cell tables above for sanity (any cell with")
    lines.append("   top-3 dropping more than 2pp deserves a closer look).")
    lines.append("2. Apply the recommended `_DEFAULT_K_FACTORS` dict to")
    lines.append("   `src/climbing_elo/engine/elo.py`.")
    lines.append(
        "3. Re-run the full backtest: `uv run python scripts/run_backtest.py --db data/climbing_elo.db`."
    )
    lines.append(
        "4. Trigger a prod re-backfill: `gh workflow run scrape-supabase.yml --repo milwil-2/climb-elo`."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_grid(spec: str) -> tuple[float, ...]:
    parts = [p.strip() for p in spec.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError(
            "grid must be a non-empty comma-separated list"
        )
    try:
        vals = tuple(float(p) for p in parts)
    except ValueError as e:
        raise argparse.ArgumentTypeError(f"invalid grid value: {e}") from e
    return vals


def _parse_disciplines(spec: str) -> tuple[Discipline, ...]:
    aliases = {
        "lead": Discipline.LEAD,
        "boulder": Discipline.BOULDER,
        "speed": Discipline.SPEED,
    }
    parts = [p.strip().lower() for p in spec.split(",") if p.strip()]
    out: list[Discipline] = []
    for p in parts:
        if p not in aliases:
            raise argparse.ArgumentTypeError(f"unknown discipline {p!r}")
        out.append(aliases[p])
    if not out:
        raise argparse.ArgumentTypeError("at least one discipline required")
    return tuple(out)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Coordinate-descent regrid of K_FACTOR_TABLE under EloConfig (#80).",
    )
    p.add_argument(
        "--db",
        required=True,
        type=Path,
        help="Path to source SQLite DB (copied; never mutated).",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=Path("docs") / "K_REGRID_REPORT.md",
        help="Path to write the markdown report (default: docs/K_REGRID_REPORT.md).",
    )
    p.add_argument(
        "--grid",
        type=_parse_grid,
        default=DEFAULT_GRID,
        help=(
            "Comma-separated multiplicative sweep points around each cell's "
            "current value (default: 0.5,0.75,1.0,1.5,2.0)."
        ),
    )
    p.add_argument(
        "--passes",
        type=int,
        default=DEFAULT_PASSES,
        help="Number of coordinate-descent passes (default: 2).",
    )
    p.add_argument(
        "--n-sims",
        type=int,
        default=DEFAULT_N_SIMS,
        help="Monte Carlo simulations per round (default: 2000).",
    )
    p.add_argument(
        "--rng-seed",
        type=int,
        default=DEFAULT_RNG_SEED,
        help="RNG seed for reproducibility (default: 2026).",
    )
    p.add_argument(
        "--holdout-seasons",
        type=int,
        default=DEFAULT_HOLDOUT_SEASONS,
        help="Trailing seasons to hold out (default: 2).",
    )
    p.add_argument(
        "--mu-p95-min",
        type=float,
        default=DEFAULT_MU_P95_MIN,
        help="Lower edge of target μ-p95 band (default: 1900).",
    )
    p.add_argument(
        "--mu-p95-max",
        type=float,
        default=DEFAULT_MU_P95_MAX,
        help="Upper edge of target μ-p95 band (default: 2200).",
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE_PP,
        help="Minimum top-3 improvement (pp) to accept a change (default: 0).",
    )
    p.add_argument(
        "--disciplines",
        type=_parse_disciplines,
        default=(Discipline.LEAD, Discipline.BOULDER),
        help="Comma-separated disciplines to score (default: lead,boulder).",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-trial progress lines.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

    if not args.db.exists():
        print(f"error: source DB not found: {args.db}", file=sys.stderr)
        return 2

    run_regrid(
        source_db=args.db,
        output_path=args.output,
        grid=tuple(args.grid),
        passes=args.passes,
        n_sims=args.n_sims,
        rng_seed=args.rng_seed,
        holdout_seasons=args.holdout_seasons,
        mu_p95_min=args.mu_p95_min,
        mu_p95_max=args.mu_p95_max,
        tolerance_pp=args.tolerance,
        disciplines=tuple(args.disciplines),
        progress=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
