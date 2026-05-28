#!/usr/bin/env python3
"""MOV constants grid search for the post-Glicko-2 (#51) effective-K regime.

Background
----------

Issue #53 introduced 538-style gap-conditioned margin-of-victory damping:

    multiplier = base · softening / (max(Δμ, 0) / rating_scale + softening)

The two constants ::

    MOV_RATING_SCALE = 400.0   # Δμ at which damping factor reaches softening/(1+softening)
    MOV_SOFTENING    = 2.2     # controls how fast damping kicks in; lower → harsher

were first-pass values picked under the pre-Glicko-2 regime. Today's #80 K
regrid re-tuned the K table for the new variable effective-K regime; the MOV
constants haven't been swept since.

Strategy
--------

Full **2-D Cartesian grid** over ``(rating_scale, softening)``. Unlike the K
table (16 cells × 7 grid points → coordinate descent) the MOV parameters are
only two scalars — a 6 × 5 = 30-point grid runs in 10-30 min and is
exhaustively the right answer.

Per grid point, the script:

1. Builds an :class:`EloConfig` with the candidate ``(mov_rating_scale,
   mov_softening)`` (no monkey-patching).
2. Restores a pristine copy of the source DB to a temp working copy.
3. Wipes ratings / rating-history and re-runs backfill for each discipline
   with ``config=trial_config``.
4. Scores every holdout round via :func:`engine.evaluation.score_split_events`
   reusing :func:`compute_podium_probabilities` for probabilistic metrics.
5. Computes the **μ-range stats** (min / p50 / p95 / p99 / max) from the
   trained Rating rows — the secondary acceptance gate alongside top-3 hit
   rate.

The winner maximises ``top-3 hit-rate`` (primary), breaking ties on lower
``log_loss_podium``, subject to ``μ-p95`` lying in the target band
``[--mu-p95-min, --mu-p95-max]`` (default 1900-2200, same as the K regrid).

Output
------

* ``docs/MOV_REGRID_REPORT.md`` — grid sweep table + the recommended values
  block as a paste-ready ``EloConfig`` snippet.

Does **not** modify ``src/climbing_elo/engine/elo.py``. Applying the
recommended values is a separate, reviewed manual step (small ~5-line PR).

Speed is excluded by default: the bracket-native Speed model (#56) in
``engine/speed.py`` doesn't use the 538 MOV damping at all, so sweeping the
constants against Speed data adds no signal.

Usage
-----

::

    # Default 6×5 grid (~10-30 min on the local DB).
    uv run python scripts/regrid_mov_factors.py --db data/climbing_elo.db

    # Faster smoke run — smaller grid + fewer sims.
    uv run python scripts/regrid_mov_factors.py --db data/climbing_elo.db \
        --rating-scale-grid 200,400,800 --softening-grid 1.0,2.2,4.5 \
        --n-sims 1000

    # Custom output path:
    uv run python scripts/regrid_mov_factors.py --db data/climbing_elo.db \
        --output docs/MOV_REGRID_REPORT.md
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

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from climbing_elo.database import get_session_factory
from climbing_elo.engine.backfill import run_backfill
from climbing_elo.engine.elo import DEFAULT_CONFIG, EloConfig
from climbing_elo.engine.evaluation import (
    BacktestDataset,
    BacktestRunner,
    HoldoutMode,
    _aggregate_metrics,
    score_split_events,
)
from climbing_elo.models import (
    Discipline,
    Rating,
    RatingHistory,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

DEFAULT_RATING_SCALE_GRID: tuple[float, ...] = (
    200.0,
    300.0,
    400.0,
    500.0,
    600.0,
    800.0,
)
DEFAULT_SOFTENING_GRID: tuple[float, ...] = (1.0, 1.5, 2.2, 3.0, 4.5)
DEFAULT_HOLDOUT_SEASONS = 2
DEFAULT_N_SIMS = 2_000
DEFAULT_RNG_SEED = 2026
DEFAULT_MU_P95_MIN = 1900.0
DEFAULT_MU_P95_MAX = 2200.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MovTrial:
    """One (rating_scale, softening) trial's metric outcome."""

    rating_scale: float
    softening: float
    top3_hit_rate: float
    top1_hit_rate: float
    log_loss_podium: float
    log_loss_win: float
    brier_podium: float
    mu_min: float
    mu_p50: float
    mu_p95: float
    mu_p99: float
    mu_max: float
    in_band: bool


@dataclass
class MovRegridReport:
    """End-to-end regrid output."""

    started_at: str
    finished_at: str
    rating_scale_grid: tuple[float, ...]
    softening_grid: tuple[float, ...]
    n_sims: int
    rng_seed: int
    holdout_seasons: int
    mu_p95_min: float
    mu_p95_max: float
    disciplines: tuple[Discipline, ...]
    baseline_rating_scale: float
    baseline_softening: float
    recommended_rating_scale: float
    recommended_softening: float
    initial_metrics: dict[str, Any] = field(default_factory=dict)
    final_metrics: dict[str, Any] = field(default_factory=dict)
    trials: list[MovTrial] = field(default_factory=list)


# ---------------------------------------------------------------------------
# DB harness — copy + isolated session for one trial
# ---------------------------------------------------------------------------


class _WorkingDB:
    """Maintain a pristine DB copy + a working copy that can be reset cheaply.

    Lifecycle (identical to the K regrid's :class:`_WorkingDB`):

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
        self._tmpdir = Path(tempfile.mkdtemp(prefix="climbing_elo_mov_regrid_"))
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


def _percentiles(values: list[float], qs: Iterable[float]) -> list[float]:
    """Inclusive percentiles via :func:`statistics.quantiles` (n=100)."""
    if not values:
        return [float("nan")] * len(list(qs))
    if len(values) == 1:
        return [values[0]] * len(list(qs))
    qs = list(qs)
    quants = statistics.quantiles(values, n=100, method="inclusive")
    out = []
    for q in qs:
        if q <= 0:
            out.append(min(values))
        elif q >= 100:
            out.append(max(values))
        else:
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

    Strategy mirrors the K regrid trial runner: pre-run backfill with the
    custom config (the harness's own backfill call would otherwise drop our
    config), then hand the session to :class:`BacktestRunner` with
    ``in_memory_session=session``. The runner's idempotency guard in
    :func:`run_backfill` will skip every already-rated round, so the scoring
    stage sees our trained state.
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
            engine = runner.engine_factory(session)
            preds = score_split_events(
                session,
                engine,
                split,
                discipline,
                n_simulations=n_sims,
                rng_seed=rng_seed,
            )
            all_predictions.extend(preds)

        mu_stats = _collect_mu_stats(session, disciplines)

    metrics = _aggregate_metrics(all_predictions)
    metrics.update(mu_stats)
    return metrics


# ---------------------------------------------------------------------------
# Grid enumeration + winner selection
# ---------------------------------------------------------------------------


def _enumerate_grid(
    rating_scale_grid: tuple[float, ...],
    softening_grid: tuple[float, ...],
) -> list[tuple[float, float]]:
    """Cartesian product over (rating_scale, softening), in stable order."""
    return [(rs, sf) for rs in rating_scale_grid for sf in softening_grid]


def _in_band(p95: float, lo: float, hi: float) -> bool:
    if p95 != p95:  # NaN
        return False
    return lo <= p95 <= hi


def _pick_winner(
    trials: list[MovTrial],
    mu_p95_min: float,
    mu_p95_max: float,
) -> MovTrial:
    """Pick the best trial.

    Prefers in-band p95 trials. Among in-band candidates, picks max top-3 hit
    rate; ties broken by lower podium log-loss. Falls back to the trial whose
    p95 is closest to the band's centre if no candidate lands in the band.
    """
    if not trials:
        raise ValueError("Cannot pick a winner from an empty trial list")

    in_band = [t for t in trials if _in_band(t.mu_p95, mu_p95_min, mu_p95_max)]

    if in_band:
        # Primary: max top-3. Tie-break: min log-loss-podium.
        return max(
            in_band,
            key=lambda t: (t.top3_hit_rate, -t.log_loss_podium),
        )

    # No in-band candidates: pick whichever has the closest μ-p95 to band centre.
    centre = (mu_p95_min + mu_p95_max) / 2.0
    return min(trials, key=lambda t: abs(t.mu_p95 - centre))


# ---------------------------------------------------------------------------
# Main grid driver
# ---------------------------------------------------------------------------


def run_regrid(
    source_db: Path,
    output_path: Path,
    rating_scale_grid: tuple[float, ...] = DEFAULT_RATING_SCALE_GRID,
    softening_grid: tuple[float, ...] = DEFAULT_SOFTENING_GRID,
    n_sims: int = DEFAULT_N_SIMS,
    rng_seed: int = DEFAULT_RNG_SEED,
    holdout_seasons: int = DEFAULT_HOLDOUT_SEASONS,
    mu_p95_min: float = DEFAULT_MU_P95_MIN,
    mu_p95_max: float = DEFAULT_MU_P95_MAX,
    disciplines: tuple[Discipline, ...] = (Discipline.LEAD, Discipline.BOULDER),
    progress: bool = True,
) -> MovRegridReport:
    """Cartesian grid search of (mov_rating_scale, mov_softening) against backtest.

    See module docstring for design overview.

    Args:
        source_db: Path to the source SQLite DB (copied; never mutated).
        output_path: Where the markdown report is written.
        rating_scale_grid: Candidate MOV_RATING_SCALE values to sweep.
        softening_grid: Candidate MOV_SOFTENING values to sweep.
        n_sims: Monte Carlo simulations per round.
        rng_seed: Seed for reproducible MC draws.
        holdout_seasons: Trailing seasons to hold out.
        mu_p95_min, mu_p95_max: Target μ-p95 band — winners must land here.
        disciplines: Which disciplines to score (defaults to LEAD + BOULDER —
            Speed has its own bracket model and is excluded by default).
        progress: Print one line per trial when True.

    Returns:
        :class:`MovRegridReport` with per-trial results + recommended values.
    """
    started_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    grid_points = _enumerate_grid(rating_scale_grid, softening_grid)
    baseline_rs = DEFAULT_CONFIG.mov_rating_scale
    baseline_sf = DEFAULT_CONFIG.mov_softening
    # Pin a base K table so MOV is the only varying knob across trials.
    base_k_table = copy.deepcopy(DEFAULT_CONFIG.k_factor_table)

    with _WorkingDB(source_db) as db:
        if progress:
            print(
                f"[mov-regrid] disciplines={[d.value for d in disciplines]} "
                f"holdout_seasons={holdout_seasons} n_sims={n_sims} "
                f"grid_points={len(grid_points)} "
                f"(rating_scale={list(rating_scale_grid)}, "
                f"softening={list(softening_grid)})",
                flush=True,
            )
            print("[mov-regrid] computing initial baseline metrics...", flush=True)

        initial = _run_trial(
            db,
            EloConfig(
                k_factor_table=copy.deepcopy(base_k_table),
                mov_rating_scale=baseline_rs,
                mov_softening=baseline_sf,
            ),
            disciplines,
            holdout_seasons,
            n_sims,
            rng_seed,
        )
        if progress:
            print(
                f"[mov-regrid] initial (rs={baseline_rs:.0f}, sf={baseline_sf:.2f}): "
                f"top-3={initial['hit_rate_top3']:.4f} "
                f"μ p50={initial['mu_p50']:.0f} p95={initial['mu_p95']:.0f} "
                f"p99={initial['mu_p99']:.0f}",
                flush=True,
            )

        trials: list[MovTrial] = []
        for idx, (rs, sf) in enumerate(grid_points, 1):
            cfg = EloConfig(
                k_factor_table=copy.deepcopy(base_k_table),
                mov_rating_scale=rs,
                mov_softening=sf,
            )
            metrics = _run_trial(
                db,
                cfg,
                disciplines,
                holdout_seasons,
                n_sims,
                rng_seed,
            )
            trial = MovTrial(
                rating_scale=rs,
                softening=sf,
                top3_hit_rate=float(metrics["hit_rate_top3"]),
                top1_hit_rate=float(metrics["hit_rate_top1"]),
                log_loss_podium=float(metrics["log_loss_podium"]),
                log_loss_win=float(metrics["log_loss_win"]),
                brier_podium=float(metrics["brier_podium"]),
                mu_min=float(metrics["mu_min"]),
                mu_p50=float(metrics["mu_p50"]),
                mu_p95=float(metrics["mu_p95"]),
                mu_p99=float(metrics["mu_p99"]),
                mu_max=float(metrics["mu_max"]),
                in_band=_in_band(float(metrics["mu_p95"]), mu_p95_min, mu_p95_max),
            )
            trials.append(trial)
            if progress:
                band_mark = "✓" if trial.in_band else " "
                print(
                    f"  [{idx:3d}/{len(grid_points)}] rs={rs:5.0f} sf={sf:.2f} "
                    f"{band_mark} top3={trial.top3_hit_rate:.4f} "
                    f"LLpod={trial.log_loss_podium:.4f} "
                    f"μp95={trial.mu_p95:.0f}",
                    flush=True,
                )

        winner = _pick_winner(trials, mu_p95_min, mu_p95_max)
        if progress:
            print(
                f"[mov-regrid] winner: rs={winner.rating_scale:.0f} "
                f"sf={winner.softening:.2f} top3={winner.top3_hit_rate:.4f} "
                f"μp95={winner.mu_p95:.0f}",
                flush=True,
            )
            print(
                "[mov-regrid] computing final metrics under recommended MOV constants...",
                flush=True,
            )
        final = _run_trial(
            db,
            EloConfig(
                k_factor_table=copy.deepcopy(base_k_table),
                mov_rating_scale=winner.rating_scale,
                mov_softening=winner.softening,
            ),
            disciplines,
            holdout_seasons,
            n_sims,
            rng_seed,
        )

    finished_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

    report = MovRegridReport(
        started_at=started_at,
        finished_at=finished_at,
        rating_scale_grid=tuple(rating_scale_grid),
        softening_grid=tuple(softening_grid),
        n_sims=n_sims,
        rng_seed=rng_seed,
        holdout_seasons=holdout_seasons,
        mu_p95_min=mu_p95_min,
        mu_p95_max=mu_p95_max,
        disciplines=disciplines,
        baseline_rating_scale=baseline_rs,
        baseline_softening=baseline_sf,
        recommended_rating_scale=winner.rating_scale,
        recommended_softening=winner.softening,
        initial_metrics=initial,
        final_metrics=final,
        trials=trials,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(_render_markdown(report))
    if progress:
        print(f"[mov-regrid] wrote {output_path}", flush=True)

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


def _render_markdown(report: MovRegridReport) -> str:
    lines: list[str] = []
    lines.append("# MOV constants regrid report (Issue #85)")
    lines.append("")
    lines.append(f"- Started: `{report.started_at}`")
    lines.append(f"- Finished: `{report.finished_at}`")
    lines.append(f"- Disciplines: {', '.join(d.value for d in report.disciplines)}")
    lines.append(f"- Holdout seasons: {report.holdout_seasons}")
    lines.append(f"- MC simulations per round: {report.n_sims}")
    lines.append(f"- RNG seed: {report.rng_seed}")
    lines.append(f"- MOV_RATING_SCALE grid: {list(report.rating_scale_grid)}")
    lines.append(f"- MOV_SOFTENING grid: {list(report.softening_grid)}")
    lines.append(
        f"- μ-p95 target band: [{report.mu_p95_min:.0f}, {report.mu_p95_max:.0f}]"
    )
    lines.append(
        f"- Baseline: rating_scale=**{report.baseline_rating_scale:.0f}**, "
        f"softening=**{report.baseline_softening:.2f}**"
    )
    lines.append(
        f"- Recommended: rating_scale=**{report.recommended_rating_scale:.0f}**, "
        f"softening=**{report.recommended_softening:.2f}**"
    )
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

    lines.append("## Grid sweep results")
    lines.append("")
    lines.append(
        "Rows = `MOV_RATING_SCALE`, columns = `MOV_SOFTENING`. Cell shows "
        "**top-3 hit rate** (μ-p95). ✓ marks in-band trials; ★ marks the "
        "recommended cell."
    )
    lines.append("")

    header = (
        "| rating_scale \\ softening | "
        + " | ".join(f"{sf:.2f}" for sf in report.softening_grid)
        + " |"
    )
    sep = "|---|" + "|".join("---" for _ in report.softening_grid) + "|"
    lines.append(header)
    lines.append(sep)

    # Build a lookup by (rs, sf) for fast cell rendering.
    by_point: dict[tuple[float, float], MovTrial] = {
        (t.rating_scale, t.softening): t for t in report.trials
    }
    rec_key = (report.recommended_rating_scale, report.recommended_softening)

    for rs in report.rating_scale_grid:
        cells: list[str] = []
        for sf in report.softening_grid:
            t = by_point.get((rs, sf))
            if t is None:
                cells.append("—")
                continue
            band_mark = " ✓" if t.in_band else ""
            rec_mark = " ★" if (rs, sf) == rec_key else ""
            cells.append(f"{t.top3_hit_rate:.4f} ({t.mu_p95:.0f}){band_mark}{rec_mark}")
        lines.append(f"| **{rs:.0f}** | " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Full per-trial table")
    lines.append("")
    lines.append(
        "| rating_scale | softening | Top-3 | Top-1 | LL win | LL podium | "
        "Brier podium | μ p50 | μ p95 | μ p99 | In band |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for t in report.trials:
        mark = " ←" if (t.rating_scale, t.softening) == rec_key else ""
        lines.append(
            f"| {t.rating_scale:.0f} | {t.softening:.2f} | "
            f"{_fmt_float(t.top3_hit_rate)} | "
            f"{_fmt_float(t.top1_hit_rate)} | "
            f"{_fmt_float(t.log_loss_win)} | "
            f"{_fmt_float(t.log_loss_podium)} | "
            f"{_fmt_float(t.brier_podium)} | "
            f"{_fmt_float(t.mu_p50, 0)} | "
            f"{_fmt_float(t.mu_p95, 0)} | "
            f"{_fmt_float(t.mu_p99, 0)} | "
            f"{'yes' if t.in_band else 'no'} |{mark}"
        )
    lines.append("")

    lines.append("## Recommended values (paste into `EloConfig` defaults)")
    lines.append("")
    lines.append("```python")
    lines.append(
        f"# Recommended MOV constants (regrid {report.finished_at[:10]} "
        "against backtest):"
    )
    rs_old = report.baseline_rating_scale
    rs_new = report.recommended_rating_scale
    sf_old = report.baseline_softening
    sf_new = report.recommended_softening
    rs_comment = (
        f"  # was {rs_old:.1f}" if abs(rs_new - rs_old) > 1e-9 else "  # unchanged"
    )
    sf_comment = (
        f"  # was {sf_old:.2f}" if abs(sf_new - sf_old) > 1e-9 else "  # unchanged"
    )
    lines.append(f"mov_rating_scale: float = {rs_new:.1f}{rs_comment}")
    lines.append(f"mov_softening: float = {sf_new:.2f}{sf_comment}")
    lines.append("```")
    lines.append("")
    lines.append("## Next steps")
    lines.append("")
    lines.append("1. Review the grid above — any in-band cell within ~1pp top-3 of")
    lines.append("   the recommended one is essentially equivalent and could be")
    lines.append("   picked instead if it has nicer round numbers.")
    lines.append("2. Apply the recommended `mov_rating_scale` / `mov_softening`")
    lines.append("   defaults to `EloConfig` in `src/climbing_elo/engine/elo.py`.")
    lines.append(
        "3. Re-run the full backtest: `uv run python scripts/run_backtest.py "
        "--db data/climbing_elo.db`."
    )
    lines.append(
        "4. Trigger a prod re-backfill: `gh workflow run scrape-supabase.yml "
        "--repo milwil-2/climb-elo`."
    )
    lines.append("")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _parse_float_grid(spec: str) -> tuple[float, ...]:
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
        description=(
            "Cartesian grid search of (MOV_RATING_SCALE, MOV_SOFTENING) "
            "under EloConfig (#85)."
        ),
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
        default=Path("docs") / "MOV_REGRID_REPORT.md",
        help="Path to write the markdown report (default: docs/MOV_REGRID_REPORT.md).",
    )
    p.add_argument(
        "--rating-scale-grid",
        type=_parse_float_grid,
        default=DEFAULT_RATING_SCALE_GRID,
        help=(
            "Comma-separated MOV_RATING_SCALE candidates "
            "(default: 200,300,400,500,600,800)."
        ),
    )
    p.add_argument(
        "--softening-grid",
        type=_parse_float_grid,
        default=DEFAULT_SOFTENING_GRID,
        help=(
            "Comma-separated MOV_SOFTENING candidates (default: 1.0,1.5,2.2,3.0,4.5)."
        ),
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
        "--disciplines",
        type=_parse_disciplines,
        default=(Discipline.LEAD, Discipline.BOULDER),
        help=(
            "Comma-separated disciplines to score (default: lead,boulder). "
            "Speed has its own bracket model and is excluded by default."
        ),
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
        rating_scale_grid=tuple(args.rating_scale_grid),
        softening_grid=tuple(args.softening_grid),
        n_sims=args.n_sims,
        rng_seed=args.rng_seed,
        holdout_seasons=args.holdout_seasons,
        mu_p95_min=args.mu_p95_min,
        mu_p95_max=args.mu_p95_max,
        disciplines=tuple(args.disciplines),
        progress=not args.quiet,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
