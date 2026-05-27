#!/usr/bin/env python3
"""Backtest the climbing ELO model across the metric × stratification matrix.

This script is a thin CLI shim around :class:`BacktestRunner` in
``engine/evaluation.py``. The legacy single-metric implementation (172 lines,
Lead-only, with a destructive ``delete()`` against the production Ratings
table) has been replaced by a probabilistic harness that:

  - Runs against a private *copy* of the production DB (state-safe).
  - Scores ``log-loss / Brier / calibration / Spearman / top-K hit rates`` on
    every holdout round.
  - Stratifies by tenure, tier, round, discipline, season, and field size.
  - Writes JSON + markdown reports to ``data/backtests/<timestamp>/``.

The default invocation reproduces the legacy 2-season holdout shape, but
across Lead AND Boulder, with the full metric matrix.

Usage
-----

::

    # Default — 2-season holdout, current variant, Lead + Boulder
    uv run python scripts/run_backtest.py

    # Pick a single discipline
    uv run python scripts/run_backtest.py --discipline lead

    # Pick a variant (Issue #38 will add baselines: random, persistence, ...)
    uv run python scripts/run_backtest.py --variant current

    # Pick an OOS mode (Issue #39 will add walk-forward, leave-one-out, ...)
    uv run python scripts/run_backtest.py --oos holdout

    # Shrink MC budget for a fast smoke test
    uv run python scripts/run_backtest.py --n-sims 2000
"""

from __future__ import annotations

import argparse
import logging
import sys

from climbing_elo.engine.evaluation import (
    BACKTEST_VARIANTS,
    OOS_MODES,
    BacktestDataset,
    BacktestRunner,
    HoldoutMode,
    make_default_output_dir,
)
from climbing_elo.models import Discipline

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")


DISCIPLINE_ALIASES = {
    "lead": Discipline.LEAD,
    "boulder": Discipline.BOULDER,
    "speed": Discipline.SPEED,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backtest the climbing ELO model with probabilistic metrics."
    )
    p.add_argument(
        "--discipline",
        choices=sorted(DISCIPLINE_ALIASES) + ["all"],
        default="all",
        help="Which discipline(s) to score (default: lead + boulder).",
    )
    p.add_argument(
        "--variant",
        choices=sorted(BACKTEST_VARIANTS.keys()),
        default="current",
        help="Rating engine variant (Issue #38 adds baselines).",
    )
    p.add_argument(
        "--oos",
        choices=sorted(OOS_MODES.keys()),
        default="holdout",
        help="Out-of-sample mode (Issue #39 adds walk-forward etc.).",
    )
    p.add_argument(
        "--holdout-seasons",
        type=int,
        default=2,
        help="Number of trailing seasons to hold out (holdout mode only).",
    )
    p.add_argument(
        "--from-season",
        type=int,
        default=None,
        help="Walk-forward mode: inclusive lower bound for the eval season.",
    )
    p.add_argument(
        "--to-season",
        type=int,
        default=None,
        help="Walk-forward mode: inclusive upper bound for the eval season.",
    )
    p.add_argument(
        "--season",
        type=int,
        default=None,
        help=(
            "leave-one-event-out mode: season to rotate over (default: "
            "most recent season present in the DB)."
        ),
    )
    p.add_argument(
        "--athlete-id",
        type=int,
        default=None,
        help="leave-one-athlete-out mode: athlete to evaluate cold-start for.",
    )
    p.add_argument(
        "--tenure",
        type=int,
        default=5,
        help=(
            "leave-one-athlete-out mode: number of earliest events to "
            "hide from training (default: 5)."
        ),
    )
    p.add_argument(
        "--n-sims",
        type=int,
        default=10_000,
        help="Monte Carlo simulations per round (default: 10000).",
    )
    p.add_argument(
        "--rng-seed",
        type=int,
        default=42,
        help="Seed for reproducible MC draws.",
    )
    p.add_argument(
        "--output-dir",
        default=None,
        help="Directory for report.json + report.md (default: data/backtests/<utc-timestamp>).",
    )
    p.add_argument(
        "--db",
        required=True,
        help=(
            "Path to source SQLite DB (read-only — a copy is used internally). "
            "The backtest harness only supports a SQLite source today; for "
            "Postgres-backed data, snapshot to SQLite first."
        ),
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.discipline == "all":
        disciplines = (Discipline.LEAD, Discipline.BOULDER)
    else:
        disciplines = (DISCIPLINE_ALIASES[args.discipline],)

    # Build OOS mode from registry, threading mode-specific flags.
    if args.oos == "holdout":
        oos_mode = HoldoutMode(n_seasons=args.holdout_seasons)
    elif args.oos == "walk-forward":
        oos_mode = OOS_MODES["walk-forward"](
            from_season=args.from_season,
            to_season=args.to_season,
        )
    elif args.oos == "leave-one-event-out":
        oos_mode = OOS_MODES["leave-one-event-out"](season=args.season)
    elif args.oos == "leave-one-athlete-out":
        if args.athlete_id is None:
            raise SystemExit("--oos leave-one-athlete-out requires --athlete-id")
        oos_mode = OOS_MODES["leave-one-athlete-out"](
            athlete_id=args.athlete_id,
            tenure=args.tenure,
        )
    else:
        # Future modes — default-construct from the registry.
        oos_mode = OOS_MODES[args.oos]()

    from pathlib import Path

    output_dir = Path(args.output_dir) if args.output_dir else make_default_output_dir()

    dataset = BacktestDataset(
        disciplines=disciplines,
        n_simulations=args.n_sims,
        rng_seed=args.rng_seed,
        source_db_path=Path(args.db),
    )

    with BacktestRunner(
        dataset=dataset,
        variant=args.variant,
        oos_mode=oos_mode,
        output_dir=output_dir,
    ) as runner:
        report = runner.run()

    # Print a one-line summary so cron/CI can grep for it.
    agg = report.aggregate
    print(f"Backtest report -> {output_dir}")
    print(
        f"  rounds={agg.get('n_rounds')} | "
        f"LL win={agg.get('log_loss_win'):.4f} | "
        f"LL podium={agg.get('log_loss_podium'):.4f} | "
        f"top-3 hit={agg.get('hit_rate_top3'):.4f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
