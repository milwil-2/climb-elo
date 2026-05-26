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

from climbing_elo.database import DEFAULT_DB_PATH
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
        default=str(DEFAULT_DB_PATH),
        help="Path to source SQLite DB (read-only — a copy is used internally).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.discipline == "all":
        disciplines = (Discipline.LEAD, Discipline.BOULDER)
    else:
        disciplines = (DISCIPLINE_ALIASES[args.discipline],)

    # Build OOS mode from registry. Holdout mode accepts n_seasons.
    if args.oos == "holdout":
        oos_mode = HoldoutMode(n_seasons=args.holdout_seasons)
    else:
        # Future modes (Issue #39) — default-construct from the registry.
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
