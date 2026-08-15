"""Pre-registered importance-weight sweep for the ``g2pl`` challenger.

Protocol (fixed BEFORE looking at any results — see PR #191's verdict thread):

- **Tune folds**: walk-forward eval seasons 2019-2022 (the confirmation span is
  never touched during selection).
- **Confirmation folds**: walk-forward eval seasons 2023-2025, run exactly once,
  for the single selected config.
- **Grid** (24 configs): ``w_scale`` x ``mov_mode`` x ``field_normalization_exponent``.
  ``w_scale`` multiplies every cell of the seeded importance-weight table — the
  evidence-rate knob the #191 verdict identified as the likeliest mis-set.
  ``mov_mode`` doubles as the #84 A/B (margin-as-outcome vs pure outcome).
- **Selection rule** (tune folds): a config *passes tune* iff it beats the
  ``current`` tune baseline on BOTH primaries (podium log-loss lower AND top-3
  hit rate higher). Among passers: highest top-3, tie-break lowest podium
  log-loss. No passers -> the sweep is over, park the challenger (exit 2).
- **Ship rule** (confirmation folds, applied once): beat ``current`` on both
  primaries AND mean Spearman >= 95% of current's. Pass -> exit 0; fail -> exit 1.

Every run: seed 42, 10k MC sims, Lead + Boulder, the leak-fixed harness (#192).

Usage:
    uv run python scripts/sweep_g2pl.py --db <sqlite path> --out <results.json>
    uv run python scripts/sweep_g2pl.py --db <sqlite path> --smoke   # 2 configs, 1 season
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from climbing_elo.engine.evaluation import (
    BacktestDataset,
    BacktestRunner,
    register_variant,
)
from climbing_elo.engine.g2pl import (
    DEFAULT_IMPORTANCE_WEIGHTS,
    G2PLConfig,
    G2PLEngine,
)
from climbing_elo.engine.oos_modes import WalkForwardMode
from climbing_elo.models import Discipline

log = logging.getLogger("sweep_g2pl")

# --- Pre-registered protocol constants (do not tweak mid-sweep) --------------

TUNE_SPAN = (2019, 2022)
CONFIRM_SPAN = (2023, 2025)
RNG_SEED = 42
N_SIMULATIONS = 10_000
DISCIPLINES = (Discipline.LEAD, Discipline.BOULDER)

W_SCALES = (0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
MOV_MODES = ("off", "margin")
EXPONENTS = (1.0, 0.5)

SPEARMAN_BAND = 0.95  # confirmation Spearman must be >= 95% of current's

METRIC_KEYS = (
    "log_loss_podium",
    "hit_rate_top3",
    "hit_rate_top1",
    "log_loss_win",
    "brier_podium",
    "mean_spearman",
    "n_rounds",
    "n_athlete_rounds",
)


def scaled_config(w_scale: float, mov_mode: str, exponent: float) -> G2PLConfig:
    """A G2PLConfig with every importance-weight cell multiplied by w_scale."""
    weights = {
        tier: {rt: w * w_scale for rt, w in row.items()}
        for tier, row in DEFAULT_IMPORTANCE_WEIGHTS.items()
    }
    return G2PLConfig(
        importance_weights=weights,
        mov_mode=mov_mode,
        field_normalization_exponent=exponent,
    )


def config_name(w_scale: float, mov_mode: str, exponent: float) -> str:
    return f"g2pl[s{w_scale:g},e{exponent:g},{mov_mode}]"


def _aggregate_subset(report) -> dict:
    return {k: report.aggregate.get(k) for k in METRIC_KEYS}


def run_backtest(
    variant: str,
    db_path: str,
    span: tuple[int, int],
    params: tuple[float, str, float] | None,
) -> dict:
    """One full walk-forward backtest; returns the aggregate metric subset.

    Module-level (and re-registering its own variant) so it survives the
    ProcessPoolExecutor pickle boundary — each worker process registers the
    parameterised factory itself before the runner looks it up.
    """
    if params is not None:
        cfg = scaled_config(*params)

        def factory(session, cutoff_date=None):
            return G2PLEngine(session, cutoff_date=cutoff_date, config=cfg)

        register_variant(variant, factory)

    dataset = BacktestDataset(
        disciplines=DISCIPLINES,
        n_simulations=N_SIMULATIONS,
        rng_seed=RNG_SEED,
        source_db_path=Path(db_path),
    )
    with BacktestRunner(
        dataset=dataset,
        variant=variant,
        oos_mode=WalkForwardMode(from_season=span[0], to_season=span[1]),
    ) as runner:
        report = runner.run()
    return _aggregate_subset(report)


def _worker(job: tuple) -> tuple:
    variant, db_path, span, params = job
    metrics = run_backtest(variant, db_path, span, params)
    return (variant, params, metrics)


def beats_baseline(metrics: dict, baseline: dict) -> bool:
    """The two pre-registered primaries."""
    return (
        metrics["log_loss_podium"] < baseline["log_loss_podium"]
        and metrics["hit_rate_top3"] > baseline["hit_rate_top3"]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--db", required=True, help="Source SQLite DB path")
    parser.add_argument("--out", default=None, help="Results JSON path")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="2 configs, tune on 2022 only, no confirmation (wiring check)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    tune_span = (2022, 2022) if args.smoke else TUNE_SPAN
    grid = [(s, m, e) for s in W_SCALES for m in MOV_MODES for e in EXPONENTS]
    if args.smoke:
        grid = grid[:2]

    results: dict = {
        "protocol": {
            "tune_span": list(tune_span),
            "confirm_span": list(CONFIRM_SPAN),
            "rng_seed": RNG_SEED,
            "n_simulations": N_SIMULATIONS,
            "grid": {
                "w_scales": list(W_SCALES),
                "mov_modes": list(MOV_MODES),
                "exponents": list(EXPONENTS),
            },
            "selection": "beat current on BOTH primaries on tune; max top-3, tie-break min podium LL",
            "ship_rule": f"confirmation: beat current on both primaries AND spearman >= {SPEARMAN_BAND} x current",
            "smoke": args.smoke,
        },
        "tune": {},
        "confirmation": {},
    }

    # --- Phase 1: current baseline on the tune folds -------------------------
    log.info("phase 1/4: current baseline on tune folds %s", tune_span)
    cur_tune = run_backtest("current", args.db, tune_span, None)
    results["tune"]["current"] = cur_tune
    log.info(
        "current tune: podium LL %.4f | top-3 %.4f | spearman %.4f",
        cur_tune["log_loss_podium"],
        cur_tune["hit_rate_top3"],
        cur_tune["mean_spearman"],
    )

    # --- Phase 2: the grid, in parallel ---------------------------------------
    log.info("phase 2/4: %d g2pl configs, %d workers", len(grid), args.workers)
    jobs = [(config_name(*params), args.db, tune_span, params) for params in grid]
    tune_rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for variant, params, metrics in pool.map(_worker, jobs):
            passes = beats_baseline(metrics, cur_tune)
            tune_rows.append(
                {
                    "name": variant,
                    "params": list(params),
                    "passes_tune": passes,
                    **metrics,
                }
            )
            log.info(
                "%-28s podium LL %.4f | top-3 %.4f | top-1 %.4f | rho %.4f %s",
                variant,
                metrics["log_loss_podium"],
                metrics["hit_rate_top3"],
                metrics["hit_rate_top1"],
                metrics["mean_spearman"],
                "PASS" if passes else "",
            )
    results["tune"]["configs"] = tune_rows

    # --- Phase 3: selection ----------------------------------------------------
    passers = [r for r in tune_rows if r["passes_tune"]]
    if not passers:
        log.info("phase 3/4: NO config beat current on both primaries — parking g2pl.")
        results["selection"] = None
        _write(results, args.out)
        return 2
    winner = sorted(passers, key=lambda r: (-r["hit_rate_top3"], r["log_loss_podium"]))[
        0
    ]
    results["selection"] = winner
    log.info("phase 3/4: selected %s (of %d passers)", winner["name"], len(passers))

    if args.smoke:
        log.info("smoke mode — skipping confirmation.")
        _write(results, args.out)
        return 0

    # --- Phase 4: one confirmation run ----------------------------------------
    log.info("phase 4/4: confirmation on %s (run once)", CONFIRM_SPAN)
    cur_confirm = run_backtest("current", args.db, CONFIRM_SPAN, None)
    win_confirm = run_backtest(
        winner["name"], args.db, CONFIRM_SPAN, tuple(winner["params"])
    )
    ship = beats_baseline(win_confirm, cur_confirm) and (
        win_confirm["mean_spearman"] >= SPEARMAN_BAND * cur_confirm["mean_spearman"]
    )
    results["confirmation"] = {
        "current": cur_confirm,
        "winner": {"name": winner["name"], **win_confirm},
        "ship_rule_passes": ship,
    }
    log.info(
        "VERDICT: %s | winner podium LL %.4f vs current %.4f | top-3 %.4f vs %.4f | rho %.4f vs %.4f",
        "SHIP RULE PASSES" if ship else "ship rule fails",
        win_confirm["log_loss_podium"],
        cur_confirm["log_loss_podium"],
        win_confirm["hit_rate_top3"],
        cur_confirm["hit_rate_top3"],
        win_confirm["mean_spearman"],
        cur_confirm["mean_spearman"],
    )
    _write(results, args.out)
    return 0 if ship else 1


def _write(results: dict, out: str | None) -> None:
    if out:
        path = Path(out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
        log.info("results -> %s", path)


if __name__ == "__main__":
    sys.exit(main())
