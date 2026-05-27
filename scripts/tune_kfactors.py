#!/usr/bin/env python3
"""Grid search to tune K-factor scaling and MARGIN_CAP.

For each combination of hyperparameters:
  1. Build a custom :class:`EloConfig` for the trial.
  2. Clear ratings and run backfill on training events only (passing the
     custom config via the new ``config=`` parameter on
     :func:`calculate_round_updates` — Issue #83 Target 3).
  3. Evaluate podium hit-rate on holdout event finals.

Prints all results sorted by ELO hit-rate, then outputs the best configuration.

NOTE
----
Post-#51 (Glicko-2 RD integration) the ``PROVISIONAL_K_MULTIPLIER`` knob is
retired — Glicko-2 handles cold start natively via high initial φ. A dedicated
follow-up (#80) tracks a proper Glicko-2-era regrid sweep that also varies the
GLICKO2_SIGMA_INACTIVITY / GLICKO2_TAU axes; that sweep should be implemented
against :class:`EloConfig` rather than monkey-patching module globals.
"""

import itertools
import sys
from dataclasses import replace
from datetime import date
from typing import Any

from sqlalchemy import delete, func, select

from climbing_elo.database import init_db
from climbing_elo.engine.backfill import run_backfill
from climbing_elo.engine.elo import DEFAULT_CONFIG, EloConfig
from climbing_elo.models import (
    Discipline,
    Event,
    EventTier,
    Rating,
    RatingHistory,
    Result,
    RoundType,
)

HOLDOUT_SEASONS = 2

# Base K-factor table (current defaults)
BASE_K_TABLE = {
    EventTier.OLYMPICS: {
        RoundType.FINAL: 48.0,
        RoundType.SEMI: 36.0,
        RoundType.QUALIFICATION: 18.0,
    },
    EventTier.WORLD_CHAMPIONSHIP: {
        RoundType.FINAL: 40.0,
        RoundType.SEMI: 30.0,
        RoundType.QUALIFICATION: 15.0,
    },
    EventTier.WORLD_CUP: {
        RoundType.FINAL: 32.0,
        RoundType.SEMI: 24.0,
        RoundType.QUALIFICATION: 12.0,
    },
    EventTier.CONTINENTAL: {
        RoundType.FINAL: 24.0,
        RoundType.SEMI: 18.0,
        RoundType.QUALIFICATION: 9.0,
    },
}

# Grid axes
K_SCALE_VALUES = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
MARGIN_CAP_VALUES = [1.5, 2.0, 2.5]
# Provisional K-multiplier was retired in #51; the axis is preserved as a
# single-valued list so the existing total_combos arithmetic still works.
PROVISIONAL_K_MULT_VALUES = [1.0]


def scale_k_table(scale: float) -> dict:
    """Return a new K-factor table with all values multiplied by scale."""
    scaled = {}
    for tier, rounds in BASE_K_TABLE.items():
        scaled[tier] = {rt: v * scale for rt, v in rounds.items()}
    return scaled


def build_config(k_scale: float, margin_cap: float) -> EloConfig:
    """Build an :class:`EloConfig` for one grid-search trial."""
    return replace(
        DEFAULT_CONFIG,
        margin_cap=margin_cap,
        k_factor_table=scale_k_table(k_scale),
    )


def run_evaluation(
    session,
    holdout_events: list,
    cutoff_date: date,
    config: EloConfig = DEFAULT_CONFIG,
) -> tuple[float, float]:
    """Run backfill on training data and evaluate holdout finals.

    Returns (elo_rate, baseline_rate) as percentages.
    """
    # Clear ratings
    session.execute(delete(RatingHistory))
    session.execute(delete(Rating))
    session.commit()

    # Run backfill on training only
    run_backfill(session, Discipline.LEAD, end_date=cutoff_date, config=config)

    # Snapshot training-end ratings
    training_ratings: dict[int, float] = {}
    training_n_events: dict[int, int] = {}
    for rating in session.execute(
        select(Rating).where(Rating.discipline == Discipline.LEAD)
    ).scalars():
        training_ratings[rating.athlete_id] = rating.mu
        training_n_events[rating.athlete_id] = rating.n_events

    elo_hits = 0
    baseline_hits = 0
    total_finals = 0

    for event in holdout_events:
        finals = [r for r in event.rounds if r.round_type == RoundType.FINAL]
        for rnd in finals:
            results = list(
                session.execute(
                    select(Result)
                    .where(Result.round_id == rnd.id, ~Result.dns)
                    .order_by(Result.rank.asc())
                ).scalars()
            )
            if len(results) < 3:
                continue

            actual_podium = {r.athlete_id for r in results[:3]}

            pre_event_ratings = {
                res.athlete_id: training_ratings.get(res.athlete_id, 1500.0)
                for res in results
            }
            elo_top3 = set(
                sorted(pre_event_ratings, key=pre_event_ratings.get, reverse=True)[:3]
            )

            event_counts = {
                res.athlete_id: training_n_events.get(res.athlete_id, 0)
                for res in results
            }
            baseline_top3 = set(
                sorted(event_counts, key=event_counts.get, reverse=True)[:3]
            )

            elo_hits += int(len(elo_top3 & actual_podium) > 0)
            baseline_hits += int(len(baseline_top3 & actual_podium) > 0)
            total_finals += 1

    if total_finals == 0:
        return 0.0, 0.0

    return elo_hits / total_finals * 100, baseline_hits / total_finals * 100


def main() -> None:
    SessionFactory = init_db()

    with SessionFactory() as session:
        max_season = session.execute(
            select(func.max(Event.season)).where(Event.discipline == Discipline.LEAD)
        ).scalar()

        if max_season is None:
            print("No Lead events in database.")
            sys.exit(1)

        cutoff_season = max_season - HOLDOUT_SEASONS + 1
        cutoff_date = date(cutoff_season, 1, 1)

        holdout_events = (
            session.execute(
                select(Event)
                .where(
                    Event.discipline == Discipline.LEAD,
                    Event.start_date >= cutoff_date,
                )
                .order_by(Event.start_date.asc())
            )
            .scalars()
            .all()
        )

        training_count = session.execute(
            select(func.count(Event.id)).where(
                Event.discipline == Discipline.LEAD,
                Event.start_date < cutoff_date,
            )
        ).scalar()

        total_combos = (
            len(K_SCALE_VALUES)
            * len(MARGIN_CAP_VALUES)
            * len(PROVISIONAL_K_MULT_VALUES)
        )

        print(f"Grid search: {total_combos} combinations")
        print(f"  K-scale:        {K_SCALE_VALUES}")
        print(f"  MARGIN_CAP:     {MARGIN_CAP_VALUES}")
        print(f"  PROV_K_MULT:    {PROVISIONAL_K_MULT_VALUES}")
        print(
            f"  Training events: {training_count}, holdout events: {len(holdout_events)}"
        )
        print()

        results_log: list[dict[str, Any]] = []
        combo_num = 0

        for k_scale, margin_cap, prov_mult in itertools.product(
            K_SCALE_VALUES, MARGIN_CAP_VALUES, PROVISIONAL_K_MULT_VALUES
        ):
            combo_num += 1

            # Build a custom EloConfig for this trial — no monkey-patching.
            trial_config = build_config(k_scale, margin_cap)
            # prov_mult is a no-op post-#51; retained for log compat.
            _ = prov_mult

            elo_rate, baseline_rate = run_evaluation(
                session, holdout_events, cutoff_date, config=trial_config
            )
            delta = elo_rate - baseline_rate

            results_log.append(
                {
                    "k_scale": k_scale,
                    "margin_cap": margin_cap,
                    "prov_mult": prov_mult,
                    "elo_rate": elo_rate,
                    "baseline_rate": baseline_rate,
                    "delta": delta,
                }
            )

            status = "PASS" if delta >= 15 else "FAIL"
            print(
                f"  [{combo_num:3d}/{total_combos}] "
                f"k={k_scale:.2f} cap={margin_cap:.1f} prov={prov_mult:.1f} → "
                f"ELO={elo_rate:.1f}% base={baseline_rate:.1f}% "
                f"Δ={delta:+.1f}pp [{status}]"
            )

        # No global state to restore — trials used per-trial EloConfig
        # instances rather than monkey-patching module globals.

        # Sort by ELO hit-rate descending
        results_log.sort(key=lambda x: (-x["elo_rate"], -x["delta"]))

        print()
        print("=" * 65)
        print("TOP 10 CONFIGURATIONS (by ELO hit-rate):")
        print("-" * 65)
        print(
            f"{'rank':>4}  {'k_scale':>7}  {'cap':>5}  {'prov':>5}  "
            f"{'ELO%':>6}  {'BASE%':>6}  {'delta':>7}"
        )
        print("-" * 65)
        for i, r in enumerate(results_log[:10], 1):
            print(
                f"  {i:>2}.  {r['k_scale']:>7.2f}  {r['margin_cap']:>5.1f}  "
                f"{r['prov_mult']:>5.1f}  "
                f"{r['elo_rate']:>6.1f}  {r['baseline_rate']:>6.1f}  "
                f"{r['delta']:>+7.1f}pp"
            )

        best = results_log[0]
        print()
        print("=" * 65)
        print("BEST CONFIGURATION:")
        print(f"  K-scale multiplier:        {best['k_scale']}")
        print(f"  MARGIN_CAP:                {best['margin_cap']}")
        print(f"  ELO podium hit-rate:       {best['elo_rate']:.1f}%")
        print(f"  Baseline hit-rate:         {best['baseline_rate']:.1f}%")
        print(f"  Delta:                     {best['delta']:+.1f} pp")
        print()
        print("Scaled K-factor table:")
        scaled = scale_k_table(best["k_scale"])
        for tier, rounds in scaled.items():
            for rt, v in rounds.items():
                print(f"  {tier.value} / {rt.value}: {v:.1f}")


if __name__ == "__main__":
    main()
