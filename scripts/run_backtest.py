#!/usr/bin/env python3
"""Backtest the Lead ELO model against historical results.

Splits data into training (all but last N seasons) and holdout (last N seasons).
For each holdout event final, predicts top-3 by pre-event ELO and compares to
actual podium. Reports hit-rate vs. an attendance-based baseline.

Methodology:
  1. Clear all ratings and rating history.
  2. Run backfill on training events only (start_date < cutoff).
  3. For each holdout event final, use the ELO ratings frozen at the end of
     training to predict the podium (athletes with no history get mu=1500).
  4. Baseline: predict by n_events attended (proxy for accumulated points).
"""
import logging
import sys
from datetime import date

from sqlalchemy import delete, func, select

from climbing_elo.database import init_db
from climbing_elo.engine.backfill import run_backfill
from climbing_elo.models import (
    Discipline,
    Event,
    Rating,
    RatingHistory,
    Result,
    RoundType,
)

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

HOLDOUT_SEASONS = 2


def main() -> None:
    SessionFactory = init_db()

    with SessionFactory() as session:
        max_season = session.execute(
            select(func.max(Event.season)).where(Event.discipline == Discipline.LEAD)
        ).scalar()

        if max_season is None:
            print("No Lead events in database. Run scrape_ifsc.py first.")
            sys.exit(1)

        cutoff_season = max_season - HOLDOUT_SEASONS + 1
        cutoff_date = date(cutoff_season, 1, 1)

        print(f"Max season:    {max_season}")
        print(f"Cutoff season: {cutoff_season}  (events before {cutoff_date} are training)")
        print(f"Training:      seasons < {cutoff_season}")
        print(f"Holdout:       seasons >= {cutoff_season}")
        print()

        # ------------------------------------------------------------------
        # 1. Clear all ratings so we start fresh
        # ------------------------------------------------------------------
        session.execute(delete(RatingHistory))
        session.execute(delete(Rating))
        session.commit()

        # ------------------------------------------------------------------
        # 2. Run backfill on training events only (end_date = cutoff_date)
        # ------------------------------------------------------------------
        training_count = session.execute(
            select(func.count(Event.id)).where(
                Event.discipline == Discipline.LEAD,
                Event.start_date < cutoff_date,
            )
        ).scalar()

        holdout_events = session.execute(
            select(Event)
            .where(
                Event.discipline == Discipline.LEAD,
                Event.start_date >= cutoff_date,
            )
            .order_by(Event.start_date.asc())
        ).scalars().all()

        print(f"Training events: {training_count}")
        print(f"Holdout events:  {len(holdout_events)}")
        print()

        report = run_backfill(session, Discipline.LEAD, end_date=cutoff_date)
        if report.errors:
            print(f"Backfill errors: {len(report.errors)}")

        # Snapshot training-end ratings (mu per athlete) for holdout evaluation.
        training_ratings: dict[int, float] = {}
        training_n_events: dict[int, int] = {}
        for rating in session.execute(
            select(Rating).where(Rating.discipline == Discipline.LEAD)
        ).scalars():
            training_ratings[rating.athlete_id] = rating.mu
            training_n_events[rating.athlete_id] = rating.n_events

        # ------------------------------------------------------------------
        # 3. Evaluate holdout finals using training-end ratings
        # ------------------------------------------------------------------
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

                # ELO prediction: use training-end mu (default 1500 if unknown)
                pre_event_ratings: dict[int, float] = {
                    res.athlete_id: training_ratings.get(res.athlete_id, 1500.0)
                    for res in results
                }
                elo_top3 = set(
                    sorted(pre_event_ratings, key=pre_event_ratings.get, reverse=True)[:3]
                )

                # Baseline: predict by most events attended
                event_counts: dict[int, int] = {
                    res.athlete_id: training_n_events.get(res.athlete_id, 0)
                    for res in results
                }
                baseline_top3 = set(
                    sorted(event_counts, key=event_counts.get, reverse=True)[:3]
                )

                elo_hit = len(elo_top3 & actual_podium) > 0
                baseline_hit = len(baseline_top3 & actual_podium) > 0

                elo_hits += int(elo_hit)
                baseline_hits += int(baseline_hit)
                total_finals += 1

                event_label = f"{event.name} ({event.season}) {rnd.gender.value}"
                elo_marker = "+" if elo_hit else "-"
                base_marker = "+" if baseline_hit else "-"
                print(f"  [ELO:{elo_marker}] [BASE:{base_marker}] {event_label}")

        if total_finals == 0:
            print("\nNo holdout finals to evaluate.")
            sys.exit(0)

        elo_rate = elo_hits / total_finals * 100
        baseline_rate = baseline_hits / total_finals * 100
        delta = elo_rate - baseline_rate

        print(f"\n{'='*55}")
        print(f"Results ({total_finals} finals evaluated):")
        print(f"  ELO podium hit-rate:      {elo_rate:.1f}%")
        print(f"  Baseline podium hit-rate: {baseline_rate:.1f}%")
        print(f"  Delta:                    {delta:+.1f} pp")
        print(f"  Target:                   +15 pp")
        print(f"  {'PASS' if delta >= 15 else 'BELOW TARGET'}")


if __name__ == "__main__":
    main()
