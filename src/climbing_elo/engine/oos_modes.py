"""Out-of-sample evaluation modes for the backtest harness (Issue #39).

This module plugs three additional :class:`OOSMode` implementations into the
registry defined in :mod:`climbing_elo.engine.evaluation`:

* :class:`WalkForwardMode` — train through season *N*, evaluate season *N+1*,
  advance one season, repeat. Yields one :class:`TrainEvalSplit` per
  consecutive ``(N, N+1)`` pair within the available season range. Aggregating
  across folds reveals how predictive accuracy evolves as the rating system
  matures.
* :class:`LeaveOneEventOutMode` — within a single season, hide each event in
  turn from training, predict it, rotate. Produces one split per event.
  Computationally heavier than holdout / walk-forward: each split forces a
  fresh ``run_backfill`` call inside :class:`BacktestRunner`, so callers
  should expect *N* full backfills for a season with *N* events.
* :class:`LeaveOneAthleteOutMode` — hide a single athlete's first *K* events
  from training (cold-start scenario). The evaluation split contains exactly
  those *K* events. This is the cold-start diagnostic R1 (Glicko-2) will
  evaluate against. The runner additionally writes a ``convergence_trace.json``
  artifact next to the standard report so downstream tooling can compare how
  fast different engines converge from their cold-start prior.

All three modes register themselves into ``OOS_MODES`` at import time. To
activate them, :mod:`climbing_elo.engine.evaluation` imports this module via a
single ``from . import oos_modes`` line.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from climbing_elo.engine.evaluation import (
    TrainEvalSplit,
    register_oos_mode,
)
from climbing_elo.models import Athlete, Discipline, Event, Result, Round


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seasons_for_discipline(session: Session, discipline: Discipline) -> list[int]:
    """Return sorted distinct seasons that have at least one event."""
    rows = session.execute(
        select(Event.season)
        .where(Event.discipline == discipline)
        .group_by(Event.season)
        .order_by(Event.season.asc())
    ).scalars()
    return list(rows)


def _events_in_season(
    session: Session, discipline: Discipline, season: int
) -> list[Event]:
    """Return events for a discipline+season ordered by start_date."""
    return list(
        session.execute(
            select(Event)
            .where(
                Event.discipline == discipline,
                Event.season == season,
            )
            .order_by(Event.start_date.asc(), Event.id.asc())
        ).scalars()
    )


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------


@dataclass
class WalkForwardMode:
    """Walk-forward season-by-season evaluation.

    For each season ``N`` in the available range (optionally restricted via
    ``from_season`` / ``to_season``), produces a split that trains on
    everything through end of season ``N`` and evaluates every event in
    season ``N+1``. Folds where ``N+1`` has no events are skipped.

    Parameters
    ----------
    from_season:
        Inclusive lower bound for the *eval* season (i.e. the first ``N+1``
        considered). If ``None``, walks from the second-earliest season.
    to_season:
        Inclusive upper bound for the *eval* season. If ``None``, walks up
        to the latest season present.
    """

    from_season: int | None = None
    to_season: int | None = None

    def name(self) -> str:
        if self.from_season is None and self.to_season is None:
            return "walk-forward"
        lo = self.from_season if self.from_season is not None else "*"
        hi = self.to_season if self.to_season is not None else "*"
        return f"walk-forward-{lo}-{hi}"

    def splits(
        self,
        session: Session,
        discipline: Discipline,
    ) -> list[TrainEvalSplit]:
        seasons = _seasons_for_discipline(session, discipline)
        if len(seasons) < 2:
            return []

        out: list[TrainEvalSplit] = []
        for i in range(len(seasons) - 1):
            train_season = seasons[i]
            eval_season = seasons[i + 1]
            if self.from_season is not None and eval_season < self.from_season:
                continue
            if self.to_season is not None and eval_season > self.to_season:
                continue

            eval_events = _events_in_season(session, discipline, eval_season)
            if not eval_events:
                continue

            # train_end_date is the start_date of the earliest eval event —
            # backfill includes events strictly before this date.
            train_end_date = min(e.start_date for e in eval_events)
            out.append(
                TrainEvalSplit(
                    label=f"walk-forward-{train_season}-to-{eval_season}",
                    train_end_date=train_end_date,
                    eval_event_ids=tuple(e.id for e in eval_events),
                )
            )
        return out


# ---------------------------------------------------------------------------
# Leave-one-event-out
# ---------------------------------------------------------------------------


@dataclass
class LeaveOneEventOutMode:
    """Within a season, hide each event in turn from training.

    For each event *E* in the chosen season (most recent complete season by
    default), produces a split with ``train_end_date = E.start_date`` and
    ``eval_event_ids = (E.id,)``. Each split therefore evaluates one event
    against ratings computed from every event strictly before it (across
    *all* seasons, not just the chosen one — full training history is used).

    .. warning::

        Computationally heavier than holdout / walk-forward — each split
        triggers an independent ``run_backfill`` call inside
        :class:`BacktestRunner`. Callers should expect *N* runs of
        ``run_backfill`` for a season with *N* events.

    Parameters
    ----------
    season:
        Season to rotate over. If ``None``, defaults to the most recent
        complete season (the latest ``Event.season`` in the DB for the
        discipline).
    """

    season: int | None = None

    def name(self) -> str:
        if self.season is None:
            return "leave-one-event-out"
        return f"leave-one-event-out-{self.season}"

    def splits(
        self,
        session: Session,
        discipline: Discipline,
    ) -> list[TrainEvalSplit]:
        target_season = self.season
        if target_season is None:
            target_season = session.execute(
                select(func.max(Event.season)).where(Event.discipline == discipline)
            ).scalar()
        if target_season is None:
            return []

        events = _events_in_season(session, discipline, target_season)
        if not events:
            return []

        out: list[TrainEvalSplit] = []
        for ev in events:
            out.append(
                TrainEvalSplit(
                    label=f"leave-one-event-out-{target_season}-event-{ev.id}",
                    train_end_date=ev.start_date,
                    eval_event_ids=(ev.id,),
                )
            )
        return out


# ---------------------------------------------------------------------------
# Leave-one-athlete-out (cold-start convergence diagnostic)
# ---------------------------------------------------------------------------


@dataclass
class LeaveOneAthleteOutMode:
    """Cold-start diagnostic — hide a single athlete's first *K* events.

    The training cutoff is set so that none of the athlete's first ``tenure``
    events are present during backfill. The eval split is exactly those *K*
    events. This produces a "convergence trace" showing how the rating engine
    converges from its cold-start prior toward the athlete's true skill as
    they accumulate results.

    The :class:`BacktestRunner` recognises this mode (via the
    :attr:`emit_convergence_trace` flag) and writes an additional
    ``convergence_trace.json`` artifact alongside the standard report.

    Parameters
    ----------
    athlete_id:
        ID of the athlete whose cold-start to evaluate.
    tenure:
        Number of earliest events to hide from training (default ``5``).
        Must be ``>= 1``.
    """

    athlete_id: int
    tenure: int = 5

    # Marker consumed by BacktestRunner to enable convergence-trace emission.
    emit_convergence_trace: bool = field(default=True, init=False)

    def name(self) -> str:
        return f"leave-one-athlete-out-{self.athlete_id}-K{self.tenure}"

    def _athlete_events(
        self,
        session: Session,
        discipline: Discipline,
    ) -> list[Event]:
        """Return all events the athlete competed in, ordered by start_date."""
        stmt = (
            select(Event)
            .join(Round, Round.event_id == Event.id)
            .join(Result, Result.round_id == Round.id)
            .where(
                Event.discipline == discipline,
                Result.athlete_id == self.athlete_id,
            )
            .group_by(Event.id)
            .order_by(Event.start_date.asc(), Event.id.asc())
        )
        return list(session.execute(stmt).scalars())

    def splits(
        self,
        session: Session,
        discipline: Discipline,
    ) -> list[TrainEvalSplit]:
        if self.tenure < 1:
            return []
        events = self._athlete_events(session, discipline)
        if not events:
            return []
        first_k = events[: self.tenure]
        if not first_k:
            return []

        # train_end_date excludes the athlete's first event (and everything
        # on/after that date) so no information about them leaks into the
        # training pass.
        cutoff = min(e.start_date for e in first_k)
        return [
            TrainEvalSplit(
                label=f"leave-one-athlete-out-{self.athlete_id}-K{self.tenure}",
                train_end_date=cutoff,
                eval_event_ids=tuple(e.id for e in first_k),
            )
        ]


# ---------------------------------------------------------------------------
# Convergence trace artifact
# ---------------------------------------------------------------------------


def build_convergence_trace(
    session: Session,
    mode: LeaveOneAthleteOutMode,
    discipline: Discipline,
    predictions: list[Any],
) -> dict[str, Any]:
    """Construct the convergence-trace artifact dict.

    Parameters
    ----------
    session:
        Working session — used to read the athlete's name.
    mode:
        The :class:`LeaveOneAthleteOutMode` instance whose split was scored.
    discipline:
        Discipline scored.
    predictions:
        The list of :class:`climbing_elo.engine.evaluation.RoundPrediction`
        objects produced for this split. We pull the athlete's predicted μ
        and actual finishing rank out of each round.

    Returns
    -------
    dict
        A JSON-serialisable mapping with the schema documented in the
        issue:

        ``{"athlete_id", "athlete_name", "discipline", "tenure_hidden",
        "trace": [{event_id, event_date, predicted_mu, actual_finish_rank,
        field_size}, ...]}``
    """
    athlete = session.get(Athlete, mode.athlete_id)
    athlete_name = athlete.name if athlete is not None else "<unknown>"

    # Group predictions by event_id; if multiple rounds in the same event
    # have the athlete, the *latest* round (final > semi > qualification)
    # wins so the trace reflects the athlete's furthest progression.
    round_order = {"qualification": 0, "semi": 1, "final": 2}
    by_event: dict[int, dict[str, Any]] = {}
    event_dates: dict[int, date] = {}

    for rp in predictions:
        # Find athlete in this round.
        ath_record = next(
            (a for a in rp.athletes if a["athlete_id"] == mode.athlete_id),
            None,
        )
        if ath_record is None:
            continue
        ord_key = round_order.get(rp.round_type, 0)
        prev = by_event.get(rp.event_id)
        prev_ord = round_order.get(prev["round_type"], -1) if prev else -1
        if prev is None or ord_key >= prev_ord:
            by_event[rp.event_id] = {
                "event_id": rp.event_id,
                "predicted_mu": ath_record["mu"],
                "actual_finish_rank": ath_record["actual_rank"],
                "field_size": rp.field_size,
                "round_type": rp.round_type,
            }

    # Resolve event dates from the DB for stable ordering / schema.
    for eid in by_event:
        ev = session.get(Event, eid)
        if ev is not None:
            event_dates[eid] = ev.start_date

    trace = []
    for eid in sorted(by_event.keys(), key=lambda i: (event_dates.get(i, date.min), i)):
        rec = by_event[eid]
        trace.append(
            {
                "event_id": rec["event_id"],
                "event_date": event_dates.get(eid, date.min).isoformat(),
                "predicted_mu": rec["predicted_mu"],
                "actual_finish_rank": rec["actual_finish_rank"],
                "field_size": rec["field_size"],
            }
        )

    return {
        "athlete_id": mode.athlete_id,
        "athlete_name": athlete_name,
        "discipline": discipline.value,
        "tenure_hidden": mode.tenure,
        "trace": trace,
    }


def write_convergence_trace(
    output_dir: Path,
    trace: dict[str, Any],
) -> Path:
    """Write the convergence-trace JSON to ``output_dir/convergence_trace.json``.

    Returns the path written. Uses ``sort_keys=True`` for byte-stable output
    (matches the main report's serialisation policy).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "convergence_trace.json"
    path.write_text(json.dumps(trace, sort_keys=True, indent=2, default=str) + "\n")
    return path


# ---------------------------------------------------------------------------
# Registry wiring — runs at import time
# ---------------------------------------------------------------------------


register_oos_mode("walk-forward", WalkForwardMode)
register_oos_mode("leave-one-event-out", LeaveOneEventOutMode)
register_oos_mode("leave-one-athlete-out", LeaveOneAthleteOutMode)


__all__ = [
    "WalkForwardMode",
    "LeaveOneEventOutMode",
    "LeaveOneAthleteOutMode",
    "build_convergence_trace",
    "write_convergence_trace",
]
