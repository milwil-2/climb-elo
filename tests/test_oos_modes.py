"""Tests for out-of-sample evaluation modes (Issue #39).

Coverage:

* Walk-forward: produces one split per consecutive ``(N, N+1)`` season pair,
  respects ``--from-season`` / ``--to-season`` clamping.
* Leave-one-event-out: produces one split per event in the chosen season,
  defaults to the most recent season when unspecified.
* Leave-one-athlete-out: produces one split covering the athlete's first
  ``K`` events; ``BacktestRunner`` emits ``convergence_trace.json`` with the
  agreed schema.
* Registry: all three modes register at import time.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
from sqlalchemy.orm import Session

from climbing_elo.engine.evaluation import (
    OOS_MODES,
    BacktestDataset,
    BacktestRunner,
)
from climbing_elo.engine.oos_modes import (
    LeaveOneAthleteOutMode,
    LeaveOneEventOutMode,
    WalkForwardMode,
)
from climbing_elo.models import (
    Athlete,
    Discipline,
    Event,
    EventTier,
    Gender,
    Result,
    Round,
    RoundType,
)


# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------


def _seed_event(
    session: Session,
    name: str,
    when: date,
    athletes: list[Athlete],
    final_order: list[int],
    discipline: Discipline = Discipline.LEAD,
    tier: EventTier = EventTier.WORLD_CUP,
) -> Event:
    event = Event(
        name=name,
        tier=tier,
        season=when.year,
        start_date=when,
        discipline=discipline,
    )
    session.add(event)
    session.flush()
    rnd = Round(
        event_id=event.id,
        round_type=RoundType.FINAL,
        gender=Gender.M,
        athlete_count=len(final_order),
    )
    session.add(rnd)
    session.flush()
    for rank, idx in enumerate(final_order, 1):
        session.add(Result(round_id=rnd.id, athlete_id=athletes[idx].id, rank=rank))
    session.flush()
    return event


@pytest.fixture
def six_athletes(db_session):
    athletes = []
    for name in ["Alpha", "Beta", "Gamma", "Delta", "Epsilon", "Zeta"]:
        a = Athlete(name=name, gender=Gender.M)
        db_session.add(a)
        athletes.append(a)
    db_session.flush()
    return athletes


@pytest.fixture
def three_season_dataset(db_session, six_athletes):
    """3 seasons × 2 events each."""
    _seed_event(db_session, "WC1 22", date(2022, 3, 1), six_athletes, [0, 1, 2, 3])
    _seed_event(db_session, "WC2 22", date(2022, 7, 1), six_athletes, [0, 1, 2, 3])
    _seed_event(db_session, "WC1 23", date(2023, 3, 1), six_athletes, [1, 0, 2, 3])
    _seed_event(db_session, "WC2 23", date(2023, 7, 1), six_athletes, [0, 2, 1, 3])
    _seed_event(db_session, "WC1 24", date(2024, 3, 1), six_athletes, [2, 0, 1, 3])
    _seed_event(db_session, "WC2 24", date(2024, 7, 1), six_athletes, [1, 2, 0, 3])
    db_session.commit()
    return six_athletes


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


def test_all_three_modes_registered():
    assert "walk-forward" in OOS_MODES
    assert "leave-one-event-out" in OOS_MODES
    assert "leave-one-athlete-out" in OOS_MODES


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------


def test_walk_forward_yields_one_split_per_consecutive_pair(
    db_session, three_season_dataset
):
    mode = WalkForwardMode()
    splits = mode.splits(db_session, Discipline.LEAD)
    # 3 seasons → 2 (N, N+1) folds: (2022→2023), (2023→2024).
    assert len(splits) == 2
    assert splits[0].label == "walk-forward-2022-to-2023"
    assert splits[1].label == "walk-forward-2023-to-2024"
    # Eval events for 2023 must be the two 2023 events, in order.
    assert len(splits[0].eval_event_ids) == 2
    # train_end_date must precede every eval event.
    assert splits[0].train_end_date == date(2023, 3, 1)


def test_walk_forward_respects_from_to_season(db_session, three_season_dataset):
    mode = WalkForwardMode(from_season=2024, to_season=2024)
    splits = mode.splits(db_session, Discipline.LEAD)
    assert len(splits) == 1
    assert splits[0].label == "walk-forward-2023-to-2024"


def test_walk_forward_empty_when_single_season(db_session, six_athletes):
    _seed_event(db_session, "WC1 22", date(2022, 3, 1), six_athletes, [0, 1, 2, 3])
    db_session.commit()
    assert WalkForwardMode().splits(db_session, Discipline.LEAD) == []


def test_walk_forward_name():
    assert WalkForwardMode().name() == "walk-forward"
    assert (
        WalkForwardMode(from_season=2023, to_season=2024).name()
        == "walk-forward-2023-2024"
    )


# ---------------------------------------------------------------------------
# Leave-one-event-out
# ---------------------------------------------------------------------------


def test_leave_one_event_out_one_split_per_event(db_session, three_season_dataset):
    mode = LeaveOneEventOutMode(season=2023)
    splits = mode.splits(db_session, Discipline.LEAD)
    assert len(splits) == 2
    # Each split evaluates exactly one event.
    for s in splits:
        assert len(s.eval_event_ids) == 1
    # train_end_date for each split should equal that event's start_date.
    dates = sorted(s.train_end_date for s in splits)
    assert dates == [date(2023, 3, 1), date(2023, 7, 1)]


def test_leave_one_event_out_defaults_to_latest_season(
    db_session, three_season_dataset
):
    mode = LeaveOneEventOutMode()  # season=None → max season
    splits = mode.splits(db_session, Discipline.LEAD)
    # 2024 has 2 events.
    assert len(splits) == 2
    for s in splits:
        assert s.label.startswith("leave-one-event-out-2024-event-")


def test_leave_one_event_out_empty_when_no_events(db_session):
    mode = LeaveOneEventOutMode(season=2099)
    assert mode.splits(db_session, Discipline.LEAD) == []


# ---------------------------------------------------------------------------
# Leave-one-athlete-out
# ---------------------------------------------------------------------------


def test_leave_one_athlete_out_single_split_first_k(db_session, three_season_dataset):
    target = three_season_dataset[1]  # Beta, in all 6 events
    mode = LeaveOneAthleteOutMode(athlete_id=target.id, tenure=3)
    splits = mode.splits(db_session, Discipline.LEAD)
    assert len(splits) == 1
    assert len(splits[0].eval_event_ids) == 3
    # Cutoff is the date of the athlete's first ever event (excludes it).
    assert splits[0].train_end_date == date(2022, 3, 1)


def test_leave_one_athlete_out_empty_when_no_events(db_session, six_athletes):
    mode = LeaveOneAthleteOutMode(athlete_id=six_athletes[0].id, tenure=3)
    assert mode.splits(db_session, Discipline.LEAD) == []


def test_leave_one_athlete_out_name():
    assert (
        LeaveOneAthleteOutMode(athlete_id=42, tenure=5).name()
        == "leave-one-athlete-out-42-K5"
    )


def test_leave_one_athlete_out_clamps_to_available_events(
    db_session, three_season_dataset
):
    """If K exceeds the athlete's actual event count, the split uses what it has."""
    target = three_season_dataset[0]  # Alpha — in 6 events
    mode = LeaveOneAthleteOutMode(athlete_id=target.id, tenure=1000)
    splits = mode.splits(db_session, Discipline.LEAD)
    assert len(splits) == 1
    assert len(splits[0].eval_event_ids) == 6


# ---------------------------------------------------------------------------
# Convergence trace artifact (BacktestRunner integration)
# ---------------------------------------------------------------------------


def test_convergence_trace_written_on_leave_one_athlete_out(
    db_session, three_season_dataset, tmp_path: Path
):
    target = three_season_dataset[0]  # Alpha
    dataset = BacktestDataset(
        disciplines=(Discipline.LEAD,),
        n_simulations=200,
        rng_seed=1,
    )
    out = tmp_path / "report-dir"
    with BacktestRunner(
        dataset=dataset,
        variant="current",
        oos_mode=LeaveOneAthleteOutMode(athlete_id=target.id, tenure=3),
        output_dir=out,
        in_memory_session=db_session,
    ) as runner:
        runner.run()

    trace_path = out / "convergence_trace.json"
    assert trace_path.exists(), "convergence_trace.json not emitted"

    trace = json.loads(trace_path.read_text())
    # Schema assertions.
    assert trace["athlete_id"] == target.id
    assert trace["athlete_name"] == "Alpha"
    assert trace["discipline"] == "L"
    assert trace["tenure_hidden"] == 3
    assert isinstance(trace["trace"], list)
    assert len(trace["trace"]) >= 1
    for entry in trace["trace"]:
        assert {
            "event_id",
            "event_date",
            "predicted_mu",
            "actual_finish_rank",
            "field_size",
        } <= set(entry.keys())
    # Trace ordered by event_date.
    dates = [e["event_date"] for e in trace["trace"]]
    assert dates == sorted(dates)


def test_convergence_trace_not_emitted_for_holdout(
    db_session, three_season_dataset, tmp_path: Path
):
    """Holdout (the default mode) must not emit a convergence trace."""
    from climbing_elo.engine.evaluation import HoldoutMode

    dataset = BacktestDataset(
        disciplines=(Discipline.LEAD,),
        n_simulations=200,
        rng_seed=1,
    )
    out = tmp_path / "holdout-report"
    with BacktestRunner(
        dataset=dataset,
        variant="current",
        oos_mode=HoldoutMode(n_seasons=1),
        output_dir=out,
        in_memory_session=db_session,
    ) as runner:
        runner.run()

    assert not (out / "convergence_trace.json").exists()


# ---------------------------------------------------------------------------
# End-to-end smoke: walk-forward + leave-one-event-out via runner
# ---------------------------------------------------------------------------


def test_walk_forward_runs_end_to_end(db_session, three_season_dataset):
    dataset = BacktestDataset(
        disciplines=(Discipline.LEAD,),
        n_simulations=200,
        rng_seed=1,
    )
    with BacktestRunner(
        dataset=dataset,
        variant="current",
        oos_mode=WalkForwardMode(),
        in_memory_session=db_session,
    ) as runner:
        report = runner.run()
    # 2 walk-forward folds.
    assert len(report.splits) == 2
    assert report.oos_mode == "walk-forward"


def test_leave_one_event_out_runs_end_to_end(db_session, three_season_dataset):
    dataset = BacktestDataset(
        disciplines=(Discipline.LEAD,),
        n_simulations=200,
        rng_seed=1,
    )
    with BacktestRunner(
        dataset=dataset,
        variant="current",
        oos_mode=LeaveOneEventOutMode(season=2024),
        in_memory_session=db_session,
    ) as runner:
        report = runner.run()
    assert len(report.splits) == 2
    assert report.oos_mode == "leave-one-event-out-2024"
