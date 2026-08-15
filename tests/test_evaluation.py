"""Tests for the backtest evaluation harness.

Coverage targets (from issue #37 acceptance criteria):
  - Smoke test on a fixture asserting report shape.
  - Reproducibility: two consecutive runs produce byte-identical JSON
    (after stripping the wall-clock ``generated_at`` field).
  - State safety: the source DB's ``Rating`` / ``RatingHistory`` tables are
    untouched after a run, even when the runner mutates the working copy.
  - Metric primitives: log-loss / Brier / Spearman / calibration on
    known-answer inputs.
  - Variant registry: register a no-op engine and confirm it is invoked.
"""

from __future__ import annotations

import json
import math
from datetime import date
from pathlib import Path

import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from climbing_elo.engine.backfill import run_backfill
from climbing_elo.engine.evaluation import (
    BACKTEST_VARIANTS,
    BacktestDataset,
    BacktestRunner,
    CurrentEloEngine,
    HoldoutMode,
    OOS_MODES,
    RatingForecast,
    _brier,
    _calibration_buckets,
    _field_size_bucket,
    _log_loss,
    _spearman,
    _tenure_bucket,
    register_oos_mode,
    register_variant,
    render_markdown,
)
from climbing_elo.models import (
    Athlete,
    Base,
    Discipline,
    Event,
    EventTier,
    Gender,
    Rating,
    RatingHistory,
    Result,
    Round,
    RoundType,
)


# ---------------------------------------------------------------------------
# Metric primitives
# ---------------------------------------------------------------------------


def test_log_loss_perfect_prediction_is_near_zero():
    # Predict 0.999 for actual=1, 0.001 for actual=0.
    loss = _log_loss([1, 0], [0.999, 0.001])
    assert loss < 0.01


def test_log_loss_worst_prediction_is_large():
    loss = _log_loss([1, 0], [0.001, 0.999])
    assert loss > 5.0


def test_log_loss_empty_is_nan():
    assert math.isnan(_log_loss([], []))


def test_brier_score_matches_known_value():
    # 2 outcomes, predicted 0.6 / 0.4, actuals 1 / 0 → ((0.6-1)^2 + (0.4-0)^2)/2 = 0.16
    b = _brier([1, 0], [0.6, 0.4])
    assert abs(b - 0.16) < 1e-9


def test_spearman_monotonic_is_one():
    rho = _spearman([1.0, 2.0, 3.0, 4.0], [10.0, 20.0, 30.0, 40.0])
    assert abs(rho - 1.0) < 1e-9


def test_spearman_reversed_is_minus_one():
    rho = _spearman([1.0, 2.0, 3.0, 4.0], [40.0, 30.0, 20.0, 10.0])
    assert abs(rho - (-1.0)) < 1e-9


def test_spearman_handles_ties():
    # Two ties on each side — should still be a finite number, not NaN.
    rho = _spearman([1.0, 1.0, 2.0, 3.0], [1.0, 1.0, 2.0, 3.0])
    assert not math.isnan(rho)
    assert rho > 0.0


def test_calibration_buckets_count_matches_total():
    y = [1, 0, 1, 1, 0]
    p = [0.9, 0.1, 0.7, 0.6, 0.2]
    buckets = _calibration_buckets(y, p, n_buckets=5)
    assert sum(b["count"] for b in buckets) == len(y)
    # Sanity: total buckets returned should always be n_buckets.
    assert len(buckets) == 5


def test_tenure_bucket_thresholds():
    assert _tenure_bucket(0) == "0"
    assert _tenure_bucket(1) == "1-3"
    assert _tenure_bucket(3) == "1-3"
    assert _tenure_bucket(4) == "4-10"
    assert _tenure_bucket(10) == "4-10"
    assert _tenure_bucket(11) == "11-30"
    assert _tenure_bucket(30) == "11-30"
    assert _tenure_bucket(31) == "30+"
    assert _tenure_bucket(1000) == "30+"


def test_field_size_bucket_thresholds():
    assert _field_size_bucket(3) == "<=8"
    assert _field_size_bucket(8) == "<=8"
    assert _field_size_bucket(9) == "9-20"
    assert _field_size_bucket(25) == "21-40"
    assert _field_size_bucket(80) == "41+"


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
def four_athletes(db_session):
    athletes = []
    for name in ["Alpha", "Beta", "Gamma", "Delta"]:
        a = Athlete(name=name, gender=Gender.M)
        db_session.add(a)
        athletes.append(a)
    db_session.flush()
    return athletes


@pytest.fixture
def small_lead_dataset(db_session, four_athletes):
    """3 training events + 2 holdout events spanning two seasons."""
    # Training (season 2023)
    _seed_event(
        db_session, "WC Innsbruck 23", date(2023, 3, 1), four_athletes, [0, 1, 2, 3]
    )
    _seed_event(
        db_session, "WC Chamonix 23", date(2023, 5, 1), four_athletes, [0, 1, 2, 3]
    )
    _seed_event(
        db_session, "WC Briancon 23", date(2023, 7, 1), four_athletes, [0, 1, 2, 3]
    )
    # Holdout (season 2024)
    _seed_event(
        db_session, "WC Innsbruck 24", date(2024, 3, 1), four_athletes, [0, 1, 2, 3]
    )
    _seed_event(
        db_session, "WC Chamonix 24", date(2024, 5, 1), four_athletes, [1, 0, 2, 3]
    )
    db_session.commit()
    return four_athletes


# ---------------------------------------------------------------------------
# Smoke test: in-memory runner end-to-end
# ---------------------------------------------------------------------------


def test_smoke_runs_end_to_end_in_memory(db_session, small_lead_dataset):
    dataset = BacktestDataset(
        disciplines=(Discipline.LEAD,),
        n_simulations=500,  # tiny for speed
        rng_seed=42,
    )
    with BacktestRunner(
        dataset=dataset,
        variant="current",
        oos_mode=HoldoutMode(n_seasons=1),
        in_memory_session=db_session,
    ) as runner:
        report = runner.run()

    # Report shape.
    assert report.variant == "current"
    assert report.oos_mode == "holdout-1s"
    assert report.disciplines == ["L"]
    assert len(report.splits) == 1
    split = report.splits[0]
    assert split["discipline"] == "L"
    assert "metrics" in split
    assert split["metrics"]["n_rounds"] >= 1

    # Aggregate carries the stratification cube.
    agg = report.aggregate
    strata = agg["stratifications"]
    for key in (
        "by_tier",
        "by_round",
        "by_discipline",
        "by_season",
        "by_field_size",
        "by_tenure",
    ):
        assert key in strata, f"missing stratification {key!r}"


def test_prebackfilled_source_does_not_leak_future_ratings(
    db_session, small_lead_dataset
):
    """#192: when the source DB is already fully backfilled (the normal case
    for prod snapshots), the harness must recompute ratings to the training
    cutoff — the idempotent backfill alone would silently keep the end-state
    ratings, handing the 'current' variant knowledge of the holdout events."""
    # Simulate a prod snapshot: backfill EVERYTHING, holdout season included.
    run_backfill(db_session, Discipline.LEAD)
    db_session.commit()
    end_state = {
        r.athlete_id: (r.mu, r.n_events)
        for r in db_session.execute(
            select(Rating).where(Rating.discipline == Discipline.LEAD)
        ).scalars()
    }
    assert all(n == 5 for _, n in end_state.values()), "expected 5 events end-state"

    dataset = BacktestDataset(
        disciplines=(Discipline.LEAD,),
        n_simulations=200,
        rng_seed=42,
    )
    with BacktestRunner(
        dataset=dataset,
        variant="current",
        oos_mode=HoldoutMode(n_seasons=1),
        in_memory_session=db_session,
    ) as runner:
        runner.run()

    # In-memory mode leaves the session at the (single) split's training-end
    # state: the three 2023 events only — never the pre-backfilled end state.
    trained = {
        r.athlete_id: (r.mu, r.n_events)
        for r in db_session.execute(
            select(Rating).where(Rating.discipline == Discipline.LEAD)
        ).scalars()
    }
    assert all(n == 3 for _, n in trained.values()), (
        f"training-end state must cover only 2023 events, got {trained}"
    )
    assert any(abs(trained[a][0] - end_state[a][0]) > 1e-6 for a in trained), (
        "μ at training cutoff should differ from end-state μ"
    )


def test_markdown_renders_without_error(db_session, small_lead_dataset):
    dataset = BacktestDataset(
        disciplines=(Discipline.LEAD,),
        n_simulations=200,
        rng_seed=42,
    )
    with BacktestRunner(
        dataset=dataset,
        variant="current",
        oos_mode=HoldoutMode(n_seasons=1),
        in_memory_session=db_session,
    ) as runner:
        report = runner.run()

    md = render_markdown(report)
    assert "# Backtest report" in md
    assert "## Aggregate metrics" in md
    assert "## Stratifications" in md
    assert "Log-loss" in md


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def _seed_two_season_dataset_in_db(db_path: Path) -> None:
    """Build a fresh on-disk DB with the same data we use for in-memory runs."""
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    session = factory()
    athletes: list[Athlete] = []
    for name in ["Alpha", "Beta", "Gamma", "Delta"]:
        a = Athlete(name=name, gender=Gender.M)
        session.add(a)
        athletes.append(a)
    session.flush()
    _seed_event(session, "WC Innsbruck 23", date(2023, 3, 1), athletes, [0, 1, 2, 3])
    _seed_event(session, "WC Chamonix 23", date(2023, 5, 1), athletes, [0, 1, 2, 3])
    _seed_event(session, "WC Briancon 23", date(2023, 7, 1), athletes, [0, 1, 2, 3])
    _seed_event(session, "WC Innsbruck 24", date(2024, 3, 1), athletes, [0, 1, 2, 3])
    _seed_event(session, "WC Chamonix 24", date(2024, 5, 1), athletes, [1, 0, 2, 3])
    session.commit()
    session.close()


def _strip_generated_at(payload: dict) -> dict:
    payload = dict(payload)
    payload.pop("generated_at", None)
    return payload


def test_two_runs_produce_identical_json(tmp_path: Path):
    """Two consecutive runs against an identical source DB must match byte-for-byte."""
    src_db = tmp_path / "source.db"
    _seed_two_season_dataset_in_db(src_db)

    dataset = BacktestDataset(
        disciplines=(Discipline.LEAD,),
        n_simulations=500,
        rng_seed=42,
        source_db_path=src_db,
    )

    with BacktestRunner(
        dataset=dataset,
        variant="current",
        oos_mode=HoldoutMode(n_seasons=1),
    ) as runner1:
        r1 = runner1.run()

    with BacktestRunner(
        dataset=dataset,
        variant="current",
        oos_mode=HoldoutMode(n_seasons=1),
    ) as runner2:
        r2 = runner2.run()

    j1 = json.loads(r1.to_json())
    j2 = json.loads(r2.to_json())
    assert _strip_generated_at(j1) == _strip_generated_at(j2)


# ---------------------------------------------------------------------------
# State safety
# ---------------------------------------------------------------------------


def test_production_db_untouched_after_run(tmp_path: Path):
    """Running the harness against a source DB must not mutate that DB.

    We compare every Rating/RatingHistory row before and after — if the
    harness "leaks" into the source the test fails.
    """
    src_db = tmp_path / "source.db"
    _seed_two_season_dataset_in_db(src_db)

    # Pre-populate the source DB with a sentinel Rating row so we can detect
    # accidental overwrites (the legacy backtest issued delete(Rating)).
    engine = create_engine(f"sqlite:///{src_db}")
    factory = sessionmaker(bind=engine)
    with factory() as s:
        a = s.execute(select(Athlete)).scalars().first()
        sentinel = Rating(
            athlete_id=a.id,
            discipline=Discipline.LEAD,
            mu=2500.0,  # implausible value — easy to spot if overwritten
            sigma=42.0,
            n_events=999,
            provisional=False,
        )
        s.add(sentinel)
        s.commit()
        before_rows = sorted(
            (r.athlete_id, r.discipline.value, r.mu, r.sigma, r.n_events)
            for r in s.execute(select(Rating)).scalars()
        )
        before_history_count = s.execute(select(RatingHistory)).all()

    dataset = BacktestDataset(
        disciplines=(Discipline.LEAD,),
        n_simulations=200,
        rng_seed=7,
        source_db_path=src_db,
    )
    with BacktestRunner(
        dataset=dataset,
        variant="current",
        oos_mode=HoldoutMode(n_seasons=1),
    ) as runner:
        runner.run()

    with factory() as s:
        after_rows = sorted(
            (r.athlete_id, r.discipline.value, r.mu, r.sigma, r.n_events)
            for r in s.execute(select(Rating)).scalars()
        )
        after_history_count = s.execute(select(RatingHistory)).all()

    assert before_rows == after_rows, (
        "Backtest mutated the source Rating table — state safety broken!"
    )
    assert len(before_history_count) == len(after_history_count), (
        "Backtest mutated the source RatingHistory table."
    )


# ---------------------------------------------------------------------------
# Variant + OOS registration
# ---------------------------------------------------------------------------


class _DummyEngine:
    """Constant μ=1500/σ=200 forecast for every athlete."""

    def __init__(self, session: Session):
        self.session = session
        self.calls = 0

    def name(self) -> str:
        return "dummy"

    def predict(
        self, athletes_in_round, discipline: Discipline
    ) -> dict[int, RatingForecast]:
        self.calls += 1
        return {
            aid: RatingForecast(athlete_id=aid, mu=1500.0, sigma=200.0, n_events=5)
            for aid in athletes_in_round
        }


def test_variant_registration_invokes_custom_engine(db_session, small_lead_dataset):
    register_variant("dummy_test", _DummyEngine)
    try:
        assert "dummy_test" in BACKTEST_VARIANTS
        dataset = BacktestDataset(
            disciplines=(Discipline.LEAD,),
            n_simulations=200,
            rng_seed=1,
        )
        with BacktestRunner(
            dataset=dataset,
            variant="dummy_test",
            oos_mode=HoldoutMode(n_seasons=1),
            in_memory_session=db_session,
        ) as runner:
            report = runner.run()
        assert report.variant == "dummy_test"
    finally:
        BACKTEST_VARIANTS.pop("dummy_test", None)


def test_unknown_variant_raises():
    with pytest.raises(ValueError, match="Unknown variant"):
        BacktestRunner(
            dataset=BacktestDataset(disciplines=(Discipline.LEAD,)),
            variant="this_variant_does_not_exist",
        )


def test_oos_mode_registry_holdout_present():
    assert "holdout" in OOS_MODES
    mode = OOS_MODES["holdout"](n_seasons=2)
    assert mode.name() == "holdout-2s"


def test_register_oos_mode_round_trip():
    class _NoopMode:
        def name(self) -> str:
            return "noop"

        def splits(self, session, discipline):
            return []

    register_oos_mode("noop_test", _NoopMode)
    try:
        assert "noop_test" in OOS_MODES
        assert OOS_MODES["noop_test"]().name() == "noop"
    finally:
        OOS_MODES.pop("noop_test", None)


# ---------------------------------------------------------------------------
# CurrentEloEngine snapshot behaviour
# ---------------------------------------------------------------------------


def test_current_engine_defaults_for_unknown_athletes(db_session):
    engine = CurrentEloEngine(db_session)
    forecasts = engine.predict([42, 99], Discipline.LEAD)
    assert forecasts[42].mu == 1500.0
    assert forecasts[42].sigma == 350.0
    assert forecasts[42].n_events == 0


def test_current_engine_reads_existing_ratings(db_session, four_athletes):
    db_session.add(
        Rating(
            athlete_id=four_athletes[0].id,
            discipline=Discipline.LEAD,
            mu=1800.0,
            sigma=120.0,
            n_events=15,
            provisional=False,
        )
    )
    db_session.commit()
    engine = CurrentEloEngine(db_session)
    forecasts = engine.predict(
        [four_athletes[0].id, four_athletes[1].id], Discipline.LEAD
    )
    assert forecasts[four_athletes[0].id].mu == 1800.0
    assert forecasts[four_athletes[0].id].n_events == 15
    # Unrated athlete falls back to defaults.
    assert forecasts[four_athletes[1].id].mu == 1500.0
