"""Tests for the backtest baseline engines (Issue #38).

Each baseline must:

  - Register itself in ``BACKTEST_VARIANTS`` at import time.
  - Conform to the :class:`RatingEngine` protocol from ``engine.evaluation``.
  - Produce sensible numeric forecasts (not NaN, μ in a reasonable range,
    σ > 0).
  - Run end-to-end through :class:`BacktestRunner` without crashing.

We also include a tiny side-by-side benchmark
(:func:`test_stripped_elo_vs_current_on_known_dataset`) that documents
whether the engine's bells-and-whistles actually buy us anything.
"""

from __future__ import annotations

import math
from datetime import date

import pytest
from sqlalchemy.orm import Session

from climbing_elo.engine.baselines import (
    AscentStatsEngine,
    IFSCOfficialEngine,
    PersistenceEngine,
    RandomEngine,
    StrippedEloEngine,
    _StrippedConfig,
    _RankSnapshotEngine,
)
from climbing_elo.engine.evaluation import (
    BACKTEST_VARIANTS,
    BacktestDataset,
    BacktestRunner,
    HoldoutMode,
    RatingForecast,
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
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _seed_final(
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
    for name in ["A", "B", "C", "D", "E", "F"]:
        a = Athlete(name=name, gender=Gender.M)
        db_session.add(a)
        athletes.append(a)
    db_session.flush()
    return athletes


@pytest.fixture
def baseline_dataset(db_session, six_athletes):
    """Two training seasons + one holdout season, deterministic ordering.

    Athletes A and B consistently finish 1–2 in training events; C, D, E, F
    cycle through 3–6. The holdout-season event repeats that order, so any
    engine that has learned "A and B are strong" should out-predict random.
    """
    base_orders_train = [
        [0, 1, 2, 3, 4, 5],
        [0, 1, 3, 2, 5, 4],
        [1, 0, 2, 4, 3, 5],
        [0, 1, 4, 5, 2, 3],
        [1, 0, 3, 5, 2, 4],
    ]
    _seed_final(
        db_session, "WC 22a", date(2022, 4, 1), six_athletes, base_orders_train[0]
    )
    _seed_final(
        db_session, "WC 22b", date(2022, 6, 1), six_athletes, base_orders_train[1]
    )
    _seed_final(
        db_session, "WC 22c", date(2022, 8, 1), six_athletes, base_orders_train[2]
    )
    _seed_final(
        db_session, "WC 23a", date(2023, 4, 1), six_athletes, base_orders_train[3]
    )
    _seed_final(
        db_session, "WC 23b", date(2023, 6, 1), six_athletes, base_orders_train[4]
    )
    # Holdout season — exact same dominance pattern.
    _seed_final(
        db_session, "WC 24a", date(2024, 4, 1), six_athletes, [0, 1, 2, 3, 4, 5]
    )
    _seed_final(
        db_session, "WC 24b", date(2024, 6, 1), six_athletes, [1, 0, 2, 3, 4, 5]
    )
    db_session.commit()
    return six_athletes


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_all_baselines_registered():
    for name in (
        "random",
        "persistence",
        "ifsc_official",
        "ascentstats",
        "stripped_elo",
    ):
        assert name in BACKTEST_VARIANTS, f"{name!r} not registered"


# ---------------------------------------------------------------------------
# RandomEngine
# ---------------------------------------------------------------------------


def test_random_engine_basic_predict(db_session, six_athletes):
    engine = RandomEngine(db_session)
    aids = [a.id for a in six_athletes]
    forecasts = engine.predict(aids, Discipline.LEAD)
    assert set(forecasts.keys()) == set(aids)
    for f in forecasts.values():
        assert isinstance(f, RatingForecast)
        assert not math.isnan(f.mu)
        assert RandomEngine.MU_LOW <= f.mu <= RandomEngine.MU_HIGH
        assert f.sigma > 0


def test_random_engine_seeded_deterministic(db_session, six_athletes):
    """Two engines built against the same session produce identical μ."""
    aids = [a.id for a in six_athletes]
    e1 = RandomEngine(db_session)
    e2 = RandomEngine(db_session)
    f1 = e1.predict(aids, Discipline.LEAD)
    f2 = e2.predict(aids, Discipline.LEAD)
    for aid in aids:
        assert f1[aid].mu == f2[aid].mu


def test_random_engine_end_to_end(db_session, baseline_dataset):
    dataset = BacktestDataset(
        disciplines=(Discipline.LEAD,), n_simulations=200, rng_seed=1
    )
    with BacktestRunner(
        dataset=dataset,
        variant="random",
        oos_mode=HoldoutMode(n_seasons=1),
        in_memory_session=db_session,
    ) as runner:
        report = runner.run()
    assert report.variant == "random"
    assert report.aggregate["n_rounds"] >= 1


# ---------------------------------------------------------------------------
# PersistenceEngine
# ---------------------------------------------------------------------------


def test_persistence_engine_unknown_athlete_returns_defaults(db_session, six_athletes):
    engine = PersistenceEngine(db_session)
    forecasts = engine.predict([six_athletes[0].id], Discipline.LEAD)
    f = forecasts[six_athletes[0].id]
    # No results in the DB yet — defaults expected.
    assert f.mu == 1500.0
    assert f.n_events == 0


def test_persistence_engine_recent_finish_drives_mu(db_session, six_athletes):
    # Two events: athlete 0 wins both, athlete 5 finishes last in both.
    _seed_final(
        db_session, "WC 22a", date(2022, 4, 1), six_athletes, [0, 1, 2, 3, 4, 5]
    )
    _seed_final(
        db_session, "WC 22b", date(2022, 6, 1), six_athletes, [0, 1, 2, 3, 4, 5]
    )
    db_session.commit()
    engine = PersistenceEngine(db_session)
    fc = engine.predict([a.id for a in six_athletes], Discipline.LEAD)
    # Winner has higher μ than last-place finisher.
    assert fc[six_athletes[0].id].mu > fc[six_athletes[5].id].mu
    # All seen athletes have n_events >= 1.
    for a in six_athletes:
        assert fc[a.id].n_events >= 1


def test_persistence_engine_end_to_end(db_session, baseline_dataset):
    dataset = BacktestDataset(
        disciplines=(Discipline.LEAD,), n_simulations=200, rng_seed=2
    )
    with BacktestRunner(
        dataset=dataset,
        variant="persistence",
        oos_mode=HoldoutMode(n_seasons=1),
        in_memory_session=db_session,
    ) as runner:
        report = runner.run()
    assert report.variant == "persistence"
    assert report.aggregate["n_rounds"] >= 1


# ---------------------------------------------------------------------------
# IFSCOfficialEngine
# ---------------------------------------------------------------------------


def test_ifsc_official_unmatched_athletes_get_defaults(db_session, six_athletes):
    """Athletes whose names aren't in any IFSC ranking fixture fall through
    to the default-μ branch (no rank → no opinion)."""
    engine = IFSCOfficialEngine(db_session)
    fc = engine.predict([a.id for a in six_athletes], Discipline.LEAD)
    for a in six_athletes:
        assert fc[a.id].mu == 1500.0
        assert fc[a.id].sigma == 350.0
        assert fc[a.id].n_events == 0


def test_ifsc_official_matches_known_athlete_and_derives_mu(db_session):
    """Seed athletes who appear in the most recent IFSC boulder fixture
    and confirm the engine produces rank-derived μ values (not defaults).

    The engine probes years descending from the current calendar year and
    uses the first year for which a fixture exists — currently 2025. Pick
    athletes whose rank in 2025 is unambiguous: Anraku #1 (high μ),
    Elias Arriagada Kruger #20 (low μ)."""
    a = Athlete(name="Sorato Anraku", gender=Gender.M)
    db_session.add(a)
    b = Athlete(name="Elias Arriagada Kruger", gender=Gender.M)  # rank 20 in 2025 BM
    db_session.add(b)
    # And one that isn't in the fixture at all.
    c = Athlete(name="Unknown Climber", gender=Gender.M)
    db_session.add(c)
    db_session.commit()

    engine = IFSCOfficialEngine(db_session)
    fc = engine.predict([a.id, b.id, c.id], Discipline.BOULDER)

    # Rank-1 athlete should sit well above default.
    assert fc[a.id].mu > 1500.0
    # Rank-20 sits well below.
    assert fc[b.id].mu < 1500.0
    # Unknown stays at default.
    assert fc[c.id].mu == 1500.0
    # And the rank-1 athlete should be ahead of the rank-20 athlete.
    assert fc[a.id].mu > fc[b.id].mu


def test_ifsc_official_handles_name_normalization(db_session):
    """Diacritic-stripped + override-keyed names still match.

    AscentStats / Wikipedia use 'Anze Peharc' (no diacritic) — our DB might
    store the Slovenian form. The normalize_name helper handles both
    directions.
    """
    a = Athlete(name="Anže Peharc", gender=Gender.M)  # rank 14 in 2024 BM
    db_session.add(a)
    db_session.commit()

    engine = IFSCOfficialEngine(db_session)
    fc = engine.predict([a.id], Discipline.BOULDER)
    # If normalization worked, we picked up rank 14 (≈ default territory)
    # and definitely have an n_events > 0 from the rank-table presence.
    assert fc[a.id].n_events > 0


def test_ifsc_official_end_to_end(db_session, baseline_dataset):
    dataset = BacktestDataset(
        disciplines=(Discipline.LEAD,), n_simulations=200, rng_seed=3
    )
    with BacktestRunner(
        dataset=dataset,
        variant="ifsc_official",
        oos_mode=HoldoutMode(n_seasons=1),
        in_memory_session=db_session,
    ) as runner:
        report = runner.run()
    assert report.variant == "ifsc_official"


# ---------------------------------------------------------------------------
# AscentStatsEngine
# ---------------------------------------------------------------------------


def test_ascentstats_matches_known_athlete_boulder(db_session):
    """Garnbret + Anraku are #1 in the 2026 boulder women / men fixtures."""
    janja = Athlete(name="Janja Garnbret", gender=Gender.F)
    anraku = Athlete(name="Sorato Anraku", gender=Gender.M)
    db_session.add_all([janja, anraku])
    db_session.commit()

    engine = AscentStatsEngine(db_session)
    fc = engine.predict([janja.id, anraku.id], Discipline.BOULDER)
    # Both are rank-1 in their respective gender brackets.
    assert fc[janja.id].mu > 1500.0
    assert fc[anraku.id].mu > 1500.0


def test_ascentstats_returns_defaults_for_lead(db_session):
    """AscentStats only publishes Boulder — Lead/Speed should fall through
    to the default-μ branch even for athletes whose names are in the
    boulder snapshot."""
    a = Athlete(name="Janja Garnbret", gender=Gender.F)
    db_session.add(a)
    db_session.commit()

    engine = AscentStatsEngine(db_session)
    fc = engine.predict([a.id], Discipline.LEAD)
    assert fc[a.id].mu == 1500.0
    assert fc[a.id].sigma == 350.0


def test_ascentstats_end_to_end(db_session, baseline_dataset):
    dataset = BacktestDataset(
        disciplines=(Discipline.LEAD,), n_simulations=200, rng_seed=5
    )
    with BacktestRunner(
        dataset=dataset,
        variant="ascentstats",
        oos_mode=HoldoutMode(n_seasons=1),
        in_memory_session=db_session,
    ) as runner:
        report = runner.run()
    assert report.variant == "ascentstats"


# ---------------------------------------------------------------------------
# StrippedEloEngine
# ---------------------------------------------------------------------------


def test_stripped_elo_basic_predict(db_session, baseline_dataset):
    engine = StrippedEloEngine(db_session)
    aids = [a.id for a in baseline_dataset]
    fc = engine.predict(aids, Discipline.LEAD)
    for aid in aids:
        f = fc[aid]
        assert not math.isnan(f.mu)
        # Stripped engine freezes σ at DEFAULT_SIGMA=350.
        assert f.sigma == 350.0


def test_stripped_elo_winners_have_higher_mu(db_session, baseline_dataset):
    """Athletes that consistently finished 1–2 in training should be the
    top-μ athletes under the stripped engine too."""
    engine = StrippedEloEngine(db_session)
    aids = [a.id for a in baseline_dataset]
    fc = engine.predict(aids, Discipline.LEAD)
    # Athletes 0 and 1 dominated training; expect them in the top-2 by μ.
    by_mu = sorted(aids, key=lambda aid: fc[aid].mu, reverse=True)
    top2 = set(by_mu[:2])
    assert top2 == {baseline_dataset[0].id, baseline_dataset[1].id}


def test_stripped_elo_does_not_leak_module_constants(db_session, baseline_dataset):
    """Verify that constructing the stripped engine doesn't leave the
    production ELO constants permanently mutated."""
    from climbing_elo.engine import elo as elo_mod

    before = (
        elo_mod.MARGIN_CAP,
        elo_mod.PROVISIONAL_K_MULTIPLIER,
        elo_mod.SIGMA_FLOOR,
        elo_mod.SIGMA_CEILING,
        elo_mod.SIGMA_CONVERGENCE_FACTOR,
    )
    engine = StrippedEloEngine(db_session)
    engine.predict([a.id for a in baseline_dataset], Discipline.LEAD)
    after = (
        elo_mod.MARGIN_CAP,
        elo_mod.PROVISIONAL_K_MULTIPLIER,
        elo_mod.SIGMA_FLOOR,
        elo_mod.SIGMA_CEILING,
        elo_mod.SIGMA_CONVERGENCE_FACTOR,
    )
    assert before == after, "stripped engine leaked module constants!"


def test_stripped_elo_config_is_actually_stripped():
    cfg = _StrippedConfig()
    assert cfg.margin_cap == 1.0
    assert cfg.provisional_k_multiplier == 1.0
    # σ floor == ceiling == default → no decay, no convergence headroom.
    assert cfg.sigma_floor == cfg.sigma_ceiling == 350.0
    assert cfg.sigma_convergence_factor == 1.0


def test_stripped_elo_end_to_end(db_session, baseline_dataset):
    dataset = BacktestDataset(
        disciplines=(Discipline.LEAD,), n_simulations=200, rng_seed=4
    )
    with BacktestRunner(
        dataset=dataset,
        variant="stripped_elo",
        oos_mode=HoldoutMode(n_seasons=1),
        in_memory_session=db_session,
    ) as runner:
        report = runner.run()
    assert report.variant == "stripped_elo"
    assert report.aggregate["n_rounds"] >= 1


# ---------------------------------------------------------------------------
# Side-by-side: stripped_elo vs current
# ---------------------------------------------------------------------------


def test_stripped_elo_vs_current_on_known_dataset(baseline_dataset, tmp_path):
    """Compare current vs stripped on a held-out season.

    This test is the heart of issue #38's acceptance criterion: if
    ``stripped_elo`` performs *at least as well* as ``current`` on any
    metric we measure, that feature isn't earning its keep. The test
    documents the gap rather than enforcing a direction — we want the
    negative result to be visible.
    """
    # Build a fresh on-disk DB so the two runs use their own working copies
    # without colliding via the in-memory session.
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    src = tmp_path / "side_by_side.db"
    engine = create_engine(f"sqlite:///{src}")
    from climbing_elo.models import Base

    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    session = factory()
    # Replicate the baseline_dataset structure on disk.
    athletes: list[Athlete] = []
    for name in ["A", "B", "C", "D", "E", "F"]:
        a = Athlete(name=name, gender=Gender.M)
        session.add(a)
        athletes.append(a)
    session.flush()
    orders = [
        ("WC 22a", date(2022, 4, 1), [0, 1, 2, 3, 4, 5]),
        ("WC 22b", date(2022, 6, 1), [0, 1, 3, 2, 5, 4]),
        ("WC 22c", date(2022, 8, 1), [1, 0, 2, 4, 3, 5]),
        ("WC 23a", date(2023, 4, 1), [0, 1, 4, 5, 2, 3]),
        ("WC 23b", date(2023, 6, 1), [1, 0, 3, 5, 2, 4]),
        ("WC 24a", date(2024, 4, 1), [0, 1, 2, 3, 4, 5]),
        ("WC 24b", date(2024, 6, 1), [1, 0, 2, 3, 4, 5]),
    ]
    for name, when, order in orders:
        _seed_final(session, name, when, athletes, order)
    session.commit()
    session.close()

    dataset = BacktestDataset(
        disciplines=(Discipline.LEAD,),
        n_simulations=2_000,
        rng_seed=42,
        source_db_path=src,
    )
    with BacktestRunner(
        dataset=dataset,
        variant="current",
        oos_mode=HoldoutMode(n_seasons=1),
    ) as r:
        current_report = r.run()
    with BacktestRunner(
        dataset=dataset,
        variant="stripped_elo",
        oos_mode=HoldoutMode(n_seasons=1),
    ) as r:
        stripped_report = r.run()

    cur_agg = current_report.aggregate
    str_agg = stripped_report.aggregate

    # Both variants must produce real numbers — no NaN, no missing keys.
    for agg in (cur_agg, str_agg):
        assert not math.isnan(agg["log_loss_podium"])
        assert not math.isnan(agg["brier_podium"])
        assert agg["n_rounds"] >= 1

    # Document the gap (helpful for the PR-body discussion).  This print
    # surfaces in pytest -s output and is not a hard assertion: a stripped
    # engine that matches current is a useful negative result.
    print(
        "\nstripped_elo vs current on synthetic holdout:\n"
        f"  current : LL pod={cur_agg['log_loss_podium']:.4f} | "
        f"brier pod={cur_agg['brier_podium']:.4f} | "
        f"top-3 hit={cur_agg['hit_rate_top3']:.4f}\n"
        f"  stripped: LL pod={str_agg['log_loss_podium']:.4f} | "
        f"brier pod={str_agg['brier_podium']:.4f} | "
        f"top-3 hit={str_agg['hit_rate_top3']:.4f}"
    )
