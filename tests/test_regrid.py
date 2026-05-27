"""Tests for the K_FACTOR_TABLE regrid sweep script (#80).

The regrid is exploratory — it doesn't ship engine code, just a script that
writes a markdown report. These tests focus on:

* End-to-end smoke: ``run_regrid`` completes against a tiny synthetic SQLite
  DB and writes a non-empty report.
* Coverage: the recommended K table contains every cell that exists in the
  module default.
* No monkey-patching: ``elo.K_FACTOR_TABLE`` and ``DEFAULT_CONFIG.k_factor_table``
  are unchanged after a regrid run — all sweeps are driven through fresh
  :class:`EloConfig` instances.
"""

from __future__ import annotations

import copy
from datetime import date
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from climbing_elo.engine import elo as elo_module
from climbing_elo.models import (
    Athlete,
    Base,
    Discipline,
    Event,
    EventTier,
    Gender,
    Result,
    Round,
    RoundType,
)
from scripts.regrid_k_factors import (
    ALL_ROUND_TYPES,
    ALL_TIERS,
    CellRecommendation,
    RegridReport,
    _pick_winner,
    _populated_cells,
    run_regrid,
)


# ---------------------------------------------------------------------------
# Synthetic dataset — small enough to backfill in <1s.
# ---------------------------------------------------------------------------


def _seed_synthetic_db(db_path: Path) -> None:
    """Seed a SQLite file with 6 athletes × 4 events / discipline for LEAD.

    The shape is identical to the production schema; the regrid harness
    only requires real events / rounds / results — it doesn't care about
    other dimensions like external rankings.
    """
    engine = create_engine(f"sqlite:///{db_path}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    session = factory()

    athletes = []
    for name in ["A", "B", "C", "D", "E", "F"]:
        a = Athlete(name=name, gender=Gender.M)
        session.add(a)
        athletes.append(a)
    session.flush()

    # 4 events across 4 different years so the holdout-2s split has both
    # training events (years 1+2) and eval events (years 3+4).
    schedule = [
        (date(2022, 6, 1), "WC 22"),
        (date(2023, 6, 1), "WC 23"),
        (date(2024, 6, 1), "WC 24"),
        (date(2025, 6, 1), "WC 25"),
    ]
    # Vary the finishing order per event so ratings actually move.
    orders = [
        [0, 1, 2, 3, 4, 5],
        [1, 0, 3, 2, 4, 5],
        [0, 2, 1, 3, 5, 4],
        [2, 0, 1, 4, 3, 5],
    ]
    for (d, name), order in zip(schedule, orders):
        event = Event(
            name=name,
            tier=EventTier.WORLD_CUP,
            season=d.year,
            start_date=d,
            discipline=Discipline.LEAD,
        )
        session.add(event)
        session.flush()

        # Two rounds: qual + final. Same finishing order in both for simplicity.
        for rt, count in [(RoundType.QUALIFICATION, 6), (RoundType.FINAL, 6)]:
            rnd = Round(
                event_id=event.id,
                round_type=rt,
                gender=Gender.M,
                athlete_count=count,
            )
            session.add(rnd)
            session.flush()
            for rank, athlete_idx in enumerate(order, 1):
                session.add(
                    Result(
                        round_id=rnd.id,
                        athlete_id=athletes[athlete_idx].id,
                        rank=rank,
                    )
                )
        session.flush()

    session.commit()
    session.close()
    engine.dispose()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_regrid_smoke_runs_against_synthetic_db(tmp_path: Path) -> None:
    """End-to-end smoke: regrid completes on a tiny DB and writes a report."""
    db_path = tmp_path / "synthetic.db"
    _seed_synthetic_db(db_path)
    report_path = tmp_path / "report.md"

    report = run_regrid(
        source_db=db_path,
        output_path=report_path,
        grid=(0.5, 1.0, 2.0),
        passes=1,
        n_sims=100,
        rng_seed=42,
        holdout_seasons=2,
        disciplines=(Discipline.LEAD,),
        progress=False,
    )

    # Returned report is well-formed.
    assert isinstance(report, RegridReport)
    assert report.grid == (0.5, 1.0, 2.0)
    assert report.passes == 1
    assert report.disciplines == (Discipline.LEAD,)

    # Markdown landed and is non-empty.
    assert report_path.exists()
    text = report_path.read_text()
    assert "K_FACTOR_TABLE regrid report" in text
    assert "Recommended K_FACTOR_TABLE" in text
    assert "_DEFAULT_K_FACTORS" in text


def test_recommended_table_covers_every_default_cell(tmp_path: Path) -> None:
    """The recommended K table must contain every (tier, round_type) cell.

    Empty cells (no rounds in the test DB) should be held at the current
    default value, not omitted from the output.
    """
    db_path = tmp_path / "synthetic.db"
    _seed_synthetic_db(db_path)
    report_path = tmp_path / "report.md"

    report = run_regrid(
        source_db=db_path,
        output_path=report_path,
        grid=(0.5, 1.0, 2.0),
        passes=1,
        n_sims=50,
        rng_seed=42,
        disciplines=(Discipline.LEAD,),
        progress=False,
    )

    # Every cell present in DEFAULT_CONFIG must be present in the recommendation.
    for tier in elo_module.DEFAULT_CONFIG.k_factor_table:
        assert tier in report.recommended_k_table, (
            f"Missing tier {tier} in recommended table"
        )
        for rt in elo_module.DEFAULT_CONFIG.k_factor_table[tier]:
            assert rt in report.recommended_k_table[tier], (
                f"Missing {tier}/{rt} in recommended table"
            )


def test_regrid_does_not_monkey_patch_module_globals(tmp_path: Path) -> None:
    """Issue #83 T3 contract — sweep trials must NOT mutate elo module state.

    Snapshots ``elo.K_FACTOR_TABLE`` and ``DEFAULT_CONFIG.k_factor_table``
    around the regrid call and asserts both are byte-identical afterwards.
    """
    db_path = tmp_path / "synthetic.db"
    _seed_synthetic_db(db_path)
    report_path = tmp_path / "report.md"

    pre_k_table = copy.deepcopy(elo_module.K_FACTOR_TABLE)
    pre_default = copy.deepcopy(elo_module.DEFAULT_CONFIG.k_factor_table)

    run_regrid(
        source_db=db_path,
        output_path=report_path,
        grid=(0.5, 2.0),
        passes=1,
        n_sims=50,
        rng_seed=42,
        disciplines=(Discipline.LEAD,),
        progress=False,
    )

    assert elo_module.K_FACTOR_TABLE == pre_k_table, (
        "elo.K_FACTOR_TABLE was mutated by the regrid run"
    )
    assert elo_module.DEFAULT_CONFIG.k_factor_table == pre_default, (
        "DEFAULT_CONFIG.k_factor_table was mutated by the regrid run"
    )


def test_populated_cells_filter(tmp_path: Path) -> None:
    """Only (tier, round_type) pairs that appear in the DB are reported active."""
    db_path = tmp_path / "synthetic.db"
    _seed_synthetic_db(db_path)
    populated = _populated_cells(db_path, [Discipline.LEAD])
    assert (EventTier.WORLD_CUP, RoundType.QUALIFICATION) in populated
    assert (EventTier.WORLD_CUP, RoundType.FINAL) in populated
    # Synthetic data has no SEMI rounds, no OLYMPICS, no CONTINENTAL.
    assert (EventTier.WORLD_CUP, RoundType.SEMI) not in populated
    assert (EventTier.OLYMPICS, RoundType.FINAL) not in populated
    assert (EventTier.CONTINENTAL, RoundType.FINAL) not in populated


def test_pick_winner_prefers_in_band_over_out_of_band() -> None:
    """When some trials land in [min,max], the picker must prefer them."""
    from scripts.regrid_k_factors import SweepResult

    trials = [
        SweepResult(
            tier=EventTier.WORLD_CUP,
            round_type=RoundType.FINAL,
            multiplier=0.5,
            k_value=10.0,
            top3_hit_rate=0.85,
            log_loss_podium=0.2,
            log_loss_win=0.1,
            brier_podium=0.05,
            mu_min=1000,
            mu_p50=1500,
            mu_p95=2000,  # in band
            mu_p99=2100,
            mu_max=2200,
        ),
        SweepResult(
            tier=EventTier.WORLD_CUP,
            round_type=RoundType.FINAL,
            multiplier=2.0,
            k_value=40.0,
            top3_hit_rate=0.90,  # better top-3, but out of band
            log_loss_podium=0.2,
            log_loss_win=0.1,
            brier_podium=0.05,
            mu_min=1000,
            mu_p50=1500,
            mu_p95=3000,  # OUT of band
            mu_p99=3500,
            mu_max=4000,
        ),
    ]
    winner, held = _pick_winner(
        cell_baseline_score=0.80,
        cell_baseline_p95=2500.0,  # baseline out of band
        trials=trials,
        mu_p95_min=1900.0,
        mu_p95_max=2200.0,
        tolerance_pp=0.0,
    )
    assert winner.k_value == 10.0  # in-band winner picked despite lower top-3
    assert held is False


def test_pick_winner_holds_when_no_signal() -> None:
    """Cells where every trial produces identical metrics must hold at 1.0x.

    This guards against the "noise winner" pathology where empty-data cells
    flip K randomly between zero-impact multipliers.
    """
    from scripts.regrid_k_factors import SweepResult

    trials = [
        SweepResult(
            tier=EventTier.OLYMPICS,
            round_type=RoundType.SEMI,
            multiplier=mult,
            k_value=mult * 36.0,
            top3_hit_rate=0.85,  # identical across trials
            log_loss_podium=0.2,
            log_loss_win=0.1,
            brier_podium=0.05,
            mu_min=1000,
            mu_p50=1500,
            mu_p95=2050,  # identical
            mu_p99=2100,
            mu_max=2200,
        )
        for mult in (0.5, 1.0, 2.0)
    ]
    winner, held = _pick_winner(
        cell_baseline_score=0.85,
        cell_baseline_p95=2050.0,
        trials=trials,
        mu_p95_min=1900.0,
        mu_p95_max=2200.0,
        tolerance_pp=0.0,
    )
    assert held is True
    assert winner.multiplier == 1.0


def test_cell_recommendation_unchanged_empty_cells_emit_no_trials(
    tmp_path: Path,
) -> None:
    """Cells with no DB data should be in the report but carry no trials."""
    db_path = tmp_path / "synthetic.db"
    _seed_synthetic_db(db_path)
    report = run_regrid(
        source_db=db_path,
        output_path=tmp_path / "out.md",
        grid=(0.5, 2.0),
        passes=1,
        n_sims=50,
        rng_seed=42,
        disciplines=(Discipline.LEAD,),
        progress=False,
    )

    # Find a known-empty cell — OLYMPICS / SEMI.
    olympics_semi: CellRecommendation | None = next(
        (
            c
            for c in report.cells
            if c.tier == EventTier.OLYMPICS and c.round_type == RoundType.SEMI
        ),
        None,
    )
    assert olympics_semi is not None
    assert olympics_semi.trials == []
    assert olympics_semi.held_unchanged is True
    assert olympics_semi.recommended_k == olympics_semi.baseline_k


def test_default_cells_cover_full_tier_round_matrix() -> None:
    """Sanity: the ALL_TIERS × ALL_ROUND_TYPES product must cover every default cell."""
    matrix = {(t, rt) for t in ALL_TIERS for rt in ALL_ROUND_TYPES}
    default = {
        (t, rt)
        for t, inner in elo_module.DEFAULT_CONFIG.k_factor_table.items()
        for rt in inner
    }
    assert default.issubset(matrix)
