"""Tests for the MOV constants regrid grid-search script (#85).

Like the K regrid, the MOV grid search is exploratory — it doesn't ship engine
code, just a script that writes a markdown report. These tests focus on:

* End-to-end smoke: ``run_regrid`` completes against a tiny synthetic SQLite
  DB and writes a non-empty report.
* Grid enumeration: the 2-D Cartesian product over rating_scale × softening
  is correctly enumerated and stable.
* No monkey-patching: ``DEFAULT_CONFIG.mov_rating_scale`` /
  ``DEFAULT_CONFIG.mov_softening`` are unchanged after a grid run — sweeps are
  driven through fresh :class:`EloConfig` instances.
* Winner picker: prefers in-band trials and breaks ties by log-loss.
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
from scripts.regrid_mov_factors import (
    DEFAULT_RATING_SCALE_GRID,
    DEFAULT_SOFTENING_GRID,
    MovRegridReport,
    MovTrial,
    _enumerate_grid,
    _pick_winner,
    run_regrid,
)


# ---------------------------------------------------------------------------
# Synthetic dataset — small enough to backfill in <1s.
# ---------------------------------------------------------------------------


def _seed_synthetic_db(db_path: Path) -> None:
    """Seed a SQLite file with 6 athletes × 4 events / discipline for LEAD.

    Shape mirrors ``tests/test_regrid.py::_seed_synthetic_db`` — the MOV grid
    harness only requires real events / rounds / results.
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


def test_mov_regrid_smoke_runs_against_synthetic_db(tmp_path: Path) -> None:
    """End-to-end smoke: grid search completes on a tiny DB and writes a report."""
    db_path = tmp_path / "synthetic.db"
    _seed_synthetic_db(db_path)
    report_path = tmp_path / "report.md"

    report = run_regrid(
        source_db=db_path,
        output_path=report_path,
        rating_scale_grid=(200.0, 400.0, 800.0),
        softening_grid=(1.0, 2.2, 4.5),
        n_sims=100,
        rng_seed=42,
        holdout_seasons=2,
        disciplines=(Discipline.LEAD,),
        progress=False,
    )

    assert isinstance(report, MovRegridReport)
    assert report.rating_scale_grid == (200.0, 400.0, 800.0)
    assert report.softening_grid == (1.0, 2.2, 4.5)
    assert report.disciplines == (Discipline.LEAD,)
    # 3 × 3 = 9 trials
    assert len(report.trials) == 9

    assert report_path.exists()
    text = report_path.read_text()
    assert "MOV constants regrid report" in text
    assert "Recommended values" in text
    assert "mov_rating_scale" in text
    assert "mov_softening" in text


def test_enumerate_grid_cartesian_product_stable_order() -> None:
    """The grid enumeration must be stable: rating_scale outer, softening inner."""
    rs_grid = (100.0, 200.0, 300.0)
    sf_grid = (1.0, 2.0)
    points = _enumerate_grid(rs_grid, sf_grid)
    assert points == [
        (100.0, 1.0),
        (100.0, 2.0),
        (200.0, 1.0),
        (200.0, 2.0),
        (300.0, 1.0),
        (300.0, 2.0),
    ]


def test_grid_default_constants_size() -> None:
    """The default grid is 6 × 5 = 30 combinations as documented."""
    assert len(DEFAULT_RATING_SCALE_GRID) == 6
    assert len(DEFAULT_SOFTENING_GRID) == 5
    # Current values should both be inside their respective grids so the
    # baseline is always evaluated as one of the trials.
    assert elo_module.DEFAULT_CONFIG.mov_rating_scale in DEFAULT_RATING_SCALE_GRID
    assert elo_module.DEFAULT_CONFIG.mov_softening in DEFAULT_SOFTENING_GRID


def test_mov_regrid_does_not_monkey_patch_module_globals(tmp_path: Path) -> None:
    """Issue #83 T3 contract — grid trials must NOT mutate elo module state.

    Snapshots ``DEFAULT_CONFIG.mov_rating_scale``, ``DEFAULT_CONFIG.mov_softening``,
    ``MOV_RATING_SCALE``, and ``MOV_SOFTENING`` around the regrid call and
    asserts they are unchanged afterwards.
    """
    db_path = tmp_path / "synthetic.db"
    _seed_synthetic_db(db_path)
    report_path = tmp_path / "report.md"

    pre_rs = elo_module.DEFAULT_CONFIG.mov_rating_scale
    pre_sf = elo_module.DEFAULT_CONFIG.mov_softening
    pre_module_rs = elo_module.MOV_RATING_SCALE
    pre_module_sf = elo_module.MOV_SOFTENING
    pre_default_k = copy.deepcopy(elo_module.DEFAULT_CONFIG.k_factor_table)

    run_regrid(
        source_db=db_path,
        output_path=report_path,
        rating_scale_grid=(200.0, 800.0),
        softening_grid=(1.0, 4.5),
        n_sims=50,
        rng_seed=42,
        disciplines=(Discipline.LEAD,),
        progress=False,
    )

    assert elo_module.DEFAULT_CONFIG.mov_rating_scale == pre_rs
    assert elo_module.DEFAULT_CONFIG.mov_softening == pre_sf
    assert elo_module.MOV_RATING_SCALE == pre_module_rs
    assert elo_module.MOV_SOFTENING == pre_module_sf
    assert elo_module.DEFAULT_CONFIG.k_factor_table == pre_default_k


def test_pick_winner_prefers_in_band_over_out_of_band() -> None:
    """When some trials land in [min, max], the picker must prefer them."""
    trials = [
        MovTrial(
            rating_scale=400.0,
            softening=2.2,
            top3_hit_rate=0.85,
            top1_hit_rate=0.3,
            log_loss_podium=0.6,
            log_loss_win=0.2,
            brier_podium=0.05,
            mu_min=1000,
            mu_p50=1500,
            mu_p95=2000,  # in band
            mu_p99=2100,
            mu_max=2200,
            in_band=True,
        ),
        MovTrial(
            rating_scale=800.0,
            softening=1.0,
            top3_hit_rate=0.90,  # better top-3, but out of band
            top1_hit_rate=0.4,
            log_loss_podium=0.5,
            log_loss_win=0.2,
            brier_podium=0.05,
            mu_min=1000,
            mu_p50=1500,
            mu_p95=3000,  # OUT of band
            mu_p99=3500,
            mu_max=4000,
            in_band=False,
        ),
    ]
    winner = _pick_winner(trials, mu_p95_min=1900.0, mu_p95_max=2200.0)
    assert winner.rating_scale == 400.0  # in-band candidate wins
    assert winner.softening == 2.2


def test_pick_winner_ties_broken_by_log_loss() -> None:
    """Among in-band trials with identical top-3, lower log_loss_podium wins."""
    common = dict(
        top1_hit_rate=0.3,
        log_loss_win=0.2,
        brier_podium=0.05,
        mu_min=1000,
        mu_p50=1500,
        mu_p95=2000,
        mu_p99=2100,
        mu_max=2200,
        in_band=True,
    )
    trials = [
        MovTrial(
            rating_scale=300.0,
            softening=1.5,
            top3_hit_rate=0.85,
            log_loss_podium=0.700,  # worse
            **common,
        ),
        MovTrial(
            rating_scale=500.0,
            softening=3.0,
            top3_hit_rate=0.85,
            log_loss_podium=0.600,  # better tie-break
            **common,
        ),
    ]
    winner = _pick_winner(trials, mu_p95_min=1900.0, mu_p95_max=2200.0)
    assert winner.rating_scale == 500.0
    assert winner.softening == 3.0


def test_pick_winner_fallback_when_no_in_band() -> None:
    """When no trial lands in band, the picker chooses the closest p95 to centre."""
    common = dict(
        top1_hit_rate=0.3,
        log_loss_podium=0.5,
        log_loss_win=0.2,
        brier_podium=0.05,
        mu_min=1000,
        mu_p50=1500,
        mu_p99=2500,
        mu_max=3000,
        in_band=False,
    )
    trials = [
        MovTrial(
            rating_scale=200.0,
            softening=4.5,
            top3_hit_rate=0.90,
            mu_p95=1500,  # below band
            **common,
        ),
        MovTrial(
            rating_scale=800.0,
            softening=1.0,
            top3_hit_rate=0.80,
            mu_p95=2800,  # above band — but closer to centre=2050
            **common,
        ),
        MovTrial(
            rating_scale=400.0,
            softening=2.2,
            top3_hit_rate=0.85,
            mu_p95=1700,  # below band — closest to centre=2050
            **common,
        ),
    ]
    # centre = (1900 + 2200) / 2 = 2050; |1700 - 2050| = 350 is smallest.
    winner = _pick_winner(trials, mu_p95_min=1900.0, mu_p95_max=2200.0)
    assert winner.rating_scale == 400.0


def test_report_includes_grid_table_and_recommendation_block(tmp_path: Path) -> None:
    """The markdown report must contain the per-cell grid + recommended snippet."""
    db_path = tmp_path / "synthetic.db"
    _seed_synthetic_db(db_path)
    report_path = tmp_path / "report.md"

    run_regrid(
        source_db=db_path,
        output_path=report_path,
        rating_scale_grid=(200.0, 400.0),
        softening_grid=(1.0, 2.2),
        n_sims=50,
        rng_seed=42,
        disciplines=(Discipline.LEAD,),
        progress=False,
    )

    text = report_path.read_text()
    # Grid heading + per-trial table.
    assert "Grid sweep results" in text
    assert "Full per-trial table" in text
    # Recommendation block.
    assert "Recommended values (paste into `EloConfig` defaults)" in text
    # The recommended cell marker must appear somewhere in the grid table.
    assert "★" in text
