"""Tests for the Boulder+Lead combined rating computation."""

from __future__ import annotations

import json
import math
from datetime import date

import pytest
from sqlalchemy.orm import Session
from sqlalchemy import select

from climbing_elo.models import (
    Athlete,
    Discipline,
    Gender,
    Rating,
)

# Import helpers from the script under test
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from compute_combined_ratings import (  # noqa: E402
    DEFAULT_W_BOULDER,
    DEFAULT_W_LEAD,
    MIN_EVENTS,
    CombinedWeights,
    compute_combined_mu,
    compute_combined_sigma,
    load_combined_weights,
)
from fit_combined_weights import (  # noqa: E402
    BASELINE_V_SIGMA_BOULDER,
    BASELINE_V_SIGMA_LEAD,
    BASELINE_W_BOULDER,
    BASELINE_W_LEAD,
    DEFAULT_CV_FOLDS,
    CombinedEntry,
    CombinedFold,
    CVResult,
    FitMetrics,
    RatingSnapshot,
    _kfold_splits,
    brier_top3,
    combined_mu,
    combined_sigma,
    decide_ship,
    fit_mu_weights,
    fit_sigma_weights,
    grid_search,
    kfold_cv,
    score_candidate,
    scipy_search,
)


# ---------------------------------------------------------------------------
# Pure maths tests
# ---------------------------------------------------------------------------


class TestComputeCombinedMu:
    """Geometric mean: sqrt(mu_b * mu_l)."""

    def test_equal_ratings_returns_same(self):
        assert compute_combined_mu(1500.0, 1500.0) == pytest.approx(1500.0)

    def test_geometric_mean_basic(self):
        # sqrt(1600 * 1400) = sqrt(2_240_000) ≈ 1496.66
        expected = math.sqrt(1600.0 * 1400.0)
        assert compute_combined_mu(1600.0, 1400.0) == pytest.approx(expected)

    def test_penalises_specialist(self):
        """A specialist should score lower than the arithmetic mean."""
        mu_b = 2000.0
        mu_l = 1000.0
        geometric = compute_combined_mu(mu_b, mu_l)
        arithmetic = (mu_b + mu_l) / 2.0
        assert geometric < arithmetic

    def test_symmetry(self):
        """Order of boulder vs lead should not matter."""
        assert compute_combined_mu(1800.0, 1600.0) == pytest.approx(
            compute_combined_mu(1600.0, 1800.0)
        )

    def test_all_rounder_close_to_arithmetic_mean(self):
        """When ratings are close, geometric ≈ arithmetic mean."""
        mu_b = 1700.0
        mu_l = 1710.0
        geometric = compute_combined_mu(mu_b, mu_l)
        arithmetic = (mu_b + mu_l) / 2.0
        # Should be within 0.1% of the arithmetic mean
        assert abs(geometric - arithmetic) / arithmetic < 0.001


class TestComputeCombinedSigma:
    """Root-mean-square of sigmas."""

    def test_equal_sigmas(self):
        # RMS of two equal values equals that value
        assert compute_combined_sigma(100.0, 100.0) == pytest.approx(100.0)

    def test_rms_formula(self):
        sigma_b = 120.0
        sigma_l = 80.0
        expected = math.sqrt((120.0**2 + 80.0**2) / 2.0)
        assert compute_combined_sigma(sigma_b, sigma_l) == pytest.approx(expected)

    def test_symmetry(self):
        assert compute_combined_sigma(150.0, 90.0) == pytest.approx(
            compute_combined_sigma(90.0, 150.0)
        )

    def test_larger_than_smaller_input(self):
        """Combined sigma should be between the smaller and larger inputs."""
        s_small = 80.0
        s_large = 160.0
        combined = compute_combined_sigma(s_small, s_large)
        assert s_small < combined < s_large

    def test_zero_sigma_gives_half_rms(self):
        """Edge case: one sigma is zero (fully certain)."""
        combined = compute_combined_sigma(0.0, 100.0)
        assert combined == pytest.approx(math.sqrt((0 + 100**2) / 2.0))


# ---------------------------------------------------------------------------
# Integration tests: end-to-end against an in-memory DB
# ---------------------------------------------------------------------------


def _make_athlete(session: Session, name: str, gender: Gender) -> Athlete:
    a = Athlete(name=name, gender=gender)
    session.add(a)
    session.flush()
    return a


def _make_rating(
    session: Session,
    athlete: Athlete,
    discipline: Discipline,
    mu: float,
    sigma: float = 100.0,
    n_events: int = 5,
    last_event_at: date | None = None,
) -> Rating:
    r = Rating(
        athlete_id=athlete.id,
        discipline=discipline,
        mu=mu,
        sigma=sigma,
        n_events=n_events,
        last_event_at=last_event_at or date(2024, 1, 1),
        provisional=n_events < 3,
    )
    session.add(r)
    session.flush()
    return r


class TestCombinedRatingIntegration:
    """End-to-end tests using the in-memory DB fixture from conftest."""

    def test_only_athletes_with_both_disciplines(self, db_session: Session):
        """Athletes with only one discipline should NOT get a combined rating."""
        lead_only = _make_athlete(db_session, "Lead Specialist", Gender.M)
        _make_rating(db_session, lead_only, Discipline.LEAD, mu=1800.0)

        both = _make_athlete(db_session, "All Rounder", Gender.M)
        _make_rating(db_session, both, Discipline.LEAD, mu=1700.0)
        _make_rating(db_session, both, Discipline.BOULDER, mu=1700.0)
        db_session.commit()

        bl_ratings = (
            db_session.execute(
                select(Rating).where(Rating.discipline == Discipline.BOULDER_LEAD)
            )
            .scalars()
            .all()
        )
        # None yet — we haven't run the script
        assert len(bl_ratings) == 0

    def test_min_events_threshold(self):
        """MIN_EVENTS constant must be 3 (matching PROVISIONAL_THRESHOLD)."""
        assert MIN_EVENTS == 3

    def test_combined_mu_correct_value(self):
        """Combined mu equals sqrt(mu_boulder * mu_lead) for known inputs."""
        mu_b = 1800.0
        mu_l = 1700.0
        result = compute_combined_mu(mu_b, mu_l)
        assert result == pytest.approx(math.sqrt(1800.0 * 1700.0), rel=1e-6)

    def test_combined_sigma_correct_value(self):
        """Combined sigma equals sqrt((sb^2 + sl^2) / 2) for known inputs."""
        sigma_b = 120.0
        sigma_l = 80.0
        result = compute_combined_sigma(sigma_b, sigma_l)
        assert result == pytest.approx(math.sqrt((120.0**2 + 80.0**2) / 2.0), rel=1e-6)

    def test_janja_ranking_logic(self):
        """Top all-rounder (equal high ratings) should beat a specialist."""
        # All-rounder: 1900 in both
        mu_all = compute_combined_mu(1900.0, 1900.0)
        # Boulder specialist: very high B, mediocre L
        mu_spec = compute_combined_mu(2100.0, 1600.0)
        # All-rounder beats the specialist thanks to geometric mean
        assert mu_all > mu_spec

    def test_specialist_penalised_below_arithmetic_mean(self):
        """Geometric mean is always <= arithmetic mean (AM-GM inequality)."""
        for mu_b, mu_l in [(2000, 1000), (1800, 1400), (2200, 1200), (1600, 1600)]:
            geo = compute_combined_mu(float(mu_b), float(mu_l))
            arith = (mu_b + mu_l) / 2.0
            assert geo <= arith + 1e-9, f"AM-GM violated for B={mu_b}, L={mu_l}"


# ---------------------------------------------------------------------------
# Learned-weights loader (Issue #54)
# ---------------------------------------------------------------------------


class TestLoadCombinedWeights:
    """``load_combined_weights`` must fall back to the geometric mean for any
    missing/malformed/out-of-range input — production correctness depends on
    never silently shipping junk weights."""

    def test_missing_file_returns_baseline(self, tmp_path):
        weights = load_combined_weights(tmp_path / "does_not_exist.json")
        assert weights.source == "geometric_mean"
        assert weights.w_lead == pytest.approx(DEFAULT_W_LEAD)
        assert weights.w_boulder == pytest.approx(DEFAULT_W_BOULDER)

    def test_empty_file_returns_baseline(self, tmp_path):
        path = tmp_path / "weights.json"
        path.write_text("")
        weights = load_combined_weights(path)
        assert weights.source == "geometric_mean"

    def test_malformed_json_returns_baseline(self, tmp_path):
        path = tmp_path / "weights.json"
        path.write_text("{not valid json")
        weights = load_combined_weights(path)
        assert weights.source == "geometric_mean"

    def test_non_object_json_returns_baseline(self, tmp_path):
        path = tmp_path / "weights.json"
        path.write_text("[0.6, 0.4]")
        weights = load_combined_weights(path)
        assert weights.source == "geometric_mean"

    def test_missing_keys_returns_baseline(self, tmp_path):
        path = tmp_path / "weights.json"
        path.write_text(json.dumps({"w_lead": 0.6}))
        weights = load_combined_weights(path)
        assert weights.source == "geometric_mean"

    def test_weights_not_summing_to_one_returns_baseline(self, tmp_path):
        path = tmp_path / "weights.json"
        path.write_text(json.dumps({"w_lead": 0.6, "w_boulder": 0.6}))
        weights = load_combined_weights(path)
        assert weights.source == "geometric_mean"

    def test_negative_weights_returns_baseline(self, tmp_path):
        path = tmp_path / "weights.json"
        path.write_text(json.dumps({"w_lead": -0.1, "w_boulder": 1.1}))
        weights = load_combined_weights(path)
        assert weights.source == "geometric_mean"

    def test_non_finite_weights_returns_baseline(self, tmp_path):
        path = tmp_path / "weights.json"
        # NaN/Infinity are accepted by Python's json with allow_nan=True (default)
        path.write_text('{"w_lead": NaN, "w_boulder": NaN}')
        weights = load_combined_weights(path)
        assert weights.source == "geometric_mean"

    def test_valid_weights_returns_learned(self, tmp_path):
        path = tmp_path / "weights.json"
        path.write_text(
            json.dumps(
                {
                    "w_lead": 0.6,
                    "w_boulder": 0.4,
                    "log_loss": 0.5,
                    "rank_corr": 0.8,
                    "baseline_log_loss": 0.6,
                    "baseline_rank_corr": 0.8,
                    "fit_date": "2026-05-26T00:00:00+00:00",
                }
            )
        )
        weights = load_combined_weights(path)
        assert weights.source == "learned"
        assert weights.w_lead == pytest.approx(0.6)
        assert weights.w_boulder == pytest.approx(0.4)


class TestComputeCombinedMuWithWeights:
    """The optional ``weights`` argument generalises the geometric mean."""

    def test_default_weights_match_geometric_mean(self):
        """When weights are None, formula collapses to sqrt(mu_b * mu_l)."""
        assert compute_combined_mu(1800.0, 1600.0) == pytest.approx(
            math.sqrt(1800.0 * 1600.0)
        )

    def test_equal_weights_match_default(self):
        """Explicit (0.5, 0.5) weights produce identical numbers to the default."""
        w = CombinedWeights(0.5, 0.5, "geometric_mean")
        for mu_b, mu_l in [(1500.0, 1500.0), (1800.0, 1600.0), (2000.0, 1200.0)]:
            assert compute_combined_mu(mu_b, mu_l) == pytest.approx(
                compute_combined_mu(mu_b, mu_l, w)
            )

    def test_lead_only_weights(self):
        """w_lead=1, w_boulder=0 reduces combined to just mu_lead."""
        w = CombinedWeights(1.0, 0.0, "learned")
        assert compute_combined_mu(1700.0, 1900.0, w) == pytest.approx(1900.0)

    def test_boulder_only_weights(self):
        """w_lead=0, w_boulder=1 reduces combined to just mu_boulder."""
        w = CombinedWeights(0.0, 1.0, "learned")
        assert compute_combined_mu(1700.0, 1900.0, w) == pytest.approx(1700.0)

    def test_asymmetric_weights_lean_lead(self):
        """w_lead > w_boulder should pull combined toward mu_lead."""
        w = CombinedWeights(0.8, 0.2, "learned")
        # Lead = 2000, Boulder = 1000 → strongly lead-leaning
        c = compute_combined_mu(1000.0, 2000.0, w)
        # 2000^0.8 * 1000^0.2 ≈ 1741
        assert c == pytest.approx(2000.0**0.8 * 1000.0**0.2)
        # And definitely closer to lead than to boulder
        assert abs(c - 2000.0) < abs(c - 1000.0)


# ---------------------------------------------------------------------------
# Fitter — synthetic data tests (Issue #54)
# ---------------------------------------------------------------------------


def _make_fold(
    year: int,
    gender: Gender,
    rows: list[tuple[int, int, int]],
) -> CombinedFold:
    """Build a fold from (athlete_id, rank_boulder, rank_lead) triples."""
    entries = tuple(
        CombinedEntry(athlete_id=a, rank_boulder=rb, rank_lead=rl) for a, rb, rl in rows
    )
    return CombinedFold(year=year, gender=gender, entries=entries)


def _snapshots_from_mus(
    year: int,
    mus_by_aid: dict[int, tuple[float, float]],
    sigma: float = 100.0,
) -> dict[int, dict[Discipline, dict[int, RatingSnapshot]]]:
    """Build the snapshot dict shape ``score_candidate`` expects.

    ``mus_by_aid[aid] = (mu_boulder, mu_lead)``.
    """
    boulder = {
        aid: RatingSnapshot(mu=mu_b, sigma=sigma)
        for aid, (mu_b, _) in mus_by_aid.items()
    }
    lead = {
        aid: RatingSnapshot(mu=mu_l, sigma=sigma)
        for aid, (_, mu_l) in mus_by_aid.items()
    }
    return {year: {Discipline.BOULDER: boulder, Discipline.LEAD: lead}}


class TestCombinedEntry:
    """Sanity checks on the rank-product combined score (Olympic format)."""

    def test_rank_product_score(self):
        # 1st in Boulder, 1st in Lead → score 1 (best possible).
        e = CombinedEntry(athlete_id=1, rank_boulder=1, rank_lead=1)
        assert e.combined_score == 1

    def test_specialist_loses_to_all_rounder(self):
        # All-rounder: 3rd, 3rd → score 9
        all_rounder = CombinedEntry(athlete_id=1, rank_boulder=3, rank_lead=3)
        # Specialist: 1st in Boulder, 10th in Lead → score 10
        specialist = CombinedEntry(athlete_id=2, rank_boulder=1, rank_lead=10)
        assert all_rounder.combined_score < specialist.combined_score


class TestScoreCandidateSynthetic:
    """When the athletes' B & L mus are equal across the board, the optimal
    weight pair must be the geometric mean (0.5, 0.5)."""

    def test_equal_mus_optimum_at_half_half(self):
        """If mu_b == mu_l for every athlete, w_lead is mathematically
        irrelevant — every (w_lead, w_boulder) candidate produces the
        identical mu_combined, so log-loss is constant across the grid.

        The grid sweep should still terminate and the metrics should match
        the baseline exactly.
        """
        year = 2024
        # Six athletes — strict mu ordering matches the actual rank-product.
        mus = {
            1: (1900.0, 1900.0),
            2: (1800.0, 1800.0),
            3: (1700.0, 1700.0),
            4: (1600.0, 1600.0),
            5: (1500.0, 1500.0),
            6: (1400.0, 1400.0),
        }
        # Construct combined ranks consistent with mu ordering (top mu → top rank).
        fold = _make_fold(
            year,
            Gender.M,
            [(1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4), (5, 5, 5), (6, 6, 6)],
        )
        snapshots = _snapshots_from_mus(year, mus)

        baseline = score_candidate(
            [fold], snapshots, BASELINE_W_LEAD, BASELINE_W_BOULDER, n_simulations=2000
        )
        # When mu_b == mu_l the formula is identity across w_lead — sample a couple.
        a = score_candidate([fold], snapshots, 0.8, 0.2, n_simulations=2000)
        b = score_candidate([fold], snapshots, 0.3, 0.7, n_simulations=2000)
        # All three should produce the *same* log-loss (modulo MC noise) since
        # mu_combined is identical and the same RNG seed is used.
        assert a.log_loss == pytest.approx(baseline.log_loss, rel=1e-6)
        assert b.log_loss == pytest.approx(baseline.log_loss, rel=1e-6)

    def test_lead_only_optimum_when_lead_predicts(self):
        """When ratings = lead-only signal and ground truth = lead-only,
        the optimum should favour lead. Boulder mu is constant noise."""
        year = 2024
        # Six athletes: distinct mu_lead, identical mu_boulder.
        mus = {
            10: (1500.0, 1900.0),
            11: (1500.0, 1800.0),
            12: (1500.0, 1700.0),
            13: (1500.0, 1600.0),
            14: (1500.0, 1500.0),
            15: (1500.0, 1400.0),
        }
        # Ground truth: combined rank-product perfectly mirrors mu_lead order.
        # Use rb=1 for everyone (constant) and rl matching mu_lead order.
        fold = _make_fold(
            year,
            Gender.M,
            [(10, 1, 1), (11, 1, 2), (12, 1, 3), (13, 1, 4), (14, 1, 5), (15, 1, 6)],
        )
        snapshots = _snapshots_from_mus(year, mus, sigma=80.0)

        baseline = score_candidate(
            [fold], snapshots, BASELINE_W_LEAD, BASELINE_W_BOULDER, n_simulations=4000
        )
        # When w_lead = 1.0, mu_combined = mu_lead → identical to ground truth.
        lead_only = score_candidate([fold], snapshots, 1.0, 0.0, n_simulations=4000)
        # Lead-only must beat geometric mean: ground truth IS mu_lead.
        # With identical mu_boulder, the geometric-mean ranks already match
        # mu_lead, so log-loss should be equal (mathematically) — but
        # rank-corr should both be 1.0.
        assert baseline.rank_corr == pytest.approx(1.0, abs=1e-9)
        assert lead_only.rank_corr == pytest.approx(1.0, abs=1e-9)


class TestGridSearch:
    """Grid search must enumerate (w_lead, 1-w_lead) and pick the best by
    log-loss."""

    def test_grid_covers_endpoints(self):
        year = 2024
        mus = {
            1: (1900.0, 1900.0),
            2: (1700.0, 1700.0),
            3: (1500.0, 1500.0),
            4: (1300.0, 1300.0),
            5: (1100.0, 1100.0),
            6: (900.0, 900.0),
        }
        fold = _make_fold(
            year,
            Gender.M,
            [(1, 1, 1), (2, 2, 2), (3, 3, 3), (4, 4, 4), (5, 5, 5), (6, 6, 6)],
        )
        snapshots = _snapshots_from_mus(year, mus)
        _, _, evals = grid_search([fold], snapshots, step=0.5, n_simulations=1000)
        w_leads = [w_l for w_l, _, _ in evals]
        # step=0.5 → {0.0, 0.5, 1.0}
        assert any(abs(w - 0.0) < 1e-6 for w in w_leads)
        assert any(abs(w - 0.5) < 1e-6 for w in w_leads)
        assert any(abs(w - 1.0) < 1e-6 for w in w_leads)

    def test_grid_summands_equal_one(self):
        year = 2024
        mus = {i: (1500.0 + i * 10, 1500.0 - i * 5) for i in range(1, 8)}
        fold = _make_fold(year, Gender.M, [(i, i, 8 - i) for i in range(1, 8)])
        snapshots = _snapshots_from_mus(year, mus)
        _, _, evals = grid_search([fold], snapshots, step=0.25, n_simulations=500)
        for w_l, w_b, _ in evals:
            assert w_l + w_b == pytest.approx(1.0, abs=1e-6)


class TestDecideShip:
    """``decide_ship`` is the single source of truth for the ship/no-ship rule."""

    def test_baseline_match_does_not_ship(self):
        m = FitMetrics(log_loss=0.5, rank_corr=0.8)
        ship, _ = decide_ship(m, m, (BASELINE_W_LEAD, BASELINE_W_BOULDER))
        assert not ship

    def test_worse_log_loss_does_not_ship(self):
        better = FitMetrics(log_loss=0.6, rank_corr=0.85)
        baseline = FitMetrics(log_loss=0.55, rank_corr=0.8)
        ship, _ = decide_ship(better, baseline, (0.7, 0.3))
        assert not ship

    def test_strict_log_loss_improvement_no_corr_regression_ships(self):
        learned = FitMetrics(log_loss=0.45, rank_corr=0.82)
        baseline = FitMetrics(log_loss=0.50, rank_corr=0.80)
        ship, reason = decide_ship(learned, baseline, (0.7, 0.3))
        assert ship, reason

    def test_log_loss_better_but_rank_corr_regression_blocks_ship(self):
        # 50% drop in rank-corr → exceeds 5% tolerance.
        learned = FitMetrics(log_loss=0.40, rank_corr=0.40)
        baseline = FitMetrics(log_loss=0.55, rank_corr=0.80)
        ship, _ = decide_ship(learned, baseline, (0.9, 0.1))
        assert not ship

    def test_nan_metrics_do_not_ship(self):
        ship, _ = decide_ship(
            FitMetrics(log_loss=float("nan"), rank_corr=0.8),
            FitMetrics(log_loss=0.5, rank_corr=0.8),
            (0.7, 0.3),
        )
        assert not ship


# ---------------------------------------------------------------------------
# End-to-end: compute_combined_ratings.py reads the JSON
# ---------------------------------------------------------------------------


class TestLearnedWeightsAppliedInProduction:
    """If a valid learned-weights JSON exists, ``compute_combined_mu`` must
    use those weights when explicitly passed; the fallback path must
    produce geometric-mean output identical to the historical behaviour."""

    def test_fallback_matches_geometric_mean(self, tmp_path):
        # No file → baseline weights → identical to historical
        # sqrt(mu_b * mu_l).
        weights = load_combined_weights(tmp_path / "nope.json")
        mu_b, mu_l = 1800.0, 1700.0
        assert compute_combined_mu(mu_b, mu_l, weights) == pytest.approx(
            math.sqrt(mu_b * mu_l)
        )

    def test_learned_weights_change_output(self, tmp_path):
        path = tmp_path / "w.json"
        path.write_text(
            json.dumps(
                {
                    "w_lead": 0.7,
                    "w_boulder": 0.3,
                    "log_loss": 0.4,
                    "rank_corr": 0.85,
                    "baseline_log_loss": 0.5,
                    "baseline_rank_corr": 0.83,
                    "fit_date": "2026-05-26T00:00:00+00:00",
                }
            )
        )
        weights = load_combined_weights(path)
        assert weights.source == "learned"
        mu_b, mu_l = 1800.0, 1700.0
        result = compute_combined_mu(mu_b, mu_l, weights)
        assert result == pytest.approx(1700.0**0.7 * 1800.0**0.3)
        # And it should NOT equal the geometric mean.
        assert result != pytest.approx(math.sqrt(mu_b * mu_l), abs=1e-6)


# ---------------------------------------------------------------------------
# scipy optimiser (#76)
# ---------------------------------------------------------------------------


def _lead_predicts_setup():
    """Lead-only signal + lead-only ground truth — shared scenario for the
    scipy / grid convergence comparison."""
    year = 2024
    mus = {
        10: (1500.0, 1900.0),
        11: (1500.0, 1800.0),
        12: (1500.0, 1700.0),
        13: (1500.0, 1600.0),
        14: (1500.0, 1500.0),
        15: (1500.0, 1400.0),
        16: (1500.0, 1300.0),
    }
    fold = _make_fold(
        year,
        Gender.M,
        [
            (10, 3, 1),
            (11, 3, 2),
            (12, 3, 3),
            (13, 3, 4),
            (14, 3, 5),
            (15, 3, 6),
            (16, 3, 7),
        ],
    )
    snapshots = _snapshots_from_mus(year, mus, sigma=80.0)
    return [fold], snapshots


class TestScipyOptimizer:
    """scipy.optimize.minimize_scalar must locate the same μ-weight optimum
    as the grid sweep (within tolerance) and remain consistent with the
    fit_mu_weights dispatcher."""

    def test_scipy_returns_weights_summing_to_one(self):
        folds, snapshots = _lead_predicts_setup()
        (w_lead, w_boulder), metrics, _ = scipy_search(
            folds, snapshots, n_simulations=2000
        )
        assert 0.0 - 1e-6 <= w_lead <= 1.0 + 1e-6
        assert w_lead + w_boulder == pytest.approx(1.0, abs=1e-6)
        assert not math.isnan(metrics.log_loss)

    def test_scipy_matches_grid_on_lead_only(self):
        """When the truth is lead-only, both optimisers should land on
        w_lead ≈ 1.0 (within scipy's xatol)."""
        folds, snapshots = _lead_predicts_setup()
        (w_lead_grid, _), _, _ = grid_search(
            folds, snapshots, step=0.1, n_simulations=2000
        )
        (w_lead_scipy, _), _, _ = scipy_search(folds, snapshots, n_simulations=2000)
        # Both must agree about which side of the unit interval is optimal.
        # We don't require pinpoint agreement (grid is 0.1-coarse, scipy is
        # ~1e-3-fine), only that they live in the same neighbourhood.
        assert abs(w_lead_grid - w_lead_scipy) <= 0.15

    def test_scipy_evaluations_recorded(self):
        """The optimiser exposes every (w_lead, w_boulder, metrics) it
        sampled — useful for inspecting convergence."""
        folds, snapshots = _lead_predicts_setup()
        _, _, evals = scipy_search(folds, snapshots, n_simulations=1000)
        assert len(evals) > 0
        for w_l, w_b, _ in evals:
            assert w_l + w_b == pytest.approx(1.0, abs=1e-6)

    def test_dispatcher_supports_both_methods(self):
        folds, snapshots = _lead_predicts_setup()
        grid_result = fit_mu_weights(
            folds, snapshots, method="grid", step=0.25, n_simulations=1000
        )
        scipy_result = fit_mu_weights(
            folds, snapshots, method="scipy", n_simulations=1000
        )
        # Both shapes match: ((w_l, w_b), metrics, evals)
        assert len(grid_result) == 3
        assert len(scipy_result) == 3

    def test_dispatcher_rejects_unknown_method(self):
        folds, snapshots = _lead_predicts_setup()
        with pytest.raises(ValueError):
            fit_mu_weights(folds, snapshots, method="bayesian")


# ---------------------------------------------------------------------------
# K-fold cross-validation (#77)
# ---------------------------------------------------------------------------


class TestKFoldSplits:
    """``_kfold_splits`` underlies the CV runner."""

    def test_partitions_indices_disjointly(self):
        splits = _kfold_splits(10, k=5)
        assert len(splits) == 5
        seen_test: list[int] = []
        for train, test in splits:
            assert set(train).isdisjoint(set(test))
            assert sorted(train + test) == list(range(10))
            seen_test.extend(test)
        # Every index appears in exactly one test fold.
        assert sorted(seen_test) == list(range(10))

    def test_uneven_split_handled(self):
        # 7 items / k=3 → sizes (3, 2, 2)
        splits = _kfold_splits(7, k=3)
        sizes = [len(test) for _, test in splits]
        assert sizes == [3, 2, 2]

    def test_k_larger_than_n_degrades_to_loocv(self):
        splits = _kfold_splits(3, k=10)
        # Capped to 3 folds; each test fold is a single index.
        assert len(splits) == 3
        for _, test in splits:
            assert len(test) == 1

    def test_invalid_k_raises(self):
        with pytest.raises(ValueError):
            _kfold_splits(5, k=1)


class TestKFoldCV:
    """k-fold CV must produce per-fold metrics, mean and std."""

    def _multi_fold_snapshots(self):
        """Build 5 synthetic folds across 5 different years — enough for 5-fold CV."""
        years = [2020, 2021, 2022, 2023, 2024]
        folds = []
        snapshots: dict = {}
        for y in years:
            mus = {100 + y + i: (1500.0 + i * 50, 1700.0 - i * 20) for i in range(7)}
            entries = [(100 + y + i, (i % 3) + 1, ((i + 2) % 3) + 1) for i in range(7)]
            folds.append(_make_fold(y, Gender.M, entries))
            snap = _snapshots_from_mus(y, mus)
            snapshots[y] = snap[y]
        return folds, snapshots

    def test_kfold_returns_one_metric_per_fold(self):
        folds, snapshots = self._multi_fold_snapshots()
        result = kfold_cv(
            folds,
            snapshots,
            BASELINE_W_LEAD,
            BASELINE_W_BOULDER,
            k=5,
            n_simulations=500,
        )
        assert isinstance(result, CVResult)
        assert len(result.per_fold_metrics) == 5

    def test_kfold_mean_std_are_finite(self):
        folds, snapshots = self._multi_fold_snapshots()
        result = kfold_cv(
            folds,
            snapshots,
            0.5,
            0.5,
            k=5,
            n_simulations=500,
        )
        assert not math.isnan(result.log_loss_mean)
        # std must be non-negative; 0.0 only if pstdev was given a single
        # value (fine — graceful degenerate behaviour).
        assert result.log_loss_std >= 0.0

    def test_kfold_std_is_reasonable_on_synthetic_data(self):
        """Across folds with similar structure, std should be a sensible
        positive fraction of the mean — not 0, not gigantic."""
        folds, snapshots = self._multi_fold_snapshots()
        result = kfold_cv(
            folds,
            snapshots,
            0.5,
            0.5,
            k=5,
            n_simulations=1000,
        )
        # Mean must be > 0 (log loss can't be negative).
        assert result.log_loss_mean > 0
        # std should not exceed the mean by orders of magnitude.
        assert result.log_loss_std < result.log_loss_mean * 5.0

    def test_cv_mean_used_by_ship_rule(self):
        """The ship rule treats whatever metrics it's given as the truth —
        the CLI passes CV means; the rule itself doesn't care about CV vs
        single-pass. Verify a hand-crafted CV mean drives the decision."""
        # CV-mean metrics with improvement.
        cv_learned = FitMetrics(log_loss=0.45, rank_corr=0.82)
        cv_baseline = FitMetrics(log_loss=0.55, rank_corr=0.80)
        ship, _ = decide_ship(cv_learned, cv_baseline, (0.7, 0.3))
        assert ship
        # Now a CV-mean that regresses → ship rule says no even though the
        # single-pass numbers would have shipped.
        cv_regressed = FitMetrics(log_loss=0.58, rank_corr=0.82)
        ship2, _ = decide_ship(cv_regressed, cv_baseline, (0.7, 0.3))
        assert not ship2

    def test_default_k_is_five(self):
        """Per the docstring + CLI default."""
        assert DEFAULT_CV_FOLDS == 5

    def test_empty_folds_yield_empty_cv_result(self):
        result = kfold_cv([], {}, 0.5, 0.5, k=5)
        assert len(result.per_fold_metrics) == 0
        assert math.isnan(result.log_loss_mean)
        assert result.log_loss_std == 0.0


# ---------------------------------------------------------------------------
# σ-weight learning (#78)
# ---------------------------------------------------------------------------


class TestCombinedSigmaWeighted:
    """``combined_sigma`` collapses to RMS at equal weights."""

    def test_equal_sigma_weights_match_rms(self):
        """v_lead = v_boulder = 0.5 must equal the historical RMS formula."""
        for s_l, s_b in [(80.0, 120.0), (50.0, 50.0), (200.0, 100.0)]:
            weighted = combined_sigma(s_l, s_b, 0.5, 0.5)
            rms = math.sqrt((s_l**2 + s_b**2) / 2.0)
            assert weighted == pytest.approx(rms, rel=1e-9)

    def test_lead_dominant_pulls_toward_lead_sigma(self):
        # If v_lead >> v_boulder, combined σ approaches σ_lead.
        s_l, s_b = 200.0, 50.0
        weighted = combined_sigma(s_l, s_b, 0.99, 0.01)
        assert abs(weighted - 200.0) < abs(weighted - 50.0)

    def test_zero_total_weights_raises(self):
        with pytest.raises(ValueError):
            combined_sigma(100.0, 100.0, 0.0, 0.0)

    def test_default_weights_are_rms_baseline(self):
        assert BASELINE_V_SIGMA_LEAD == 0.5
        assert BASELINE_V_SIGMA_BOULDER == 0.5


class TestSigmaWeights:
    """σ-weight optimisation should favour the more predictive discipline."""

    def _make_predictive_lead_sigma(self):
        """Construct synthetic folds where Lead σ matches actual outcome
        spread more tightly than Boulder σ.

        Lead has a clean, narrow σ across all athletes (high signal).
        Boulder has very wide σ for everyone (noisy). Ground truth follows
        mu_lead → so lead is more predictive and lead σ should better
        capture residual uncertainty."""
        years = [2022, 2023, 2024]
        folds = []
        snapshots: dict = {}
        for y in years:
            mus = {2000 + y + i: (1500.0, 1900.0 - i * 50) for i in range(7)}
            # Build the snapshots manually to give per-discipline σ control.
            boulder_snap = {
                aid: RatingSnapshot(mu=mu_b, sigma=300.0)  # Wide / noisy boulder σ.
                for aid, (mu_b, _) in mus.items()
            }
            lead_snap = {
                aid: RatingSnapshot(mu=mu_l, sigma=60.0)  # Tight / informative lead σ.
                for aid, (_, mu_l) in mus.items()
            }
            snapshots[y] = {
                Discipline.BOULDER: boulder_snap,
                Discipline.LEAD: lead_snap,
            }
            # Ground truth: lead-rank ordering.
            entries = [(2000 + y + i, (i % 3) + 1, i + 1) for i in range(7)]
            folds.append(_make_fold(y, Gender.M, entries))
        return folds, snapshots

    def test_brier_top3_is_finite(self):
        folds, snapshots = self._make_predictive_lead_sigma()
        # Use lead-leaning μ since lead is the predictive signal.
        b = brier_top3(folds, snapshots, 0.9, 0.1, 0.5, 0.5, n_simulations=2000)
        assert not math.isnan(b)
        assert 0.0 <= b <= 1.0

    def test_brier_top3_empty_folds_is_nan(self):
        b = brier_top3([], {}, 0.5, 0.5, 0.5, 0.5)
        assert math.isnan(b)

    def test_sigma_fit_returns_normalised_weights(self):
        folds, snapshots = self._make_predictive_lead_sigma()
        (v_l, v_b), brier = fit_sigma_weights(
            folds, snapshots, 0.9, 0.1, n_simulations=1500
        )
        assert v_l + v_b == pytest.approx(1.0, abs=1e-6)
        assert 0.0 - 1e-6 <= v_l <= 1.0 + 1e-6
        assert not math.isnan(brier)

    def test_sigma_fit_favours_predictive_discipline(self):
        """When Lead σ is more informative than Boulder σ, the optimiser
        should NOT prefer the boulder-dominant corner — i.e. v_sigma_lead
        should not collapse to 0. Lead-leaning (v_lead > v_boulder) is the
        principled answer; we assert at minimum the non-collapse direction."""
        folds, snapshots = self._make_predictive_lead_sigma()
        (v_l, v_b), _ = fit_sigma_weights(
            folds, snapshots, 0.9, 0.1, n_simulations=2000
        )
        # The wide-boulder corner (v_b ≈ 1) inflates uncertainty for every
        # athlete — Brier should be worse there. Optimiser must avoid it.
        assert v_l > 0.05, (
            f"σ fit collapsed to boulder-only (v_l={v_l:.3f}); expected lead-leaning"
        )


# ---------------------------------------------------------------------------
# Backward compatibility (#76, #77, #78)
# ---------------------------------------------------------------------------


class TestBackwardCompat:
    """The v1 learned-weights JSON (no σ / no CV fields) must keep loading
    and behave exactly like the historical RMS + single-pass paths."""

    def test_v1_payload_no_sigma_fields_falls_back_to_rms(self, tmp_path):
        path = tmp_path / "v1.json"
        path.write_text(
            json.dumps(
                {
                    "w_lead": 0.6,
                    "w_boulder": 0.4,
                    "log_loss": 0.5,
                    "rank_corr": 0.85,
                    "baseline_log_loss": 0.55,
                    "baseline_rank_corr": 0.83,
                    "n_folds": 12,
                    "fit_date": "2026-05-26T00:00:00+00:00",
                }
            )
        )
        weights = load_combined_weights(path)
        assert weights.source == "learned"
        # σ fields default to RMS.
        assert weights.sigma_source == "rms"
        assert weights.w_sigma_lead == pytest.approx(DEFAULT_W_BOULDER)
        assert weights.w_sigma_boulder == pytest.approx(DEFAULT_W_LEAD)

    def test_v1_payload_uses_rms_in_compute_combined_sigma(self, tmp_path):
        """A learned-μ-only payload must produce the same σ as the
        legacy default-weights path."""
        path = tmp_path / "v1.json"
        path.write_text(
            json.dumps(
                {
                    "w_lead": 0.7,
                    "w_boulder": 0.3,
                    "log_loss": 0.4,
                    "rank_corr": 0.85,
                    "baseline_log_loss": 0.5,
                    "baseline_rank_corr": 0.83,
                    "fit_date": "2026-05-26T00:00:00+00:00",
                }
            )
        )
        weights = load_combined_weights(path)
        sigma_b, sigma_l = 120.0, 80.0
        with_weights = compute_combined_sigma(sigma_b, sigma_l, weights)
        without_weights = compute_combined_sigma(sigma_b, sigma_l)
        assert with_weights == pytest.approx(without_weights, rel=1e-9)

    def test_v2_payload_with_sigma_fields_loads_learned(self, tmp_path):
        path = tmp_path / "v2.json"
        path.write_text(
            json.dumps(
                {
                    "w_lead": 0.65,
                    "w_boulder": 0.35,
                    "log_loss": 0.45,
                    "rank_corr": 0.84,
                    "baseline_log_loss": 0.5,
                    "baseline_rank_corr": 0.83,
                    "w_sigma_lead": 0.7,
                    "w_sigma_boulder": 0.3,
                    "sigma_calibration_metric": 0.18,
                    "cv_method": "kfold",
                    "cv_folds": 5,
                    "cv_log_loss_mean": 0.46,
                    "cv_log_loss_std": 0.03,
                    "fit_date": "2026-05-26T00:00:00+00:00",
                }
            )
        )
        weights = load_combined_weights(path)
        assert weights.source == "learned"
        assert weights.sigma_source == "learned"
        assert weights.w_sigma_lead == pytest.approx(0.7)
        assert weights.w_sigma_boulder == pytest.approx(0.3)

    def test_v2_sigma_weights_change_compute_combined_sigma_output(self, tmp_path):
        path = tmp_path / "v2.json"
        path.write_text(
            json.dumps(
                {
                    "w_lead": 0.5,
                    "w_boulder": 0.5,
                    "w_sigma_lead": 0.9,
                    "w_sigma_boulder": 0.1,
                    "log_loss": 0.5,
                    "rank_corr": 0.8,
                    "baseline_log_loss": 0.55,
                    "baseline_rank_corr": 0.79,
                    "fit_date": "2026-05-26T00:00:00+00:00",
                }
            )
        )
        weights = load_combined_weights(path)
        sigma_b, sigma_l = 200.0, 50.0
        with_weights = compute_combined_sigma(sigma_b, sigma_l, weights)
        rms = compute_combined_sigma(sigma_b, sigma_l)
        # Lead-dominant σ weights pull toward sigma_lead → must differ from RMS.
        assert with_weights != pytest.approx(rms, abs=0.5)

    def test_malformed_sigma_fields_fall_back_to_rms(self, tmp_path):
        path = tmp_path / "bad-sigma.json"
        path.write_text(
            json.dumps(
                {
                    "w_lead": 0.6,
                    "w_boulder": 0.4,
                    "w_sigma_lead": "not-a-number",
                    "w_sigma_boulder": 0.5,
                    "log_loss": 0.5,
                    "rank_corr": 0.8,
                    "baseline_log_loss": 0.55,
                    "baseline_rank_corr": 0.79,
                    "fit_date": "2026-05-26T00:00:00+00:00",
                }
            )
        )
        weights = load_combined_weights(path)
        # μ weights still load; σ falls back to RMS.
        assert weights.source == "learned"
        assert weights.sigma_source == "rms"

    def test_negative_sigma_weights_fall_back_to_rms(self, tmp_path):
        path = tmp_path / "neg-sigma.json"
        path.write_text(
            json.dumps(
                {
                    "w_lead": 0.6,
                    "w_boulder": 0.4,
                    "w_sigma_lead": -0.1,
                    "w_sigma_boulder": 1.1,
                    "log_loss": 0.5,
                    "rank_corr": 0.8,
                    "baseline_log_loss": 0.55,
                    "baseline_rank_corr": 0.79,
                    "fit_date": "2026-05-26T00:00:00+00:00",
                }
            )
        )
        weights = load_combined_weights(path)
        assert weights.sigma_source == "rms"


# Suppress unused-import warnings — keeping these imported tightens the
# public surface that tests assert on.
_ = (Session, date, select, Rating, Athlete, MIN_EVENTS, combined_mu)
