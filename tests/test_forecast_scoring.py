"""Tests for the forecast scoring engine (``engine/forecast_scoring.py``)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from sqlalchemy.orm import Session

from climbing_elo.engine.elo import engine_version_tag
from climbing_elo.engine.forecast_scoring import score_forecast
from climbing_elo.models import (
    Athlete,
    Event,
    EventForecast,
    EventForecastScore,
    Gender,
    Result,
    Round,
    RoundType,
)


def _insert_forecast(
    db_session: Session,
    *,
    event: Event,
    athlete: Athlete,
    prob_qualify: float = 1.0,
    prob_reach_semi: float = 0.5,
    prob_reach_final: float = 0.3,
    prob_podium: float = 0.1,
    prob_win: float = 0.05,
    expected_rank: float = 4.0,
    mu_at_forecast: float = 1500.0,
    sigma_at_forecast: float = 100.0,
    n_simulations: int = 1000,
    roster_source: str = "confirmed",
    is_backfill: bool = False,
    engine_version: str | None = None,
) -> EventForecast:
    row = EventForecast(
        event_id=event.id,
        gender=Gender.M,
        athlete_id=athlete.id,
        prob_qualify=prob_qualify,
        prob_reach_semi=prob_reach_semi,
        prob_reach_final=prob_reach_final,
        prob_podium=prob_podium,
        prob_win=prob_win,
        expected_rank=expected_rank,
        mu_at_forecast=mu_at_forecast,
        sigma_at_forecast=sigma_at_forecast,
        n_simulations=n_simulations,
        roster_source=roster_source,
        is_backfill=is_backfill,
        engine_version=engine_version or engine_version_tag(),
        generated_at=datetime.now(timezone.utc),
    )
    db_session.add(row)
    db_session.flush()
    return row


def _add_final_results(
    db_session: Session,
    event: Event,
    athletes: list[Athlete],
    ranks: list[int],
) -> Round:
    """Seed a FINAL round with ranks for the given athletes."""
    rnd = Round(
        event_id=event.id,
        round_type=RoundType.FINAL,
        athlete_count=len(athletes),
        gender=Gender.M,
    )
    db_session.add(rnd)
    db_session.flush()
    for athlete, rank in zip(athletes, ranks):
        db_session.add(
            Result(
                round_id=rnd.id,
                athlete_id=athlete.id,
                rank=rank,
                dns=False,
            )
        )
    db_session.flush()
    return rnd


def _add_qual_results(
    db_session: Session,
    event: Event,
    athletes: list[Athlete],
) -> Round:
    rnd = Round(
        event_id=event.id,
        round_type=RoundType.QUALIFICATION,
        athlete_count=len(athletes),
        gender=Gender.M,
    )
    db_session.add(rnd)
    db_session.flush()
    for i, athlete in enumerate(athletes):
        db_session.add(
            Result(
                round_id=rnd.id,
                athlete_id=athlete.id,
                rank=i + 1,
                dns=False,
            )
        )
    db_session.flush()
    return rnd


def test_score_returns_none_when_no_forecast(
    db_session: Session,
    sample_event: Event,
):
    assert (
        score_forecast(db_session, sample_event.id, Gender.M, is_backfill=False) is None
    )


def test_score_returns_none_when_no_final_results(
    db_session: Session,
    sample_event: Event,
    eight_athletes_with_ratings: list[Athlete],
):
    # Insert forecast rows but only a qualification round (no final).
    for athlete in eight_athletes_with_ratings:
        _insert_forecast(db_session, event=sample_event, athlete=athlete)
    _add_qual_results(db_session, sample_event, eight_athletes_with_ratings)
    db_session.commit()

    assert score_forecast(db_session, sample_event.id, Gender.M) is None


def test_perfect_prediction_zero_brier(
    db_session: Session,
    sample_event: Event,
    eight_athletes: list[Athlete],
):
    # Manually craft forecasts where prob_podium=1.0 for athletes who
    # actually finish on the podium and 0.0 for everyone else. Brier should
    # be exactly 0 for the podium stage.
    podium_athletes = eight_athletes[:3]
    other_athletes = eight_athletes[3:]
    for i, athlete in enumerate(podium_athletes):
        _insert_forecast(
            db_session,
            event=sample_event,
            athlete=athlete,
            prob_qualify=1.0,
            prob_reach_semi=1.0,
            prob_reach_final=1.0,
            prob_podium=1.0,
            prob_win=1.0 if i == 0 else 0.0,
            expected_rank=float(i + 1),
        )
    for j, athlete in enumerate(other_athletes):
        _insert_forecast(
            db_session,
            event=sample_event,
            athlete=athlete,
            prob_qualify=1.0,
            prob_reach_semi=1.0,
            prob_reach_final=1.0,
            prob_podium=0.0,
            prob_win=0.0,
            expected_rank=float(4 + j),
        )

    # Actual final-round ranks line up with our forecast.
    _add_final_results(
        db_session,
        sample_event,
        eight_athletes,
        ranks=[1, 2, 3, 4, 5, 6, 7, 8],
    )
    db_session.commit()

    score = score_forecast(db_session, sample_event.id, Gender.M)
    assert score is not None
    assert score.brier_podium == pytest.approx(0.0, abs=1e-9)
    assert score.brier_win == pytest.approx(0.0, abs=1e-9)
    # Predicted top-3 matches actual top-3.
    assert score.top3_intersection == 3


def test_coin_flip_brier_quarter(
    db_session: Session,
    sample_event: Event,
    eight_athletes: list[Athlete],
):
    # All probs = 0.5. Whatever y∈{0,1} is, (0.5 - y)² = 0.25.
    for i, athlete in enumerate(eight_athletes):
        _insert_forecast(
            db_session,
            event=sample_event,
            athlete=athlete,
            prob_qualify=0.5,
            prob_reach_semi=0.5,
            prob_reach_final=0.5,
            prob_podium=0.5,
            prob_win=0.5,
            expected_rank=float(i + 1),
        )
    # Half the athletes podium / win.
    _add_final_results(
        db_session,
        sample_event,
        eight_athletes,
        ranks=[1, 2, 3, 4, 5, 6, 7, 8],
    )
    db_session.commit()

    score = score_forecast(db_session, sample_event.id, Gender.M)
    assert score is not None
    # 3 of 8 athletes finished podium; (0.5 - y)² == 0.25 regardless of y.
    assert score.brier_podium == pytest.approx(0.25, abs=1e-9)
    # 1 of 8 won; Brier still 0.25 across the board.
    assert score.brier_win == pytest.approx(0.25, abs=1e-9)


def test_top3_intersection_partial(
    db_session: Session,
    sample_event: Event,
    eight_athletes: list[Athlete],
):
    # Predict athletes 0, 1, 2 as top-3 with prob_podium=0.9; everyone else
    # gets 0.1. Then arrange actual finals so only athlete 0 lands in top-3,
    # giving |predicted_top3 ∩ actual_top3| = 1.
    for i, athlete in enumerate(eight_athletes):
        is_predicted_top3 = i < 3
        _insert_forecast(
            db_session,
            event=sample_event,
            athlete=athlete,
            prob_podium=0.9 if is_predicted_top3 else 0.1,
            prob_win=0.4 if i == 0 else 0.05,
            prob_reach_final=0.95 if is_predicted_top3 else 0.2,
            prob_reach_semi=0.95 if is_predicted_top3 else 0.3,
            expected_rank=float(i + 1),
        )
    # Actual: athlete 0 wins, then athletes 5 and 6 take 2nd/3rd.
    actual_ranks = [1, 4, 5, 6, 7, 2, 3, 8]
    _add_final_results(db_session, sample_event, eight_athletes, actual_ranks)
    db_session.commit()

    score = score_forecast(db_session, sample_event.id, Gender.M)
    assert score is not None
    assert score.top3_intersection == 1
    # Top-8 intersection covers all 8 since all 8 finished, but the
    # forecast list only has 8 entries so cap=8.
    assert score.top8_intersection == 8


def test_score_upserts_one_row(
    db_session: Session,
    sample_event: Event,
    eight_athletes: list[Athlete],
):
    for i, athlete in enumerate(eight_athletes):
        _insert_forecast(
            db_session,
            event=sample_event,
            athlete=athlete,
            prob_podium=0.5,
            prob_win=0.1,
            expected_rank=float(i + 1),
        )
    _add_final_results(
        db_session,
        sample_event,
        eight_athletes,
        ranks=[1, 2, 3, 4, 5, 6, 7, 8],
    )
    db_session.commit()

    score_forecast(db_session, sample_event.id, Gender.M)
    db_session.commit()
    score_forecast(db_session, sample_event.id, Gender.M)
    db_session.commit()

    rows = (
        db_session.query(EventForecastScore)
        .filter_by(event_id=sample_event.id, gender=Gender.M, is_backfill=False)
        .all()
    )
    assert len(rows) == 1


def test_score_engine_version_mirrors_forecast(
    db_session: Session,
    sample_event: Event,
    eight_athletes: list[Athlete],
):
    # Stamp the forecast rows with a synthetic engine version and make sure
    # the score row mirrors it (not whatever the current engine returns).
    synthetic_version = "abc123def456-deadbee"
    for i, athlete in enumerate(eight_athletes):
        _insert_forecast(
            db_session,
            event=sample_event,
            athlete=athlete,
            prob_podium=0.4,
            prob_win=0.1,
            expected_rank=float(i + 1),
            engine_version=synthetic_version,
        )
    _add_final_results(
        db_session,
        sample_event,
        eight_athletes,
        ranks=[1, 2, 3, 4, 5, 6, 7, 8],
    )
    db_session.commit()

    score = score_forecast(db_session, sample_event.id, Gender.M)
    assert score is not None
    assert score.engine_version == synthetic_version


def test_score_logloss_finite(
    db_session: Session,
    sample_event: Event,
    eight_athletes: list[Athlete],
):
    # Edge-case prob values clamp to ε so log-loss stays finite.
    for i, athlete in enumerate(eight_athletes):
        _insert_forecast(
            db_session,
            event=sample_event,
            athlete=athlete,
            prob_podium=0.0 if i >= 3 else 1.0,
            prob_win=0.0 if i != 0 else 1.0,
            prob_reach_final=0.0 if i >= 3 else 1.0,
            prob_reach_semi=0.0 if i >= 3 else 1.0,
            expected_rank=float(i + 1),
        )
    _add_final_results(
        db_session,
        sample_event,
        eight_athletes,
        ranks=[1, 2, 3, 4, 5, 6, 7, 8],
    )
    db_session.commit()

    score = score_forecast(db_session, sample_event.id, Gender.M)
    assert score is not None
    # All four log-loss values are finite (clipped at ε).
    for attr in ("logloss_semi", "logloss_final", "logloss_podium", "logloss_win"):
        val = getattr(score, attr)
        assert val == val  # NaN check
        assert val < 100.0  # not blown up to ∞
