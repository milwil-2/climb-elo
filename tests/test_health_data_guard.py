"""Tests for the data + rating health guard (Issue #118)."""

from __future__ import annotations

from datetime import date

from climbing_elo.models import (
    Athlete,
    Discipline,
    Event,
    EventTier,
    Gender,
    Rating,
    Result,
    Round,
    RoundType,
)
from scripts.health_data_guard import run_checks


def _boulder_round(session) -> Round:
    ath = Athlete(name="Test Climber", gender=Gender.M)
    ev = Event(
        name="Test WC",
        season=2016,
        discipline=Discipline.BOULDER,
        tier=EventTier.WORLD_CUP,
        start_date=date(2016, 6, 1),
    )
    session.add_all([ath, ev])
    session.flush()
    rnd = Round(event_id=ev.id, round_type=RoundType.QUALIFICATION, gender=Gender.M)
    session.add(rnd)
    session.flush()
    return rnd, ath


def _add_result(session, rnd, ath, raw, norm):
    session.add(
        Result(
            round_id=rnd.id,
            athlete_id=ath.id,
            rank=1,
            raw_score=raw,
            score_normalized=norm,
        )
    )
    session.flush()


def test_guard_passes_when_score_correctly_normalized(db_session):
    rnd, ath = _boulder_round(db_session)
    # "0t 4b10" → 0 tops, 4 zones, 10 zone-attempts → 390 (the #115 form, parsed)
    _add_result(db_session, rnd, ath, "0t 4b10", 390.0)
    failures, warnings = run_checks(db_session)
    assert failures == []


def test_guard_flags_recoverable_parse_failure(db_session):
    """#115 class: raw IS parseable but stored score_normalized is NULL → FAIL."""
    rnd, ath = _boulder_round(db_session)
    _add_result(db_session, rnd, ath, "0t 4b10", None)
    failures, _ = run_checks(db_session)
    assert any("parse-failure" in f and "B" in f for f in failures), failures


def test_guard_ignores_genuinely_unparseable_null(db_session):
    """A NULL whose raw is genuinely unparseable (parser also returns None) is OK."""
    rnd, ath = _boulder_round(db_session)
    _add_result(db_session, rnd, ath, "not-a-score", None)
    failures, _ = run_checks(db_session)
    assert failures == []


def test_guard_skips_dns_dnf(db_session):
    rnd, ath = _boulder_round(db_session)
    db_session.add(
        Result(
            round_id=rnd.id,
            athlete_id=ath.id,
            rank=999,
            raw_score="DNS",
            score_normalized=None,
            dns=True,
        )
    )
    db_session.flush()
    failures, _ = run_checks(db_session)
    assert failures == []


def test_guard_mismatch_warn_requires_full_scan(db_session):
    """Check 2 (stored≠recomputed WARN) only runs on the weekly full scan."""
    rnd, ath = _boulder_round(db_session)
    # "0t 4b10" recomputes to 390; store 999 → mismatch.
    _add_result(db_session, rnd, ath, "0t 4b10", 999.0)
    _, warnings_full = run_checks(db_session, full_scan=True)
    assert any("scale-drift" in w for w in warnings_full), warnings_full
    _, warnings_daily = run_checks(db_session, full_scan=False)
    assert warnings_daily == []


def test_guard_recoverable_check_runs_without_full_scan(db_session):
    """Check 1 (NULL-but-parseable FAIL) runs daily regardless of full_scan."""
    rnd, ath = _boulder_round(db_session)
    _add_result(db_session, rnd, ath, "0t 4b10", None)
    failures, _ = run_checks(db_session, full_scan=False)
    assert any("parse-failure" in f for f in failures), failures


def test_guard_flags_sigma_floor(db_session):
    """#95 class: a rating at the σ-floor → FAIL."""
    ath = Athlete(name="X", gender=Gender.M)
    db_session.add(ath)
    db_session.flush()
    db_session.add(
        Rating(athlete_id=ath.id, discipline=Discipline.BOULDER, mu=2000.0, sigma=50.0)
    )
    db_session.flush()
    failures, _ = run_checks(db_session)
    assert any("σ-floor" in f for f in failures), failures


def test_guard_flags_mu_p95_out_of_band(db_session):
    """Lead μ-p95 far below the elite band → FAIL (band disciplines only)."""
    for i in range(5):
        a = Athlete(name=f"A{i}", gender=Gender.M)
        db_session.add(a)
        db_session.flush()
        db_session.add(
            Rating(athlete_id=a.id, discipline=Discipline.LEAD, mu=1500.0, sigma=200.0)
        )
    db_session.flush()
    failures, _ = run_checks(db_session)
    assert any("μ-p95" in f and "L" in f for f in failures), failures


def test_guard_boulder_band_is_wider_than_lead(db_session):
    """μ-p95 of 1880 passes Boulder's post-#117 band (1850, 2200) but would
    fail Lead's (1900, 2200) - the bands are per-discipline."""
    for i in range(5):
        a = Athlete(name=f"B{i}", gender=Gender.M)
        db_session.add(a)
        db_session.flush()
        db_session.add(
            Rating(
                athlete_id=a.id, discipline=Discipline.BOULDER, mu=1880.0, sigma=200.0
            )
        )
        db_session.add(
            Rating(athlete_id=a.id, discipline=Discipline.LEAD, mu=1880.0, sigma=200.0)
        )
    db_session.flush()
    failures, _ = run_checks(db_session)
    assert not any("μ-p95" in f and "B:" in f for f in failures), failures
    assert any("μ-p95" in f and "L:" in f for f in failures), failures


def test_guard_speed_p95_is_informational(db_session):
    """Speed has no band — a low μ-p95 must not FAIL."""
    for i in range(5):
        a = Athlete(name=f"S{i}", gender=Gender.M)
        db_session.add(a)
        db_session.flush()
        db_session.add(
            Rating(athlete_id=a.id, discipline=Discipline.SPEED, mu=1500.0, sigma=200.0)
        )
    db_session.flush()
    failures, _ = run_checks(db_session)
    assert failures == []
