"""Tests for engine/likely_roster.py (Issue #33, updated #62).

Covers:
- Pre-season (empty DB) → returns []
- Athlete with 1 non-DNS event in season → included
- Athlete with 0 events in season (high historical μ) → excluded
- Gender separation: women's events don't count toward men's roster
- Tier filtering: continental/world-championship events excluded
- Cap: more than cap eligible athletes → returns top-cap by μ
- DNS exclusion: DNS results don't count as participation
"""

from __future__ import annotations

from datetime import date, timedelta

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from climbing_elo.engine.likely_roster import likely_competitors
from climbing_elo.models import (
    Athlete,
    Base,
    Discipline,
    Event,
    EventTier,
    Gender,
    Rating,
    Result,
    Round,
    RoundType,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    return factory()


def add_athlete(
    session,
    name: str,
    gender: Gender,
    mu: float = 1500.0,
    discipline: Discipline = Discipline.LEAD,
    n_events: int = 10,
) -> Athlete:
    a = Athlete(name=name, gender=gender)
    session.add(a)
    session.flush()
    session.add(
        Rating(
            athlete_id=a.id,
            discipline=discipline,
            mu=mu,
            sigma=200.0,
            n_events=n_events,
            provisional=False,
        )
    )
    session.flush()
    return a


def add_wc_event(
    session,
    discipline: Discipline,
    season: int,
    index: int,
    tier: EventTier = EventTier.WORLD_CUP,
) -> Event:
    ev = Event(
        name=f"{discipline.value} WC {season} #{index}",
        tier=tier,
        season=season,
        start_date=date(season, 1, 1) + timedelta(days=index * 30),
        discipline=discipline,
    )
    session.add(ev)
    session.flush()
    return ev


def add_result(
    session, event: Event, athlete: Athlete, gender: Gender, dns: bool = False
) -> None:
    """Add a qualification round result for athlete at event."""
    from sqlalchemy import select

    rnd = session.execute(
        select(Round).where(
            Round.event_id == event.id,
            Round.round_type == RoundType.QUALIFICATION,
            Round.gender == gender,
        )
    ).scalar_one_or_none()
    if rnd is None:
        rnd = Round(
            event_id=event.id,
            round_type=RoundType.QUALIFICATION,
            gender=gender,
            athlete_count=0,
        )
        session.add(rnd)
        session.flush()

    from sqlalchemy import select as sel

    existing = session.execute(
        sel(Result).where(
            Result.round_id == rnd.id,
            Result.athlete_id == athlete.id,
        )
    ).scalar_one_or_none()
    if existing is None:
        session.add(
            Result(
                round_id=rnd.id,
                athlete_id=athlete.id,
                rank=None if dns else 1,
                dns=dns,
            )
        )
        session.flush()


# ---------------------------------------------------------------------------
# Test: pre-season — empty DB → always returns []
# ---------------------------------------------------------------------------


class TestPreSeason:
    def test_empty_db_returns_empty_list(self):
        """With no events at all, return empty list — don't fabricate a roster."""
        session = make_session()
        session.commit()

        result = likely_competitors(session, Discipline.LEAD, 2026, Gender.M)
        assert result == []

    def test_no_season_events_returns_empty_even_with_high_mu_athletes(self):
        """Athletes with high historical mu but no current-season events are excluded."""
        session = make_session()
        add_athlete(session, "RetiredStar", Gender.M, mu=2000.0)
        add_athlete(session, "AnotherStar", Gender.M, mu=1900.0)
        session.commit()

        result = likely_competitors(session, Discipline.LEAD, 2026, Gender.M)
        assert result == []

    def test_events_in_different_season_return_empty(self):
        """Events from a prior season don't count; still returns []."""
        session = make_session()
        athlete = add_athlete(session, "OldTimer", Gender.M, mu=1800.0)
        # event from 2025, not 2026
        ev = add_wc_event(session, Discipline.LEAD, 2025, 1)
        add_result(session, ev, athlete, Gender.M)
        session.commit()

        result = likely_competitors(session, Discipline.LEAD, 2026, Gender.M)
        assert result == []


# ---------------------------------------------------------------------------
# Test: current-season participation → included
# ---------------------------------------------------------------------------


class TestCurrentSeasonParticipation:
    def test_athlete_with_one_event_included(self):
        """Athlete with exactly 1 non-DNS WC event this season is included."""
        session = make_session()
        athlete = add_athlete(session, "OneEvent", Gender.M, mu=1600.0)
        ev = add_wc_event(session, Discipline.LEAD, 2026, 1)
        add_result(session, ev, athlete, Gender.M)
        session.commit()

        result = likely_competitors(session, Discipline.LEAD, 2026, Gender.M)
        assert athlete.id in result

    def test_athlete_with_multiple_events_included(self):
        """Athlete attending several events this season is included."""
        session = make_session()
        athlete = add_athlete(session, "Regular", Gender.M, mu=1700.0)
        for i in range(1, 6):
            ev = add_wc_event(session, Discipline.LEAD, 2026, i)
            add_result(session, ev, athlete, Gender.M)
        session.commit()

        result = likely_competitors(session, Discipline.LEAD, 2026, Gender.M)
        assert athlete.id in result

    def test_high_mu_athlete_without_season_events_excluded(self):
        """An athlete with a high historical mu but zero current-season events is excluded."""
        session = make_session()
        retired_star = add_athlete(session, "RetiredStar", Gender.M, mu=2000.0)
        active = add_athlete(session, "Active", Gender.M, mu=1500.0)
        ev = add_wc_event(session, Discipline.LEAD, 2026, 1)
        add_result(session, ev, active, Gender.M)
        session.commit()

        result = likely_competitors(session, Discipline.LEAD, 2026, Gender.M)
        assert active.id in result
        assert retired_star.id not in result

    def test_ordered_by_mu_descending(self):
        """Results are ordered by mu descending."""
        session = make_session()
        low = add_athlete(session, "Low", Gender.M, mu=1300.0)
        high = add_athlete(session, "High", Gender.M, mu=1800.0)
        mid = add_athlete(session, "Mid", Gender.M, mu=1550.0)
        ev = add_wc_event(session, Discipline.LEAD, 2026, 1)
        for a in [low, high, mid]:
            add_result(session, ev, a, Gender.M)
        session.commit()

        result = likely_competitors(session, Discipline.LEAD, 2026, Gender.M)
        assert result.index(high.id) < result.index(mid.id) < result.index(low.id)


# ---------------------------------------------------------------------------
# Test: gender separation
# ---------------------------------------------------------------------------


class TestGenderSeparation:
    def test_womens_results_not_counted_for_men(self):
        """Women's WC results must NOT cause male athletes to appear in men's roster."""
        session = make_session()
        male = add_athlete(session, "Male", Gender.M, mu=1600.0)
        female = add_athlete(session, "Female", Gender.F, mu=1700.0)
        ev = add_wc_event(session, Discipline.LEAD, 2026, 1)
        # only female has a result this season
        add_result(session, ev, female, Gender.F)
        session.commit()

        men_result = likely_competitors(session, Discipline.LEAD, 2026, Gender.M)
        assert male.id not in men_result
        assert men_result == []

        women_result = likely_competitors(session, Discipline.LEAD, 2026, Gender.F)
        assert female.id in women_result

    def test_gender_filter_applied_to_results(self):
        """Requesting men only returns men even when women competed at same event."""
        session = make_session()
        male = add_athlete(session, "Male", Gender.M, mu=1600.0)
        female = add_athlete(session, "Female", Gender.F, mu=1900.0)
        ev = add_wc_event(session, Discipline.LEAD, 2026, 1)
        add_result(session, ev, male, Gender.M)
        add_result(session, ev, female, Gender.F)
        session.commit()

        men_result = likely_competitors(session, Discipline.LEAD, 2026, Gender.M)
        women_result = likely_competitors(session, Discipline.LEAD, 2026, Gender.F)

        assert male.id in men_result
        assert female.id not in men_result
        assert female.id in women_result
        assert male.id not in women_result


# ---------------------------------------------------------------------------
# Test: tier filtering — only WORLD_CUP events count
# ---------------------------------------------------------------------------


class TestTierFiltering:
    def test_continental_events_not_counted(self):
        """Continental-tier results don't qualify an athlete for the likely roster."""
        session = make_session()
        athlete = add_athlete(session, "Continental", Gender.M, mu=1600.0)
        for i in range(1, 4):
            ev = add_wc_event(
                session, Discipline.LEAD, 2026, i, tier=EventTier.CONTINENTAL
            )
            add_result(session, ev, athlete, Gender.M)
        session.commit()

        result = likely_competitors(session, Discipline.LEAD, 2026, Gender.M)
        assert athlete.id not in result
        assert result == []

    def test_world_championship_not_counted(self):
        """World Championship results don't qualify an athlete."""
        session = make_session()
        athlete = add_athlete(session, "WCChamp", Gender.M, mu=1600.0)
        for i in range(1, 4):
            ev = add_wc_event(
                session,
                Discipline.LEAD,
                2026,
                i,
                tier=EventTier.WORLD_CHAMPIONSHIP,
            )
            add_result(session, ev, athlete, Gender.M)
        session.commit()

        result = likely_competitors(session, Discipline.LEAD, 2026, Gender.M)
        assert athlete.id not in result
        assert result == []

    def test_wc_tier_counts_but_continental_does_not(self):
        """Athlete with mixed tiers is included because they have ≥1 WC result."""
        session = make_session()
        athlete = add_athlete(session, "Mixed", Gender.M, mu=1600.0)
        cont_ev = add_wc_event(
            session, Discipline.LEAD, 2026, 1, tier=EventTier.CONTINENTAL
        )
        add_result(session, cont_ev, athlete, Gender.M)
        wc_ev = add_wc_event(session, Discipline.LEAD, 2026, 2)
        add_result(session, wc_ev, athlete, Gender.M)
        session.commit()

        result = likely_competitors(session, Discipline.LEAD, 2026, Gender.M)
        assert athlete.id in result


# ---------------------------------------------------------------------------
# Test: cap enforcement
# ---------------------------------------------------------------------------


class TestCap:
    def test_cap_limits_returned_athletes(self):
        """When more athletes than cap qualify, only cap are returned."""
        session = make_session()
        athletes = [
            add_athlete(session, f"Athlete{i}", Gender.M, mu=1500.0 + i)
            for i in range(10)
        ]
        ev = add_wc_event(session, Discipline.LEAD, 2026, 1)
        for a in athletes:
            add_result(session, ev, a, Gender.M)
        session.commit()

        result = likely_competitors(session, Discipline.LEAD, 2026, Gender.M, cap=5)
        assert len(result) == 5

    def test_cap_returns_highest_mu_athletes(self):
        """When capping, the highest-mu athletes are returned."""
        session = make_session()
        athletes = [
            add_athlete(session, f"Athlete{i}", Gender.M, mu=float(1000 + i * 100))
            for i in range(6)
        ]
        ev = add_wc_event(session, Discipline.LEAD, 2026, 1)
        for a in athletes:
            add_result(session, ev, a, Gender.M)
        session.commit()

        result = likely_competitors(session, Discipline.LEAD, 2026, Gender.M, cap=3)
        assert len(result) == 3
        # Top 3 by mu: athletes[5] (1500), athletes[4] (1400), athletes[3] (1300)
        assert athletes[5].id in result
        assert athletes[4].id in result
        assert athletes[3].id in result
        assert athletes[0].id not in result


# ---------------------------------------------------------------------------
# Test: DNS exclusion
# ---------------------------------------------------------------------------


class TestDNSExclusion:
    def test_dns_only_athlete_excluded(self):
        """An athlete with only DNS results is not included."""
        session = make_session()
        dns_only = add_athlete(session, "DNSOnly", Gender.M, mu=1800.0)
        ev = add_wc_event(session, Discipline.LEAD, 2026, 1)
        add_result(session, ev, dns_only, Gender.M, dns=True)
        session.commit()

        result = likely_competitors(session, Discipline.LEAD, 2026, Gender.M)
        assert dns_only.id not in result
        assert result == []

    def test_dns_does_not_count_as_participation(self):
        """Mixed scenario: DNS-only athlete excluded, athlete with real result included."""
        session = make_session()
        dns_only = add_athlete(session, "DNSOnly", Gender.M, mu=1600.0)
        regular = add_athlete(session, "Regular", Gender.M, mu=1500.0)
        ev = add_wc_event(session, Discipline.LEAD, 2026, 1)
        add_result(session, ev, dns_only, Gender.M, dns=True)
        add_result(session, ev, regular, Gender.M, dns=False)
        session.commit()

        result = likely_competitors(session, Discipline.LEAD, 2026, Gender.M)
        assert dns_only.id not in result
        assert regular.id in result

    def test_athlete_with_dns_and_real_result_included(self):
        """Athlete DNS at one event but competed at another is included."""
        session = make_session()
        athlete = add_athlete(session, "SometimesDNS", Gender.M, mu=1600.0)
        ev1 = add_wc_event(session, Discipline.LEAD, 2026, 1)
        ev2 = add_wc_event(session, Discipline.LEAD, 2026, 2)
        add_result(session, ev1, athlete, Gender.M, dns=True)
        add_result(session, ev2, athlete, Gender.M, dns=False)
        session.commit()

        result = likely_competitors(session, Discipline.LEAD, 2026, Gender.M)
        assert athlete.id in result
