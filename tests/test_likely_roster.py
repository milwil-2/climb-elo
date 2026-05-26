"""Tests for engine/likely_roster.py (Issue #33).

Covers:
- Empty season: no events yet → returns top-cap-by-μ fallback
- Early season: 2 events (< 3) → returns top-cap-by-μ fallback
- Mid season: 5 events; 80% attendance included, 40% excluded
- Boundary: exactly 60% (3/5) included, just below (2/5) excluded
- Gender separation: women's events don't count toward men's threshold
- Tier filtering: continental events excluded from World Cup denominator
- Cap: more than cap eligible athletes → returns top-cap by μ
- DNS exclusion: athletes marked DNS don't count as participating
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
    # Reuse existing round if any.
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

    # Skip duplicate results (unique constraint on round_id + athlete_id).
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
# Test: empty season — no events yet → top-cap-by-μ fallback
# ---------------------------------------------------------------------------


class TestEmptySeason:
    def test_empty_season_returns_top_by_mu(self):
        """With no season events, fall back to top athletes by μ."""
        session = make_session()
        a1 = add_athlete(session, "A", Gender.M, mu=1800.0)
        a2 = add_athlete(session, "B", Gender.M, mu=1600.0)
        add_athlete(session, "C", Gender.M, mu=1400.0)
        session.commit()

        result = likely_competitors(session, Discipline.LEAD, 2026, Gender.M)

        assert len(result) > 0
        # Should be ordered by mu descending
        assert a1.id in result
        assert a2.id in result
        assert result.index(a1.id) < result.index(a2.id)

    def test_empty_season_excludes_wrong_gender(self):
        """Fallback must only return athletes of the requested gender."""
        session = make_session()
        male = add_athlete(session, "Male", Gender.M, mu=1700.0)
        female = add_athlete(session, "Female", Gender.F, mu=1900.0)
        session.commit()

        result = likely_competitors(session, Discipline.LEAD, 2026, Gender.M)

        assert male.id in result
        assert female.id not in result

    def test_empty_season_returns_empty_when_no_athletes(self):
        """A truly empty DB returns an empty list."""
        session = make_session()
        session.commit()

        result = likely_competitors(session, Discipline.LEAD, 2026, Gender.M)
        assert result == []


# ---------------------------------------------------------------------------
# Test: early season (< min_events_for_threshold) → top-cap-by-μ fallback
# ---------------------------------------------------------------------------


class TestEarlySeason:
    def test_two_events_uses_fallback(self):
        """With only 2 events (< 3), use the top-by-μ fallback."""
        session = make_session()
        a1 = add_athlete(session, "TopMu", Gender.M, mu=1900.0)
        a2 = add_athlete(session, "LowAttend", Gender.M, mu=1300.0)

        ev1 = add_wc_event(session, Discipline.LEAD, 2026, 1)
        ev2 = add_wc_event(session, Discipline.LEAD, 2026, 2)
        # a2 attended both; a1 attended neither
        add_result(session, ev1, a2, Gender.M)
        add_result(session, ev2, a2, Gender.M)
        session.commit()

        result = likely_competitors(session, Discipline.LEAD, 2026, Gender.M)

        # Fallback: top by mu, so a1 (mu=1900) should come before a2 (mu=1300)
        assert a1.id in result
        assert result.index(a1.id) < result.index(a2.id)


# ---------------------------------------------------------------------------
# Test: mid season — attendance threshold logic
# ---------------------------------------------------------------------------


class TestMidSeason:
    def _setup_five_events(
        self, session, discipline=Discipline.LEAD, season=2026, gender=Gender.M
    ):
        """Return 5 World Cup events for the given season."""
        return [add_wc_event(session, discipline, season, i) for i in range(1, 6)]

    def test_80_percent_included(self):
        """Athlete attending 4/5 events (80%) is included."""
        session = make_session()
        regular = add_athlete(session, "Regular", Gender.M, mu=1600.0)
        events = self._setup_five_events(session)
        for ev in events[:4]:  # 4 of 5 = 80%
            add_result(session, ev, regular, Gender.M)
        session.commit()

        result = likely_competitors(session, Discipline.LEAD, 2026, Gender.M)
        assert regular.id in result

    def test_40_percent_excluded(self):
        """Athlete attending 2/5 events (40%) is excluded.

        A background athlete attends all 5 events so all 5 are marked 'finished'
        (have results in the DB), giving a true denominator of 5.
        """
        session = make_session()
        irregular = add_athlete(session, "Irregular", Gender.M, mu=1600.0)
        background = add_athlete(session, "Background", Gender.M, mu=1700.0)
        events = self._setup_five_events(session)
        # background attends all 5 → denominator = 5
        for ev in events:
            add_result(session, ev, background, Gender.M)
        for ev in events[:2]:  # 2 of 5 = 40%
            add_result(session, ev, irregular, Gender.M)
        session.commit()

        result = likely_competitors(session, Discipline.LEAD, 2026, Gender.M)
        assert irregular.id not in result

    def test_both_in_same_season(self):
        """80% athlete included; 40% athlete excluded — both in same DB."""
        session = make_session()
        regular = add_athlete(session, "Regular", Gender.M, mu=1700.0)
        irregular = add_athlete(session, "Irregular", Gender.M, mu=1500.0)
        events = self._setup_five_events(session)
        # regular attends 4/5 (80%) → included
        for ev in events[:4]:
            add_result(session, ev, regular, Gender.M)
        # irregular attends 2/5 (40%) → excluded; regular's attendance also
        # covers events 0-3, so all 5 events have ≥1 result in the DB
        for ev in events[4:]:
            add_result(session, ev, regular, Gender.M)  # complete denominator
        for ev in events[:2]:
            add_result(session, ev, irregular, Gender.M)
        session.commit()

        result = likely_competitors(session, Discipline.LEAD, 2026, Gender.M)
        assert regular.id in result
        assert irregular.id not in result


# ---------------------------------------------------------------------------
# Test: boundary — exactly 60% included, just below 60% excluded
# ---------------------------------------------------------------------------


class TestBoundary:
    def _five_events(self, session):
        return [add_wc_event(session, Discipline.LEAD, 2026, i) for i in range(1, 6)]

    def test_exactly_60_percent_included(self):
        """Athlete with exactly 3/5 events (60%) meets the threshold → included."""
        session = make_session()
        athlete = add_athlete(session, "Boundary", Gender.M, mu=1600.0)
        events = self._five_events(session)
        for ev in events[:3]:  # 3/5 = 60%
            add_result(session, ev, athlete, Gender.M)
        session.commit()

        result = likely_competitors(session, Discipline.LEAD, 2026, Gender.M)
        assert athlete.id in result

    def test_just_below_60_percent_excluded(self):
        """Athlete with 2/5 events (40%) is below threshold → excluded.

        A background athlete ensures all 5 events have results so the
        denominator is correctly 5.
        """
        session = make_session()
        athlete = add_athlete(session, "JustBelow", Gender.M, mu=1600.0)
        background = add_athlete(session, "Background", Gender.M, mu=1700.0)
        events = self._five_events(session)
        for ev in events:
            add_result(session, ev, background, Gender.M)
        for ev in events[:2]:  # 2/5 = 40%
            add_result(session, ev, athlete, Gender.M)
        session.commit()

        result = likely_competitors(session, Discipline.LEAD, 2026, Gender.M)
        assert athlete.id not in result


# ---------------------------------------------------------------------------
# Test: gender separation
# ---------------------------------------------------------------------------


class TestGenderSeparation:
    def test_womens_events_dont_count_for_men(self):
        """A woman's attendance in Women's rounds must NOT count toward Men's threshold.

        A background male athlete attends all 5 events so the men's denominator
        is 5.  The test male attends only 2/5 (40%) → below threshold → excluded
        from men's likely roster even though women have results in all 5 events.
        """
        session = make_session()
        male = add_athlete(session, "Male", Gender.M, mu=1600.0)
        background_male = add_athlete(session, "BackgroundM", Gender.M, mu=1400.0)
        female = add_athlete(session, "Female", Gender.F, mu=1700.0)

        events = [add_wc_event(session, Discipline.LEAD, 2026, i) for i in range(1, 6)]

        # background male attends all 5 men's events → denominator = 5
        for ev in events:
            add_result(session, ev, background_male, Gender.M)

        # test male attends 2/5 men's rounds (40%) → below threshold
        for ev in events[:2]:
            add_result(session, ev, male, Gender.M)

        # Female attends 5/5 women's rounds (100%)
        for ev in events:
            add_result(session, ev, female, Gender.F)

        session.commit()

        men_result = likely_competitors(session, Discipline.LEAD, 2026, Gender.M)
        # male attended only 2/5 men's events → should be excluded
        assert male.id not in men_result

        women_result = likely_competitors(session, Discipline.LEAD, 2026, Gender.F)
        # female attended 5/5 → should be included
        assert female.id in women_result


# ---------------------------------------------------------------------------
# Test: tier filtering — continental events excluded
# ---------------------------------------------------------------------------


class TestTierFiltering:
    def test_continental_events_dont_count(self):
        """Continental events must be excluded from the World Cup denominator."""
        session = make_session()
        athlete = add_athlete(session, "Continental", Gender.M, mu=1600.0)

        # 3 continental events + 2 WC events → denominator should be 2 (< 3 = fallback)
        for i in range(1, 4):
            ev = add_wc_event(
                session, Discipline.LEAD, 2026, i, tier=EventTier.CONTINENTAL
            )
            add_result(session, ev, athlete, Gender.M)

        for i in range(4, 6):
            ev = add_wc_event(
                session, Discipline.LEAD, 2026, i, tier=EventTier.WORLD_CUP
            )
            add_result(session, ev, athlete, Gender.M)

        session.commit()

        # Only 2 WC events exist → below min_events_for_threshold → fallback mode
        # In fallback mode athlete is included if they have n_events >= 3
        result = likely_competitors(session, Discipline.LEAD, 2026, Gender.M)
        # athlete has n_events=10 in ratings, so should appear in fallback
        assert athlete.id in result

    def test_world_championship_not_counted(self):
        """World championship events are also not World Cup tier → excluded."""
        session = make_session()
        athlete = add_athlete(session, "WC_champ", Gender.M, mu=1600.0)

        # 5 world championship events — denominator for WC should be 0 → fallback
        for i in range(1, 6):
            ev = add_wc_event(
                session,
                Discipline.LEAD,
                2026,
                i,
                tier=EventTier.WORLD_CHAMPIONSHIP,
            )
            add_result(session, ev, athlete, Gender.M)

        session.commit()

        # 0 WC events → fallback
        result = likely_competitors(session, Discipline.LEAD, 2026, Gender.M)
        # athlete in fallback because they have n_events=10
        assert athlete.id in result


# ---------------------------------------------------------------------------
# Test: cap enforcement
# ---------------------------------------------------------------------------


class TestCap:
    def test_cap_limits_returned_athletes(self):
        """When more athletes than cap meet the threshold, only cap are returned."""
        session = make_session()
        athletes = [
            add_athlete(session, f"Athlete{i}", Gender.M, mu=1500.0 + i)
            for i in range(10)
        ]
        events = [add_wc_event(session, Discipline.LEAD, 2026, i) for i in range(1, 6)]
        # All 10 athletes attend all 5 events
        for athlete in athletes:
            for ev in events:
                add_result(session, ev, athlete, Gender.M)
        session.commit()

        # Cap at 5
        result = likely_competitors(session, Discipline.LEAD, 2026, Gender.M, cap=5)
        assert len(result) == 5

    def test_cap_returns_highest_mu_athletes(self):
        """When capping, the highest-μ athletes are returned."""
        session = make_session()
        athletes = [
            add_athlete(session, f"Athlete{i}", Gender.M, mu=float(1000 + i * 100))
            for i in range(6)
        ]
        events = [add_wc_event(session, Discipline.LEAD, 2026, i) for i in range(1, 6)]
        for athlete in athletes:
            for ev in events:
                add_result(session, ev, athlete, Gender.M)
        session.commit()

        result = likely_competitors(session, Discipline.LEAD, 2026, Gender.M, cap=3)
        assert len(result) == 3
        # Top 3 by mu: athletes[5], [4], [3]
        assert athletes[5].id in result
        assert athletes[4].id in result
        assert athletes[3].id in result
        assert athletes[0].id not in result


# ---------------------------------------------------------------------------
# Test: DNS exclusion
# ---------------------------------------------------------------------------


class TestDNSExclusion:
    def test_dns_does_not_count_as_participation(self):
        """A DNS entry must NOT count toward an athlete's event attendance."""
        session = make_session()
        dns_only = add_athlete(session, "DNSOnly", Gender.M, mu=1600.0)
        regular = add_athlete(session, "Regular", Gender.M, mu=1500.0)

        events = [add_wc_event(session, Discipline.LEAD, 2026, i) for i in range(1, 6)]

        # dns_only is present in the DB but DNS in all 5 events → 0 actual participations
        for ev in events:
            add_result(session, ev, dns_only, Gender.M, dns=True)

        # regular attends 3/5 events (60%) → meets threshold
        for ev in events[:3]:
            add_result(session, ev, regular, Gender.M, dns=False)

        session.commit()

        result = likely_competitors(session, Discipline.LEAD, 2026, Gender.M)
        assert dns_only.id not in result
        assert regular.id in result

    def test_mixed_dns_and_finishes(self):
        """Only non-DNS appearances count; mixed athlete (2 DNS + 2 finishes) of 5 is excluded."""
        session = make_session()
        mixed = add_athlete(session, "Mixed", Gender.M, mu=1600.0)
        events = [add_wc_event(session, Discipline.LEAD, 2026, i) for i in range(1, 6)]

        # 2 real participations + 2 DNS = only 2 count → 2/5 = 40% → excluded
        add_result(session, events[0], mixed, Gender.M, dns=False)
        add_result(session, events[1], mixed, Gender.M, dns=False)
        add_result(session, events[2], mixed, Gender.M, dns=True)
        add_result(session, events[3], mixed, Gender.M, dns=True)
        # events[4]: no result at all
        session.commit()

        result = likely_competitors(session, Discipline.LEAD, 2026, Gender.M)
        assert mixed.id not in result
