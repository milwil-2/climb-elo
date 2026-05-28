"""Integration test for the backfill pipeline."""

from datetime import date

from sqlalchemy import select

from climbing_elo.engine.backfill import run_backfill
from climbing_elo.models import (
    Athlete,
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


def _seed_event(session, name, event_date, athletes, final_order):
    """Create an event with a final round and seed results."""
    event = Event(
        name=name,
        tier=EventTier.WORLD_CUP,
        season=event_date.year,
        start_date=event_date,
        discipline=Discipline.LEAD,
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

    for rank, athlete_idx in enumerate(final_order, 1):
        session.add(
            Result(
                round_id=rnd.id,
                athlete_id=athletes[athlete_idx].id,
                rank=rank,
            )
        )
    session.flush()
    return event


def test_backfill_three_events(db_session):
    """Run backfill over 3 events and verify ratings converge sensibly."""
    athletes = []
    for name in ["Alpha", "Beta", "Gamma", "Delta"]:
        a = Athlete(name=name, gender=Gender.M)
        db_session.add(a)
        athletes.append(a)
    db_session.flush()

    # Event 1: Alpha > Beta > Gamma > Delta
    _seed_event(db_session, "WC Innsbruck", date(2024, 3, 1), athletes, [0, 1, 2, 3])
    # Event 2: Beta > Alpha > Delta > Gamma
    _seed_event(db_session, "WC Chamonix", date(2024, 5, 1), athletes, [1, 0, 3, 2])
    # Event 3: Alpha > Alpha again > Gamma > Beta — wait, can't repeat.
    # Event 3: Alpha > Gamma > Beta > Delta
    _seed_event(db_session, "WC Briançon", date(2024, 7, 1), athletes, [0, 2, 1, 3])
    db_session.commit()

    report = run_backfill(db_session, Discipline.LEAD)

    assert report.events_processed == 3
    assert report.rounds_processed == 3
    assert len(report.athletes_rated) == 4
    assert len(report.errors) == 0

    ratings = {
        r.athlete_id: r
        for r in db_session.execute(
            select(Rating).where(Rating.discipline == Discipline.LEAD)
        ).scalars()
    }

    # Alpha won 2 of 3 events and placed 2nd in the other — should be top
    alpha_id = athletes[0].id
    beta_id = athletes[1].id
    delta_id = athletes[3].id

    assert ratings[alpha_id].mu > ratings[beta_id].mu
    assert ratings[beta_id].mu > ratings[delta_id].mu

    # All should have n_events = 3
    for r in ratings.values():
        assert r.n_events == 3
        assert r.provisional is False  # >= PROVISIONAL_THRESHOLD

    # Rating history should have entries
    history_count = db_session.execute(select(RatingHistory)).all()
    assert len(history_count) >= 12  # 4 athletes × 3 events


def test_backfill_reproducibility(db_session):
    """Running backfill twice on the same data should produce identical results."""
    athletes = []
    for name in ["X", "Y", "Z"]:
        a = Athlete(name=name, gender=Gender.M)
        db_session.add(a)
        athletes.append(a)
    db_session.flush()

    _seed_event(db_session, "Test WC 1", date(2024, 1, 1), athletes, [0, 1, 2])
    _seed_event(db_session, "Test WC 2", date(2024, 6, 1), athletes, [2, 0, 1])
    db_session.commit()

    run_backfill(db_session, Discipline.LEAD)
    first_run = {
        r.athlete_id: r.mu
        for r in db_session.execute(
            select(Rating).where(Rating.discipline == Discipline.LEAD)
        ).scalars()
    }

    # Reset ratings and history
    for r in db_session.execute(select(Rating)).scalars():
        db_session.delete(r)
    for rh in db_session.execute(select(RatingHistory)).scalars():
        db_session.delete(rh)
    db_session.commit()

    run_backfill(db_session, Discipline.LEAD)
    second_run = {
        r.athlete_id: r.mu
        for r in db_session.execute(
            select(Rating).where(Rating.discipline == Discipline.LEAD)
        ).scalars()
    }

    for aid in first_run:
        assert abs(first_run[aid] - second_run[aid]) < 0.0001, (
            f"Athlete {aid}: first={first_run[aid]}, second={second_run[aid]}"
        )


def test_backfill_writes_tpb_rows(db_session):
    """Issue #90: backfill writes one kind='tpb' RatingHistory row per athlete
    per event, pointing at the final round, and the deltas sum to zero."""
    athletes = []
    for name in ["A1", "A2", "A3", "A4"]:
        a = Athlete(name=name, gender=Gender.M)
        db_session.add(a)
        athletes.append(a)
    db_session.flush()

    event = _seed_event(
        db_session, "WC TPB Test", date(2024, 4, 1), athletes, [0, 1, 2, 3]
    )
    db_session.commit()

    run_backfill(db_session, Discipline.LEAD)

    tpb_rows = list(
        db_session.execute(
            select(RatingHistory).where(
                RatingHistory.event_id == event.id,
                RatingHistory.kind == "tpb",
            )
        ).scalars()
    )
    assert len(tpb_rows) == 4

    total_delta = sum(r.mu_after - r.mu_before for r in tpb_rows)
    assert abs(total_delta) < 1e-6, f"TPB deltas not zero-sum: {total_delta}"

    by_athlete = {r.athlete_id: r for r in tpb_rows}
    winner_id = athletes[0].id
    loser_id = athletes[3].id
    assert by_athlete[winner_id].mu_after > by_athlete[winner_id].mu_before
    assert by_athlete[loser_id].mu_after < by_athlete[loser_id].mu_before

    pair_rows = list(
        db_session.execute(
            select(RatingHistory).where(
                RatingHistory.event_id == event.id,
                RatingHistory.kind == "pair",
            )
        ).scalars()
    )
    assert len(pair_rows) == 4


def test_backfill_tpb_idempotent(db_session):
    """Re-running backfill must NOT duplicate TPB rows (uq constraint guards it)."""
    athletes = []
    for name in ["B1", "B2", "B3"]:
        a = Athlete(name=name, gender=Gender.M)
        db_session.add(a)
        athletes.append(a)
    db_session.flush()

    event = _seed_event(
        db_session, "WC Idempotent", date(2024, 4, 1), athletes, [0, 1, 2]
    )
    db_session.commit()

    run_backfill(db_session, Discipline.LEAD)
    run_backfill(db_session, Discipline.LEAD)

    tpb_rows = list(
        db_session.execute(
            select(RatingHistory).where(
                RatingHistory.event_id == event.id,
                RatingHistory.kind == "tpb",
            )
        ).scalars()
    )
    assert len(tpb_rows) == 3  # one per athlete, not six


def _seed_multi_round_event(session, name, event_date, athletes, ranks_per_round):
    """Create an event with qualification + final rounds. Helper for #89 Fix 3
    regression test. `ranks_per_round` is a dict {RoundType: [athlete_idx,...]}.
    """
    event = Event(
        name=name,
        tier=EventTier.WORLD_CUP,
        season=event_date.year,
        start_date=event_date,
        discipline=Discipline.LEAD,
    )
    session.add(event)
    session.flush()

    for round_type, ordering in ranks_per_round.items():
        rnd = Round(
            event_id=event.id,
            round_type=round_type,
            gender=Gender.M,
            athlete_count=len(ordering),
        )
        session.add(rnd)
        session.flush()
        for rank, athlete_idx in enumerate(ordering, 1):
            session.add(
                Result(
                    round_id=rnd.id,
                    athlete_id=athletes[athlete_idx].id,
                    rank=rank,
                )
            )
    session.flush()
    return event


def test_backfill_multi_round_event_does_not_reinflate_sigma(db_session):
    """Issue #89 Fix 3: round 2 of a multi-round event must NOT re-inflate σ.

    Pre-fix bug: ``Rating.last_event_at`` was only updated once per event (at
    the end of the event loop), so when round 2 of the same event ran,
    ``glicko2_inflate_phi`` saw the *prior event's* date and applied a large
    gap inflation, clamping σ back to the ceiling. Per-round σ shrinkage was
    wiped out on every subsequent round of the same event.

    Fix: in the per-round update loop, set ``ratings_cache[aid].last_event_at
    = event.start_date`` so the next round of the same event sees zero gap.

    This test enforces the invariant by checking that the final round's
    ``sigma_before`` exactly matches the qualification round's ``sigma_after``
    (i.e. no inflation between rounds of the same event).
    """
    from climbing_elo.models import RoundType as RT

    athletes = []
    for name in ["MR1", "MR2", "MR3", "MR4"]:
        a = Athlete(name=name, gender=Gender.M)
        db_session.add(a)
        athletes.append(a)
    db_session.flush()

    # Two events 6 months apart so the FIRST round of event 2 *will* inflate
    # σ legitimately (gap > 30-day grace). The bug we're catching is round 2
    # of event 2 also re-inflating.
    _seed_multi_round_event(
        db_session,
        "WC Innsbruck",
        date(2024, 1, 1),
        athletes,
        {RT.QUALIFICATION: [0, 1, 2, 3], RT.FINAL: [0, 1, 2, 3]},
    )
    _seed_multi_round_event(
        db_session,
        "WC Chamonix",
        date(2024, 7, 1),  # 6 months later — beyond the 30-day grace
        athletes,
        {RT.QUALIFICATION: [1, 0, 3, 2], RT.FINAL: [1, 0, 3, 2]},
    )
    db_session.commit()

    run_backfill(db_session, Discipline.LEAD)

    # Pull all pair-kind history rows for athlete[0] across both events,
    # ordered chronologically by event then round.
    rows = list(
        db_session.execute(
            select(RatingHistory, Round, Event)
            .join(Round, RatingHistory.round_id == Round.id)
            .join(Event, RatingHistory.event_id == Event.id)
            .where(
                RatingHistory.athlete_id == athletes[0].id,
                RatingHistory.kind == "pair",
            )
            .order_by(Event.start_date, Round.round_type)
        ).all()
    )
    assert len(rows) == 4, "Expected 4 pair rows (2 events × 2 rounds)"

    # For each event, the second round's sigma_before must equal the first
    # round's sigma_after — i.e. no σ re-inflation between rounds of the
    # same event. Sort by the canonical round ORDER (qual → semi → final),
    # NOT by enum value (which is alphabetical: "final" sorts before
    # "qualification" — that's a classic gotcha).
    from climbing_elo.engine.backfill import ROUND_ORDER

    event_groups: dict[int, list] = {}
    for rh, rnd, ev in rows:
        event_groups.setdefault(ev.id, []).append((rnd.round_type, rh))

    for ev_id, group in event_groups.items():
        group.sort(key=lambda x: ROUND_ORDER[x[0]])
        assert len(group) == 2, f"Event {ev_id}: expected 2 rounds, got {len(group)}"
        qual_after = group[0][1].sigma_after
        final_before = group[1][1].sigma_before
        assert abs(final_before - qual_after) < 1e-6, (
            f"Event {ev_id}: σ re-inflated between rounds — qualification "
            f"σ_after={qual_after:.3f}, final σ_before={final_before:.3f}. "
            f"This is the #89 Fix 3 regression — see engine/backfill.py."
        )

    # Sanity: σ should be visibly shrinking across events. After 4 round-
    # updates on athlete[0] (the consistent winner), σ should be well below
    # the ceiling.
    final_sigma = rows[-1][0].sigma_after
    assert final_sigma < 300.0, (
        f"After 4 round-updates σ_after={final_sigma:.1f}; expected meaningful "
        f"shrinkage well below the 350 ceiling."
    )
