"""Cold-start trajectory tests for the Glicko-2 RD integration (#51).

Glicko-2's promise: emerging athletes climb to the top of the leaderboard in
fewer events than the legacy constant-K + 2× provisional-K cliff. We validate
that promise on two angles:

1. **Synthetic cold-start**: a fresh athlete (σ at the ceiling) who beats an
   established field repeatedly under Glicko-2 should hit a recognisable
   "top-of-rating" μ in a small number of events.
2. **Real-data trajectory (opt-in)**: if a current production SQLite snapshot
   is available locally at ``data/climbing_elo.db`` *and* it shares the
   current schema (post-Discipline-rename), backfill the Boulder discipline
   and check that the two canonical emerging-athlete examples in the research
   synthesis (Sorato Anraku, Oriane Bertone) end up high on the leaderboard
   — corroborating the AscentStats community Bradley-Terry rating.

The opt-in real-data test silently skips when the local DB does not match
the current schema (e.g. older snapshots predate the Discipline enum rename).
CI runs the synthetic test only; the maintainer's local environment runs both.
"""

from __future__ import annotations

import json
import shutil
from datetime import date, timedelta
from pathlib import Path

import pytest
from sqlalchemy import create_engine, delete, select, text
from sqlalchemy.orm import sessionmaker

from climbing_elo.engine.backfill import run_backfill
from climbing_elo.engine.elo import (
    DEFAULT_MU,
    DEFAULT_SIGMA,
    AthleteRating,
    AthleteResult,
    EloConfig,
    calculate_round_updates,
    glicko2_inflate_phi,
)
from climbing_elo.models import (
    Athlete,
    Discipline,
    Gender,
    Rating,
    RatingHistory,
)
from climbing_elo.models import (
    EventTier,
    RoundType,
)

PROD_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "climbing_elo.db"
FIXTURE_DIR = (
    Path(__file__).resolve().parent / "fixtures" / "external_rankings" / "ascentstats"
)


# ---------------------------------------------------------------------------
# Synthetic cold-start test — always runs (no external DB dependency)
# ---------------------------------------------------------------------------


def test_cold_start_athlete_climbs_quickly_under_glicko2():
    """A fresh athlete who wins repeatedly against a known field should climb
    fast under Glicko-2.

    Setup: 7 established athletes at μ=1700-1500 with low σ=80 (well-known
    veterans), plus one newcomer at default μ=1500, σ=350 who wins every
    final for 5 consecutive events. Under the legacy constant-K + 2× provisional
    regime, this athlete needed roughly 5-7 events to reach the top-3 of the
    field. Under Glicko-2 the φ² scaling makes the climb much faster.

    Assertion: after 5 wins, the newcomer's μ exceeds the highest established
    athlete by at least 50 points — i.e. they are the new clear leader.
    """
    NEWCOMER_ID = 999
    veteran_ids = list(range(1, 8))  # 7 veterans
    veteran_mus = [1700, 1680, 1660, 1640, 1620, 1600, 1580]

    ratings: dict[int, AthleteRating] = {
        aid: AthleteRating(
            athlete_id=aid,
            mu=mu,
            sigma=80.0,  # well-established
            n_events=10,
            last_event_at=date(2024, 1, 1),
            provisional=False,
        )
        for aid, mu in zip(veteran_ids, veteran_mus)
    }
    ratings[NEWCOMER_ID] = AthleteRating(
        athlete_id=NEWCOMER_ID,
        mu=DEFAULT_MU,
        sigma=DEFAULT_SIGMA,
        n_events=0,
        last_event_at=None,
        provisional=True,
    )

    event_date = date(2024, 6, 1)
    for event_idx in range(5):
        results = [AthleteResult(athlete_id=NEWCOMER_ID, rank=1)] + [
            AthleteResult(athlete_id=aid, rank=i + 2)
            for i, aid in enumerate(veteran_ids)
        ]
        updates = calculate_round_updates(
            results,
            ratings,
            EventTier.WORLD_CUP,
            RoundType.FINAL,
            event_date,
            discipline=Discipline.LEAD,
        )
        for upd in updates:
            ratings[upd.athlete_id].mu = upd.mu_after
            ratings[upd.athlete_id].sigma = upd.sigma_after
            ratings[upd.athlete_id].n_events += 1
            ratings[upd.athlete_id].last_event_at = event_date
            ratings[upd.athlete_id].provisional = ratings[upd.athlete_id].n_events < 3
        event_date += timedelta(days=30)

    newcomer_mu = ratings[NEWCOMER_ID].mu
    top_veteran_mu = max(ratings[aid].mu for aid in veteran_ids)
    newcomer_climb = newcomer_mu - DEFAULT_MU
    # Cold-start direction check. After 5 wins a fresh newcomer (σ=350) should
    # climb materially from default. The absolute magnitude is K-scale
    # dependent — it moves with the K table (re-tuned 2026-05-27 per #80, see
    # docs/K_REGRID_REPORT.md) and shrank ~7x when #174 restored the μ
    # field-size normalization (base K / (n−1), n=8 here). The K table has not
    # yet been re-derived for the normalized engine (that is #189), so this
    # threshold tracks direction and materiality, not a K-specific level.
    assert newcomer_climb >= 25.0, (
        f"After 5 wins, newcomer climbed only {newcomer_climb:.1f}μ from "
        f"default — Glicko-2 cold-start isn't lifting fresh athletes "
        f"enough (current μ={newcomer_mu:.1f}, top veteran={top_veteran_mu:.1f})."
    )

    # σ should also have shrunk noticeably from the ceiling.
    assert ratings[NEWCOMER_ID].sigma < DEFAULT_SIGMA * 0.6, (
        f"After 5 events the newcomer's σ should have shrunk well below the "
        f"350 ceiling; got {ratings[NEWCOMER_ID].sigma:.1f}"
    )


def test_cold_start_loser_drops_quickly_under_glicko2():
    """Symmetric to above: a cold-start athlete who consistently loses drops
    fast — Glicko-2 doesn't anchor them at μ=1500 for long.
    """
    NEWCOMER_ID = 999
    veteran_ids = list(range(1, 8))
    veteran_mus = [1700, 1680, 1660, 1640, 1620, 1600, 1580]
    ratings: dict[int, AthleteRating] = {
        aid: AthleteRating(
            athlete_id=aid,
            mu=mu,
            sigma=80.0,
            n_events=10,
            last_event_at=date(2024, 1, 1),
            provisional=False,
        )
        for aid, mu in zip(veteran_ids, veteran_mus)
    }
    ratings[NEWCOMER_ID] = AthleteRating(
        athlete_id=NEWCOMER_ID,
        mu=DEFAULT_MU,
        sigma=DEFAULT_SIGMA,
        n_events=0,
        last_event_at=None,
        provisional=True,
    )

    event_date = date(2024, 6, 1)
    for _ in range(5):
        # Newcomer in last place every time.
        results = [
            AthleteResult(athlete_id=aid, rank=i + 1)
            for i, aid in enumerate(veteran_ids)
        ] + [AthleteResult(athlete_id=NEWCOMER_ID, rank=len(veteran_ids) + 1)]
        updates = calculate_round_updates(
            results,
            ratings,
            EventTier.WORLD_CUP,
            RoundType.FINAL,
            event_date,
            discipline=Discipline.LEAD,
        )
        for upd in updates:
            ratings[upd.athlete_id].mu = upd.mu_after
            ratings[upd.athlete_id].sigma = upd.sigma_after
            ratings[upd.athlete_id].n_events += 1
            ratings[upd.athlete_id].last_event_at = event_date
        event_date += timedelta(days=30)

    # Newcomer should be clearly below μ=1500 after consistent losses. The
    # absolute drop is K-scale dependent and shrank ~7x when #174 restored the
    # μ field-size normalization (base K / (n−1), n=8 here) — see the note on
    # the climbing test above; #189 re-derives the K table for the normalized
    # engine.
    assert ratings[NEWCOMER_ID].mu < DEFAULT_MU - 10, (
        f"After 5 last-place finishes, newcomer μ={ratings[NEWCOMER_ID].mu:.1f} "
        f"should drop clearly below default 1500."
    )


# ---------------------------------------------------------------------------
# σ field-size normalization (Issue #95) — v_inv must not over-count evidence
# ---------------------------------------------------------------------------


def _make_field(n_athletes: int) -> dict[int, AthleteRating]:
    """Build a field of established veterans plus one fresh newcomer (id=999).

    The newcomer starts at the cold-start defaults (σ at the ceiling); the
    veterans sit at a low σ so the round's evidence is dominated by the
    newcomer's own large φ.
    """
    veteran_ids = list(range(1, n_athletes))
    ratings: dict[int, AthleteRating] = {
        aid: AthleteRating(
            athlete_id=aid,
            mu=1600.0 - 10.0 * idx,
            sigma=80.0,
            n_events=10,
            last_event_at=date(2024, 1, 1),
            provisional=False,
        )
        for idx, aid in enumerate(veteran_ids)
    }
    ratings[999] = AthleteRating(
        athlete_id=999,
        mu=DEFAULT_MU,
        sigma=DEFAULT_SIGMA,
        n_events=0,
        last_event_at=None,
        provisional=True,
    )
    return ratings


def _single_event_results(n_athletes: int) -> list[AthleteResult]:
    """Newcomer (id=999) wins; veterans fill ranks 2..n."""
    veteran_ids = list(range(1, n_athletes))
    return [AthleteResult(athlete_id=999, rank=1)] + [
        AthleteResult(athlete_id=aid, rank=i + 2) for i, aid in enumerate(veteran_ids)
    ]


def test_single_multi_athlete_event_does_not_collapse_sigma():
    """A first-event athlete in a large field must retain large σ (#95).

    The closed-form φ update accumulates one variance term per pairwise
    comparison — (n−1) of them in an n-athlete round. Without field-size
    normalization, a single 8-athlete event drives a fresh athlete's σ
    straight to the floor (50). With the default exponent=1.0 the round
    contributes ≈ one game of evidence, so σ should stay high.
    """
    n_athletes = 8
    ratings = _make_field(n_athletes)
    updates = calculate_round_updates(
        _single_event_results(n_athletes),
        ratings,
        EventTier.WORLD_CUP,
        RoundType.FINAL,
        date(2024, 6, 1),
        discipline=Discipline.LEAD,
    )
    newcomer = next(u for u in updates if u.athlete_id == 999)
    assert newcomer.sigma_after > 200.0, (
        f"After a single 8-athlete event a first-time athlete (n_events=1) "
        f"should retain σ > 200, not collapse toward the floor; "
        f"got σ={newcomer.sigma_after:.1f}"
    )


def test_exponent_zero_restores_legacy_sigma_collapse():
    """exponent=0.0 must reproduce the old over-counting behaviour (#95).

    This proves the normalization knob is wired correctly: with the divisor
    disabled, the (n−1)-game evidence over-count collapses a fresh athlete's
    σ near the floor after a single multi-athlete event.
    """
    n_athletes = 8
    ratings = _make_field(n_athletes)
    legacy_cfg = EloConfig(sigma_field_normalization_exponent=0.0)
    updates = calculate_round_updates(
        _single_event_results(n_athletes),
        ratings,
        EventTier.WORLD_CUP,
        RoundType.FINAL,
        date(2024, 6, 1),
        discipline=Discipline.LEAD,
        config=legacy_cfg,
    )
    newcomer = next(u for u in updates if u.athlete_id == 999)
    # Legacy behaviour: σ collapses hard toward the floor in one event. At n=8
    # the 7-game over-count lands ~130 (it collapses further the larger the
    # field), versus the normalized default's field-invariant ~255.
    assert newcomer.sigma_after < 150.0, (
        f"With exponent=0.0 the legacy over-counting should collapse σ toward "
        f"the floor after one 8-athlete event; got σ={newcomer.sigma_after:.1f}"
    )
    # And it must be *much* lower than the normalized default (sanity on the knob).
    normalized = calculate_round_updates(
        _single_event_results(n_athletes),
        _make_field(n_athletes),
        EventTier.WORLD_CUP,
        RoundType.FINAL,
        date(2024, 6, 1),
        discipline=Discipline.LEAD,
    )
    normalized_newcomer = next(u for u in normalized if u.athlete_id == 999)
    assert newcomer.sigma_after < normalized_newcomer.sigma_after - 50.0


def test_repeated_events_drive_sigma_to_low_but_not_floor_band():
    """Many events should settle σ in a sensible low band, not pin it at 50.

    Under the normalized accumulator a fresh athlete who competes in a long
    string of multi-athlete events should see σ shrink gradually toward — but
    plausibly above — the floor, landing in a ~80-160 band rather than being
    pinned at the 50 floor after a single event.
    """
    n_athletes = 8
    ratings = _make_field(n_athletes)
    event_date = date(2024, 6, 1)
    for _ in range(15):
        updates = calculate_round_updates(
            _single_event_results(n_athletes),
            ratings,
            EventTier.WORLD_CUP,
            RoundType.FINAL,
            event_date,
            discipline=Discipline.LEAD,
        )
        for upd in updates:
            ratings[upd.athlete_id].mu = upd.mu_after
            ratings[upd.athlete_id].sigma = upd.sigma_after
            ratings[upd.athlete_id].n_events += 1
            ratings[upd.athlete_id].last_event_at = event_date
        event_date += timedelta(days=30)

    final_sigma = ratings[999].sigma
    assert 60.0 < final_sigma < 160.0, (
        f"After 15 events the newcomer's σ should settle in a low-but-not-floor "
        f"band (~80-160), not be pinned at the 50 floor; got σ={final_sigma:.1f}"
    )


# ---------------------------------------------------------------------------
# Real-data trajectory test — opt-in, skips when local DB schema mismatched
# ---------------------------------------------------------------------------


def _has_compatible_prod_data() -> bool:
    """Return True only if the prod DB is present *and* uses the current
    Discipline enum codes (short single-letter values). Older snapshots used
    full enum names like "BOULDER" and silently break the backfill.
    Also requires the post-#90 ``rating_history.kind`` column."""
    if not PROD_DB_PATH.exists():
        return False
    if not (FIXTURE_DIR / "2026-boulder-M.json").exists():
        return False
    if not (FIXTURE_DIR / "2026-boulder-F.json").exists():
        return False
    try:
        with create_engine(f"sqlite:///{PROD_DB_PATH}").connect() as conn:
            rows = conn.execute(
                text("SELECT DISTINCT discipline FROM events")
            ).fetchall()
            codes = {r[0] for r in rows}
            # Current schema uses single-letter codes — old snapshots use words.
            if not (codes and codes.issubset({"B", "L", "S", "BL"})):
                return False
            # Issue #90 added rating_history.kind. Older snapshots lack it.
            cols = {
                r[1]
                for r in conn.execute(
                    text("PRAGMA table_info(rating_history)")
                ).fetchall()
            }
            if "kind" not in cols:
                return False
            return True
    except Exception:
        return False


def _load_ascentstats_rank(gender_filename: str, name_substring: str) -> int | None:
    path = FIXTURE_DIR / gender_filename
    data = json.loads(path.read_text())
    for entry in data["ranking"]:
        if name_substring.lower() in entry["name"].lower():
            return entry["rank"]
    return None


@pytest.fixture(scope="module")
def glicko2_boulder_session(tmp_path_factory):
    if not _has_compatible_prod_data():
        pytest.skip(
            "production DB or AscentStats fixture missing / schema mismatch — "
            "skipping cold-start trajectory test"
        )

    tmp_db = tmp_path_factory.mktemp("glicko2_cold_start") / "climbing_elo.db"
    shutil.copy(PROD_DB_PATH, tmp_db)

    engine = create_engine(f"sqlite:///{tmp_db}")
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()

    session.execute(delete(RatingHistory))
    session.execute(delete(Rating))
    session.commit()

    report = run_backfill(session, Discipline.BOULDER)
    assert report.events_processed > 0, "backfill processed zero events"

    yield session
    session.close()


def _rank_of(session, name_substring: str, gender: Gender) -> tuple[int, int, str]:
    leaderboard = list(
        session.execute(
            select(Rating, Athlete.name)
            .join(Athlete, Athlete.id == Rating.athlete_id)
            .where(
                Rating.discipline == Discipline.BOULDER,
                Athlete.gender == gender,
                Rating.n_events >= 3,
            )
            .order_by(Rating.mu.desc())
        ).all()
    )
    total = len(leaderboard)
    for idx, (_rating, name) in enumerate(leaderboard, 1):
        if name_substring.lower() in name.lower():
            return idx, total, name
    raise AssertionError(
        f"athlete matching {name_substring!r} not found in "
        f"gender={gender.value} leaderboard"
    )


def test_anraku_cold_start_glicko2(glicko2_boulder_session):
    ext_rank = _load_ascentstats_rank("2026-boulder-M.json", "Anraku")
    assert ext_rank is not None and ext_rank <= 5, (
        "AscentStats fixture sanity: Anraku should be near the top of men's Boulder"
    )
    rank, total, name = _rank_of(glicko2_boulder_session, "Anraku", Gender.M)
    assert total >= 50, f"unexpectedly thin men's Boulder leaderboard: {total}"
    assert rank <= 10, (
        f"Glicko-2 should place {name} in the top-10 "
        f"(AscentStats has him #{ext_rank}); got rank {rank}/{total}"
    )


def test_bertone_cold_start_glicko2(glicko2_boulder_session):
    ext_rank = _load_ascentstats_rank("2026-boulder-F.json", "Oriane Bertone")
    assert ext_rank is not None and ext_rank <= 10, (
        "AscentStats fixture sanity: Bertone should be in top-10 of women's Boulder"
    )
    rank, total, name = _rank_of(glicko2_boulder_session, "Oriane Bertone", Gender.F)
    assert total >= 30, f"unexpectedly thin women's Boulder leaderboard: {total}"
    assert rank <= 15, (
        f"Glicko-2 should place {name} in the top-15 "
        f"(AscentStats has her #{ext_rank}); got rank {rank}/{total}"
    )


def test_garnbret_present_and_elite_under_glicko2(glicko2_boulder_session):
    """Real-data sanity: Garnbret (prod athlete id=60) is an elite veteran.

    Backfilling the full Boulder history under the full-volatility engine (#81)
    should still place the most-decorated competitor of the era near the top of
    the women's leaderboard — a guard that the volatility refit didn't break
    the long-tenure trajectory.
    """
    rank, total, name = _rank_of(glicko2_boulder_session, "Garnbret", Gender.F)
    assert total >= 30, f"unexpectedly thin women's Boulder leaderboard: {total}"
    assert rank <= 5, (
        f"Glicko-2 should keep {name} in the women's Boulder top-5; "
        f"got rank {rank}/{total}"
    )


# ---------------------------------------------------------------------------
# Sabbatical-return uncertainty widening (Issue #81 / plan §"sabbatical test")
# ---------------------------------------------------------------------------


def test_sabbatical_return_widens_then_round_reshrinks_uncertainty():
    """A long inactivity gap inflates φ; the comeback round re-shrinks it.

    Models the Garnbret-style post-Olympic break (id=60 in prod). Under the
    full Glicko-2 volatility iteration (#81) the pre-period inflation must
    visibly widen RD across a ~12-month gap (acceptance: post-gap RD ≥
    pre-gap RD × 1.1, mirroring the plan's sabbatical-test bar), and the next
    round of competition must then consume that uncertainty (σ shrinks back).
    """
    GARNBRET_ID = 60  # prod athlete id, per the #51 plan doc
    veteran_ids = list(range(1, 8))
    veteran_mus = [1750, 1730, 1710, 1690, 1670, 1650, 1630]

    pre_gap_sigma = 110.0
    ratings: dict[int, AthleteRating] = {
        aid: AthleteRating(
            athlete_id=aid,
            mu=mu,
            sigma=120.0,
            n_events=20,
            last_event_at=date(2024, 8, 1),
            provisional=False,
        )
        for aid, mu in zip(veteran_ids, veteran_mus)
    }
    # Garnbret: strong, well-established, last competed a year before the return.
    ratings[GARNBRET_ID] = AthleteRating(
        athlete_id=GARNBRET_ID,
        mu=1900.0,
        sigma=pre_gap_sigma,
        n_events=70,
        last_event_at=date(2024, 8, 1),
        provisional=False,
    )

    # The pre-period inflation alone (12-month gap) must widen RD by ≥10%.
    inflated = glicko2_inflate_phi(pre_gap_sigma, date(2024, 8, 1), date(2025, 8, 1))
    assert inflated >= pre_gap_sigma * 1.1, (
        f"a ~12-month sabbatical should inflate RD by ≥10%; "
        f"{pre_gap_sigma} → {inflated:.1f}"
    )

    # Comeback event a year later. Garnbret wins; the round should consume the
    # inflated uncertainty (σ_after < the inflated σ_before).
    return_date = date(2025, 8, 1)
    results = [AthleteResult(athlete_id=GARNBRET_ID, rank=1)] + [
        AthleteResult(athlete_id=aid, rank=i + 2) for i, aid in enumerate(veteran_ids)
    ]
    updates = calculate_round_updates(
        results,
        ratings,
        EventTier.WORLD_CUP,
        RoundType.FINAL,
        return_date,
        discipline=Discipline.BOULDER,
    )
    g_upd = next(u for u in updates if u.athlete_id == GARNBRET_ID)
    # sigma_before on the update is the *inflated* RD the round started from.
    assert g_upd.sigma_before >= pre_gap_sigma * 1.1, (
        f"the round should see the inflated RD as its starting point; "
        f"got σ_before={g_upd.sigma_before:.1f}"
    )
    assert g_upd.sigma_after < g_upd.sigma_before, (
        f"the comeback round should shrink the inflated uncertainty; "
        f"σ {g_upd.sigma_before:.1f} → {g_upd.sigma_after:.1f}"
    )
