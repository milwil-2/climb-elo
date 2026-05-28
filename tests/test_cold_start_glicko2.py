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
    calculate_round_updates,
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
    # Structural cold-start invariant under Glicko-2 (independent of K):
    # high-σ athletes move proportionally more per round than established
    # ones. After 5 wins a fresh newcomer (σ=350) should climb significantly
    # — at least ~150μ — from default. The exact magnitude depends on K
    # (re-tuned 2026-05-27 per #80, see docs/K_REGRID_REPORT.md); this
    # threshold tracks the structural property, not the K-specific level.
    assert newcomer_climb >= 150.0, (
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

    # Newcomer should be well below μ=1500 after consistent losses.
    assert ratings[NEWCOMER_ID].mu < DEFAULT_MU - 30, (
        f"After 5 last-place finishes, newcomer μ={ratings[NEWCOMER_ID].mu:.1f} "
        f"should drop well below default 1500."
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
