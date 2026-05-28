"""Tests for HTML/static utility routes (Issue #73) and the rich profile (#86).

Covers:
- GET /favicon.ico returns 200 with an image content-type (no longer 500/404).
- base.html includes a <link rel="icon"> tag so browser tabs render the icon.
- GET /athletes/{id} renders for an athlete with full metadata.
- GET /athletes/{id} renders for an athlete with no photo / metrics / combined.
- GET /athletes/{id} returns 404 for a non-existent athlete.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

import climbing_elo.api.routes as _routes
import climbing_elo.database as _db
from climbing_elo.api.app import create_app
from climbing_elo.models import (
    Athlete,
    Base,
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


# ---------------------------------------------------------------------------
# /favicon.ico
# ---------------------------------------------------------------------------


def test_favicon_route_returns_200():
    """GET /favicon.ico must return 200 — no more 500s on browser tab opens."""
    app = create_app()
    with TestClient(app) as tc:
        r = tc.get("/favicon.ico")
    assert r.status_code == 200, f"expected 200, got {r.status_code}: {r.text[:200]}"


def test_favicon_route_has_image_content_type():
    """The favicon response must declare an image content-type."""
    app = create_app()
    with TestClient(app) as tc:
        r = tc.get("/favicon.ico")
    ctype = r.headers.get("content-type", "")
    accepted = (
        "image/x-icon",
        "image/png",
        "image/svg+xml",
        "image/vnd.microsoft.icon",
    )
    assert any(ctype.startswith(t) for t in accepted), (
        f"expected one of {accepted}, got {ctype!r}"
    )


def test_favicon_response_has_body():
    """The favicon response must include a non-empty body."""
    app = create_app()
    with TestClient(app) as tc:
        r = tc.get("/favicon.ico")
    assert len(r.content) > 0


# ---------------------------------------------------------------------------
# base.html <link rel="icon">
# ---------------------------------------------------------------------------


def test_base_html_includes_favicon_link_tag():
    """base.html must declare a <link rel="icon"> tag in its <head> block."""
    base_html_path = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "climbing_elo"
        / "templates"
        / "base.html"
    )
    text = base_html_path.read_text(encoding="utf-8")
    assert 'rel="icon"' in text, 'base.html missing <link rel="icon"> tag'
    # The link must reference the static favicon (not an external URL).
    assert "favicon" in text.lower()


# ---------------------------------------------------------------------------
# /athletes/{id}  — rich profile page (Issue #86)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def profile_db_path(tmp_path_factory):
    return tmp_path_factory.mktemp("profile_db") / "test.db"


@pytest.fixture(scope="module")
def profile_factory(profile_db_path):
    """Seed a small DB with three athletes covering all profile scenarios.

    - Full athlete:   photo + height + wingspan + ratings in B/L/BL + history.
    - Minimal athlete: no photo, no metrics, only one discipline rating.
    - Solo athlete:    has Lead rating but no BL (combined section absent).
    """
    engine = create_engine(f"sqlite:///{profile_db_path}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    session = factory()

    # ── Athletes
    full = Athlete(
        name="Sora Climber",
        gender=Gender.M,
        nationality="JPN",
        year_of_birth=2006,
        photo_url="https://example.com/sora.jpg",
        height_cm=168,
        wingspan_cm=181,
    )
    minimal = Athlete(name="Minimal Climber", gender=Gender.F, nationality="USA")
    solo = Athlete(
        name="Solo Lead", gender=Gender.M, nationality="ITA", year_of_birth=1993
    )
    session.add_all([full, minimal, solo])
    session.flush()

    # ── Events (Olympics, World Champs, World Cup — to exercise marker logic)
    ev_olympics = Event(
        name="Paris Olympics",
        tier=EventTier.OLYMPICS,
        country="FRA",
        season=2024,
        start_date=date(2024, 8, 1),
        discipline=Discipline.BOULDER_LEAD,
    )
    ev_wch = Event(
        name="Bern World Championships",
        tier=EventTier.WORLD_CHAMPIONSHIP,
        country="SUI",
        season=2023,
        start_date=date(2023, 8, 1),
        discipline=Discipline.LEAD,
    )
    ev_wc_b = Event(
        name="Innsbruck World Cup",
        tier=EventTier.WORLD_CUP,
        country="AUT",
        season=2024,
        start_date=date(2024, 6, 1),
        discipline=Discipline.BOULDER,
    )
    ev_wc_l = Event(
        name="Briancon World Cup",
        tier=EventTier.WORLD_CUP,
        country="FRA",
        season=2024,
        start_date=date(2024, 7, 1),
        discipline=Discipline.LEAD,
    )
    session.add_all([ev_olympics, ev_wch, ev_wc_b, ev_wc_l])
    session.flush()

    # ── Rounds
    rounds = []
    for ev in (ev_olympics, ev_wch, ev_wc_b, ev_wc_l):
        rnd = Round(
            event_id=ev.id,
            round_type=RoundType.FINAL,
            gender=Gender.M,
            athlete_count=2,
        )
        session.add(rnd)
        rounds.append(rnd)
    session.flush()

    # ── Results: full + solo competed in everything. minimal has none.
    for rnd in rounds:
        session.add(
            Result(round_id=rnd.id, athlete_id=full.id, rank=1, raw_score="TOP")
        )
        session.add(
            Result(round_id=rnd.id, athlete_id=solo.id, rank=2, raw_score="34+")
        )
    session.flush()

    # ── Ratings: full has B/L/BL; solo has L only; minimal has L only.
    session.add(
        Rating(
            athlete_id=full.id,
            discipline=Discipline.LEAD,
            mu=2050.0,
            sigma=110.0,
            n_events=12,
            provisional=False,
            last_event_at=date(2024, 8, 1),
        )
    )
    session.add(
        Rating(
            athlete_id=full.id,
            discipline=Discipline.BOULDER,
            mu=2000.0,
            sigma=120.0,
            n_events=10,
            provisional=False,
            last_event_at=date(2024, 6, 1),
        )
    )
    session.add(
        Rating(
            athlete_id=full.id,
            discipline=Discipline.BOULDER_LEAD,
            mu=2024.8,  # ≈ sqrt(2050 * 2000)
            sigma=115.0,
            n_events=10,
            provisional=False,
            last_event_at=date(2024, 8, 1),
        )
    )
    session.add(
        Rating(
            athlete_id=solo.id,
            discipline=Discipline.LEAD,
            mu=1800.0,
            sigma=125.0,
            n_events=8,
            provisional=False,
            last_event_at=date(2024, 7, 1),
        )
    )
    session.add(
        Rating(
            athlete_id=minimal.id,
            discipline=Discipline.LEAD,
            mu=1600.0,
            sigma=200.0,
            n_events=3,
            provisional=False,
        )
    )
    session.flush()

    # ── Rating history (1 round per event for full + solo)
    for ev, rnd in zip((ev_olympics, ev_wch, ev_wc_b, ev_wc_l), rounds):
        session.add(
            RatingHistory(
                athlete_id=full.id,
                event_id=ev.id,
                round_id=rnd.id,
                mu_before=2000.0,
                mu_after=2050.0,
                sigma_before=120.0,
                sigma_after=110.0,
                contributing_pairs=[
                    {
                        "opponent_id": solo.id,
                        "result": 1.0,
                        "expected": 0.7,
                        "actual": 1.0,
                        "delta": 15.0,
                        "margin_multiplier": 1.0,
                    }
                ],
            )
        )
        session.add(
            RatingHistory(
                athlete_id=solo.id,
                event_id=ev.id,
                round_id=rnd.id,
                mu_before=1820.0,
                mu_after=1800.0,
                sigma_before=130.0,
                sigma_after=125.0,
                contributing_pairs=[
                    {
                        "opponent_id": full.id,
                        "result": 0.0,
                        "expected": 0.3,
                        "actual": 0.0,
                        "delta": -15.0,
                        "margin_multiplier": 1.0,
                    }
                ],
            )
        )
    # Issue #90 / #36 regression: a TPB row for `full` on the lead WC event.
    # kind='tpb' stores contributing_pairs as a DICT (not a list of pair-dicts),
    # and it's added last so it has the highest id for its event. The athlete
    # profile's "recent ELO changes" opponents logic must NOT pick this row
    # (iterating a dict yields string keys → p.get() crash). See
    # routes.py:v2_athlete_profile.
    session.add(
        RatingHistory(
            athlete_id=full.id,
            event_id=ev_wc_l.id,
            round_id=rounds[3].id,
            mu_before=2050.0,
            mu_after=2062.0,
            sigma_before=110.0,
            sigma_after=110.0,
            kind="tpb",
            contributing_pairs={
                "rank": 1,
                "gross_bonus": 12.0,
                "debit": 0.0,
                "tier": "world_cup",
            },
        )
    )
    session.commit()
    session.close()
    return factory


@pytest.fixture(scope="module")
def profile_client(profile_db_path, profile_factory):
    """TestClient bound to the seeded profile DB."""
    original_session = _routes._session
    original_get_engine = _db.get_engine

    def patched_session():
        return profile_factory()

    def patched_get_engine(db_path=None):
        return create_engine(f"sqlite:///{profile_db_path}")

    _routes._session = patched_session  # type: ignore[assignment]
    _db.get_engine = patched_get_engine  # type: ignore[assignment]

    app = create_app()
    tc = TestClient(app)

    yield tc

    _db.get_engine = original_get_engine
    _routes._session = original_session


def _athlete_id_by_name(factory, name: str) -> int:
    """Look up the seeded athlete's local id."""
    from sqlalchemy import select

    with factory() as session:
        ath = session.execute(select(Athlete).where(Athlete.name == name)).scalar_one()
        return ath.id


def test_profile_route_full_athlete_renders_200(profile_client, profile_factory):
    """A fully-populated athlete renders 200 and includes all key sections."""
    aid = _athlete_id_by_name(profile_factory, "Sora Climber")
    r = profile_client.get(f"/athletes/{aid}")
    assert r.status_code == 200, r.text[:500]
    html = r.text

    # Header content
    assert "Sora Climber" in html
    assert "JPN" in html
    assert "Born 2006" in html
    # Photo URL is rendered into an <img>
    assert "https://example.com/sora.jpg" in html
    # Body metrics card (header div)
    assert ">Body metrics</div>" in html
    assert "168" in html  # height
    assert "181" in html  # wingspan
    # Current ratings card — three rows
    assert ">Current ratings</div>" in html
    assert "2050.0" in html  # mu lead
    assert "2000.0" in html  # mu boulder
    # Combined breakdown section present
    assert ">Combined (Boulder + Lead) breakdown</div>" in html
    # Recent ELO changes
    assert ">Recent ELO changes</div>" in html
    # Full event history
    assert ">Event history</div>" in html
    assert "Paris Olympics" in html
    assert "Bern World Championships" in html


def test_profile_route_with_tpb_row_renders_200(profile_client, profile_factory):
    """Regression (#36): an athlete whose latest rating_history row for an
    event is a TPB row (kind='tpb', dict contributing_pairs) must still render
    200. Before the fix the opponents logic in v2_athlete_profile picked the
    highest-id row regardless of kind and iterated the dict's string keys,
    raising 'str' object has no attribute 'get'. The profile fixture seeds
    such a TPB row for 'Sora Climber' on the Briancon World Cup event."""
    aid = _athlete_id_by_name(profile_factory, "Sora Climber")
    r = profile_client.get(f"/athletes/{aid}")
    assert r.status_code == 200, r.text[:500]
    # The opponents list must still come from the pair row (Solo Lead), never
    # from the TPB row's dict payload.
    assert "Solo Lead" in r.text


def test_profile_route_minimal_athlete_no_photo_no_metrics(
    profile_client, profile_factory
):
    """An athlete with no photo / metrics still renders cleanly."""
    aid = _athlete_id_by_name(profile_factory, "Minimal Climber")
    r = profile_client.get(f"/athletes/{aid}")
    assert r.status_code == 200, r.text[:500]
    html = r.text

    # Fallback "No photo" text instead of a broken <img>
    assert "No photo" in html
    # No "Body metrics" card when all metric columns are NULL — we look for the
    # rendered section-header div, not the substring (which appears in CSS).
    assert ">Body metrics</div>" not in html
    # No broken <img src=""> — we only emit <img> when photo_url is non-NULL
    assert 'src=""' not in html


def test_profile_route_no_combined_rating_no_combined_section(
    profile_client, profile_factory
):
    """Athletes without BOULDER_LEAD rating must not show the combined section."""
    aid = _athlete_id_by_name(profile_factory, "Solo Lead")
    r = profile_client.get(f"/athletes/{aid}")
    assert r.status_code == 200, r.text[:500]
    assert "Combined (Boulder + Lead) breakdown" not in r.text


def test_profile_route_includes_chart_data(profile_client, profile_factory):
    """Chart.js script + rating-history payload must be inline on the page."""
    aid = _athlete_id_by_name(profile_factory, "Sora Climber")
    r = profile_client.get(f"/athletes/{aid}")
    assert r.status_code == 200
    html = r.text

    # Chart.js is loaded from base.html
    assert "chart.umd.min.js" in html
    # Our chart instantiation is present
    assert "athlete-chart" in html
    # Rating-history payload is exposed as a data attribute for JS to parse
    assert "data-rating-history" in html
    # Markers (Olympics + WCh) make it into the payload
    assert "olympics" in html.lower()


def test_profile_route_404_for_unknown_athlete(profile_client):
    """A nonexistent athlete returns 404."""
    r = profile_client.get("/athletes/999999")
    assert r.status_code == 404


# ---------------------------------------------------------------------------
# /leaderboard — dual view (Issue #91)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def leaderboard_db_path(tmp_path_factory):
    return tmp_path_factory.mktemp("leaderboard_db") / "test.db"


@pytest.fixture(scope="module")
def leaderboard_factory(leaderboard_db_path):
    """Seed a DB with four athletes that exercise every #91 view branch.

    - ``Active Ace``: competed last month → present in active + all + legacy.
    - ``On Break``:   last competed 18 months ago → absent from active,
      present in all + legacy.
    - ``Long Gone``:  last competed 5 years ago (no manual flag) → absent
      from active + all, present in legacy.
    - ``Flagged``:    competed recently but ``retired_at`` set → absent from
      all, present in legacy and in active (active filter does not consult
      ``retired_at``).
    """
    from datetime import timedelta

    engine = create_engine(f"sqlite:///{leaderboard_db_path}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    session = factory()

    today = date.today()

    athletes = [
        ("Active Ace", None, today - timedelta(days=30), 2100.0),
        ("On Break", None, today - timedelta(days=540), 2050.0),  # ~18 mo
        ("Long Gone", None, today - timedelta(days=int(5 * 365.25)), 2000.0),
        ("Flagged", today - timedelta(days=10), today - timedelta(days=15), 1950.0),
    ]
    name_to_id: dict[str, int] = {}
    for name, retired, _, _ in athletes:
        a = Athlete(
            name=name,
            gender=Gender.M,
            nationality="USA",
            retired_at=retired,
        )
        session.add(a)
        session.flush()
        name_to_id[name] = a.id

    for name, _, last_event, mu in athletes:
        session.add(
            Rating(
                athlete_id=name_to_id[name],
                discipline=Discipline.LEAD,
                mu=mu,
                sigma=110.0,
                n_events=10,
                provisional=False,
                last_event_at=last_event,
            )
        )
    session.commit()
    session.close()
    return factory


@pytest.fixture(scope="module")
def leaderboard_client(leaderboard_db_path, leaderboard_factory):
    original_session = _routes._session
    original_get_engine = _db.get_engine

    def patched_session():
        return leaderboard_factory()

    def patched_get_engine(db_path=None):
        return create_engine(f"sqlite:///{leaderboard_db_path}")

    _routes._session = patched_session  # type: ignore[assignment]
    _db.get_engine = patched_get_engine  # type: ignore[assignment]

    app = create_app()
    tc = TestClient(app)
    yield tc

    _db.get_engine = original_get_engine
    _routes._session = original_session


def test_leaderboard_default_view_is_active(leaderboard_client):
    """No ``view`` query param → only active-window athletes are shown."""
    r = leaderboard_client.get("/leaderboard?disc=L&gender=M")
    assert r.status_code == 200, r.text[:500]
    html = r.text
    # Active Ace must be present; On Break and Long Gone must be hidden.
    assert "Active Ace" in html
    assert "On Break" not in html
    assert "Long Gone" not in html


def test_leaderboard_view_all_shows_more_than_active(leaderboard_client):
    """``view=all`` includes on-break athletes but still hides retirees."""
    r_active = leaderboard_client.get("/leaderboard?disc=L&gender=M&view=active")
    r_all = leaderboard_client.get("/leaderboard?disc=L&gender=M&view=all")
    assert r_active.status_code == 200
    assert r_all.status_code == 200

    # On Break appears only in the all-time view.
    assert "On Break" not in r_active.text
    assert "On Break" in r_all.text
    # Active Ace is present in both.
    assert "Active Ace" in r_active.text
    assert "Active Ace" in r_all.text
    # Long Gone (>5y gap) is hidden in both views.
    assert "Long Gone" not in r_active.text
    assert "Long Gone" not in r_all.text
    # Flagged athlete (manual retired_at) is absent from all-time.
    assert "Flagged" not in r_all.text


def test_leaderboard_view_legacy_shows_everyone(leaderboard_client):
    """``view=legacy`` is the pre-#91 unfiltered behaviour."""
    r = leaderboard_client.get("/leaderboard?disc=L&gender=M&view=legacy")
    assert r.status_code == 200
    html = r.text
    for name in ("Active Ace", "On Break", "Long Gone", "Flagged"):
        assert name in html, f"legacy view missing {name!r}"


def test_leaderboard_view_invalid_falls_back_to_active(leaderboard_client):
    """HTML route is forgiving: a bogus ``view`` value just defaults to active."""
    r = leaderboard_client.get("/leaderboard?disc=L&gender=M&view=junk")
    assert r.status_code == 200
    # Same content as the default active view.
    assert "Active Ace" in r.text
    assert "On Break" not in r.text


def test_leaderboard_all_view_renders_activity_badges(leaderboard_client):
    """In all-time view, each row is tagged "Active" or "Inactive Xmo"."""
    r = leaderboard_client.get("/leaderboard?disc=L&gender=M&view=all")
    assert r.status_code == 200
    html = r.text
    assert "activity-badge" in html
    # Active Ace got the "Active" tag
    assert ">Active<" in html
    # On Break got an Inactive label like "Inactive 17mo" or similar
    assert "Inactive" in html


def test_leaderboard_view_toggle_renders(leaderboard_client):
    """The Active/All-time toggle links are present in the HTML."""
    r = leaderboard_client.get("/leaderboard?disc=L&gender=M")
    assert r.status_code == 200
    html = r.text
    assert "view=active" in html
    assert "view=all" in html
    # The default view's toggle has ``active`` class on the Active button.
    assert 'view=active"\n         class="active"' in html or 'class="active"' in html
