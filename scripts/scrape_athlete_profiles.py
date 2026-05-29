#!/usr/bin/env python3
"""Populate ``Athlete.photo_url`` and body-metric columns from the IFSC API.

Issue #86 — manual one-time / occasional refresh job.

For every locally-stored athlete, this script:

1. Discovers the athlete's *IFSC* athlete_id by walking recent event result
   payloads (IFSC results carry ``firstname``/``lastname``/``athlete_id``).
2. Calls ``/api/v1/athletes/{ifsc_id}`` via
   :func:`climbing_elo.scraper.ifsc_api.scrape_athlete_profile` to fetch the
   photo URL and body metrics.
3. Updates the local ``Athlete`` row in place. Existing non-NULL values are
   preserved if the IFSC payload doesn't carry a value (``year_of_birth`` is
   only filled in when the row currently has ``NULL``).

Idempotent — running again only refreshes rows whose IFSC payload has changed.

Usage::

    uv run python scripts/scrape_athlete_profiles.py --only-missing   # skip athletes with photo_url set (daily cron)
    uv run python scripts/scrape_athlete_profiles.py --force          # re-scrape every athlete (one-time refresh)
    uv run python scripts/scrape_athlete_profiles.py --only-missing --limit 20  # first 20 missing only
    uv run python scripts/scrape_athlete_profiles.py --athlete-id 5   # one specific row (implicit --force)
    uv run python scripts/scrape_athlete_profiles.py --only-missing --since-year 2022  # only walk events from 2022+

Either ``--only-missing`` or ``--force`` is required (no implicit default — write
operations on prod should be explicit). The script is intentionally rate-limited
(``--delay``, default 0.5s) and reads sequentially. With ~2,700 athletes and
0.5s/request it takes ~25 min end-to-end on a full ``--force`` run; the daily
``--only-missing`` cron is normally just the day's new athletes.
"""

from __future__ import annotations

import argparse
import difflib
import logging
import sys
import time
from pathlib import Path

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

# Ensure src/ is importable when run directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from climbing_elo.database import init_db
from climbing_elo.models import Athlete, Event, Gender
from climbing_elo.scraper.ifsc_api import (
    BASE_URL,
    HEADERS,
    REQUEST_DELAY,
    _api_get,
    normalize_name,
    scrape_athlete_profile,
)

#: difflib similarity floor for the fuzzy name-resolution fallback. Names below
#: this ratio are not matched (too risky — would mis-assign a stranger's photo).
#: 0.90 keeps spelling/diacritic drift in scope while rejecting distinct people.
FUZZY_MATCH_THRESHOLD = 0.90

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)


class IfscIdIndex:
    """Name → IFSC-athlete-id lookup with tiered fallback (Issue #103).

    Holds three views of the harvested ``(name, gender) → ifsc_id`` mapping:

    - ``exact``      keyed by the raw ``"firstname lastname"`` string + gender.
    - ``normalized`` keyed by :func:`normalize_name` output + gender (folds
      accents, hyphens, casing and whitespace).
    - ``norm_any_gender`` keyed by normalized name only (gender-agnostic
      fallback for the rare rows whose gender disagrees between our DB and the
      IFSC payload).

    :meth:`resolve` walks the tiers in increasing looseness and finally tries a
    ``difflib`` fuzzy match within the same gender, so spelling drift that the
    deterministic folds miss still resolves.
    """

    def __init__(self) -> None:
        self.exact: dict[tuple[str, Gender], int] = {}
        self.normalized: dict[tuple[str, Gender], int] = {}
        self.norm_any_gender: dict[str, int] = {}
        # Per-gender list of (normalized_name, ifsc_id) for fuzzy matching.
        self._by_gender: dict[Gender, list[tuple[str, int]]] = {
            Gender.M: [],
            Gender.F: [],
        }

    def add(self, firstname: str, lastname: str, gender: Gender, ifsc_id: int) -> None:
        raw = f"{firstname} {lastname}".strip()
        norm = normalize_name(raw)
        # setdefault: events are walked most-recent-first, so the first id wins.
        self.exact.setdefault((raw, gender), ifsc_id)
        if norm:
            if (norm, gender) not in self.normalized:
                self.normalized[(norm, gender)] = ifsc_id
                self._by_gender[gender].append((norm, ifsc_id))
            self.norm_any_gender.setdefault(norm, ifsc_id)

    def __len__(self) -> int:
        return len(self.exact)

    def resolve(self, name: str, gender: Gender) -> tuple[int | None, str]:
        """Resolve a stored athlete name to an IFSC id.

        Returns ``(ifsc_id_or_None, tier_label)`` where ``tier_label`` is one of
        ``"exact"`` / ``"normalized"`` / ``"cross_gender"`` / ``"fuzzy"`` /
        ``"unresolved"`` for dry-run reporting.
        """
        if (name, gender) in self.exact:
            return self.exact[(name, gender)], "exact"

        norm = normalize_name(name)
        if not norm:
            return None, "unresolved"
        if (norm, gender) in self.normalized:
            return self.normalized[(norm, gender)], "normalized"
        if norm in self.norm_any_gender:
            return self.norm_any_gender[norm], "cross_gender"

        candidates = self._by_gender.get(gender, [])
        if candidates:
            names = [c[0] for c in candidates]
            best = difflib.get_close_matches(
                norm, names, n=1, cutoff=FUZZY_MATCH_THRESHOLD
            )
            if best:
                idx = names.index(best[0])
                return candidates[idx][1], "fuzzy"

        return None, "unresolved"


def _build_ifsc_id_index(
    session: Session,
    client: httpx.Client,
    since_year: int | None,
    delay: float,
) -> IfscIdIndex:
    """Build an :class:`IfscIdIndex` by re-walking event payloads.

    Local Athlete rows don't store the IFSC id, so we have to recover it from
    the upstream API. We walk every event in the local DB (optionally filtered
    by season), fetch the result payload for each round, and harvest the
    ``athlete_id`` field next to the ``firstname``/``lastname`` pair.

    Args
    ----
    since_year:
        When set, only events with ``season >= since_year`` are queried. Pass
        ``None`` (the script default) to walk every season — required to
        recover IDs for athletes whose last competition predates the old 2018
        floor (Issue #103: ~744 missing-photo athletes last competed <2018).
    delay:
        Seconds to sleep between API requests (rate limiting).
    """
    stmt = select(Event)
    if since_year is not None:
        stmt = stmt.where(Event.season >= since_year)
    events = list(session.execute(stmt.order_by(Event.start_date.desc())).scalars())

    log.info("Walking %d events to discover IFSC athlete IDs...", len(events))

    # Build a quick lookup: discipline → IFSC d_cat_id mapping isn't stored
    # locally, so we use the events index endpoint per season instead.
    # Simpler: derive (firstname, lastname) tuples from each event's result.

    index = IfscIdIndex()
    seen_event_count = 0

    seasons_seen: dict[int, list[dict]] = {}  # season -> league d_cats

    for ev in events:
        # We need the IFSC event_id + dcat_id. Local DB doesn't have them, so
        # walk the season league to find them.
        season = ev.season
        if season not in seasons_seen:
            seasons = _api_get(client, "/api/v1/")
            if not isinstance(seasons, dict):
                seasons_seen[season] = []
                continue
            wc_url = None
            for s in seasons.get("seasons", []):
                if s.get("name") == str(season):
                    for lg in s.get("leagues", []):
                        if lg.get("league_id") == 1:
                            wc_url = lg.get("url")
                            break
                    break
            if not wc_url:
                seasons_seen[season] = []
                continue
            time.sleep(delay)
            league = _api_get(client, wc_url)
            seasons_seen[season] = (
                league.get("events", []) if isinstance(league, dict) else []
            )

        ifsc_events = seasons_seen[season]
        matched = next((e for e in ifsc_events if e.get("event") == ev.name), None)
        if not matched:
            continue

        for dc in matched.get("d_cats", []):
            if dc.get("status") != "finished":
                continue
            event_id = matched.get("event_id")
            dcat_id = dc.get("id")
            if not event_id or not dcat_id:
                continue

            time.sleep(delay)
            payload = _api_get(client, f"/api/v1/events/{event_id}/result/{dcat_id}")
            if not isinstance(payload, dict):
                continue
            for entry in payload.get("ranking", []):
                firstname = (entry.get("firstname") or "").strip()
                lastname = (entry.get("lastname") or "").strip()
                ifsc_id = entry.get("athlete_id")
                # IFSC d_cat names "BOULDER Women" / "BOULDER Men" — "Women"
                # contains "men", so check Women first.
                gender_str = "M" if "men" in (dc.get("name") or "").lower() else "F"
                if "women" in (dc.get("name") or "").lower():
                    gender_str = "F"
                if not (firstname and lastname and ifsc_id):
                    continue
                index.add(firstname, lastname, Gender(gender_str), int(ifsc_id))

        seen_event_count += 1
        if seen_event_count % 25 == 0:
            log.info(
                "Walked %d/%d events, %d IFSC IDs discovered so far...",
                seen_event_count,
                len(events),
                len(index),
            )

    log.info("Discovered IFSC IDs for %d (name, gender) pairs.", len(index))
    return index


def _update_athlete(athlete: Athlete, payload: dict) -> bool:
    """Apply ``payload`` to ``athlete`` in place. Return True iff anything changed."""
    changed = False
    if "photo_url" in payload and payload["photo_url"] != athlete.photo_url:
        athlete.photo_url = payload["photo_url"]
        changed = True
    if "height_cm" in payload and payload["height_cm"] != athlete.height_cm:
        athlete.height_cm = payload["height_cm"]
        changed = True
    if "weight_kg" in payload and payload["weight_kg"] != athlete.weight_kg:
        athlete.weight_kg = payload["weight_kg"]
        changed = True
    if "wingspan_cm" in payload and payload["wingspan_cm"] != athlete.wingspan_cm:
        athlete.wingspan_cm = payload["wingspan_cm"]
        changed = True
    # Only fill year_of_birth when it's currently NULL — the scraper already
    # populates it on first ingest, no point clobbering.
    if athlete.year_of_birth is None and "year_of_birth" in payload:
        athlete.year_of_birth = payload["year_of_birth"]
        changed = True
    return changed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Populate Athlete.photo_url + body metrics from the IFSC API (Issue #86)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Stop after refreshing this many athletes (default: all).",
    )
    parser.add_argument(
        "--athlete-id",
        type=int,
        default=None,
        help="Refresh only this local Athlete.id (skips the IFSC-id discovery walk).",
    )
    parser.add_argument(
        "--since-year",
        type=int,
        default=2018,
        help="Only walk events from this season onward during IFSC-id discovery. "
        "Default: 2018 (cron-safe: keeps the daily scrape-supabase.yml walk "
        "well inside the 60-min Actions budget). Issue #103: the 2018 floor "
        "strands ~744 missing-photo athletes whose last competition predated "
        "2018. To harvest them, run the ONE-TIME full backfill LOCALLY with an "
        "early floor, e.g. --since-year 2000 (slow; not for CI).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve IFSC ids and report projected coverage WITHOUT calling the "
        "profile endpoint or writing to the DB. Use to validate matching gains.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=REQUEST_DELAY,
        help=f"Seconds to sleep between API requests (default: {REQUEST_DELAY}).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--only-missing",
        action="store_true",
        help="Only scrape athletes whose photo_url is NULL. Cheap steady-state mode "
        "for the daily workflow — once a row has a photo it's left alone.",
    )
    mode.add_argument(
        "--force",
        action="store_true",
        help="Re-scrape every athlete even if photo_url is already set. Use when "
        "IFSC has updated photos and you want to refresh them all.",
    )
    args = parser.parse_args()

    # Explicit > implicit for write operations on prod. --athlete-id is its own
    # explicit choice (the user named a single row), so it's exempt.
    if not args.only_missing and not args.force and args.athlete_id is None:
        parser.error(
            "must pass one of --only-missing (skip rows with photo_url set) or "
            "--force (re-scrape everything). Refusing to run without an explicit choice."
        )

    SessionFactory = init_db()

    with httpx.Client(base_url=BASE_URL, headers=HEADERS, timeout=20) as client:
        with SessionFactory() as session:
            if args.athlete_id is not None:
                target = session.get(Athlete, args.athlete_id)
                if target is None:
                    log.error("Athlete id %s not found in local DB", args.athlete_id)
                    sys.exit(1)
                athletes = [target]
            else:
                stmt = select(Athlete).order_by(Athlete.id.asc())
                if args.only_missing:
                    stmt = stmt.where(Athlete.photo_url.is_(None))
                athletes = list(session.execute(stmt).scalars())
                log.info(
                    "Mode=%s — %d athletes queued for scrape.",
                    "only-missing" if args.only_missing else "force",
                    len(athletes),
                )

            id_index = _build_ifsc_id_index(
                session, client, args.since_year, args.delay
            )

            updated = 0
            missing_id = 0
            no_change = 0
            errors = 0
            # Resolution-tier tally for reporting (Issue #103).
            tier_counts: dict[str, int] = {}

            for ath in athletes:
                if args.limit is not None and updated >= args.limit:
                    break
                ifsc_id, tier = id_index.resolve(ath.name, ath.gender)
                tier_counts[tier] = tier_counts.get(tier, 0) + 1
                if ifsc_id is None:
                    missing_id += 1
                    continue

                if args.dry_run:
                    # Count a resolved id as a projected refresh; never touch the
                    # network or DB.
                    updated += 1
                    continue

                time.sleep(args.delay)
                try:
                    payload = scrape_athlete_profile(ifsc_id, client=client)
                except Exception as exc:  # pragma: no cover — defensive
                    log.warning(
                        "scrape_athlete_profile failed for %s (ifsc=%s): %s",
                        ath.name,
                        ifsc_id,
                        exc,
                    )
                    errors += 1
                    continue

                if not payload:
                    no_change += 1
                    continue
                if _update_athlete(ath, payload):
                    updated += 1
                    if updated % 25 == 0:
                        session.commit()
                        log.info(
                            "Committed %d updates so far (missing_id=%d, no_change=%d, errors=%d)...",
                            updated,
                            missing_id,
                            no_change,
                            errors,
                        )
                else:
                    no_change += 1

            if not args.dry_run:
                session.commit()

    print()
    if args.dry_run:
        resolved = sum(v for k, v in tier_counts.items() if k != "unresolved")
        queued = resolved + missing_id
        print("DRY RUN — no network profile calls, no DB writes.")
        print(f"  Athletes queued:       {queued}")
        print(f"  Resolved to IFSC id:   {resolved}")
        print(f"  Unresolved:            {missing_id}")
        print("  Resolution by tier:")
        for tier in ("exact", "normalized", "cross_gender", "fuzzy", "unresolved"):
            if tier in tier_counts:
                print(f"    {tier:<14} {tier_counts[tier]}")
        return

    print("Profile scrape complete:")
    print(f"  Athletes refreshed:    {updated}")
    print(f"  No-op (no changes):    {no_change}")
    print(f"  Missing IFSC ID:       {missing_id}")
    print(f"  Errors:                {errors}")


if __name__ == "__main__":
    main()
