"""Live event poller.

Periodically fetches `/api/v1/events/{event_id}/result/{dcat_id}` from the
IFSC API, detects new Result rows by comparing DB state against the API
response, inserts them, and publishes to the EventBus for SSE subscribers.

Mutex:
  A lock file at /tmp/climbing_elo_poller_<event_id>.lock prevents two
  processes (e.g. two uvicorn workers or a manual CLI run) from polling
  the same event simultaneously.

ELO updates:
  Deferred until the event finishes (status == "finished").  Mid-event ELO
  updates are too noisy; the backfill script runs after completion.

Security:
  - IFSC API errors are not propagated to clients.
  - rank values are type-validated before DB insert (int guard).
  - All DB writes via SQLAlchemy ORM (no raw SQL).
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

import httpx
from sqlalchemy import select
from sqlalchemy.orm import Session

from climbing_elo.database import get_session_factory
from climbing_elo.live.bus import event_bus
from climbing_elo.models import Athlete, Event, Gender, Result, Round, RoundType

log = logging.getLogger(__name__)

IFSC_BASE = "https://ifsc.results.info"
IFSC_HEADERS = {
    "User-Agent": "ClimbingELO/0.1 (research project)",
    "Referer": "https://ifsc.results.info/",
    "Accept": "application/json",
}

LOCK_DIR = Path("/tmp")

# Registry of running poller tasks: event_id → asyncio.Task
_running_tasks: dict[int, asyncio.Task] = {}
# Shutdown signals: event_id → asyncio.Event
_stop_events: dict[int, asyncio.Event] = {}


def _lock_path(event_id: int) -> Path:
    return LOCK_DIR / f"climbing_elo_poller_{event_id}.lock"


def _acquire_file_lock(event_id: int) -> Optional[int]:
    """Try to acquire the file lock for event_id.

    Returns the file descriptor (held open) or None if already locked.
    Uses O_CREAT | O_EXCL for atomic creation — standard advisory lock.
    """
    path = _lock_path(event_id)
    try:
        fd = os.open(str(path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        os.write(fd, str(os.getpid()).encode())
        return fd
    except FileExistsError:
        return None


def _release_file_lock(event_id: int, fd: int) -> None:
    try:
        os.close(fd)
    except OSError:
        pass
    try:
        _lock_path(event_id).unlink(missing_ok=True)
    except OSError:
        pass


def _parse_round_type(name: str) -> RoundType:
    name = name.lower()
    if "final" in name:
        return RoundType.FINAL
    if "semi" in name:
        return RoundType.SEMI
    return RoundType.QUALIFICATION


async def _fetch_results(
    client: httpx.AsyncClient, event_id: int, dcat_id: int
) -> Optional[dict]:
    """Async fetch of IFSC result payload; returns None on any error (don't leak to clients)."""
    url = f"{IFSC_BASE}/api/v1/events/{event_id}/result/{dcat_id}"
    try:
        resp = await client.get(url, headers=IFSC_HEADERS, timeout=15)
        if resp.status_code == 200:
            return resp.json()
        log.warning(
            "IFSC API returned HTTP %d for event %d dcat %d",
            resp.status_code,
            event_id,
            dcat_id,
        )
        return None
    except Exception as exc:
        log.error("IFSC fetch failed for event %d dcat %d: %s", event_id, dcat_id, exc)
        return None


def _current_db_keys(session: Session, event_db_id: int) -> set[tuple]:
    """Return set of (athlete_id, round_type_value, rank, raw_score) for an event."""
    rows = session.execute(
        select(Result.athlete_id, Round.round_type, Result.rank, Result.raw_score)
        .join(Round, Result.round_id == Round.id)
        .where(Round.event_id == event_db_id)
    ).all()
    return {(r.athlete_id, r.round_type.value, r.rank, r.raw_score) for r in rows}


def _get_or_create_athlete(
    session: Session,
    firstname: str,
    lastname: str,
    country: str,
    gender: Gender,
) -> Athlete:
    """Find or create an Athlete row by name+gender."""
    name = f"{firstname} {lastname}".strip()
    existing = session.execute(
        select(Athlete).where(Athlete.name == name, Athlete.gender == gender)
    ).scalar_one_or_none()
    if existing:
        return existing
    athlete = Athlete(
        name=name,
        gender=gender,
        nationality=country[:3] if country else None,
    )
    session.add(athlete)
    session.flush()
    return athlete


def _get_or_create_round(
    session: Session,
    event_db_id: int,
    round_type: RoundType,
    gender: Gender,
) -> Round:
    existing = session.execute(
        select(Round).where(
            Round.event_id == event_db_id,
            Round.round_type == round_type,
            Round.gender == gender,
        )
    ).scalar_one_or_none()
    if existing:
        return existing
    rnd = Round(
        event_id=event_db_id,
        round_type=round_type,
        gender=gender,
        athlete_count=0,
    )
    session.add(rnd)
    session.flush()
    return rnd


class LivePoller:
    """Polls IFSC API for one (event_id, dcat_id) pair and ingests new Results.

    Args:
        event_id:         IFSC event ID (used in API URL).
        dcat_id:          Discipline-category ID (used in API URL).
        interval_seconds: How often to poll (default 15 s).
        event_db_id:      Our internal DB Event.id (resolved from event_id if None).
    """

    def __init__(
        self,
        event_id: int,
        dcat_id: int,
        interval_seconds: int = 15,
        event_db_id: Optional[int] = None,
    ) -> None:
        self.event_id = event_id
        self.dcat_id = dcat_id
        self.interval_seconds = interval_seconds
        self._event_db_id = event_db_id
        self._stop_event: Optional[asyncio.Event] = None

    def _resolve_event_db_id(self, session: Session) -> Optional[int]:
        """Resolve IFSC event_id to our internal DB Event.id."""
        if self._event_db_id is not None:
            result = session.execute(
                select(Event).where(Event.id == self._event_db_id)
            ).scalar_one_or_none()
            return result.id if result else None
        # Fall back: try treating event_id as our DB primary key
        result = session.execute(
            select(Event).where(Event.id == self.event_id)
        ).scalar_one_or_none()
        return result.id if result else None

    async def run(self, stop_event: asyncio.Event, session_factory=None) -> None:
        """Main polling loop.  Runs until stop_event is set or event finishes."""
        self._stop_event = stop_event
        if session_factory is None:
            session_factory = get_session_factory()

        async with httpx.AsyncClient() as client:
            while not stop_event.is_set():
                try:
                    await self._poll_once(client, session_factory)
                except Exception as exc:
                    log.error(
                        "Unexpected error in poll loop for event %d: %s",
                        self.event_id,
                        exc,
                    )

                # If stop was requested during poll, exit immediately
                if stop_event.is_set():
                    break

                try:
                    await asyncio.wait_for(
                        asyncio.shield(stop_event.wait()),
                        timeout=self.interval_seconds,
                    )
                    # stop_event was set during wait
                    break
                except asyncio.TimeoutError:
                    pass  # Normal — interval elapsed, loop again

    async def _poll_once(self, client: httpx.AsyncClient, session_factory) -> None:
        """One poll iteration: fetch, diff, insert, publish."""
        raw = await _fetch_results(client, self.event_id, self.dcat_id)
        if raw is None:
            return  # Error already logged; do not propagate to clients

        # Check if event is finished → auto-stop
        status = raw.get("status", "")
        if status == "finished":
            log.info("Event %d is finished — stopping poller", self.event_id)
            if self._stop_event:
                self._stop_event.set()
            return

        ranking = raw.get("ranking", [])
        if not ranking:
            return

        # Determine gender from d_cat name (fall back to M)
        dcat_info = raw.get("d_cat")
        dcat_name = dcat_info.get("name", "") if isinstance(dcat_info, dict) else ""
        gender = Gender.F if "women" in dcat_name.lower() else Gender.M

        session: Session = session_factory()
        try:
            event_db_id = self._resolve_event_db_id(session)
            if event_db_id is None:
                log.warning(
                    "Cannot resolve DB event for IFSC event_id=%d; skipping poll",
                    self.event_id,
                )
                return

            existing_keys = _current_db_keys(session, event_db_id)
            new_results: list[dict] = []

            for athlete_entry in ranking:
                firstname = str(athlete_entry.get("firstname", ""))
                lastname = str(athlete_entry.get("lastname", ""))
                country = str(athlete_entry.get("country", ""))

                athlete = _get_or_create_athlete(
                    session, firstname, lastname, country, gender
                )

                for rnd_data in athlete_entry.get("rounds", []):
                    round_name = rnd_data.get("round_name", "Unknown")
                    round_type = _parse_round_type(round_name)

                    rank = rnd_data.get("rank")
                    try:
                        rank_int = int(rank) if rank is not None else None
                    except (TypeError, ValueError):
                        log.warning(
                            "Non-integer rank %r for athlete %s in event %d; skipping",
                            rank,
                            athlete.id,
                            self.event_id,
                        )
                        continue

                    score_raw = str(rnd_data.get("score") or "").strip()

                    key = (athlete.id, round_type.value, rank_int, score_raw or None)
                    if key in existing_keys:
                        continue  # Already in DB — skip

                    # Insert new Result
                    rnd = _get_or_create_round(session, event_db_id, round_type, gender)

                    # Check for existing result (race condition guard)
                    existing_result = session.execute(
                        select(Result).where(
                            Result.round_id == rnd.id,
                            Result.athlete_id == athlete.id,
                        )
                    ).scalar_one_or_none()
                    if existing_result:
                        continue

                    result = Result(
                        round_id=rnd.id,
                        athlete_id=athlete.id,
                        rank=rank_int,
                        raw_score=score_raw or None,
                        score_normalized=None,  # Defer normalization until backfill
                        dnf=False,
                        dns=(rank_int is None),
                    )
                    session.add(result)
                    existing_keys.add(key)

                    athlete_name = f"{firstname} {lastname}".strip()
                    new_results.append(
                        {
                            "type": "new_result",
                            "athlete_id": athlete.id,
                            "name": athlete_name,
                            "rank": rank_int,
                            "score": score_raw or None,
                            "round_type": round_type.value,
                        }
                    )

            if new_results:
                session.commit()
                log.info(
                    "Inserted %d new results for event %d",
                    len(new_results),
                    self.event_id,
                )
                for payload in new_results:
                    await event_bus.publish(event_db_id, payload)
            else:
                session.rollback()

        except Exception as exc:
            session.rollback()
            log.error("DB error during poll for event %d: %s", self.event_id, exc)
        finally:
            session.close()


# ---------------------------------------------------------------------------
# Module-level poller registry (for SSE router + CLI)
# ---------------------------------------------------------------------------


def is_polling(event_id: int) -> bool:
    """Return True if a poller task is alive for event_id."""
    task = _running_tasks.get(event_id)
    return task is not None and not task.done()


async def start_polling(
    event_id: int,
    dcat_id: int,
    interval_seconds: int = 15,
    event_db_id: Optional[int] = None,
) -> bool:
    """Start a poller for event_id.

    Returns True if started, False if already running or file lock held.
    """
    if is_polling(event_id):
        log.info("Poller already running for event %d", event_id)
        return False

    fd = _acquire_file_lock(event_id)
    if fd is None:
        log.warning(
            "File lock held for event %d — another process is polling", event_id
        )
        return False

    stop_ev = asyncio.Event()
    _stop_events[event_id] = stop_ev

    poller = LivePoller(
        event_id=event_id,
        dcat_id=dcat_id,
        interval_seconds=interval_seconds,
        event_db_id=event_db_id,
    )

    async def _run_and_cleanup():
        try:
            await poller.run(stop_ev)
        finally:
            _release_file_lock(event_id, fd)
            _running_tasks.pop(event_id, None)
            _stop_events.pop(event_id, None)
            log.info("Poller stopped for event %d", event_id)

    task = asyncio.get_event_loop().create_task(_run_and_cleanup())
    _running_tasks[event_id] = task
    log.info(
        "Started poller for event %d (dcat %d, interval %ds)",
        event_id,
        dcat_id,
        interval_seconds,
    )
    return True


def stop_polling(event_id: int) -> None:
    """Gracefully stop the poller for event_id."""
    stop_ev = _stop_events.get(event_id)
    if stop_ev:
        stop_ev.set()
        log.info("Stop signal sent for event %d poller", event_id)
    else:
        log.info("No running poller found for event %d", event_id)
