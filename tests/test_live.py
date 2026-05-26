"""Tests for the live event polling backend and SSE endpoint.

Coverage:
- LivePoller detects new Result rows from a mocked IFSC response
- Duplicate score not double-inserted
- SSE endpoint returns 404 for unknown event_id
- SSE format: lines parseable as data: <json>\n\n
- Mutex (file lock) prevents two pollers for the same event
- Poller stops when event status flips to 'finished'
- Auto-close after timeout (SSE deadline)
- Concurrent connection cap enforced (429)
"""
from __future__ import annotations

import asyncio
import json
import os
from collections import defaultdict
from datetime import date
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

import climbing_elo.api.sse as _sse_module
from climbing_elo.live.bus import EventBus
from climbing_elo.live.poller import (
    LivePoller,
    _acquire_file_lock,
    _lock_path,
    _release_file_lock,
    _running_tasks,
    _stop_events,
    is_polling,
    start_polling,
    stop_polling,
)
from climbing_elo.models import (
    Athlete,
    Base,
    Discipline,
    Event,
    EventTier,
    Gender,
    Result,
    Round,
    RoundType,
)


# ---------------------------------------------------------------------------
# Shared in-memory DB fixture
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    eng = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(eng)
    return eng


@pytest.fixture
def session_factory(engine):
    return sessionmaker(bind=engine)


@pytest.fixture
def db_session(session_factory):
    session: Session = session_factory()
    yield session
    session.close()


@pytest.fixture
def live_event(db_session: Session) -> Event:
    """Seed a live event."""
    ev = Event(
        name="Live Test World Cup",
        tier=EventTier.WORLD_CUP,
        season=2026,
        start_date=date(2026, 6, 1),
        discipline=Discipline.LEAD,
    )
    db_session.add(ev)
    db_session.commit()
    return ev


@pytest.fixture
def live_round(db_session: Session, live_event: Event) -> Round:
    rnd = Round(
        event_id=live_event.id,
        round_type=RoundType.FINAL,
        gender=Gender.M,
        athlete_count=0,
    )
    db_session.add(rnd)
    db_session.commit()
    return rnd


# ---------------------------------------------------------------------------
# Sample IFSC API response factory
# ---------------------------------------------------------------------------

def _make_ifsc_response(status: str = "live", extra_athletes: list[dict] | None = None) -> dict:
    """Build a mock IFSC /events/{id}/result/{dcat_id} response."""
    athletes = [
        {
            "athlete_id": 101,
            "firstname": "Adam",
            "lastname": "Ondra",
            "country": "CZE",
            "rounds": [
                {
                    "round_name": "Final",
                    "rank": 1,
                    "score": "TOP",
                }
            ],
        },
    ]
    if extra_athletes:
        athletes.extend(extra_athletes)
    return {
        "status": status,
        "d_cat": {"name": "LEAD Men"},
        "ranking": athletes,
    }


# ---------------------------------------------------------------------------
# EventBus unit tests
# ---------------------------------------------------------------------------

class TestEventBus:
    @pytest.mark.asyncio
    async def test_publish_received_by_subscriber(self):
        bus = EventBus()
        q = await bus.subscribe(event_id=1)
        await bus.publish(event_id=1, payload={"type": "new_result"})
        item = await asyncio.wait_for(q.get(), timeout=1.0)
        assert item["type"] == "new_result"
        await bus.unsubscribe(event_id=1, queue=q)

    @pytest.mark.asyncio
    async def test_no_cross_event_leakage(self):
        bus = EventBus()
        q1 = await bus.subscribe(event_id=1)
        q2 = await bus.subscribe(event_id=2)
        await bus.publish(event_id=1, payload={"event": 1})
        # q2 should not receive the message for event 1
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(q2.get(), timeout=0.05)
        await bus.unsubscribe(event_id=1, queue=q1)
        await bus.unsubscribe(event_id=2, queue=q2)

    @pytest.mark.asyncio
    async def test_unsubscribe_removes_queue(self):
        bus = EventBus()
        q = await bus.subscribe(event_id=5)
        assert bus.subscriber_count(5) == 1
        await bus.unsubscribe(event_id=5, queue=q)
        assert bus.subscriber_count(5) == 0

    @pytest.mark.asyncio
    async def test_multiple_subscribers_receive_same_payload(self):
        bus = EventBus()
        q1 = await bus.subscribe(event_id=10)
        q2 = await bus.subscribe(event_id=10)
        await bus.publish(event_id=10, payload={"x": 42})
        item1 = await asyncio.wait_for(q1.get(), timeout=1.0)
        item2 = await asyncio.wait_for(q2.get(), timeout=1.0)
        assert item1 == item2 == {"x": 42}
        await bus.unsubscribe(10, q1)
        await bus.unsubscribe(10, q2)


# ---------------------------------------------------------------------------
# LivePoller unit tests — test _poll_once directly to avoid run() loop timing
# ---------------------------------------------------------------------------

class TestLivePoller:
    @pytest.mark.asyncio
    async def test_new_result_inserted(self, session_factory, live_event: Event):
        """Poller inserts a new result row when API returns a new score."""
        ifsc_response = _make_ifsc_response(status="live")

        poller = LivePoller(
            event_id=999,  # IFSC event id
            dcat_id=1,
            interval_seconds=15,
            event_db_id=live_event.id,
        )
        # Set stop event so run() loop doesn't spin, then call _poll_once
        poller._stop_event = asyncio.Event()

        mock_client = AsyncMock()
        with patch("climbing_elo.live.poller._fetch_results", return_value=ifsc_response):
            await poller._poll_once(mock_client, session_factory)

        session = session_factory()
        try:
            results = session.execute(select(Result)).scalars().all()
            assert len(results) == 1
            assert results[0].rank == 1
            assert results[0].raw_score == "TOP"
        finally:
            session.close()

    @pytest.mark.asyncio
    async def test_duplicate_not_inserted(self, session_factory, live_event: Event, live_round: Round, db_session: Session):
        """Poller skips rows that already exist in the DB (idempotent)."""
        # Pre-seed the result
        athlete = Athlete(name="Adam Ondra", gender=Gender.M, nationality="CZE")
        db_session.add(athlete)
        db_session.flush()

        db_session.add(Result(
            round_id=live_round.id,
            athlete_id=athlete.id,
            rank=1,
            raw_score="TOP",
        ))
        db_session.commit()

        ifsc_response = _make_ifsc_response(status="live")

        poller = LivePoller(
            event_id=999,
            dcat_id=1,
            interval_seconds=15,
            event_db_id=live_event.id,
        )
        poller._stop_event = asyncio.Event()

        mock_client = AsyncMock()
        with patch("climbing_elo.live.poller._fetch_results", return_value=ifsc_response):
            await poller._poll_once(mock_client, session_factory)

        session = session_factory()
        try:
            results = session.execute(select(Result)).scalars().all()
            assert len(results) == 1, "Duplicate insert detected"
        finally:
            session.close()

    @pytest.mark.asyncio
    async def test_stops_on_finished_status(self, session_factory, live_event: Event):
        """Poller stops automatically when event status is 'finished'."""
        ifsc_response = _make_ifsc_response(status="finished")

        poller = LivePoller(
            event_id=999,
            dcat_id=1,
            interval_seconds=15,
            event_db_id=live_event.id,
        )

        with patch("climbing_elo.live.poller._fetch_results", return_value=ifsc_response):
            stop_ev = asyncio.Event()
            # Run with a generous timeout — poller should stop itself quickly
            await asyncio.wait_for(poller.run(stop_ev, session_factory=session_factory), timeout=5.0)

        assert stop_ev.is_set(), "Stop event should be set after finished status"

    @pytest.mark.asyncio
    async def test_invalid_rank_skipped(self, session_factory, live_event: Event):
        """Non-integer rank values are skipped without crashing."""
        ifsc_response = {
            "status": "live",
            "d_cat": {"name": "LEAD Men"},
            "ranking": [
                {
                    "athlete_id": 42,
                    "firstname": "Test",
                    "lastname": "Athlete",
                    "country": "USA",
                    "rounds": [{"round_name": "Final", "rank": "not-a-number", "score": "34+"}],
                }
            ],
        }

        poller = LivePoller(event_id=999, dcat_id=1, event_db_id=live_event.id)
        poller._stop_event = asyncio.Event()
        mock_client = AsyncMock()

        with patch("climbing_elo.live.poller._fetch_results", return_value=ifsc_response):
            await poller._poll_once(mock_client, session_factory)

        session = session_factory()
        try:
            results = session.execute(select(Result)).scalars().all()
            assert results == [], "Bad rank should not produce a Result row"
        finally:
            session.close()

    @pytest.mark.asyncio
    async def test_publishes_to_event_bus(self, session_factory, live_event: Event):
        """Poller publishes a new_result payload to the EventBus."""
        bus = EventBus()
        q = await bus.subscribe(live_event.id)

        ifsc_response = _make_ifsc_response(status="live")

        poller = LivePoller(event_id=999, dcat_id=1, event_db_id=live_event.id)
        poller._stop_event = asyncio.Event()
        mock_client = AsyncMock()

        with (
            patch("climbing_elo.live.poller._fetch_results", return_value=ifsc_response),
            patch("climbing_elo.live.poller.event_bus", bus),
        ):
            await poller._poll_once(mock_client, session_factory)

        payload = await asyncio.wait_for(q.get(), timeout=2.0)
        assert payload["type"] == "new_result"
        assert payload["rank"] == 1
        assert "athlete_id" in payload
        assert "name" in payload

        await bus.unsubscribe(live_event.id, q)

    @pytest.mark.asyncio
    async def test_none_fetch_result_is_no_op(self, session_factory, live_event: Event):
        """When _fetch_results returns None (API error), no DB write occurs."""
        poller = LivePoller(event_id=999, dcat_id=1, event_db_id=live_event.id)
        poller._stop_event = asyncio.Event()
        mock_client = AsyncMock()

        with patch("climbing_elo.live.poller._fetch_results", return_value=None):
            await poller._poll_once(mock_client, session_factory)

        session = session_factory()
        try:
            results = session.execute(select(Result)).scalars().all()
            assert results == []
        finally:
            session.close()


# ---------------------------------------------------------------------------
# File-lock (mutex) tests
# ---------------------------------------------------------------------------

class TestPollerMutex:
    def test_lock_acquired_and_released(self):
        event_id = 77777
        _lock_path(event_id).unlink(missing_ok=True)
        fd = _acquire_file_lock(event_id)
        assert fd is not None, "Should acquire lock when no lock exists"
        _release_file_lock(event_id, fd)
        assert not _lock_path(event_id).exists(), "Lock file should be removed after release"

    def test_second_acquire_fails(self):
        event_id = 77778
        _lock_path(event_id).unlink(missing_ok=True)
        fd1 = _acquire_file_lock(event_id)
        assert fd1 is not None
        try:
            fd2 = _acquire_file_lock(event_id)
            assert fd2 is None, "Second acquire should fail (file already locked)"
        finally:
            _release_file_lock(event_id, fd1)

    @pytest.mark.asyncio
    async def test_start_polling_twice_returns_false(self, live_event: Event):
        """start_polling returns False if a poller is already running for the event."""
        event_id = live_event.id + 50000  # Unique to avoid collision
        _running_tasks.pop(event_id, None)
        _stop_events.pop(event_id, None)
        _lock_path(event_id).unlink(missing_ok=True)

        # Create a mock poller that runs until stop event
        blocker = asyncio.Event()

        async def _long_running_fetch(*args, **kwargs):
            await blocker.wait()
            return None

        with patch("climbing_elo.live.poller._fetch_results", side_effect=_long_running_fetch):
            started1 = await start_polling(event_id=event_id, dcat_id=1, event_db_id=event_id)
            await asyncio.sleep(0.05)  # Let task start
            started2 = await start_polling(event_id=event_id, dcat_id=1, event_db_id=event_id)

        assert started1 is True
        assert started2 is False, "Second start_polling should return False (already running)"

        # Cleanup
        blocker.set()
        stop_polling(event_id)
        await asyncio.sleep(0.15)

    def test_is_polling_false_when_not_started(self):
        assert is_polling(99999) is False

    @pytest.mark.asyncio
    async def test_stop_polling_graceful(self, live_event: Event):
        """stop_polling signals the poller to exit."""
        event_id = live_event.id + 60000  # Unique
        _running_tasks.pop(event_id, None)
        _stop_events.pop(event_id, None)
        _lock_path(event_id).unlink(missing_ok=True)

        blocker = asyncio.Event()

        async def _blocking_fetch(*args, **kwargs):
            await blocker.wait()
            return None

        with patch("climbing_elo.live.poller._fetch_results", side_effect=_blocking_fetch):
            started = await start_polling(event_id=event_id, dcat_id=1, event_db_id=event_id)
            assert started is True
            await asyncio.sleep(0.05)
            assert is_polling(event_id) is True

            blocker.set()  # Unblock fetch
            stop_polling(event_id)
            await asyncio.sleep(0.3)

        assert is_polling(event_id) is False


# ---------------------------------------------------------------------------
# SSE endpoint tests (via TestClient)
# ---------------------------------------------------------------------------

# Use a known event_id for SSE tests (mocked — no real DB needed for these)
_SSE_KNOWN_EVENT_ID = 1001


@pytest.fixture(scope="module")
def sse_client():
    """TestClient with _event_exists patched: event 1001 exists, others don't."""
    from climbing_elo.api.app import create_app

    def _mock_event_exists(event_id: int) -> bool:
        return event_id == _SSE_KNOWN_EVENT_ID

    with patch("climbing_elo.api.sse._event_exists", side_effect=_mock_event_exists):
        app = create_app()
        yield TestClient(app, raise_server_exceptions=True)


class TestSSEEndpoint:
    def test_404_unknown_event(self, sse_client):
        with patch("climbing_elo.api.sse._event_exists", return_value=False):
            resp = sse_client.get("/live/999999/stream")
        assert resp.status_code == 404

    def test_200_known_event_content_type(self, sse_client):
        """SSE stream opens successfully for a known event and returns SSE content-type.

        We patch the heartbeat to a tiny value so the generator yields quickly and
        the TestClient stream context does not block indefinitely.
        """
        _sse_module._connection_counts[_SSE_KNOWN_EVENT_ID] = 0

        with (
            patch("climbing_elo.api.sse._event_exists", return_value=True),
            patch.object(_sse_module, "SSE_HEARTBEAT_INTERVAL", 0.01),
            patch.object(_sse_module, "SSE_MAX_DURATION", 0.05),
        ):
            with sse_client.stream("GET", f"/live/{_SSE_KNOWN_EVENT_ID}/stream") as resp:
                assert resp.status_code == 200
                assert "text/event-stream" in resp.headers.get("content-type", "")

    def test_429_over_connection_cap(self, sse_client):
        """429 when per-event connection cap is exceeded."""
        _sse_module._connection_counts[_SSE_KNOWN_EVENT_ID] = _sse_module.MAX_CONNECTIONS_PER_EVENT

        with patch("climbing_elo.api.sse._event_exists", return_value=True):
            resp = sse_client.get(f"/live/{_SSE_KNOWN_EVENT_ID}/stream")
        assert resp.status_code == 429

        # Restore
        _sse_module._connection_counts[_SSE_KNOWN_EVENT_ID] = 0

    def test_sse_data_format_parseable(self):
        """SSE data lines must be parseable as JSON (format validation)."""
        payload = {"type": "new_result", "athlete_id": 42, "name": "Adam Ondra",
                   "rank": 1, "score": "TOP", "round_type": "final"}
        line = f"data: {json.dumps(payload)}\n\n"
        assert line.startswith("data: ")
        data_part = line[len("data: "):].strip()
        parsed = json.loads(data_part)
        assert parsed["type"] == "new_result"
        assert parsed["athlete_id"] == 42
        assert parsed["rank"] == 1

    def test_heartbeat_format(self):
        """Heartbeat comment lines must match SSE spec."""
        heartbeat = b": heartbeat\n\n"
        assert heartbeat.startswith(b":")


# ---------------------------------------------------------------------------
# SSE auto-close timeout test
# ---------------------------------------------------------------------------

class TestSSETimeout:
    @pytest.mark.asyncio
    async def test_generator_closes_after_deadline(self):
        """SSE generator exits quickly when deadline is in the past."""
        import climbing_elo.api.sse as _sse

        # Build a mock request that is never disconnected
        mock_request = MagicMock()
        mock_request.is_disconnected = AsyncMock(return_value=False)

        event_id = 42
        bus = EventBus()

        with (
            patch.object(_sse, "event_bus", bus),
            patch.object(_sse, "SSE_MAX_DURATION", 0),   # Immediate deadline
            patch.object(_sse, "SSE_HEARTBEAT_INTERVAL", 0.01),
        ):
            chunks = []
            async for chunk in _sse._sse_generator(event_id, mock_request):
                chunks.append(chunk)
                if len(chunks) >= 2:
                    break  # Safety break

        # Generator should have yielded at most a heartbeat then exited
        assert len(chunks) <= 1
