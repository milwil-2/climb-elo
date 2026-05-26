"""Server-Sent Events endpoint for live competition updates.

GET /live/{event_id}/stream
  - Returns text/event-stream
  - Each new Result is emitted as:
        data: {"type":"new_result","athlete_id":123,...}\n\n
  - Heartbeat every 30 s (comment line: ": heartbeat\n\n")
  - Auto-closes after 4 hours (max single-event duration)
  - 404 if event_id does not exist in DB
  - 429 if per-event concurrent connection cap (100) is exceeded

Security:
  - Read-only — no client→server mutations via SSE
  - event_id validated as int by FastAPI path param (no path traversal)
  - IFSC API errors never echoed to clients
  - Structured JSON payloads only (no raw IFSC HTML)
  - Concurrent connection cap: MAX_CONNECTIONS_PER_EVENT = 100
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from sqlalchemy import select

from climbing_elo.database import get_session_factory
from climbing_elo.live.bus import event_bus
from climbing_elo.models import Event

log = logging.getLogger(__name__)

router = APIRouter(tags=["live"])

# Max concurrent SSE connections per event
MAX_CONNECTIONS_PER_EVENT = 100

# Max SSE stream lifetime (seconds) — 4 hours
SSE_MAX_DURATION = 4 * 60 * 60

# Heartbeat interval (seconds)
SSE_HEARTBEAT_INTERVAL = 30

# In-memory per-event connection counter (approximate, single-process)
_connection_counts: dict[int, int] = defaultdict(int)


def _event_exists(event_id: int) -> bool:
    """Return True if event_id exists in the DB."""
    factory = get_session_factory()
    session = factory()
    try:
        row = session.execute(
            select(Event).where(Event.id == event_id)
        ).scalar_one_or_none()
        return row is not None
    finally:
        session.close()


async def _sse_generator(event_id: int, request: Request):
    """Async generator that yields SSE-formatted byte strings."""
    queue = await event_bus.subscribe(event_id)
    deadline = asyncio.get_event_loop().time() + SSE_MAX_DURATION

    try:
        while True:
            # Check 4-hour deadline
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                log.info("SSE stream for event %d reached 4h limit", event_id)
                break

            # Check client disconnect
            if await request.is_disconnected():
                log.debug("SSE client disconnected for event %d", event_id)
                break

            wait_time = min(SSE_HEARTBEAT_INTERVAL, remaining)
            try:
                payload = await asyncio.wait_for(queue.get(), timeout=wait_time)
                # Emit structured result event — no raw IFSC data passed through
                line = f"data: {json.dumps(payload)}\n\n"
                yield line.encode()
            except asyncio.TimeoutError:
                # Heartbeat
                yield b": heartbeat\n\n"

    finally:
        await event_bus.unsubscribe(event_id, queue)


@router.get(
    "/live/{event_id}/stream",
    summary="Live result stream (SSE)",
    description=(
        "Server-Sent Events stream for a live competition. "
        "Emits `new_result` events as scores are ingested. "
        "Heartbeat every 30s. Auto-closes after 4h or on disconnect. "
        "Max 100 concurrent connections per event."
    ),
)
async def live_stream(event_id: int, request: Request):
    # Validate event exists
    if not _event_exists(event_id):
        raise HTTPException(status_code=404, detail=f"Event {event_id} not found")

    # Enforce per-event connection cap
    if _connection_counts[event_id] >= MAX_CONNECTIONS_PER_EVENT:
        raise HTTPException(
            status_code=429,
            detail=f"Too many concurrent connections for event {event_id} (max {MAX_CONNECTIONS_PER_EVENT})",
        )

    _connection_counts[event_id] += 1
    log.info(
        "SSE connection opened for event %d (total=%d)",
        event_id,
        _connection_counts[event_id],
    )

    async def _guarded_generator():
        try:
            async for chunk in _sse_generator(event_id, request):
                yield chunk
        finally:
            _connection_counts[event_id] = max(0, _connection_counts[event_id] - 1)
            log.info(
                "SSE connection closed for event %d (total=%d)",
                event_id,
                _connection_counts[event_id],
            )

    return StreamingResponse(
        _guarded_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # Disable nginx buffering for proxied deployments
        },
    )
