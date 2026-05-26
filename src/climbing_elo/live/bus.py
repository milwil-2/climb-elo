"""Asyncio pub/sub bus for live event updates.

Single global EventBus instance shared by the poller (publisher) and SSE
handlers (subscribers).  Pure in-memory — single process only (acceptable for
a single uvicorn worker deployment).

Usage:
    # Subscriber (SSE handler)
    queue = event_bus.subscribe(event_id)
    try:
        item = await asyncio.wait_for(queue.get(), timeout=30)
    finally:
        event_bus.unsubscribe(event_id, queue)

    # Publisher (poller)
    await event_bus.publish(event_id, {"type": "new_result", ...})
"""

from __future__ import annotations

import asyncio
import logging

log = logging.getLogger(__name__)


class EventBus:
    """In-process publish/subscribe for live result events."""

    def __init__(self) -> None:
        # Maps event_id → set of subscriber queues
        self._subscribers: dict[int, set[asyncio.Queue]] = {}
        self._lock = asyncio.Lock()

    async def subscribe(self, event_id: int) -> asyncio.Queue:
        """Return a new Queue that will receive payloads for event_id."""
        async with self._lock:
            if event_id not in self._subscribers:
                self._subscribers[event_id] = set()
            q: asyncio.Queue = asyncio.Queue(maxsize=256)
            self._subscribers[event_id].add(q)
            log.debug(
                "EventBus: subscriber added for event %d (total=%d)",
                event_id,
                len(self._subscribers[event_id]),
            )
            return q

    async def unsubscribe(self, event_id: int, queue: asyncio.Queue) -> None:
        """Remove a subscriber queue."""
        async with self._lock:
            subs = self._subscribers.get(event_id, set())
            subs.discard(queue)
            if not subs and event_id in self._subscribers:
                del self._subscribers[event_id]
            log.debug("EventBus: subscriber removed for event %d", event_id)

    async def publish(self, event_id: int, payload: dict) -> None:
        """Push payload to all subscribers of event_id (non-blocking, drops on full)."""
        async with self._lock:
            queues = list(self._subscribers.get(event_id, set()))
        for q in queues:
            try:
                q.put_nowait(payload)
            except asyncio.QueueFull:
                log.warning(
                    "EventBus: queue full for event %d — dropping payload", event_id
                )

    def subscriber_count(self, event_id: int) -> int:
        """Return the current number of subscribers for event_id (not async-safe, approximate)."""
        return len(self._subscribers.get(event_id, set()))


# Singleton shared across the process
event_bus = EventBus()
