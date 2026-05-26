"""Live event polling and SSE pub/sub infrastructure.

Architecture:
- LivePoller: async task that periodically fetches IFSC results and pushes new
  Result rows to the DB, broadcasting via asyncio.Queue to SSE subscribers.
- EventBus: in-memory pub/sub (one asyncio.Queue per subscriber per event_id).
- SSE endpoint (api/sse.py): subscribes to EventBus, streams events to browser.

Poller lifecycle:
  start_polling(event_id) → spawns asyncio Task if none running.
  stop_polling(event_id)  → signals graceful shutdown.
  is_polling(event_id)    → True while task is alive.

Mutex: /tmp/climbing_elo_poller_<event_id>.lock (file lock) prevents duplicate
pollers across multiple processes (e.g. two uvicorn workers, manual scripts).
"""

from climbing_elo.live.poller import (
    LivePoller,
    is_polling,
    start_polling,
    stop_polling,
)
from climbing_elo.live.bus import EventBus, event_bus

__all__ = [
    "LivePoller",
    "EventBus",
    "event_bus",
    "start_polling",
    "stop_polling",
    "is_polling",
]
