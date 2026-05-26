"""CLI for manually starting a live event poller.

Usage:
    uv run python scripts/live_poll.py --event-id 1234 --dcat-id 567
    uv run python scripts/live_poll.py --event-id 1234 --dcat-id 567 --interval 30
    uv run python scripts/live_poll.py --event-id 1234 --dcat-id 567 --event-db-id 99

Arguments:
    --event-id     IFSC API event ID (used in /api/v1/events/{id}/result/{dcat_id})
    --dcat-id      Discipline-category ID (from IFSC API)
    --interval     Poll interval in seconds (default: 15)
    --event-db-id  Our internal DB Event.id (if different from --event-id)

Press Ctrl+C to stop.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys

# Ensure src is on the path when run directly
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from climbing_elo.live.poller import LivePoller

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("live_poll")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Start a live IFSC event poller (manual ops / testing)."
    )
    parser.add_argument("--event-id", type=int, required=True, help="IFSC API event ID")
    parser.add_argument(
        "--dcat-id", type=int, required=True, help="IFSC API discipline-category ID"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=15,
        help="Poll interval in seconds (default: 15)",
    )
    parser.add_argument(
        "--event-db-id",
        type=int,
        default=None,
        help="Internal DB Event.id if different from --event-id",
    )
    return parser.parse_args()


async def main() -> None:
    args = parse_args()

    log.info(
        "Starting live poller — event_id=%d  dcat_id=%d  interval=%ds",
        args.event_id,
        args.dcat_id,
        args.interval,
    )

    stop_event = asyncio.Event()

    poller = LivePoller(
        event_id=args.event_id,
        dcat_id=args.dcat_id,
        interval_seconds=args.interval,
        event_db_id=args.event_db_id,
    )

    # Graceful shutdown on Ctrl+C
    loop = asyncio.get_event_loop()
    try:
        import signal

        def _on_sigint(*_):
            log.info("Interrupt received — stopping poller…")
            stop_event.set()

        loop.add_signal_handler(signal.SIGINT, _on_sigint)
        loop.add_signal_handler(signal.SIGTERM, _on_sigint)
    except (NotImplementedError, AttributeError):
        # Windows fallback
        pass

    await poller.run(stop_event)
    log.info("Poller finished.")


if __name__ == "__main__":
    asyncio.run(main())
