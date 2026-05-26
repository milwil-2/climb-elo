#!/usr/bin/env python3
"""Health-check CLI for the IFSC results API.

Pings ifsc.results.info and exits 0 on success, 1 on failure.
Optionally POSTs a Discord-compatible alert on failure.

Usage:
    uv run python scripts/health_check_cli.py
    uv run python scripts/health_check_cli.py --quiet
    uv run python scripts/health_check_cli.py --webhook https://discord.com/api/webhooks/...

Security notes:
  - The webhook URL is accepted via CLI flag (pass from a secret, e.g. $DISCORD_WEBHOOK_URL).
  - The URL is never echoed to stdout/stderr.
  - Alerts are rate-limited: a sentinel file ($TMPDIR/health_check_last_alert) prevents
    more than one alert per hour even if checks fail every 30 min.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALERT_COOLDOWN_SECONDS = 3600  # 1 hour — max one Discord alert per hour
SENTINEL_FILE = Path(os.environ.get("TMPDIR", "/tmp")) / "health_check_last_alert"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _should_send_alert() -> bool:
    """Return True if the cooldown period has elapsed since the last alert."""
    try:
        if SENTINEL_FILE.exists():
            last_alert = float(SENTINEL_FILE.read_text().strip())
            elapsed = time.time() - last_alert
            if elapsed < ALERT_COOLDOWN_SECONDS:
                return False
    except (ValueError, OSError):
        pass
    return True


def _record_alert_sent() -> None:
    """Write current epoch time to the sentinel file."""
    try:
        SENTINEL_FILE.write_text(str(time.time()))
    except OSError:
        pass  # Non-fatal; worst case we send an extra alert


_ALLOWED_WEBHOOK_HOSTS = ("discord.com", "discordapp.com")


def _is_allowed_webhook_url(url: str) -> bool:
    """Restrict webhook target to Discord hosts to prevent SSRF."""
    from urllib.parse import urlparse
    try:
        parsed = urlparse(url)
    except ValueError:
        return False
    if parsed.scheme != "https":
        return False
    host = (parsed.hostname or "").lower()
    return host == "discord.com" or host.endswith(".discord.com") \
        or host == "discordapp.com" or host.endswith(".discordapp.com")


def _post_discord_alert(webhook_url: str, timestamp: str) -> None:
    """POST a Discord webhook message.  Never logs the URL itself."""
    if not _is_allowed_webhook_url(webhook_url):
        print(
            "[health_check] Refusing to POST: webhook URL is not a Discord host",
            file=sys.stderr,
        )
        return
    import httpx

    payload = {
        "embeds": [
            {
                "title": "IFSC API Health-Check FAILED",
                "description": (
                    "The `ifsc.results.info` API did not respond successfully.\n\n"
                    "Check the [GitHub Actions run]"
                    "(https://github.com/milwil-2/climb-elo/actions) for details."
                ),
                "color": 0xFF0000,
                "fields": [
                    {"name": "Timestamp (UTC)", "value": timestamp, "inline": True},
                    {"name": "Endpoint", "value": "ifsc.results.info/api/v1/", "inline": True},
                ],
                "footer": {"text": "climbing-elo health monitor"},
            }
        ]
    }

    try:
        with httpx.Client(timeout=10) as client:
            resp = client.post(
                webhook_url,
                content=json.dumps(payload),
                headers={"Content-Type": "application/json"},
            )
            resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        # Print error without including the URL
        print(f"[health_check] Warning: Discord alert POST failed: {exc}", file=sys.stderr)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ping the IFSC results API; exit 1 on failure."
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress all output (useful for cron).",
    )
    parser.add_argument(
        "--webhook",
        metavar="URL",
        default=None,
        help=(
            "Discord webhook URL to POST on failure. "
            "Pass via $DISCORD_WEBHOOK_URL in CI; never hardcode. "
            "Rate-limited to 1 alert per hour."
        ),
    )
    args = parser.parse_args()

    # Lazy import so the module path works regardless of how the script is invoked
    from climbing_elo.scraper.ifsc_api import health_check

    timestamp = _now_utc().strftime("%Y-%m-%d %H:%M:%S UTC")
    healthy = health_check()

    if not args.quiet:
        status_str = "HEALTHY" if healthy else "UNHEALTHY"
        print(f"[{timestamp}] IFSC API status: {status_str}")

    if not healthy:
        if args.webhook and _should_send_alert():
            _post_discord_alert(args.webhook, timestamp)
            _record_alert_sent()
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
