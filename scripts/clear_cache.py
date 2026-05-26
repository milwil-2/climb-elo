"""Manual cache invalidation helper.

Usage::

    uv run python scripts/clear_cache.py

This imports the global ``predictions_cache`` singleton and calls ``clear()``.
It is useful after a scrape run when you want immediate freshness on the
/predictions page without waiting for the 1-hour TTL to expire.

Note: This only works when the web server and this script share the same Python
process (i.e. single-worker deployments).  In a multi-worker setup (gunicorn
with multiple workers) each worker holds its own in-memory cache; restart the
workers instead, or migrate to a shared cache backend (Issue #29).
"""

from __future__ import annotations

from climbing_elo.cache import predictions_cache


def main() -> None:
    before = len(predictions_cache)
    predictions_cache.clear()
    print(f"Cache cleared — removed {before} entr{'y' if before == 1 else 'ies'}.")


if __name__ == "__main__":
    main()
