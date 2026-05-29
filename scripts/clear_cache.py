"""Manual cache invalidation helper.

Usage::

    uv run python scripts/clear_cache.py

Clears all in-memory caches: ``predictions_cache`` (Monte Carlo results),
``likely_roster_cache`` (likely-competitor lookups), and ``html_page_cache``
(the landing / leaderboard / athletes response cache, Issue #97).  Useful after
a scrape run when you want immediate freshness without waiting for each cache's
TTL to expire.

Note: This only works when the web server and this script share the same Python
process (i.e. single-worker deployments).  In a multi-worker setup (gunicorn
with multiple workers) each worker holds its own in-memory cache; restart the
workers instead, or migrate to a shared cache backend (Issue #29).
"""

from __future__ import annotations

from climbing_elo.cache import (
    html_page_cache,
    likely_roster_cache,
    predictions_cache,
)

_CACHES = {
    "predictions_cache": predictions_cache,
    "likely_roster_cache": likely_roster_cache,
    "html_page_cache": html_page_cache,
}


def main() -> None:
    total = 0
    for name, cache in _CACHES.items():
        before = len(cache)
        cache.clear()
        total += before
        print(f"  {name}: removed {before} entr{'y' if before == 1 else 'ies'}")
    print(f"Cache cleared — removed {total} entr{'y' if total == 1 else 'ies'} total.")


if __name__ == "__main__":
    main()
